"""PyTorch Lightning training modules for detection and recognition.

- :class:`DetectionModule` wraps DBNet + DBNetLoss + Mean-Teacher consistency.
- :class:`RecognitionModule` wraps CRNNLite/SVTRLite + CTC loss.

Both support semi-supervised batches via the ``unlabeled`` key (Mean Teacher
consistency regularization with entropy-based uncertainty weighting).

Adapted from the medical imaging `train.py` structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lightning as L
    LightningModule = L.LightningModule
except ImportError:  # pragma: no cover — lightning is in `[train]` extras
    L = None  # type: ignore
    LightningModule = nn.Module  # type: ignore

from spinai_ocr.models.detection import DBNet
from spinai_ocr.models.recognition import build_recognition
from spinai_ocr.training.losses import (
    CTCRecognitionLoss,
    ConsistencyLoss,
    DBNetLoss,
)
from spinai_ocr.training.mean_teacher import MeanTeacher
from spinai_ocr.training.uncertainty import (
    entropy_weights,
    pixel_entropy,
    timestep_entropy,
)


# ---------------------------------------------------------------------------
# Shared schedule helpers
# ---------------------------------------------------------------------------


@dataclass
class OptimCfg:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_steps: int = 2000
    total_steps: int = 100_000


def _make_optimizer(params, cfg: OptimCfg) -> torch.optim.Optimizer:
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    raise ValueError(cfg.optimizer)


def _make_scheduler(optimizer: torch.optim.Optimizer, cfg: OptimCfg):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(cfg.total_steps - cfg.warmup_steps, 1)
        )
    if cfg.scheduler == "poly":
        return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=cfg.total_steps, power=0.9)
    return None


# ---------------------------------------------------------------------------
# Detection (DBNet)
# ---------------------------------------------------------------------------


class DetectionModule(LightningModule):
    def __init__(
        self,
        backbone: str = "resnet18",
        optim_cfg: OptimCfg | None = None,
        use_mean_teacher: bool = True,
        consistency_weight: float = 1.0,
        uncertainty_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        if L is None:
            raise ImportError("lightning not installed. pip install -e '.[train]'")
        self.save_hyperparameters()
        self.student = DBNet(backbone=backbone)
        self.loss_fn = DBNetLoss()
        self.use_mt = use_mean_teacher
        if use_mean_teacher:
            self.mt = MeanTeacher(self.student, total_steps=(optim_cfg or OptimCfg()).total_steps)
        self.consistency = ConsistencyLoss(kind="mse")
        self.consistency_weight = consistency_weight
        self.uncertainty_alpha = uncertainty_alpha
        self.optim_cfg = optim_cfg or OptimCfg()

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        sup = batch["labeled"]
        out = self.student(sup["image"])
        sup_losses = self.loss_fn(out, sup)
        total = sup_losses["total"]
        self.log_dict({f"train/sup_{k}": v for k, v in sup_losses.items()}, prog_bar=False)

        if self.use_mt and "unlabeled" in batch:
            unl = batch["unlabeled"]
            with torch.no_grad():
                self.mt.teacher.eval()
                teacher_out = self.mt.teacher(unl["image"])
            student_out = self.student(unl["image"])
            # consistency on prob map, weighted by teacher confidence
            t_prob = teacher_out["prob"]
            s_prob = student_out["prob"]
            weight = entropy_weights(pixel_entropy(t_prob), self.uncertainty_alpha)
            cons = F.mse_loss(s_prob * weight, t_prob.detach() * weight)
            total = total + self.consistency_weight * cons
            self.log("train/consistency", cons, prog_bar=True)
            self.mt.update_teacher()

        self.log("train/total", total, prog_bar=True)
        return total

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        out = self.student(batch["image"])
        loss = self.loss_fn(out, batch)["total"]
        self.log("val/total", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = _make_optimizer(self.student.parameters(), self.optim_cfg)
        sch = _make_scheduler(opt, self.optim_cfg)
        if sch is None:
            return opt
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}


# ---------------------------------------------------------------------------
# Recognition (CRNN / SVTR)
# ---------------------------------------------------------------------------


class RecognitionModule(LightningModule):
    def __init__(
        self,
        arch: str = "svtr_lite",
        vocab_size: int = 11300,
        input_height: int = 48,
        optim_cfg: OptimCfg | None = None,
        use_mean_teacher: bool = True,
        consistency_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if L is None:
            raise ImportError("lightning not installed. pip install -e '.[train]'")
        self.save_hyperparameters()
        self.student = build_recognition(arch, vocab_size=vocab_size, input_height=input_height)
        self.ctc = CTCRecognitionLoss(blank=0)
        self.use_mt = use_mean_teacher
        if use_mean_teacher:
            self.mt = MeanTeacher(self.student, total_steps=(optim_cfg or OptimCfg()).total_steps)
        self.consistency = ConsistencyLoss(kind="kl")
        self.consistency_weight = consistency_weight
        self.optim_cfg = optim_cfg or OptimCfg()

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        sup = batch["labeled"]
        logits = self.student(sup["image"])  # [B, T, V]
        b, t, _ = logits.shape
        input_lengths = torch.full((b,), t, dtype=torch.long, device=logits.device)
        loss = self.ctc(logits, sup["targets"], input_lengths, sup["target_lengths"])
        total = loss
        self.log("train/ctc", loss, prog_bar=True)

        if self.use_mt and "unlabeled" in batch:
            unl = batch["unlabeled"]
            with torch.no_grad():
                self.mt.teacher.eval()
                t_logits = self.mt.teacher(unl["image"])
            s_logits = self.student(unl["image"])
            # Down-weight uncertain teacher timesteps
            weight = entropy_weights(timestep_entropy(t_logits)).unsqueeze(-1)
            cons = F.mse_loss(
                F.softmax(s_logits, -1) * weight,
                F.softmax(t_logits, -1).detach() * weight,
            )
            total = total + self.consistency_weight * cons
            self.log("train/consistency", cons, prog_bar=True)
            self.mt.update_teacher()

        return total

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self.student(batch["image"])
        b, t, _ = logits.shape
        input_lengths = torch.full((b,), t, dtype=torch.long, device=logits.device)
        loss = self.ctc(logits, batch["targets"], input_lengths, batch["target_lengths"])
        self.log("val/ctc", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = _make_optimizer(self.student.parameters(), self.optim_cfg)
        sch = _make_scheduler(opt, self.optim_cfg)
        if sch is None:
            return opt
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}
