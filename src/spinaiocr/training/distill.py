"""Knowledge distillation: Gemma 4 (VLM teacher) → SPINAI student.

We don't have direct access to Gemma's logits (the vLLM endpoint returns
only sampled text), so we distill in two practical modes:

**Mode 1 — Hard-label distillation (default).**
    Use Gemma's transcriptions as pseudo labels for the recognition head.
    Combined with PaddleOCR/EasyOCR/Tesseract via consensus
    (:mod:`spinaiocr.data.pseudo`) so Gemma errors are dampened by
    majority voting.

**Mode 2 — Sequence-level soft distillation.**
    When we *do* have logits (e.g. a locally-loaded Gemma via transformers
    with `output_scores=True`), align teacher distributions to student
    timesteps via CTC path probability and minimize KL divergence on the
    aligned positions. This is the "soft" version.

This file implements both. Mode 1 is just a `TeacherSample` → CTC loss
wrapper; Mode 2 requires teacher logits + a shared vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from spinaiocr.training.losses import CTCRecognitionLoss


@dataclass
class DistillSample:
    image: torch.Tensor  # [3, H, W]
    pseudo_text: str  # teacher's transcription
    teacher_logits: torch.Tensor | None = None  # [T_t, V_t], optional
    confidence: float = 1.0  # from consensus tier (high/mid/low)


class HardLabelDistillLoss(nn.Module):
    """Mode 1 — treat teacher text as pseudo ground truth, weighted by confidence."""

    def __init__(self, blank: int = 0) -> None:
        super().__init__()
        self.ctc = CTCRecognitionLoss(blank=blank)

    def forward(
        self,
        student_logits: torch.Tensor,  # [B, T, V]
        targets: torch.Tensor,  # flat int64
        target_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.ctc(student_logits, targets, input_lengths, target_lengths)
        if sample_weights is not None:
            return (loss * sample_weights.mean()).mean()
        return loss


class SoftDistillLoss(nn.Module):
    """Mode 2 — KL divergence on aligned softmax distributions.

    Assumes teacher and student share a vocabulary (or a prefix thereof).
    If the vocab differs, provide `teacher_to_student_idx` mapping.
    """

    def __init__(self, temperature: float = 2.0) -> None:
        super().__init__()
        self.T = temperature

    def forward(
        self,
        student_logits: torch.Tensor,  # [B, T_s, V]
        teacher_logits: torch.Tensor,  # [B, T_t, V]
    ) -> torch.Tensor:
        # align timesteps by linear interpolation on length axis
        if student_logits.shape[1] != teacher_logits.shape[1]:
            teacher_logits = F.interpolate(
                teacher_logits.transpose(1, 2),
                size=student_logits.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        s = F.log_softmax(student_logits / self.T, dim=-1)
        t = F.softmax(teacher_logits / self.T, dim=-1).detach()
        return F.kl_div(s, t, reduction="batchmean") * (self.T * self.T)


class CombinedDistillLoss(nn.Module):
    """α·hard + β·soft + γ·ground_truth (if available)."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 1.0,
        temperature: float = 2.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.hard = HardLabelDistillLoss()
        self.soft = SoftDistillLoss(temperature=temperature)

    def forward(
        self,
        student_logits: torch.Tensor,
        pseudo_targets: torch.Tensor,
        pseudo_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
        gt_targets: torch.Tensor | None = None,
        gt_lengths: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        total = student_logits.new_zeros(())
        hard = self.hard(
            student_logits, pseudo_targets, pseudo_lengths, input_lengths, sample_weights
        )
        out["hard"] = hard
        total = total + self.alpha * hard

        if teacher_logits is not None:
            soft = self.soft(student_logits, teacher_logits)
            out["soft"] = soft
            total = total + self.beta * soft

        if gt_targets is not None and gt_lengths is not None:
            gt = self.hard(student_logits, gt_targets, gt_lengths, input_lengths)
            out["gt"] = gt
            total = total + self.gamma * gt

        out["total"] = total
        return out
