"""LightningDataModules for detection and recognition.

Wire these into `training/train.py` to actually call `trainer.fit(module, dm)`.

Detection batch shape:
    {"image": [B,3,H,W], "prob_map": ..., "prob_mask": ..., "thresh_map": ..., "thresh_mask": ...}

Recognition batch shape:
    {"image": [B,3,H,W], "targets": [sum_target_lengths], "target_lengths": [B], "input_lengths": [B]}

Both modules yield an optional `unlabeled` key for Mean Teacher training
when `unlabeled_root` is provided.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import lightning as L
    LDM = L.LightningDataModule
except ImportError:
    L = None  # type: ignore
    LDM = object  # type: ignore

from torch.utils.data import DataLoader, Dataset

from spinai_ocr.data.dataset import DetectionDataset, RecognitionDataset
from spinai_ocr.data.detection_augment import DetectionAugment, default_detection_augment
from spinai_ocr.training.db_gt import DBGTConfig, build_dbnet_targets
from spinai_ocr.vocab.base import Vocab, load_vocab


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _letterbox(image: np.ndarray, target: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = image.shape[:2]
    scale = target / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((target, target, 3), 255, dtype=np.uint8)
    padded[:nh, :nw] = resized
    return padded, scale, (nh, nw)


class DetectionTrainingDataset(Dataset):
    """Wraps DetectionDataset with augmentation + DBNet GT generation."""

    def __init__(
        self,
        root: str | Path,
        label_file: str | Path,
        target_size: int = 640,
        augment: DetectionAugment | None = None,
        gt_cfg: DBGTConfig | None = None,
    ) -> None:
        self.base = DetectionDataset(root, label_file)
        self.target_size = target_size
        self.augment = augment
        self.gt_cfg = gt_cfg or DBGTConfig()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.base[idx]
        img = (sample.image * 255).astype(np.uint8)
        polys = list(sample.polygons)
        texts = list(sample.texts)
        if self.augment is not None:
            img, polys, texts = self.augment(img, polys, texts)
        img, scale, _ = _letterbox(img, self.target_size)
        polys = [p * scale for p in polys]
        ignore = [t == "###" for t in texts]
        targets = build_dbnet_targets(
            (self.target_size, self.target_size), polys, ignore, self.gt_cfg
        )
        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return {
            "image": img_t,
            "prob_map": torch.from_numpy(targets["prob_map"]).unsqueeze(0),
            "prob_mask": torch.from_numpy(targets["prob_mask"]).unsqueeze(0),
            "thresh_map": torch.from_numpy(targets["thresh_map"]).unsqueeze(0),
            "thresh_mask": torch.from_numpy(targets["thresh_mask"]).unsqueeze(0),
        }


@dataclass
class DetectionDMCfg:
    train_root: str
    train_labels: str
    val_root: str
    val_labels: str
    unlabeled_root: str | None = None
    unlabeled_labels: str | None = None  # optional dummy labels file (bbox not needed)
    target_size: int = 640
    batch_size: int = 16
    num_workers: int = 4


class DetectionDataModule(LDM):  # type: ignore[misc]
    def __init__(self, cfg: DetectionDMCfg) -> None:
        if L is None:
            raise ImportError("pip install -e '.[train]'")
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = DetectionTrainingDataset(
            self.cfg.train_root,
            self.cfg.train_labels,
            target_size=self.cfg.target_size,
            augment=default_detection_augment(),
        )
        self.val_ds = DetectionTrainingDataset(
            self.cfg.val_root,
            self.cfg.val_labels,
            target_size=self.cfg.target_size,
            augment=None,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------


class RecognitionTrainingDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        label_file: str | Path,
        vocab: Vocab,
        image_height: int = 48,
        max_width: int = 320,
        augment=None,
    ) -> None:
        # _76 (iter 20): `augment` is an OCRAugment callable applied to the
        # uint8 RGB crop before tensor conversion. Keeps the existing None
        # default so callers that don't opt in are unchanged.
        self.base = RecognitionDataset(
            root, label_file, vocab=vocab, image_height=image_height, max_width=max_width
        )
        self.augment = augment

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        sample = self.base[idx]
        # RecognitionDataset returns HxWx3 float32 in [0, 1] (already /255).
        # augment callables in data/augment.py were written for uint8 [0, 255]
        # (cv2.cvtColor, cv2.erode, cv2.imencode all expect uint8). Convert
        # in/out around the augment call. Bug caught in iter 20 — raw float
        # passed to cv2 corrupted data and blew warm-start loss 0.5 -> 125.
        img = sample.image
        if self.augment is not None:
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            img_u8 = self.augment(img_u8)
            img = img_u8.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        targets = self.base.vocab.encode(sample.text)
        return {
            "image": img_t,
            "target": torch.tensor(targets, dtype=torch.long),
            "target_length": len(targets),
        }


def recognition_collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    # pad to max H and max W in batch (MSR produces variable-height images)
    max_w = max(b["image"].shape[2] for b in batch)
    max_h = max(b["image"].shape[1] for b in batch)
    imgs = torch.ones(len(batch), 3, max_h, max_w)
    for i, b in enumerate(batch):
        h = b["image"].shape[1]
        w = b["image"].shape[2]
        imgs[i, :, :h, :w] = b["image"]
    targets = torch.cat([b["target"] for b in batch])
    target_lengths = torch.tensor([b["target_length"] for b in batch], dtype=torch.long)
    return {"image": imgs, "targets": targets, "target_lengths": target_lengths}


@dataclass
class RecognitionDMCfg:
    train_root: str
    train_labels: str
    val_root: str
    val_labels: str
    vocab: str = "ko_en_v1"
    image_height: int = 48
    max_width: int = 320
    batch_size: int = 128
    num_workers: int = 4


class RecognitionDataModule(LDM):  # type: ignore[misc]
    def __init__(self, cfg: RecognitionDMCfg) -> None:
        if L is None:
            raise ImportError("pip install -e '.[train]'")
        super().__init__()
        self.cfg = cfg
        self.vocab = load_vocab(cfg.vocab)

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = RecognitionTrainingDataset(
            self.cfg.train_root,
            self.cfg.train_labels,
            self.vocab,
            image_height=self.cfg.image_height,
            max_width=self.cfg.max_width,
        )
        self.val_ds = RecognitionTrainingDataset(
            self.cfg.val_root,
            self.cfg.val_labels,
            self.vocab,
            image_height=self.cfg.image_height,
            max_width=self.cfg.max_width,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=recognition_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=recognition_collate,
        )
