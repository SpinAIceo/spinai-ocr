"""Test-time augmentation and model ensembling.

Two strategies:

1. **TTA (single model, multiple inputs)**: pass an image at several scales /
   tiny rotations; average the softmax logits. Improves CER on noisy inputs
   by 0.5-2% without retraining.

2. **Ensemble (multiple models, same input)**: average logits across
   checkpoints trained with different seeds / architectures.

Both produce `[T, V]` logits, which then go through any decoder in
`spinai_ocr.inference.decoders`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TTAConfig:
    scales: tuple[float, ...] = (1.0, 1.1, 0.9)
    rotations_deg: tuple[float, ...] = (-2.0, 0.0, 2.0)
    mirror: bool = False        # keep False for text — flipping text hurts
    weights: tuple[float, ...] | None = None  # per-view weight; None = equal


def _rotate_tensor(x: torch.Tensor, deg: float) -> torch.Tensor:
    if deg == 0:
        return x
    theta = torch.tensor([
        [np.cos(np.deg2rad(deg)), -np.sin(np.deg2rad(deg)), 0],
        [np.sin(np.deg2rad(deg)),  np.cos(np.deg2rad(deg)), 0],
    ], dtype=x.dtype, device=x.device).unsqueeze(0).repeat(x.size(0), 1, 1)
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="border")


def _scale_tensor(x: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 1.0:
        return x
    h, w = x.shape[-2:]
    new_h, new_w = max(int(h * scale), 8), max(int(w * scale), 8)
    y = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)


@torch.no_grad()
def tta_predict(
    model: torch.nn.Module,
    x: torch.Tensor,  # [B, 3, H, W]
    cfg: TTAConfig | None = None,
) -> torch.Tensor:
    """Return averaged softmax probabilities [B, T, V].

    Caller can take `.log()` for beam search or `.argmax(-1)` for greedy.
    """
    cfg = cfg or TTAConfig()
    views: list[torch.Tensor] = []
    for s in cfg.scales:
        sx = _scale_tensor(x, s)
        for r in cfg.rotations_deg:
            rx = _rotate_tensor(sx, r)
            views.append(rx)
            if cfg.mirror:
                views.append(torch.flip(rx, dims=[-1]))
    weights = cfg.weights or [1.0] * len(views)
    weights = np.asarray(weights, dtype=np.float32) / sum(weights)
    model.eval()
    agg = None
    for w, v in zip(weights, views):
        out = model(v)
        p = F.softmax(out, dim=-1)
        agg = p * w if agg is None else agg + p * w
    return agg  # [B, T, V]


@torch.no_grad()
def ensemble_predict(
    models: list[torch.nn.Module],
    x: torch.Tensor,
    weights: list[float] | None = None,
) -> torch.Tensor:
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    assert len(weights) == len(models)
    agg = None
    for w, m in zip(weights, models):
        m.eval()
        out = m(x)
        p = F.softmax(out, dim=-1)
        agg = p * w if agg is None else agg + p * w
    return agg


@torch.no_grad()
def ensemble_tta_predict(
    models: list[torch.nn.Module],
    x: torch.Tensor,
    tta_cfg: TTAConfig | None = None,
    model_weights: list[float] | None = None,
) -> torch.Tensor:
    """Combine TTA with ensemble: for each model, average TTA views; then
    average across models."""
    per_model = [tta_predict(m, x, tta_cfg) for m in models]
    if model_weights is None:
        model_weights = [1.0 / len(models)] * len(models)
    agg = None
    for w, p in zip(model_weights, per_model):
        agg = p * w if agg is None else agg + p * w
    return agg
