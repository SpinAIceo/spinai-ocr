"""Entropy-based uncertainty weighting.

For detection: per-pixel entropy of prob map → high-entropy pixels get
down-weighted when they are pseudo-labeled (they are uncertain).

For recognition: per-timestep entropy on softmax → uncertain timesteps
contribute less to CTC on pseudo-labeled data.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pixel_entropy(prob: torch.Tensor) -> torch.Tensor:
    """prob: [B, 1, H, W] sigmoid output."""
    eps = 1e-6
    p = prob.clamp(eps, 1 - eps)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def timestep_entropy(logits: torch.Tensor) -> torch.Tensor:
    """logits: [B, T, V]. Returns [B, T] entropy per timestep."""
    p = F.softmax(logits, dim=-1)
    eps = 1e-6
    return -(p * (p.clamp(eps, 1.0)).log()).sum(dim=-1)


def entropy_weights(entropy: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Map entropy to [0, 1] confidence weights: w = exp(-alpha * H).

    Low entropy (confident teacher) → weight ≈ 1.
    High entropy (unsure teacher)   → weight → 0.
    """
    return torch.exp(-alpha * entropy)
