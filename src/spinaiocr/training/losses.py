"""Training losses.

Adapted from the medical imaging multi-modality pipeline:
- Dice + CE (balanced detection objective)
- Boundary Loss (Sobel edge-upweighted CE)
- Surface Loss (signed distance transform on GT)
- CTC wrapper for recognition
- Consistency Loss for Mean Teacher semi-supervised training
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Detection losses
# ---------------------------------------------------------------------------


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            pred = pred * mask
            target = target * mask
        inter = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        return (1 - (2 * inter + self.eps) / (union + self.eps)).mean()


class BalancedBCELoss(nn.Module):
    """OHEM-style balanced BCE: keeps all positives + top-k negatives at ratio."""

    def __init__(self, negative_ratio: float = 3.0) -> None:
        super().__init__()
        self.negative_ratio = negative_ratio

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(target)
        positive = (target * mask).bool()
        negative = ((1 - target) * mask).bool()
        pos_count = int(positive.sum().item())
        neg_count = min(int(negative.sum().item()), int(pos_count * self.negative_ratio))
        if pos_count == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        loss = F.binary_cross_entropy(pred, target, reduction="none")
        pos_loss = loss[positive]
        neg_loss = loss[negative]
        if neg_count > 0 and neg_loss.numel() > 0:
            neg_loss, _ = neg_loss.topk(min(neg_count, neg_loss.numel()))
        total = pos_loss.sum() + neg_loss.sum()
        return total / (pos_count + neg_count + 1e-6)


class BoundaryLoss(nn.Module):
    """Sobel-edge-weighted BCE: upweights pixels near GT boundary.

    Adapted from the medical `boundary_loss.py`. For text detection this
    sharpens text vs non-text edges (helps DBNet unclip stability).
    """

    def __init__(self, boundary_weight: float = 5.0) -> None:
        super().__init__()
        self.boundary_weight = boundary_weight
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        ky = kx.t()
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def _edge_map(self, target: torch.Tensor) -> torch.Tensor:
        # target: [B, 1, H, W] binary
        gx = F.conv2d(target, self.kx, padding=1)
        gy = F.conv2d(target, self.ky, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-6)
        edge = (edge > 0.1).float()
        return edge

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        edge = self._edge_map(target)
        weight = 1.0 + self.boundary_weight * edge
        loss = F.binary_cross_entropy(pred, target, reduction="none")
        return (loss * weight).mean()


class SurfaceLoss(nn.Module):
    """Signed-distance-transform loss.

    Requires a per-image distance map `dist_map` computed on CPU (e.g., via
    scipy.ndimage.distance_transform_edt) and passed in. Negative inside the
    text region, positive outside.
    """

    def forward(self, pred: torch.Tensor, dist_map: torch.Tensor) -> torch.Tensor:
        return (pred * dist_map).mean()


class DBNetLoss(nn.Module):
    """Composite loss for DBNet outputs {prob, thresh, binary}."""

    def __init__(
        self,
        prob_weight: float = 1.0,
        thresh_weight: float = 10.0,
        binary_weight: float = 1.0,
        use_boundary: bool = True,
    ) -> None:
        super().__init__()
        self.prob_weight = prob_weight
        self.thresh_weight = thresh_weight
        self.binary_weight = binary_weight
        self.bce = BalancedBCELoss()
        self.dice = DiceLoss()
        self.l1 = nn.L1Loss()
        self.boundary = BoundaryLoss() if use_boundary else None

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        gt: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """gt must contain: prob_map, prob_mask, thresh_map, thresh_mask."""
        prob = outputs["prob"]
        thresh = outputs["thresh"]
        binary = outputs["binary"]

        prob_loss = self.bce(prob, gt["prob_map"], gt.get("prob_mask"))
        binary_loss = self.dice(binary, gt["prob_map"], gt.get("prob_mask"))
        # threshold map: L1 inside threshold mask region
        thresh_loss = self.l1(thresh * gt["thresh_mask"], gt["thresh_map"] * gt["thresh_mask"])

        total = (
            self.prob_weight * prob_loss
            + self.binary_weight * binary_loss
            + self.thresh_weight * thresh_loss
        )
        out = {
            "prob_loss": prob_loss,
            "binary_loss": binary_loss,
            "thresh_loss": thresh_loss,
            "total": total,
        }
        if self.boundary is not None:
            boundary_loss = self.boundary(prob, gt["prob_map"])
            out["boundary_loss"] = boundary_loss
            out["total"] = total + boundary_loss
        return out


# ---------------------------------------------------------------------------
# Recognition loss
# ---------------------------------------------------------------------------


class CTCRecognitionLoss(nn.Module):
    """Thin wrapper around nn.CTCLoss that handles variable-length targets."""

    def __init__(self, blank: int = 0, zero_infinity: bool = True) -> None:
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity)

    def forward(
        self,
        logits: torch.Tensor,  # [B, T, V]
        targets: torch.Tensor,  # [sum(target_lengths)]
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # [T, B, V]
        return self.ctc(log_probs, targets, input_lengths, target_lengths)


# ---------------------------------------------------------------------------
# Semi-supervised / Mean Teacher
# ---------------------------------------------------------------------------


class ConsistencyLoss(nn.Module):
    """MSE between student & teacher softmax outputs. Used with Mean Teacher."""

    def __init__(self, kind: str = "mse") -> None:
        super().__init__()
        self.kind = kind

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        s = F.softmax(student_logits, dim=-1)
        t = F.softmax(teacher_logits, dim=-1).detach()
        if self.kind == "mse":
            return F.mse_loss(s, t)
        if self.kind == "kl":
            return F.kl_div(F.log_softmax(student_logits, -1), t, reduction="batchmean")
        raise ValueError(self.kind)
