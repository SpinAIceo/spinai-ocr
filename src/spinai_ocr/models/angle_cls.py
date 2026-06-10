"""Orientation classifier — predicts rotation class {0, 90, 180, 270}.

Placed before detection in the pipeline when input source is uncontrolled
(scanned documents, mobile photos). A tiny CNN on a thumbnail suffices;
PaddleOCR's cls model uses a similar 4-way head on 48x192 input.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AngleClassifier(nn.Module):
    """48×192 RGB → 4-class logits."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24x96
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 12x48
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 6x24
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x).flatten(1)
        return self.head(f)


ANGLES = (0, 90, 180, 270)


@torch.no_grad()
def predict_angle(model: AngleClassifier, image: torch.Tensor) -> int:
    """image: [1,3,H,W] in [0,1]. Returns predicted angle in degrees."""
    model.eval()
    if image.shape[-2:] != (48, 192):
        image = F.interpolate(image, size=(48, 192), mode="bilinear", align_corners=False)
    logits = model(image)
    return ANGLES[int(logits.argmax(dim=-1).item())]


def rotate_to_upright(image, angle: int):
    """Rotate a numpy/PIL image so the predicted angle becomes 0."""
    import numpy as np
    from PIL import Image

    if isinstance(image, np.ndarray):
        pil = Image.fromarray(image)
    else:
        pil = image
    if angle == 0:
        return image
    # Rotate by -angle to undo
    rotated = pil.rotate(-angle, expand=True)
    if isinstance(image, np.ndarray):
        return np.asarray(rotated)
    return rotated
