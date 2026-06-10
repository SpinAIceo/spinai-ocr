"""DBNet-lite text detection head.

Paper: Real-time Scene Text Detection with Differentiable Binarization
(https://arxiv.org/abs/1911.08947)

This is a compact, educational implementation to be improved iteratively.
Full DBNet++ lives behind a feature flag once the baseline converges.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50


class FPN(nn.Module):
    def __init__(self, in_channels: list[int], out_channels: int = 256) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in in_channels]
        )
        self.smooth = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels // 4, 3, padding=1) for _ in in_channels]
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        # feats: list of [c2, c3, c4, c5] (increasing stride)
        laterals = [lat(f) for lat, f in zip(self.lateral, feats)]
        # top-down
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
        smoothed = [s(l) for s, l in zip(self.smooth, laterals)]
        # upsample all to highest resolution
        target = smoothed[0].shape[-2:]
        out = torch.cat(
            [F.interpolate(s, size=target, mode="bilinear", align_corners=False) for s in smoothed],
            dim=1,
        )
        return out


class DBNet(nn.Module):
    def __init__(self, backbone: str = "resnet18", inner: int = 256, k: int = 50) -> None:
        super().__init__()
        if backbone == "resnet18":
            bb = resnet18(weights=None)
            channels = [64, 128, 256, 512]
        elif backbone == "resnet50":
            bb = resnet50(weights=None)
            channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = bb.layer4

        self.neck = FPN(channels, out_channels=inner)
        fuse_ch = inner  # concat of (inner/4) * 4

        self.prob_head = nn.Sequential(
            nn.Conv2d(fuse_ch, fuse_ch // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(fuse_ch // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(fuse_ch // 4, fuse_ch // 4, 2, stride=2),
            nn.BatchNorm2d(fuse_ch // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(fuse_ch // 4, 1, 2, stride=2),
            nn.Sigmoid(),
        )
        self.thresh_head = nn.Sequential(
            nn.Conv2d(fuse_ch, fuse_ch // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(fuse_ch // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(fuse_ch // 4, fuse_ch // 4, 2, stride=2),
            nn.BatchNorm2d(fuse_ch // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(fuse_ch // 4, 1, 2, stride=2),
            nn.Sigmoid(),
        )
        self.k = k

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        fused = self.neck([c2, c3, c4, c5])
        prob = self.prob_head(fused)
        thresh = self.thresh_head(fused)
        # Differentiable binarization
        binary = torch.reciprocal(1 + torch.exp(-self.k * (prob - thresh)))
        return {"prob": prob, "thresh": thresh, "binary": binary}
