"""DexiNed — Dense Extreme Inception Network for Edge Detection.

Reference:
  Poma, X. S., Riba, E., & Sappa, A. (2020).
  "Dense Extreme Inception Network: Towards a Robust CNN Model for Edge Detection."
  WACV 2020. https://arxiv.org/abs/1909.01955
  Official repo: https://github.com/xavysp/DexiNed (MIT license)

This is a self-contained PyTorch implementation so users don't need to clone
the original repo. Weights must be downloaded separately from the official
release (see `spinai_ocr/models/weights.py` for helper) — the architecture
below matches the published `10_model.pth` state dict keys.

Usage:
    from spinai_ocr.models.dexined import DexiNed, load_pretrained
    model = load_pretrained("checkpoints/dexined_10_model.pth")
    edge = model.predict(image_rgb_uint8)  # returns HxW float32 in [0,1]
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _DoubleConv(nn.Module):
    def __init__(self, in_c: int, mid_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, mid_c, 3, stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.conv2 = nn.Conv2d(mid_c, out_c, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x)))


class _SingleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, stride)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class _DenseLayer(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2 = inputs
        out = self.layer(x1)
        return 0.5 * (out + x2)


class _DenseBlock(nn.Module):
    def __init__(self, n_layers: int, in_c: int, out_c: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_DenseLayer(in_c if i == 0 else out_c, out_c) for i in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer((x, skip))
        return x


class _UpSampleBlock(nn.Module):
    """Learned upsampler used at each side-output to produce full-res edge map."""

    def __init__(self, in_c: int, up_scale: int) -> None:
        super().__init__()
        self.up_scale = up_scale
        self.features = nn.ModuleList()
        for i in range(up_scale):
            out_c = 1 if i == up_scale - 1 else 16
            self.features.append(
                nn.Sequential(
                    nn.Conv2d(in_c if i == 0 else 16, 16, 1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(16, out_c, 2, stride=2),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for f in self.features:
            x = f(x)
        return x


# ---------------------------------------------------------------------------
# DexiNed
# ---------------------------------------------------------------------------


class DexiNed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block_1 = _DoubleConv(3, 32, 64, stride=2)
        self.block_2 = _DoubleConv(64, 128, 128)

        self.dblock_3 = _DenseBlock(2, 128, 256)
        self.dblock_4 = _DenseBlock(3, 256, 512)
        self.dblock_5 = _DenseBlock(3, 512, 512)
        self.dblock_6 = _DenseBlock(3, 512, 256)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.side_1 = _SingleConv(64, 128, stride=2)
        self.side_2 = _SingleConv(128, 256, stride=2)
        self.side_3 = _SingleConv(256, 512, stride=2)
        self.side_4 = _SingleConv(512, 512)
        self.side_5 = _SingleConv(512, 256)

        self.pre_dense_2 = _SingleConv(128, 256)
        self.pre_dense_3 = _SingleConv(128, 256)
        self.pre_dense_4 = _SingleConv(256, 512)
        self.pre_dense_5 = _SingleConv(512, 512)
        self.pre_dense_6 = _SingleConv(512, 256)

        self.up_block_1 = _UpSampleBlock(64, up_scale=1)
        self.up_block_2 = _UpSampleBlock(128, up_scale=1)
        self.up_block_3 = _UpSampleBlock(256, up_scale=2)
        self.up_block_4 = _UpSampleBlock(512, up_scale=3)
        self.up_block_5 = _UpSampleBlock(512, up_scale=4)
        self.up_block_6 = _UpSampleBlock(256, up_scale=4)

        self.block_cat = nn.Conv2d(6, 1, 1)

    def _match_size(self, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if t.shape[-2:] != ref.shape[-2:]:
            t = F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return t

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Block 1
        b1 = self.block_1(x)
        b1_side = self.side_1(b1)
        # Block 2
        b2 = self.block_2(b1)
        b2_down = self.maxpool(b2)
        b2_add = b2_down + b1_side
        b2_side = self.side_2(b2_add)
        # Dense Block 3
        b3_pre = self.pre_dense_3(b2_down)
        b3 = self.dblock_3(b2_add, b3_pre)
        b3_down = self.maxpool(b3)
        b3_add = b3_down + b2_side
        b3_side = self.side_3(b3_add)
        # Dense Block 4
        b4_pre = self.pre_dense_4(b3_down)
        b4 = self.dblock_4(b3_add, b4_pre)
        b4_down = self.maxpool(b4)
        b4_add = b4_down + b3_side
        b4_side = self.side_4(b4_add)
        # Dense Block 5
        b5_pre = self.pre_dense_5(b4_down)
        b5 = self.dblock_5(b4_add, b5_pre)
        b5_add = b5 + b4_side
        # Dense Block 6
        b6_pre = self.pre_dense_6(b5)
        b6 = self.dblock_6(b5_add, b6_pre)

        # Side outputs upsampled to input resolution
        out1 = self.up_block_1(b1)
        out2 = self.up_block_2(b2)
        out3 = self.up_block_3(b3)
        out4 = self.up_block_4(b4)
        out5 = self.up_block_5(b5)
        out6 = self.up_block_6(b6)

        outs = [out1, out2, out3, out4, out5, out6]
        outs = [self._match_size(o, x) for o in outs]
        fused = self.block_cat(torch.cat(outs, dim=1))
        return outs + [fused]

    # Convenience inference wrapper --------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        device: str | None = None,
    ) -> np.ndarray:
        """image: HxWx3 uint8 RGB. Returns HxW float32 edge probability in [0,1]."""
        if device is None:
            device = next(self.parameters()).device.type
        self.eval()
        # Match official preprocessing: mean-subtract BGR order.
        x = image[..., ::-1].astype(np.float32)  # RGB → BGR
        x -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
        x = x.transpose(2, 0, 1)[None]  # [1, 3, H, W]
        t = torch.from_numpy(x).to(device)
        outs = self.forward(t)
        fused = torch.sigmoid(outs[-1])[0, 0].cpu().numpy()
        return fused.astype(np.float32)


def load_pretrained(checkpoint_path: Union[str, Path], map_location: str = "cpu") -> DexiNed:
    model = DexiNed()
    state = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        import warnings

        warnings.warn(
            f"DexiNed checkpoint partially loaded "
            f"(missing={len(missing)}, unexpected={len(unexpected)})",
            stacklevel=2,
        )
    model.eval()
    return model
