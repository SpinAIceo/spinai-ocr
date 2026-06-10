"""Pretrained backbone loaders.

Two sources:
    * `timm` — ResNet, ConvNeXt, EfficientNet, MobileNet with ImageNet weights
    * `transformers` — DINOv2 ViT (pure visual, no text), CLIP vision encoder

These plug into DBNet/SVTR as feature extractors. Using ImageNet-pretrained
convnets typically gives a big early-training boost vs training from scratch.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from spinai_ocr.log import get_logger

log = get_logger("spinai_ocr.models.backbones")


@dataclass
class BackboneSpec:
    name: str                # "resnet18" | "convnext_tiny" | "dinov2_vits14" | ...
    source: str = "timm"     # "timm" | "torchvision" | "transformers"
    pretrained: bool = True
    out_channels: tuple[int, ...] | None = None  # per stage, if known


def load_timm_backbone(name: str, pretrained: bool = True, features_only: bool = True) -> nn.Module:
    try:
        import timm  # type: ignore
    except ImportError as e:
        raise ImportError("pip install timm") from e
    log.info("backbone.load source=timm name=%s pretrained=%s features_only=%s",
             name, pretrained, features_only,
             extra={"op": "backbone.load", "source": "timm", "name": name})
    model = timm.create_model(name, pretrained=pretrained, features_only=features_only)
    # Log the output feature channels so callers can size their neck properly.
    try:
        info = model.feature_info.channels()
        log.info("backbone.channels name=%s channels=%s", name, info,
                 extra={"op": "backbone.load", "channels": info})
    except Exception:  # noqa: BLE001
        pass
    return model


def load_torchvision_backbone(name: str, pretrained: bool = True) -> nn.Module:
    import torchvision.models as tv  # type: ignore

    log.info("backbone.load source=torchvision name=%s pretrained=%s",
             name, pretrained,
             extra={"op": "backbone.load", "source": "torchvision", "name": name})
    fn = getattr(tv, name, None)
    if fn is None:
        raise KeyError(f"torchvision has no model '{name}'")
    weights = "DEFAULT" if pretrained else None
    model = fn(weights=weights)
    return model


def load_dinov2(variant: str = "dinov2_vits14") -> nn.Module:
    """Meta DINOv2 — strong self-supervised ViT. Good for document
    representations when paired with a simple CTC head.
    Downloads ~86MB (ViT-S/14) on first call."""
    log.info("backbone.load source=torch.hub name=%s", variant,
             extra={"op": "backbone.load", "source": "torch.hub", "name": variant})
    return torch.hub.load("facebookresearch/dinov2", variant, pretrained=True)


# ---------------------------------------------------------------------------
# Convenience wrappers for SPINAI model heads
# ---------------------------------------------------------------------------


class TimmFPNBackbone(nn.Module):
    """Adapter: timm feature-extractor → list of (c2, c3, c4, c5) for FPN.

    Drop-in replacement for DBNet's ResNet stem+layer1..4. Slower than raw
    ResNet18 but yields better CER/IoU on noisy docs because of better
    low-level features.
    """

    def __init__(self, name: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        self.feature_extractor = load_timm_backbone(name, pretrained=pretrained, features_only=True)
        channels = self.feature_extractor.feature_info.channels()
        if len(channels) < 4:
            raise ValueError(f"backbone '{name}' yields {len(channels)} stages; need ≥4")
        self.out_channels = tuple(channels[-4:])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.feature_extractor(x)
        return feats[-4:]
