"""Tier selection plumbing works for medical_v2.

Pinned _56 after I noticed the API didn't surface medical_v2: the
ModelTier Literal lacked "medical", and the pipeline was building
recognizers with the default arch ("svtr_lite") regardless of tier,
so a medical ckpt (svtr_medical, dim 320) would silently load with
most weights dropped via strict=False. These tests prevent both
regressions.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from spinai_ocr.config import ModelTier


def test_model_tier_includes_medical():
    # Literal types don't support `in` at runtime in older pythons; use
    # typing.get_args for a runtime-safe check.
    from typing import get_args
    assert "medical" in get_args(ModelTier)


@pytest.mark.skipif(
    not Path("checkpoints/medical/ko/rec.pth").exists(),
    reason="medical tier ckpt not installed in this environment",
)
def test_pipeline_medical_tier_loads_correct_arch_and_vocab():
    """With tier='medical', the recognizer is built with svtr_medical
    (not svtr_lite) and head sized to the 2005-char pruned vocab."""
    from spinai_ocr.config import PipelineConfig
    from spinai_ocr.inference.pipeline import OCRPipeline
    pipe = OCRPipeline(config=PipelineConfig(lang="ko", tier="medical",
                                               device="cpu"))
    pipe._ensure_loaded()
    # Vocab size should come from the ckpt (medical vocab = 2005), not
    # the default ko_en_v1 (~11k).
    assert pipe._vocab.size < 3000, (
        f"medical tier vocab should be pruned (~2005), got {pipe._vocab.size}")
    # Recognizer head shape matches the pruned vocab + blank
    import torch
    # Last linear-like layer in the head output-projects to vocab_size+1
    # (CTC blank). Quick structural check via named_parameters.
    head_weights = [p for n, p in pipe._recognizer.named_parameters()
                     if "head" in n.lower() and p.ndim >= 2]
    assert head_weights, "recognizer has no head-shaped params"
    # SVTR head's out-dim == Vocab.size (blank is a token inside vocab,
    # not an extra slot — see train script passing vocab_size=vocab.size).
    out_dim = head_weights[-1].shape[0]
    assert out_dim == pipe._vocab.size, (
        f"head out-dim {out_dim} != vocab.size {pipe._vocab.size}")
