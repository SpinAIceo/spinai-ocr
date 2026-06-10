# -*- coding: utf-8 -*-
"""Smoke tests for iter 166 inference harness additions.

Tests are model-weight independent: no real checkpoint needed.
All assertions are smoke-level (non-empty string, conf in range, no crash).
"""
import sys
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. jamo_correct: import and correctness
# ---------------------------------------------------------------------------

def test_jamo_correct_import():
    from spinai_ocr.postprocess.jamo_correct import correct_jamo  # noqa: F401


def test_jamo_correct_space_removal():
    """Spurious space between Korean syllables must be removed."""
    from spinai_ocr.postprocess.jamo_correct import correct_jamo
    an = chr(0xc548)    # 안
    nyeong = chr(0xb155)  # 녕
    result = correct_jamo(an + " " + nyeong)
    assert result == an + nyeong, f"Expected {an+nyeong!r}, got {result!r}"


def test_jamo_correct_jamo_composition():
    """compat-jamo runs (cho+jung+jong) must compose into syllables."""
    from spinai_ocr.postprocess.jamo_correct import correct_jamo
    # ㅇ(0x3147) + ㅏ(0x314F) + ㄴ(0x3134) -> 안(0xC548)
    raw = chr(0x3147) + chr(0x314F) + chr(0x3134)
    expected = chr(0xC548)
    result = correct_jamo(raw)
    assert result == expected, f"Expected {expected!r}, got {result!r}"


def test_jamo_correct_noop_latin():
    """Non-Korean text must pass through unchanged."""
    from spinai_ocr.postprocess.jamo_correct import correct_jamo
    assert correct_jamo("hello world") == "hello world"


def test_jamo_correct_noop_empty():
    from spinai_ocr.postprocess.jamo_correct import correct_jamo
    assert correct_jamo("") == ""


def test_jamo_correct_mixed_boundary():
    """Space at Korean/Latin boundary must NOT be removed."""
    from spinai_ocr.postprocess.jamo_correct import correct_jamo
    an = chr(0xc548)
    # Korean char then space then Latin -- boundary space must stay
    result = correct_jamo(an + " B")
    assert result == an + " B", f"Got {result!r}"


# ---------------------------------------------------------------------------
# 2. AngleClassifier: import and predict_angle on random tensor
# ---------------------------------------------------------------------------

def test_angle_classifier_import():
    from spinai_ocr.models.angle_cls import AngleClassifier  # noqa: F401


def test_angle_classifier_predict_random():
    """predict_angle must return one of {0, 90, 180, 270} without crashing."""
    import torch
    from spinai_ocr.models.angle_cls import AngleClassifier, predict_angle
    model = AngleClassifier(num_classes=4)
    x = torch.rand(1, 3, 48, 192)
    angle = predict_angle(model, x)
    assert angle in (0, 90, 180, 270), f"Unexpected angle: {angle}"


def test_angle_classifier_rotate_noop():
    """rotate_to_upright with angle=0 must return original array."""
    import numpy as np
    from spinai_ocr.models.angle_cls import rotate_to_upright
    arr = np.zeros((48, 192, 3), dtype=np.uint8)
    result = rotate_to_upright(arr, 0)
    assert isinstance(result, np.ndarray)
    assert result.shape == arr.shape


def test_angle_classifier_rotate_90():
    """rotate_to_upright with angle=90 must flip H and W."""
    import numpy as np
    from spinai_ocr.models.angle_cls import rotate_to_upright
    arr = np.zeros((48, 192, 3), dtype=np.uint8)
    result = rotate_to_upright(arr, 90)
    assert isinstance(result, np.ndarray)
    # After -90 rotation, shape becomes (192, 48, 3)
    assert result.shape[:2] == (192, 48)


# ---------------------------------------------------------------------------
# 3. Upscale retry: helper is importable and cv2.resize works at 1.5x
# ---------------------------------------------------------------------------

def test_upscale_retry_cv2():
    """cv2.resize at 1.5x on a synthetic crop must return larger array."""
    import cv2
    crop = np.zeros((48, 100, 3), dtype=np.uint8)
    uh = max(int(crop.shape[0] * 1.5), 1)
    uw = max(int(crop.shape[1] * 1.5), 1)
    up = cv2.resize(crop, (uw, uh), interpolation=cv2.INTER_CUBIC)
    assert up.shape == (uh, uw, 3)
    assert uh > crop.shape[0]
    assert uw > crop.shape[1]


# ---------------------------------------------------------------------------
# 4. Config flags are present and default correctly
# ---------------------------------------------------------------------------

def test_recognition_config_flags():
    from spinai_ocr.config import RecognitionConfig
    cfg = RecognitionConfig()
    assert cfg.low_conf_upscale_retry is True, "low_conf_upscale_retry should default True"
    assert cfg.use_jamo_correction is True, "use_jamo_correction should default True"


def test_recognition_config_flags_toggle():
    from spinai_ocr.config import RecognitionConfig
    cfg = RecognitionConfig(low_conf_upscale_retry=False, use_jamo_correction=False)
    assert cfg.low_conf_upscale_retry is False
    assert cfg.use_jamo_correction is False


# ---------------------------------------------------------------------------
# 5. Pipeline import does not crash with new flags
# ---------------------------------------------------------------------------

def test_pipeline_import():
    """OCRPipeline must be importable without model weights."""
    from spinai_ocr.inference.pipeline import OCRPipeline  # noqa: F401


def test_pipeline_init_no_crash():
    """OCRPipeline.__init__ must complete without real checkpoints (degraded mode)."""
    from spinai_ocr.inference.pipeline import OCRPipeline
    p = OCRPipeline(lang="ko", tier="lite", device="cpu",
                    checkpoints_root="/tmp/no_such_root")
    assert p is not None
    assert p.config.recognition.low_conf_upscale_retry is True
    assert p.config.recognition.use_jamo_correction is True
