import numpy as np

from spinai_ocr.training.db_gt import build_dbnet_targets


def test_dbnet_targets_shapes():
    size = (128, 160)
    poly = np.array([[20, 20], [80, 20], [80, 40], [20, 40]], dtype=np.float32)
    out = build_dbnet_targets(size, [poly])
    assert out["prob_map"].shape == size
    assert out["prob_mask"].shape == size
    assert out["thresh_map"].shape == size
    assert out["thresh_mask"].shape == size
    assert out["prob_map"].max() == 1.0
    assert out["thresh_mask"].max() == 1.0


def test_dbnet_targets_ignore():
    size = (64, 64)
    poly = np.array([[5, 5], [30, 5], [30, 20], [5, 20]], dtype=np.float32)
    out = build_dbnet_targets(size, [poly], ignore_flags=[True])
    # ignored polygon → prob_map should be all zeros, prob_mask has hole
    assert out["prob_map"].max() == 0.0
    assert out["prob_mask"].min() == 0.0
