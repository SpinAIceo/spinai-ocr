import numpy as np
import torch

from spinaiocr.models.dexined import DexiNed


def test_dexined_forward_runs():
    model = DexiNed()
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        outs = model(x)
    # 6 side outputs + 1 fused
    assert len(outs) == 7
    for o in outs:
        assert o.shape[-2:] == x.shape[-2:]
        assert o.shape[1] == 1


def test_dexined_predict_numpy():
    model = DexiNed()
    img = (np.random.rand(64, 96, 3) * 255).astype(np.uint8)
    prob = model.predict(img)
    assert prob.shape == (64, 96)
    assert prob.dtype == np.float32
    assert 0.0 <= float(prob.min()) <= float(prob.max()) <= 1.0
