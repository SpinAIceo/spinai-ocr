import torch

from spinaiocr.training.losses import (
    BoundaryLoss,
    ConsistencyLoss,
    DBNetLoss,
    DiceLoss,
)


def test_dice_loss_perfect():
    pred = torch.ones(2, 1, 8, 8)
    target = torch.ones(2, 1, 8, 8)
    loss = DiceLoss()(pred, target)
    assert loss.item() < 1e-3


def test_dice_loss_disjoint():
    pred = torch.zeros(2, 1, 8, 8)
    target = torch.ones(2, 1, 8, 8)
    loss = DiceLoss()(pred, target)
    assert loss.item() > 0.9


def test_boundary_loss_runs():
    pred = torch.sigmoid(torch.randn(2, 1, 16, 16))
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    loss = BoundaryLoss()(pred, target)
    assert torch.isfinite(loss)


def test_dbnet_loss_runs():
    b, h, w = 2, 16, 16
    out = {
        "prob": torch.sigmoid(torch.randn(b, 1, h, w)),
        "thresh": torch.sigmoid(torch.randn(b, 1, h, w)),
        "binary": torch.sigmoid(torch.randn(b, 1, h, w)),
    }
    gt = {
        "prob_map": (torch.rand(b, 1, h, w) > 0.5).float(),
        "prob_mask": torch.ones(b, 1, h, w),
        "thresh_map": torch.rand(b, 1, h, w),
        "thresh_mask": torch.ones(b, 1, h, w),
    }
    result = DBNetLoss()(out, gt)
    assert torch.isfinite(result["total"])


def test_consistency_loss():
    a = torch.randn(4, 10, 100)
    b = torch.randn(4, 10, 100)
    loss = ConsistencyLoss()(a, b)
    assert torch.isfinite(loss) and loss.item() >= 0
