import pytest
import torch

from inv_cooking.utils.metrics import soft_iou


def test_soft_iou():
    logits = torch.FloatTensor([
        [12, 15, -10],
        [17, -50, 30],
    ])
    targets = torch.LongTensor([
        [1, 0, 0],
        [1, 0, 1],
    ])
    out = soft_iou(logits, targets)
    assert pytest.approx(0.5, abs=1e-4) == out[0].item()
    assert pytest.approx(1.0, abs=1e-4) == out[1].item()

