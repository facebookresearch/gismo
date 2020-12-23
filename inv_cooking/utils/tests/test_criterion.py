import torch

from inv_cooking.utils.criterion import soft_iou, SoftIoULoss, TargetDistributionLoss, _to_target_distribution


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
    expected = torch.tensor([[0.5], [1.0]])
    assert torch.allclose(out, expected, atol=1e-4)

    criterion = SoftIoULoss(reduction="none")
    expected = torch.tensor([[0.5], [0.0]])
    losses = criterion(logits, targets)
    assert torch.allclose(losses, expected, atol=1e-4)

    criterion = SoftIoULoss(reduction="mean")
    losses = criterion(logits, targets)
    assert torch.allclose(losses, torch.tensor(0.25), atol=1e-4)


def test_target_distribution_construction():
    targets = torch.LongTensor([
        [1, 0, 0, 0],
        [1, 0, 1, 0],
    ])
    distribution = _to_target_distribution(targets, epsilon=1e-8)
    expected = torch.tensor([
        [1.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.0000, 0.5000, 0.0000],
    ])
    assert torch.allclose(distribution, expected, atol=1e-4)


def test_target_distribution_loss():
    logits = torch.FloatTensor([
        [12, 15, -10, 3],
        [17, -50, 30, -10],
    ])
    targets = torch.LongTensor([
        [1, 0, 0, 0],
        [1, 0, 1, 0],
    ])

    td = TargetDistributionLoss(reduction="none")
    out = td(logits, targets)
    assert torch.allclose(out, torch.tensor([3.0486, 6.5000]), atol=1e-4)

    td = TargetDistributionLoss(reduction="mean")
    out = td(logits, targets)
    assert torch.allclose(out, torch.tensor(4.7743), atol=1e-4)
