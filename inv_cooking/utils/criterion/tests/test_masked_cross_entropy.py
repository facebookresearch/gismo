import torch

from inv_cooking.utils.criterion import MaskedCrossEntropyCriterion


def test_masked_cross_entropy_criterion():
    torch.random.manual_seed(0)
    vocab_size = 3
    targets = torch.LongTensor(
        [
            [1, 2, 1, 2, 0],
            [1, 2, 0, 0, 0],
        ]
    )
    logits = torch.randn(size=(targets.size(0), targets.size(1) + 1, vocab_size))
    criterion = MaskedCrossEntropyCriterion()
    out = criterion(logits, targets)
    expected = torch.tensor([1.6113, 1.6840])
    assert torch.allclose(out, expected, atol=1e-4)
