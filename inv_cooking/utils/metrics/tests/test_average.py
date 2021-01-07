import torch

from inv_cooking.utils.metrics.average import DistributedAverage, DistributedCompositeAverage


def test_distributed_average():
    metric = DistributedAverage()
    metric.update(torch.tensor([1.0, 2.0, 3.0]))
    metric.update(torch.tensor([2.0, 3.0, 4.0]))
    assert metric.compute() == torch.tensor(2.5)


def test_distributed_composite_average():
    weights = {"label": 0.75, "ignore": 0.0, "recipe": 0.25}
    metric = DistributedCompositeAverage(weights=weights, total="total")
    metric.update({
        "label": torch.tensor([1.0, 2.0, 3.0]),
        "recipe": torch.tensor([2.0, 3.0, 4.0]),
        "n_samples": 3,
    })
    metric.update({
        "label": torch.tensor([4.0]),
        "recipe": torch.tensor([5.0]),
        "n_samples": 1,
    })
    output = metric.compute()
    assert output["label"] == torch.tensor(2.5)
    assert output["recipe"] == torch.tensor(3.5)
    assert output["total"] == torch.tensor(2.75)
