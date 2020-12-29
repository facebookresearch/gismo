import torch

from inv_cooking.utils.metrics.average import DistributedAverage


def test_distributed_average():
    metric = DistributedAverage()
    metric.update(torch.tensor([1.0, 2.0, 3.0]))
    metric.update(torch.tensor([2.0, 3.0, 4.0]))
    assert metric.compute() == torch.tensor(2.5)
