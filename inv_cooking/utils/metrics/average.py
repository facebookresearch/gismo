# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import pytorch_lightning as pl
import torch


class DistributedAverage(pl.metrics.Metric):
    """
    Metric to compute the average of a value, distributed among several workers
    """

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, quantity: torch.Tensor):
        self.quantity += quantity.sum()
        self.n_samples += quantity.numel()

    def compute(self):
        return self.quantity / self.n_samples


class DistributedCompositeAverage(pl.metrics.Metric):
    """
    Metric to compute the average of a set of values, distributed among several workers
    """

    def __init__(
        self, weights: Dict[str, float], total: str, dist_sync_on_step: bool = False
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.weights = {key: weight for key, weight in weights.items() if weight > 0.0}
        self.total = total
        for key, weight in self.weights.items():
            self.add_state(key, default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, quantity: Dict[str, torch.Tensor]):
        for key, weight in self.weights.items():
            attribute = getattr(self, key)
            attribute += quantity[key].sum()
        self.n_samples += quantity["n_samples"]

    def compute(self):
        ret_dict = {self.total: 0.0}
        for key, weight in self.weights.items():
            attribute = getattr(self, key)
            ret_dict[key] = attribute / self.n_samples
            ret_dict[self.total] += self.weights[key] * ret_dict[key]
        return ret_dict
