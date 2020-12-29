# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytorch_lightning as pl
import torch


class DistributedAverage(pl.metrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, quantity: torch.Tensor):
        self.quantity += quantity.sum()
        self.n_samples += quantity.numel()

    def compute(self):
        return self.quantity / self.n_samples
