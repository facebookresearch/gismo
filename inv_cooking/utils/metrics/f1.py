# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytorch_lightning as pl
import torch

from inv_cooking.models.ingredients_predictor import label2_k_hots


class DistributedF1(pl.metrics.Metric):
    def __init__(
        self,
        which_f1: str,
        pad_value: int,
        remove_eos: bool,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        assert which_f1 in ["i_f1", "o_f1", "c_f1"]

        self.which_f1 = which_f1
        self.pad_value = pad_value
        self.remove_eos = remove_eos

        if self.which_f1 == "i_f1":
            # One F1-score cross-cateogry, but computed on each worker independently
            self.add_state("n_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("f1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        elif self.which_f1 == "o_f1":
            # One F1-score cross-category
            self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("pp", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("ap", default=torch.tensor(0.0), dist_reduce_fx="sum")
        else:
            # One F1-score for each category
            vocab_size = self.pad_value - 1
            if not self.remove_eos:
                vocab_size += 1
            self.add_state("tp", default=torch.zeros(vocab_size), dist_reduce_fx="sum")
            self.add_state("pp", default=torch.zeros(vocab_size), dist_reduce_fx="sum")
            self.add_state("ap", default=torch.zeros(vocab_size), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        assert pred.shape == gt.shape

        # convert model predictions and targets to k-hots
        y_pred = label2_k_hots(
            pred, pad_value=self.pad_value, remove_eos=self.remove_eos
        )
        y_true = label2_k_hots(gt, pad_value=self.pad_value, remove_eos=self.remove_eos)

        if self.which_f1 == "i_f1":
            self.tp = (y_pred * y_true).sum(dim=1)
            self.pp = y_pred.sum(dim=1)
            self.ap = y_true.sum(dim=1)
            self.f1 += self._computef1().sum()
            self.n_samples += y_pred.shape[0]
        elif self.which_f1 == "o_f1":
            self.tp += (y_pred * y_true).sum()
            self.pp += y_pred.sum()
            self.ap += y_true.sum()
        elif self.which_f1 == "c_f1":
            self.tp += (y_pred * y_true).sum(dim=0)
            self.pp += y_pred.sum(dim=0)
            self.ap += y_true.sum(dim=0)

    def _computef1(self, eps: float = 1e-8):
        pre = (self.tp + eps) / (self.pp + eps)
        rec = (self.tp + eps) / (self.ap + eps)
        f1 = 2 * (pre * rec) / (pre + rec)
        return f1

    def compute(self):
        if self.which_f1 == "c_f1":
            f1 = self._computef1()
            f1 = f1.mean()
        elif self.which_f1 == "o_f1":
            f1 = self._computef1()
        elif self.which_f1 == "i_f1":
            f1 = self.f1 / self.n_samples
        return f1
