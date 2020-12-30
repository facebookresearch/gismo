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
            # One F1-score cross-category, but computed on sample independently, and then averaged
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

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        """
        :param pred: predictions - shape (batch_size, num_category)
        :param gt: ground truth - shape (batch_size, num_category)
        """
        assert len(pred.shape) == 2
        assert pred.shape == gt.shape

        # convert model predictions and targets to k-hots
        y_pred = label2_k_hots(
            pred, pad_value=self.pad_value, remove_eos=self.remove_eos
        )
        y_true = label2_k_hots(gt, pad_value=self.pad_value, remove_eos=self.remove_eos)

        if self.which_f1 == "i_f1":
            tp = (y_pred * y_true).sum(dim=1)
            pp = y_pred.sum(dim=1)
            ap = y_true.sum(dim=1)
            self.f1 += compute_f1(tp, pp, ap).sum()
            self.n_samples += y_pred.shape[0]
        elif self.which_f1 == "o_f1":
            self.tp += (y_pred * y_true).sum()
            self.pp += y_pred.sum()
            self.ap += y_true.sum()
        elif self.which_f1 == "c_f1":
            self.tp += (y_pred * y_true).sum(dim=0)
            self.pp += y_pred.sum(dim=0)
            self.ap += y_true.sum(dim=0)

    def compute(self) -> torch.Tensor:
        if self.which_f1 == "i_f1":
            return self.f1 / self.n_samples
        elif self.which_f1 == "o_f1":
            return compute_f1(self.tp, self.pp, self.ap)
        elif self.which_f1 == "c_f1":
            return compute_f1(self.tp, self.pp, self.ap).mean()


def compute_f1(
    true_positive: torch.Tensor,
    pred_positive: torch.Tensor,
    real_positive: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    precision = (true_positive + eps) / (pred_positive + eps)
    recall = (true_positive + eps) / (real_positive + eps)
    return 2 * (precision * recall) / (precision + recall)
