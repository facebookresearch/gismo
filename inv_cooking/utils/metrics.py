# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


@torch.jit.script
def soft_iou(logits: torch.Tensor, target: torch.Tensor, sum_axis: int = 1, epsilon: float = 1e-8):
    probs = torch.sigmoid(logits)
    num = (probs * target).sum(sum_axis, True) + epsilon
    den = (probs + target - probs * target).sum(sum_axis, True) + epsilon
    return num / den


class SoftIoULoss(nn.Module):
    def __init__(self, reduction="none", epsilon=1e-8):
        super().__init__()
        assert reduction in ["none", "mean"]
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        loss = 1.0 - soft_iou(inputs, targets, epsilon=self.epsilon)
        if self.reduction == "mean":
            loss = loss.mean()
        return loss


class TargetDistributionLoss(nn.Module):
    def __init__(self, reduction="none", epsilon=1e-8):
        super().__init__()
        assert reduction in ["none", "mean"]
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, label_prediction, label_target):
        # create target distribution
        # check if the target is all 0s
        cardinality_target = label_target.sum(dim=-1).unsqueeze(1)
        is_empty = (cardinality_target == 0).float()

        # create flat distribution
        flat_target = 1 / label_target.size(-1)
        flat_target = (
            torch.FloatTensor(np.array(flat_target))
            .unsqueeze(-1)
            .unsqueeze(-1)
            .type_as(cardinality_target)
        )

        # divide target by number of elements and add equal prob to all elements for the empty sets
        target_distribution = (
            label_target.float() / (cardinality_target + self.epsilon)
            + is_empty * flat_target
        )

        # loss
        loss = target_distribution * torch.nn.functional.log_softmax(
            label_prediction, dim=-1
        )
        loss = -torch.sum(loss, dim=-1)

        if self.reduction == "mean":
            loss = loss.mean()

        return loss


class OverallErrorCounts:
    def __init__(self):
        self.counts: Dict[str, int] = {}
        self.reset(overall=True)

    def reset(self, overall: bool = False, per_image: bool = False):
        # reset all error counts (done at the end of each epoch)
        if overall:
            self.counts = {
                "c_tp": 0,
                "c_fp": 0,
                "c_fn": 0,
                "c_tn": 0,
                "o_tp": 0,
                "o_fp": 0,
                "o_fn": 0,
                "i_tp": 0,
                "i_fp": 0,
                "i_fn": 0,
            }
        # reset per sample error counts (done at the end of each iteration)
        if per_image:
            self.counts["i_tp"] = 0
            self.counts["i_fp"] = 0
            self.counts["i_fn"] = 0

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor, which_metrics: List[str]) -> None:
        if "o_f1" in which_metrics:
            self.counts["o_tp"] += (y_pred * y_true).sum().item()
            self.counts["o_fp"] += (y_pred * (1 - y_true)).sum().item()
            self.counts["o_fn"] += ((1 - y_pred) * y_true).sum().item()
        if "c_f1" in which_metrics:
            self.counts["c_tp"] += (y_pred * y_true).sum(0).cpu().data.numpy()
            self.counts["c_fp"] += (y_pred * (1 - y_true)).sum(0).cpu().data.numpy()
            self.counts["c_fn"] += ((1 - y_pred) * y_true).sum(0).cpu().data.numpy()
            self.counts["c_tn"] += ((1 - y_pred) * (1 - y_true)).sum(0).cpu().data.numpy()
        if "i_f1" in which_metrics:
            self.counts["i_tp"] = (y_pred * y_true).sum(1).cpu().data.numpy()
            self.counts["i_fp"] = (y_pred * (1 - y_true)).sum(1).cpu().data.numpy()
            self.counts["i_fn"] = ((1 - y_pred) * y_true).sum(1).cpu().data.numpy()

    def compute_metrics(self, which_metrics: List[str], eps: float = 1e-8):
        ret_metrics = {}

        if "o_f1" in which_metrics:
            pre = (self.counts["o_tp"] + eps) / (self.counts["o_tp"] + self.counts["o_fp"] + eps)
            rec = (self.counts["o_tp"] + eps) / (self.counts["o_tp"] + self.counts["o_fn"] + eps)
            o_f1 = 2 * (pre * rec) / (pre + rec)
            ret_metrics["o_f1"] = o_f1

        if "c_f1" in which_metrics:
            pre = (self.counts["c_tp"] + eps) / (self.counts["c_tp"] + self.counts["c_fp"] + eps)
            rec = (self.counts["c_tp"] + eps) / (self.counts["c_tp"] + self.counts["c_fn"] + eps)
            f1_perclass = 2 * (pre * rec) / (pre + rec)
            f1_perclass_avg = np.average(f1_perclass)
            ret_metrics["c_f1"] = f1_perclass_avg

        if "i_f1" in which_metrics:
            pre = (self.counts["i_tp"] + eps) / (self.counts["i_tp"] + self.counts["i_fp"] + eps)
            rec = (self.counts["i_tp"] + eps) / (self.counts["i_tp"] + self.counts["i_fn"] + eps)
            f1_i = 2 * (pre * rec) / (pre + rec)
            ret_metrics["i_f1"] = f1_i.sum()

        return ret_metrics
