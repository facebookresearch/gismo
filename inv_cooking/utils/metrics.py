# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


def DC(alphas, dataset="recipe1m"):

    if dataset == "recipe1m":
        cms = np.array(
            [
                0.0000e00,
                1.1570e04,
                2.7383e04,
                4.5583e04,
                6.2422e04,
                7.2228e04,
                7.6909e04,
                7.6319e04,
                7.0102e04,
                5.9216e04,
                4.6648e04,
                3.4192e04,
                2.3283e04,
                1.5590e04,
                1.0079e04,
                6.2400e03,
                3.7690e03,
                2.2110e03,
                1.3430e03,
                2.7000e01,
            ]
        )

    cms = cms[0 : alphas.size(-1)]
    c = sum(cms)

    cms = torch.from_numpy(cms).type_as(alphas)
    cms = cms.unsqueeze(0)

    num = alphas + cms
    den = (torch.sum(alphas, dim=-1) + c).unsqueeze(1)
    dc_alphas = num / den

    return dc_alphas


class DCLoss(nn.Module):
    def __init__(self, U, dataset, reduction="none", e=1e-8):
        super(DCLoss, self).__init__()
        assert reduction in ["none", "mean"]
        self.U = math.log(U)
        self.dataset = dataset
        self.offset = 0 if self.dataset in ["coco", "nuswide"] else 1
        self.reduction = reduction
        self.e = e

    def forward(self, input, target):
        loss = (
            nn.NLLLoss(reduction="none")(
                torch.log(DC(input, dataset=self.dataset) + self.e), target
            )
            - ((target + self.offset) * self.U).float()
        )

        if self.reduction == "mean":
            loss = loss.mean()
        return loss


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


def update_error_counts(
    error_counts: Dict[str, int],
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    which_metrics: List[str],
):

    if "o_f1" in which_metrics:
        error_counts["o_tp"] += (y_pred * y_true).sum().item()
        error_counts["o_fp"] += (y_pred * (1 - y_true)).sum().item()
        error_counts["o_fn"] += ((1 - y_pred) * y_true).sum().item()

    if "c_f1" in which_metrics:
        error_counts["c_tp"] += (y_pred * y_true).sum(0).cpu().data.numpy()
        error_counts["c_fp"] += (y_pred * (1 - y_true)).sum(0).cpu().data.numpy()
        error_counts["c_fn"] += ((1 - y_pred) * y_true).sum(0).cpu().data.numpy()
        error_counts["c_tn"] += ((1 - y_pred) * (1 - y_true)).sum(0).cpu().data.numpy()

    if "i_f1" in which_metrics:
        error_counts["i_tp"] = (y_pred * y_true).sum(1).cpu().data.numpy()
        error_counts["i_fp"] = (y_pred * (1 - y_true)).sum(1).cpu().data.numpy()
        error_counts["i_fn"] = ((1 - y_pred) * y_true).sum(1).cpu().data.numpy()


def compute_metrics(error_counts, which_metrics, eps=1e-8, weights=None):

    ret_metrics = {}

    if "o_f1" in which_metrics:
        pre = (error_counts["o_tp"] + eps) / (
            error_counts["o_tp"] + error_counts["o_fp"] + eps
        )
        rec = (error_counts["o_tp"] + eps) / (
            error_counts["o_tp"] + error_counts["o_fn"] + eps
        )

        o_f1 = 2 * (pre * rec) / (pre + rec)
        ret_metrics["o_f1"] = o_f1

    if "c_f1" in which_metrics:
        pre = (error_counts["c_tp"] + eps) / (
            error_counts["c_tp"] + error_counts["c_fp"] + eps
        )
        rec = (error_counts["c_tp"] + eps) / (
            error_counts["c_tp"] + error_counts["c_fn"] + eps
        )

        f1_perclass = 2 * (pre * rec) / (pre + rec)
        f1_perclass_avg = np.average(f1_perclass, weights=weights)
        ret_metrics["c_f1"] = f1_perclass_avg

    if "i_f1" in which_metrics:
        pre = (error_counts["i_tp"] + eps) / (
            error_counts["i_tp"] + error_counts["i_fp"] + eps
        )
        rec = (error_counts["i_tp"] + eps) / (
            error_counts["i_tp"] + error_counts["i_fn"] + eps
        )

        f1_i = 2 * (pre * rec) / (pre + rec)
        ret_metrics["i_f1"] = f1_i.sum()

    return ret_metrics
