# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
