# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn


@torch.jit.script
def soft_iou(logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Compute the Intersection over Union between the logits and the target
    :param logits: shape (batch_size, nb_labels)
    :param target: shape (batch_size, nb_labels)
    :param epsilon: to avoid division by zero
    """
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=1, keepdim=True) + epsilon
    union = (probs + target - probs * target).sum(dim=1, keepdim=True) + epsilon
    return intersection / union


class SoftIoULoss(nn.Module):
    """
    Compute the negative Intersection over Union (to increase the IoU)
    """

    def __init__(self, reduction: str = "none", epsilon: float = 1e-8):
        super().__init__()
        assert reduction in ["none", "mean"]
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 1.0 - soft_iou(inputs, targets, epsilon=self.epsilon)
        if self.reduction == "mean":
            loss = loss.mean()
        return loss


class TargetDistributionLoss(nn.Module):
    """
    Compare the target distribution with the actual distribution
    """

    def __init__(self, reduction: str = "none", epsilon: float = 1e-8):
        super().__init__()
        assert reduction in ["none", "mean"]
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the input distribution with the target distribution and return
        their negative cross-entropy (perfect match is minimum entropy)
        :param logits: shape (batch_size, nb_labels)
        :param target: shape (batch_size, nb_labels)
        """
        target_distr = _to_target_distribution(targets, epsilon=self.epsilon)
        cross_entropy = target_distr * nn.functional.log_softmax(logits, dim=-1)
        loss = -torch.sum(cross_entropy, dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()
        return loss


@torch.jit.script
def _to_target_distribution(targets: torch.Tensor, epsilon: float):
    """
    Create the target distribution associated to the target labels
    :param targets: shape (batch_size, nb_labels) where each label is set independently of the others
    :param epsilon: to avoid division by zero
    """
    nb_target_by_sample = targets.sum(dim=-1, keepdim=True)
    uniform_distribution = torch.tensor(1.0 / targets.size(-1), device=targets.device)
    return torch.where(nb_target_by_sample == 0,
                       uniform_distribution,
                       targets.float() / (nb_target_by_sample + epsilon))
