# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
# Code adapted from https://github.com/facebookresearch/image-to-set
# This source code is licensed under the MIT license found in the
# LICENSE file in https://github.com/facebookresearch/image-to-set


import torch
import torch.nn as nn


@torch.jit.script
def soft_iou(
    logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
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


class SoftIoUCriterion(nn.Module):
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
