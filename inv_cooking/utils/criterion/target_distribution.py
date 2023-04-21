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


class TargetDistributionCriterion(nn.Module):
    """
    Compute the difference between the target distribution with the actual distribution
    by computing the cross-entropy.

    The cross-entropy represent the expected message length when encoding one distributing
    with the other and is minimized when the two distributions are aligned.
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
        cross_entropy = -torch.sum(cross_entropy, dim=-1)
        if self.reduction == "mean":
            cross_entropy = cross_entropy.mean()
        return cross_entropy


@torch.jit.script
def _to_target_distribution(targets: torch.Tensor, epsilon: float):
    """
    Create the target distribution associated to the target labels
    :param targets: shape (batch_size, nb_labels) where each label is set independently of the others
    :param epsilon: to avoid division by zero
    """
    nb_target_by_sample = targets.sum(dim=-1, keepdim=True)
    uniform_distribution = torch.tensor(1.0 / targets.size(-1), device=targets.device)
    return torch.where(
        nb_target_by_sample == 0,
        uniform_distribution,
        targets.float() / (nb_target_by_sample + epsilon),
    )
