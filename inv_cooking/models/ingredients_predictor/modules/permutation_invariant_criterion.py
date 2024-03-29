# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Dict

import torch
import torch.nn as nn

from inv_cooking.models.ingredients_predictor.utils import label2_k_hots


@torch.jit.script
def knn_cross_entropy(
    xs: torch.Tensor, ys: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Find the closest YS point to each of the XS element:
    - The XS and YS points must live in a probability space (sum of axis == 1)
    - The YS points must have non zero probability on each axis (for cross entropy)
      so we use a softening mechanism (with eps) to make sure probabilities are not too low
    Return a tensor with the sum of those distances
    """
    min_distances = []
    batch_size = xs.size(0)
    nb_points = xs.size(1)
    nb_dim = xs.size(2)
    for n in range(batch_size):
        for i in range(nb_points):
            soft_ys = ys[n].clip(min=eps, max=1 - (nb_dim - 1) * eps)
            distances = (-1 * xs[n][i] * torch.log(soft_ys)).sum(dim=-1)
            min_distances.append(torch.min(distances))
    return torch.stack(min_distances).sum()


def chamfer_cross_entropy_distance(
    probs: torch.Tensor, one_hot_targets: torch.Tensor, eps: float = 1e-8
):
    d1 = knn_cross_entropy(one_hot_targets, probs, eps)
    d2 = knn_cross_entropy(probs, one_hot_targets, eps)
    return d1 + d2


class ChamferDistanceType(Enum):
    l2 = 0
    cross_entropy = 1
    unilateral_cross_entropy = 2


class ChamferDistanceCriterion(nn.Module):
    """
    Chamfer distance between points the probability distribution
    """

    def __init__(
        self,
        eos_value: int,
        pad_value: int,
        eps: float = 1e-8,
        distanceType: ChamferDistanceType = ChamferDistanceType.l2,
    ):
        super().__init__()
        self.eos_value = eos_value
        self.pad_value = pad_value
        self.eps = eps
        self.criterion_eos = nn.BCELoss(reduction="none")
        self.distanceType = distanceType

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}

        # Compute the probabilities of each label
        probs = nn.functional.softmax(logits, dim=-1)

        # Find index for eos label
        # eos probability is the one assigned to the position self.eos_value of the softmax
        # this is used with bce loss only
        eos_probs = probs[:, :, self.eos_value]

        # all zeros except position where eos is in the gt
        eos_pos = targets == self.eos_value

        # 1s for gt label positions, 0s starting from eos position in the gt
        eos_head = (targets != self.pad_value) & (targets != self.eos_value)

        # 0s for gt label positions, 1s starting from eos position in the gt
        eos_target = ~eos_head

        # select probabilities to assign to real ingredients (before EOS)
        probs = probs * eos_head.float().unsqueeze(-1)

        # select targets for real ingredients (before EOS)
        target_one_hot = self._targets_to_one_hots(
            targets, num_classes=self.pad_value + 1, remove_eos=True
        )
        target_one_hot = target_one_hot * eos_head.float().unsqueeze(-1)

        # compute the l2 chamfer distance in the probability space
        if self.distanceType == ChamferDistanceType.l2:
            from pytorch3d.loss.chamfer import chamfer_distance as chamfer_distance_l2

            chamfer_losses = chamfer_distance_l2(
                probs[:, :, 1:], target_one_hot.float()
            )
            losses["label_loss"] = chamfer_losses[0]
        elif self.distanceType == ChamferDistanceType.cross_entropy:
            losses["label_loss"] = chamfer_cross_entropy_distance(
                probs[:, :, 1:], target_one_hot.float(), self.eps
            )
        elif self.distanceType == ChamferDistanceType.unilateral_cross_entropy:
            losses["label_loss"] = knn_cross_entropy(
                target_one_hot.float(), probs[:, :, 1:], self.eps
            )

        # compute eos loss
        eos_loss = self.criterion_eos(eos_probs, eos_target.float())

        # eos loss is computed for all timesteps <= eos in gt and
        # equally penalizes the head (all 0s) and the true eos position (1)
        losses["eos_loss"] = (
            0.5
            * (eos_loss * eos_pos.float()).sum(1)
            / (eos_pos.float().sum(1) + self.eps)
            + 0.5
            * (eos_loss * eos_head.float()).sum(1)
            / (eos_head.float().sum(1) + self.eps)
        ).mean()
        return losses

    @staticmethod
    def _targets_to_one_hots(targets: torch.Tensor, num_classes: int, remove_eos: bool):
        """
        Input shape is (batch_size, max_num_ingredients + 1)
        Output shape is (batch_size, max_num_ingredients + 1, vocab_size)
        """
        one_hots = nn.functional.one_hot(targets, num_classes=num_classes)
        # Remove the pad value (not in predicted targets)
        one_hots = one_hots[:, :, :-1]
        if remove_eos:
            one_hots = one_hots[:, :, 1:]
        return one_hots


class BiPartiteAssignmentCriterion(nn.Module):
    """
    Permutation invariant loss for set prediction meant for Auto Regressive models:
    - the input is a list of predictions (batch_size, max_num_ingredients + 1, vocab_size)
    - the target of shape (batch_size, max_num_ingredients + 1) with EOS + padding after EOS
    - the end-of-sequence token indicates the end of the sequence
    - the order is eliminated by using a bipartite matching algorithm
    """

    def __init__(self, eos_value: int, pad_value: int, eps: float = 1e-8):
        super().__init__()
        self.eos_value = eos_value
        self.pad_value = pad_value
        self.eps = eps
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        from scipy.optimize import linear_sum_assignment

        losses: Dict[str, torch.Tensor] = {}

        matching_logits = []
        eos_logits = []
        matching_targets = []

        batch_size = targets.size(0)
        for i in range(batch_size):
            logit = logits[i]
            target = targets[i]
            eos_indices = torch.nonzero(target == self.eos_value)
            if len(eos_indices):
                eos_index = eos_indices.item()
                eos_logits.append(logit[eos_index:])
                target = target[:eos_index]
                logit = logit[:eos_index]
            cost_matrix = logit[:, target]  # shape is (target_count, target_count)
            cost_matrix = cost_matrix.cpu().detach().numpy()
            row_indices, col_indices = linear_sum_assignment(cost_matrix, maximize=True)
            matching_logits.append(logit[row_indices])
            matching_targets.append(target[col_indices])

        matching_logits = torch.cat(matching_logits)
        matching_targets = torch.cat(matching_targets)
        losses["label_loss"] = self.cross_entropy(matching_logits, matching_targets)

        if eos_logits:
            eos_logits = torch.cat(eos_logits)
            eos_targets = torch.zeros(size=(eos_logits.size(0),)).type_as(targets)
            losses["eos_loss"] = self.cross_entropy(eos_logits, eos_targets)
        return losses


class PooledBinaryCrossEntropy(nn.Module):
    """
    Permutation invariant loss for set prediction meant for Auto Regressive models:
    - the input is a list of predictions (batch_size, max_num_ingredients + 1, vocab_size)
    - the target of shape (batch_size, max_num_ingredients + 1) with EOS + padding after EOS
    - the end-of-sequence token indicates the end of the sequence
    - the order is eliminated by using max-pooling on the whole sequence
    """

    def __init__(self, eos_value: int, pad_value: int, eps: float = 1e-8):
        super().__init__()
        self.eos_value = eos_value
        self.pad_value = pad_value
        self.eps = eps
        self.crit = nn.BCELoss(reduction="mean")
        self.crit_eos = nn.BCELoss(reduction="none")

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}

        # Compute the probabilities of each label
        probs = nn.functional.softmax(logits, dim=-1)

        # Find index for eos label
        # eos probability is the one assigned to the position self.eos_value of the softmax
        # this is used with bce loss only
        eos_probs = probs[:, :, self.eos_value]

        # all zeros except position where eos is in the gt
        eos_pos = target == self.eos_value

        # 1s for gt label positions, 0s starting from eos position in the gt
        eos_head = (target != self.pad_value) & (target != self.eos_value)

        # 0s for gt label positions, 1s starting from eos position in the gt
        eos_target = ~eos_head

        # select transformer steps to pool (steps corresponding to set elements, i.e. labels)
        probs = probs * eos_head.float().unsqueeze(-1)

        # maximum pool
        probs, _ = torch.max(probs, dim=1)

        # compute label loss
        target_k_hot = label2_k_hots(target, self.pad_value, remove_eos=True)
        losses["label_loss"] = self.crit(probs[:, 1:], target_k_hot.float())

        # compute eos loss
        eos_loss = self.crit_eos(eos_probs, eos_target.float())

        # eos loss is computed for all timesteps <= eos in gt and
        # equally penalizes the head (all 0s) and the true eos position (1)
        losses["eos_loss"] = (
            0.5
            * (eos_loss * eos_pos.float()).sum(1)
            / (eos_pos.float().sum(1) + self.eps)
            + 0.5
            * (eos_loss * eos_head.float()).sum(1)
            / (eos_head.float().sum(1) + self.eps)
        ).mean()
        return losses
