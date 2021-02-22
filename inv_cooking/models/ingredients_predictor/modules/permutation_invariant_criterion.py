from typing import Dict

import torch
import torch.nn as nn

from inv_cooking.models.ingredients_predictor.utils import label2_k_hots


class SetPooledCrossEntropy(nn.Module):
    """
    Permutation invariant loss for set prediction meant for Auto Regressive models:
    - the input is a list of predictions (batch_size, max_num_ingredients + 1, vocab_size)
    - the end-of-sequence token indicates the end of the sequence
    - the order is eliminated by using max-pooling on the whole sequence
    """

    def __init__(self, eos_value: int, pad_value: int, eps: float = 1e-8):
        super().__init__()
        self.eos_value = eos_value
        self.pad_value = pad_value
        self.crit = nn.BCELoss(reduction="mean")
        self.crit_eos = nn.BCELoss(reduction="none")
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
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
