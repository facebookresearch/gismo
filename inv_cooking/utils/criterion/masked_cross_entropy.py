# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# All rights reserved.
# Code adapted from inversecooking
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss


class MaskedCrossEntropyCriterion(_WeightedLoss):
    """
    Masked cross entropy criterion used for the recipe generation
    """

    def __init__(self, ignore_index: int = -100, reduce: bool = False):
        super().__init__()
        self.padding_idx = ignore_index
        self.reduce = reduce

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param logits: shape (batch_size, sequence_len + 1, vocab_size) of type float
        :param targets: shape (batch_size, sequence_len) of type long
        """
        logits = logits[:, :-1, :].contiguous()
        logits = logits.view(logits.size(0) * logits.size(1), -1)
        lprobs = nn.functional.log_softmax(logits, dim=-1)

        targets_sz = targets.size()
        mask = targets.ne(self.padding_idx).float()
        targets = targets.contiguous().view(-1)

        # remove padding idx from targets to allow gathering without error (padded entries will be suppressed later)
        targets[targets == self.padding_idx] = 0

        # cross-entropy loss: gather the probabilities matching the target indices
        loss = -lprobs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze()

        # apply mask and obtain one loss per element in batch
        loss = loss.view(targets_sz)
        loss = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)

        # reduce loss across batch
        if self.reduce:
            loss = loss.mean()
        return loss
