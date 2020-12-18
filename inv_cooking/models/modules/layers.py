# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# Code from https://github.com/pytorch/fairseq
#
# This source code is licensed under the license found in the LICENSE file in
# https://github.com/pytorch/fairseq. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn


def make_positions(tensor: torch.Tensor, padding_idx: int, left_pad: bool):
    """
    Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    # creates tensor from scratch - to avoid multigpu issues
    max_pos = padding_idx + 1 + tensor.size(1)
    # if not hasattr(make_positions, 'range_buf'):
    range_buf = tensor.new()
    # make_positions.range_buf = make_positions.range_buf.type_as(tensor)
    if range_buf.numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=range_buf)
    mask = tensor.ne(padding_idx)
    positions = range_buf[: tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)

    out = tensor.clone()
    out = out.masked_scatter_(mask, positions[mask])
    return out


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, left_pad: int):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.left_pad = left_pad

    def forward(self, input: torch.Tensor, incremental_state=None):
        """
        Input is expected to be of shape (batch_size, sequence_len)
        """
        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            positions = input.data.new(1, 1).fill_(self.padding_idx + input.size(1))
        else:

            positions = make_positions(input.data, self.padding_idx, self.left_pad)
        return super().forward(positions)

    def max_positions(self):
        """Maximum number of supported positions."""
        return self.num_embeddings - self.padding_idx - 1
