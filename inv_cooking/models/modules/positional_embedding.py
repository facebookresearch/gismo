# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# Code adapted from https://github.com/pytorch/fairseq
#
# This source code is licensed under the license found in the LICENSE file in
# https://github.com/pytorch/fairseq. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import math

import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    """
    This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(
        self,
        embedding_dim: int,
        padding_idx: int,
        left_pad: bool,
        init_size: int = 1024,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer("_float_tensor", torch.FloatTensor())

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None):
        """Input is expected to be of size [bsz x seqlen]."""
        # recompute/expand embeddings if needed
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.type_as(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            return self.weights[self.padding_idx + seq_len, :].expand(bsz, 1, -1)

        positions = make_positions(input.data, self.padding_idx, self.left_pad)
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int, left_pad: int
    ):
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


def PositionalEmbedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int,
    left_pad: bool,
    learned: bool = False,
):
    if learned:
        m = LearnedPositionalEmbedding(
            num_embeddings, embedding_dim, padding_idx, left_pad
        )
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(
            embedding_dim, padding_idx, left_pad, num_embeddings
        )
    return m


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
