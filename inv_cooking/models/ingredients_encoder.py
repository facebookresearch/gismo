# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.nn as nn


class IngredientsEncoder(nn.Module):
    def __init__(
        self,
        embed_size: int,
        voc_size: int,
        dropout=0.5,
        embed_weights=None,
        scale_grad=False,
    ):
        super().__init__()
        embeddinglayer = nn.Embedding(
            num_embeddings=voc_size,
            embedding_dim=embed_size,
            padding_idx=voc_size - 1,
            scale_grad_by_freq=scale_grad,
        )
        if embed_weights is not None:
            embeddinglayer.weight.data.copy_(embed_weights)
        self.linear = embeddinglayer
        self.dropout_prob = dropout

    def forward(self, x: torch.Tensor, onehot_flag=False):
        if onehot_flag:
            embeddings = torch.matmul(x, self.linear.weight)
        else:
            embeddings = self.linear(x)
        embeddings = self._drop_out(embeddings)
        return embeddings.permute(0, 2, 1).contiguous()

    def _drop_out(self, embeddings):
        return nn.functional.dropout(
            embeddings, p=self.dropout_prob, training=self.training
        )
