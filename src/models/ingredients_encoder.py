# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IngredientsEncoder(nn.Module):
    def __init__(self, embed_size, num_classes, dropout=0.5, embed_weights=None, scale_grad=False):

        super(IngredientsEncoder, self).__init__()
        embeddinglayer = nn.Embedding(num_classes, embed_size, padding_idx=num_classes-1, scale_grad_by_freq=scale_grad)
        if embed_weights is not None:
            embeddinglayer.weight.data.copy_(embed_weights)
        self.linear = embeddinglayer
        self.dropout = dropout

    def forward(self, x, onehot_flag=False):

        if onehot_flag:
            embeddings = torch.matmul(x, self.linear.weight)
        else:
            embeddings = self.linear(x)

        embeddings = nn.functional.dropout(embeddings, p=self.dropout, training=self.training)
        embeddings = embeddings.permute(0, 2, 1).contiguous()

        return embeddings
