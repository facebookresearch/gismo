import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from inv_cooking.models.modules.multihead_attention import MultiheadAttention
from inv_cooking.models.modules.positional_embedding import (
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from inv_cooking.models.modules.transformer_decoder import Linear

class TransformerEncoderLayer(nn.Module):
    """Encoder layer block."""
    def __init__(self, embed_dim, n_att, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()

        self.dropout = dropout

        # self attention
        self.self_attn = MultiheadAttention(embed_dim, n_att, dropout=dropout)

        # feed-forward layers
        self.linear1 = Linear(embed_dim, embed_dim)
        self.linear2 = Linear(embed_dim, embed_dim)

        # layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # activations
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu

    def forward(self,
                x,
                mask,
                incremental_state):

        residual = x
        x, _ = self.self_attn(
             query=x,
             key=x,
             value=x,
             key_padding_mask=mask,
             incremental_state=incremental_state,
             static_kv=True,
        )
        x = residual + F.dropout(x, p=self.dropout, training=self.training)

        residual = self.norm1(x)
        x = self.activation(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear2(x)
        x = residual + F.dropout(x, p=self.dropout, training=self.training)
        x = self.norm2(x)
        return x


class EncoderTransformer(nn.Module):
    """Transformer encoder."""

    def __init__(
        self,
        embed_size,
        dropout=0.5,
        attention_nheads=16,
        pos_embeddings=True,
        num_layers=8,
        learned=True,
        activation="relu",
    ):
        super(EncoderTransformer, self).__init__()
        self.dropout = dropout

        if pos_embeddings:
            self.embed_positions = PositionalEmbedding(
                217, embed_size, 0, left_pad=False, learned=learned  ## TODO: 217 works with resnet50+20 ingr predictions
            )
        else:
            self.embed_positions = None

        self.embed_scale = math.sqrt(embed_size)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerEncoderLayer(
                    embed_size,
                    attention_nheads,
                    dropout=dropout,
                    activation=activation,
                )
                for i in range(num_layers)
            ]
        )

    def forward(
        self, features, masks, incremental_state=None
    ):

        # embed positions
        if self.embed_positions is not None:
            positions = self.embed_positions(
                features, incremental_state=incremental_state
            )

        x = self.embed_scale * features
        if self.embed_positions is not None:
            x += positions

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x C x T -> T x B x C
        features = features.permute(2, 0, 1)

        # encoder layers
        for layer in self.layers:
            x = layer(features, masks, incremental_state)

        # T x B x C -> B x C x T
        x = x.permute(1, 2, 0)

        return x



