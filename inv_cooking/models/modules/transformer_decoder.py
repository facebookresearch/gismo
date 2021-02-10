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
import torch.nn.functional as F

from inv_cooking.models.modules.multihead_attention import MultiheadAttention
from inv_cooking.models.modules.positional_embedding import (
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""

    def __init__(self, embed_dim, n_att, dropout=0.5, activation="relu", n_cross_attn=1):
        super().__init__()

        self.embed_dim = embed_dim
        self.dropout = dropout

        # self attention
        self.self_attn = MultiheadAttention(
            self.embed_dim,
            n_att,
            dropout=dropout,
        )

        # cross attention between encoder and decoder
        self.cross_attn = nn.ModuleList(
            [MultiheadAttention(self.embed_dim,
                                n_att,
                                dropout=dropout,
                            )
            for i in range(n_cross_attn)]
        )

        self.fc1 = Linear(self.embed_dim, self.embed_dim)
        self.fc2 = Linear(self.embed_dim, self.embed_dim)

        # layer normalizations: we have one for the cross attn, one for the self attn and one for the final module
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for i in range(3)]
        )

        # activations
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu

    def forward(
        self,
        x,
        encoder_out,
        encoder_padding_mask,
        incremental_state,
        activation="relu",
    ):

        # self attention
        residual = x
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask_future_timesteps=True,
            incremental_state=incremental_state,
            need_weights=False,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.layer_norms[0](x)

        # cross attention
        residual = x

        assert (len(encoder_out) == len(encoder_padding_mask)) and (len(encoder_out) == len(self.cross_attn))
        for i, (e, m) in enumerate(zip(encoder_out, encoder_padding_mask)):
            x, _ = self.cross_attn[i](
                query=x,
                key=e,
                value=e,
                key_padding_mask=m,
                incremental_state=incremental_state,
                static_kv=True,
            )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.layer_norms[1](x)

        # final operations
        residual = x
        x = self.activation(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.layer_norms[2](x)
        return x


class DecoderTransformer(nn.Module):
    """Transformer decoder."""

    def __init__(
        self,
        embed_size,
        vocab_size,
        dropout=0.5,
        seq_length=20,
        attention_nheads=16,
        pos_embeddings=True,
        num_layers=8,
        learned=True,
        num_cross_attn=1,
        activation="relu",
    ):
        super(DecoderTransformer, self).__init__()
        self.dropout = dropout
        self.seq_length = seq_length
        self.embed_tokens = Embedding(
            vocab_size, embed_size, padding_idx=vocab_size - 1
        )

        if pos_embeddings:
            self.embed_positions = PositionalEmbedding(
                self.seq_length+1, embed_size, 0, left_pad=False, learned=learned
            )
        else:
            self.embed_positions = None

        self.embed_scale = math.sqrt(embed_size)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(
                    embed_size,
                    attention_nheads,
                    dropout=dropout,
                    activation=activation,
                    n_cross_attn=num_cross_attn,
                )
                for i in range(num_layers)
            ]
        )

        self.linear = Linear(embed_size, vocab_size - 1)

    def forward(
        self, features, masks, captions, incremental_state=None
    ):

        if not isinstance(features, list):
            features = [features]
        if not isinstance(masks, list):
            masks = [masks]

        # B x C x T -> T x B x C
        features = [f.permute(2, 0, 1) for f in features]

        # embed positions
        if self.embed_positions is not None:
            positions = self.embed_positions(
                captions, incremental_state=incremental_state
            )
        if incremental_state is not None:
            if self.embed_positions is not None:
                positions = positions[:, -1:]
            captions = captions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(captions)
        if self.embed_positions is not None:
            x += positions

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        for layer in self.layers:
            x = layer(x, features, masks, incremental_state)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        x = self.linear(x)
        _, predicted = x.max(dim=-1)

        if incremental_state is None:
            return x, predicted
        else:
            return x

    def sample(
        self,
        features,
        masks,
        greedy=True,
        temperature=1.0,
        first_token_value=0,
        replacement=True,
    ):

        incremental_state = {}

        if not isinstance(features, list):
            features = [features]
        if not isinstance(masks, list):
            masks = [masks]

        # create dummy previous word
        assert all(f.size(0) == features[0].size(0) for f in features)
        fs = features[0].size(0)
        first_word = torch.ones(fs).type_as(features[0]) * first_token_value       

        first_word = first_word.long()
        sampled_ids = [first_word]
        logits = []
        for i in range(self.seq_length):
            # forward
            outputs = self.forward(
                features=features,
                masks=masks,
                captions=torch.stack(sampled_ids, 1),
                incremental_state=incremental_state,
            )
            outputs = outputs.squeeze(1)
            if not replacement:
                # predicted mask
                if i == 0:
                    predicted_mask = torch.zeros(outputs.shape).type_as(outputs)
                else:
                    batch_ind = [j for j in range(fs) if sampled_ids[i][j] != 0]
                    sampled_ids_new = sampled_ids[i][batch_ind]
                    predicted_mask[batch_ind, sampled_ids_new] = float("-inf")

                # mask previously selected ids
                outputs += predicted_mask

            # add outputs to list
            logits.append(outputs)

            if greedy:
                _, predicted = outputs.max(1)
                predicted = predicted.detach()
            else:
                k = 10
                prob_prev = torch.div(outputs.squeeze(1), temperature)
                prob_prev = torch.nn.functional.softmax(prob_prev, dim=-1).data

                # top k random sampling
                prob_prev_topk, indices = torch.topk(prob_prev, k=k, dim=1)
                predicted = torch.multinomial(prob_prev_topk, 1).view(-1)
                predicted = torch.index_select(indices, dim=1, index=predicted)[
                    :, 0
                ].detach()

            sampled_ids.append(predicted)
        sampled_ids = torch.stack(sampled_ids[1:], 1)
        logits = torch.stack(logits, 1)
        return sampled_ids, logits

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if "decoder.embed_positions.weights" in state_dict:
                del state_dict["decoder.embed_positions.weights"]
            if "decoder.embed_positions._float_tensor" not in state_dict:
                state_dict[
                    "decoder.embed_positions._float_tensor"
                ] = torch.FloatTensor()
        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m


def Linear(in_features: int, out_features: int, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.0)
    return m
