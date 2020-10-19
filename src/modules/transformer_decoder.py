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
from torch.nn.modules.utils import _single
import modules.utils as utils
from modules.multihead_attention import MultiheadAttention
from modules.layers import make_positions, LearnedPositionalEmbedding
from scipy.stats import entropy
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx, left_pad, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor())

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
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
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""

    def __init__(self, embed_dim, n_att, dropout=0.5, normalize_before=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.dropout = dropout
        self.relu_dropout = dropout
        self.normalize_before = normalize_before
        num_layer_norm = 3

        # self-attention on generated recipe
        self.self_attn = MultiheadAttention(
            self.embed_dim,
            n_att,
            dropout=dropout,
        )

        self.encoder_attn = MultiheadAttention(
            self.embed_dim,
            n_att,
            dropout=dropout,
        )

        self.fc1 = Linear(self.embed_dim, self.embed_dim)
        self.fc2 = Linear(self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(num_layer_norm)])

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state):

        # self attention
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
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
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)

        x, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
        )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)

        residual = x
        x = self.maybe_layer_norm(-1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(-1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class DecoderTransformer(nn.Module):
    """Transformer decoder."""

    def __init__(self,
                 embed_size,
                 vocab_size,
                 dropout=0.5,
                 seq_length=20,
                 num_instrs=15,
                 attention_nheads=16,
                 pos_embeddings=True,
                 num_layers=8,
                 learned=True,
                 normalize_before=True):
        super(DecoderTransformer, self).__init__()
        self.dropout = dropout
        self.seq_length = seq_length * num_instrs
        self.embed_tokens = Embedding(vocab_size, embed_size, padding_idx=vocab_size - 1)
        if pos_embeddings:
            self.embed_positions = PositionalEmbedding(
                1024, embed_size, 0, left_pad=False, learned=learned)
        else:
            self.embed_positions = None

        self.embed_scale = math.sqrt(embed_size)

        self.layer_norm = LayerNorm(embed_size)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(
                embed_size, attention_nheads, dropout=dropout, normalize_before=normalize_before)
            for i in range(num_layers)
        ])

        self.linear = Linear(embed_size, vocab_size - 1)

    def forward(self, features, mask, captions, incremental_state=None):

        if features is not None:
            features = features.permute(0, 2, 1)
            features = features.transpose(0, 1)
            features = self.layer_norm(features)

        if mask is not None:
            mask = (1 - mask.squeeze(1)).byte()

        # embed positions
        if self.embed_positions is not None:
            positions = self.embed_positions(captions, incremental_state=incremental_state)
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
            x = layer(x, features, mask, incremental_state)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        x = self.linear(x)
        _, predicted = x.max(dim=-1)

        if incremental_state is None:
            return x, predicted
        else:
            return x

    def sample(self,
               features,
               mask,
               greedy=True,
               temperature=1.0,
               first_token_value=0,
               replacement=True):

        incremental_state = {}

        # create dummy previous word
        fs = features.size(0)
        first_word = torch.ones(fs) * first_token_value

        first_word = first_word.to(device).long()
        sampled_ids = [first_word]
        logits = []
        for i in range(self.seq_length):
            # forward
            outputs = self.forward(features, mask, torch.stack(sampled_ids, 1), incremental_state)
            outputs = outputs.squeeze(1)
            if not replacement:
                # predicted mask
                if i == 0:
                    predicted_mask = torch.zeros(outputs.shape).float().to(device)
                else:
                    batch_ind = [j for j in range(fs) if sampled_ids[i][j] != 0]
                    sampled_ids_new = sampled_ids[i][batch_ind]
                    predicted_mask[batch_ind, sampled_ids_new] = float('-inf')

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
                predicted = torch.index_select(indices, dim=1, index=predicted)[:, 0].detach()

            sampled_ids.append(predicted)
        sampled_ids = torch.stack(sampled_ids[1:], 1)
        logits = torch.stack(logits, 1)
        return sampled_ids, logits

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'decoder.embed_positions.weights' in state_dict:
                del state_dict['decoder.embed_positions.weights']
            if 'decoder.embed_positions._float_tensor' not in state_dict:
                state_dict['decoder.embed_positions._float_tensor'] = torch.FloatTensor()
        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings)
    return m
