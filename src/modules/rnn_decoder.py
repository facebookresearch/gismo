# Copyright (c) Facebook, Inc. and its affiliates.
#
# Code partially based on https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/models/AttModel.py
#
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import random
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AttentionLayer(nn.Module):

    def __init__(self, embed_size, hidden_size):
        super(AttentionLayer, self).__init__()

        self.linear_feats = nn.Sequential(
            nn.Conv1d(embed_size, hidden_size, kernel_size=1), nn.ReLU())
        self.embed_feats = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=1))

        self.linear_hidden = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.embed_hidden = nn.Sequential(nn.Linear(hidden_size, hidden_size))

        self.att_coeffs = nn.Conv1d(hidden_size, 1, kernel_size=1)
        self.linear_out = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())

    def forward(self, features, hidden):

        features = self.linear_feats(features)
        features_embed = self.embed_feats(features)

        hidden = self.linear_hidden(hidden)
        hidden_embed = self.embed_hidden(hidden)

        hidden_repeat = hidden_embed.unsqueeze(2)
        hidden_repeat = hidden_repeat.expand(-1, -1, features_embed.size(-1))

        merged = nn.Tanh()(features_embed + hidden_repeat)
        att_coeffs = self.att_coeffs(merged)
        att_coeffs = torch.nn.functional.softmax(att_coeffs, dim=2)

        att_coeffs_exp = att_coeffs.expand(-1, features.size(1), -1)

        v = att_coeffs_exp * features

        v = v.sum(dim=-1) + hidden
        out = self.linear_out(v)

        return out, att_coeffs.squeeze()


class LSTMAtt(nn.Module):

    def __init__(self, embed_size, hidden_size):
        """Set the hyper-parameters and build the layers."""
        super(LSTMAtt, self).__init__()

        self.lstmcell = nn.LSTMCell(embed_size * 2, hidden_size)

        self.attention = AttentionLayer(embed_size, hidden_size)

        self.hidden_size = hidden_size

    def forward(self, input_feat, features, prev_word, states):

        if states is None:
            states = (torch.zeros(input_feat.size(0), self.hidden_size).cuda(),
                      torch.zeros(input_feat.size(0), self.hidden_size).cuda())

        input = torch.cat((input_feat, prev_word), 1)
        states = self.lstmcell(input, states)

        v, att_coeffs = self.attention(features, states[0])

        return v, states, att_coeffs


class DecoderRNN(nn.Module):

    def __init__(self,
                 embed_size,
                 hidden_size,
                 vocab_size,
                 dropout=0.5,
                 seq_length=20,
                 num_instrs=15):

        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(hidden_size, vocab_size - 1)
        self.core = LSTMAtt(embed_size, hidden_size)

        self.seq_length = seq_length * num_instrs
        self.dropout = dropout
        self.embed_size = embed_size

    def forward(self, features, mask, captions):

        embeddings = self.embed(captions)

        avg_feats = torch.mean(features, dim=-1)

        outputs = []
        states = None
        sampled_ids = []

        input_feat = avg_feats
        for i in range(embeddings.size(1)):

            v, states, atts = self.core(input_feat, features, embeddings[:, i], states)
            input_feat = v
            v = torch.nn.functional.dropout(v, p=self.dropout, training=self.training)
            o = self.linear(v)

            _, predicted = o.max(1)
            sampled_ids.append(predicted)

            outputs.append(o.unsqueeze(1))

        outs = torch.cat(outputs, 1)

        return outs, sampled_ids

    def sample(self,
               features,
               mask,
               greedy=True,
               temperature=1.0,
               first_token_value=0,
               replacement=True):
        """Generate captions for given image features."""
        logits = []
        avg_feats = torch.mean(features, dim=-1)

        inputs = avg_feats
        states = None
        fs = features.size(0)
        prev_word = torch.ones(fs, 1).cuda().long() * first_token_value
        sampled_ids = [prev_word]
        prev_word = self.embed(prev_word).squeeze(1)
        for i in range(self.seq_length):
            v, states, att_coeffs = self.core(inputs, features, prev_word, states)
            inputs = v
            outputs = self.linear(v)

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

            logits.append(outputs)
            # outputs = torch.nn.functional.log_softmax(outputs, dim=1)
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
            prev_word = self.embed(predicted)

        logits = torch.stack(logits, 1)
        sampled_ids = torch.stack(sampled_ids[1:], 1)

        return sampled_ids, logits
