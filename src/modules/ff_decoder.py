# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class FFDecoder(nn.Module):

    def __init__(self,
                 embed_size,
                 vocab_size,
                 hidden_size,
                 dropout=0.0,
                 pred_cardinality='none',
                 nobjects=10,
                 n_layers=1,
                 use_empty_set=False):
        super(FFDecoder, self).__init__()

        in_dim = embed_size

        # Add fully connected layers
        fc_layers = []
        for i in range(n_layers):
            fc_layers.extend([
                nn.Linear(in_dim, hidden_size, bias=False),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_size, momentum=0.01),
                nn.ReLU()
            ])
            in_dim = hidden_size

        if len(fc_layers) != 0:
            self.fc_layers = nn.Sequential(*fc_layers)
        else:
            self.fc_layers = None

        self.classifier = nn.Sequential(nn.Linear(hidden_size, vocab_size - 1))

        self.pred_cardinality = pred_cardinality
        if self.pred_cardinality != 'none':
            if use_empty_set:
                # This is to account for 0 when using cardinality prediction and dealing with datasets with empty sets
                nobjects += 1
            self.fc_cardinality = nn.Sequential(nn.Linear(hidden_size, nobjects))

    def forward(self, img_features):

        # Apply global average pooling
        feat = torch.mean(img_features, dim=-1)

        # Apply fully connected layers
        if self.fc_layers is not None:
            feat = self.fc_layers(feat)

        # Apply classifier
        logits = self.classifier(feat)

        # Apply cardinality layer
        if self.pred_cardinality == 'dc':
            return logits, nn.ReLU()(self.fc_cardinality(feat))
        elif self.pred_cardinality != 'none':
            return logits, self.fc_cardinality(feat)
        else:
            return logits, None
