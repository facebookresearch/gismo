# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Tuple

import torch
import torch.nn as nn


class FFDecoder(nn.Module):
    def __init__(
        self,
        embed_size: int,
        vocab_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        n_layers: int = 1,
    ):
        super().__init__()
        self.fc_layers = self._create_hidden_layers(
            embed_size, n_layers, hidden_size, dropout
        )
        if self.fc_layers is not None:
            in_dim = hidden_size
        else:
            in_dim = embed_size
        self.classifier = nn.Linear(in_dim, vocab_size - 1)
        self.fc_cardinality = None

    def add_cardinality_prediction(self, max_num_labels: int):
        self.fc_cardinality = nn.Linear(self.classifier.in_features, max_num_labels)

    def forward(
        self, img_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        :param img_features: features of the image - shape (N, embedding_size, sequence_length)
        """
        feat = self._average_pooling(img_features)
        if self.fc_layers is not None:
            feat = self.fc_layers(feat)
        logits = self.classifier(feat)
        if self.fc_cardinality is not None:
            return logits, self.fc_cardinality(feat)
        else:
            return logits, None

    @staticmethod
    def _average_pooling(img_features):
        return torch.mean(img_features, dim=-1)

    @staticmethod
    def _create_hidden_layers(
        in_dim: int, n_layers: int, hidden_size: int, dropout: float
    ) -> Optional[nn.Module]:
        fc_layers = []
        for i in range(n_layers):
            fc_layers.extend(
                [
                    nn.Linear(in_dim, hidden_size, bias=False),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(hidden_size, momentum=0.01),
                    nn.ReLU(),
                ]
            )
        if len(fc_layers) != 0:
            return nn.Sequential(*fc_layers)
        return None
