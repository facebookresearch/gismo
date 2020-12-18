# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple, Optional

import torch
import torch.nn as nn

from inv_cooking.config import CardinalityPredictionType


class FFDecoder(nn.Module):
    def __init__(
        self,
        embed_size: int,
        vocab_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        pred_cardinality: CardinalityPredictionType = CardinalityPredictionType.none,
        nobjects: int = 10,  ## for cardinality prediction only
        n_layers: int = 1,
    ):
        super(FFDecoder, self).__init__()

        in_dim = embed_size

        # Add fully connected layers
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
            in_dim = hidden_size

        if len(fc_layers) != 0:
            self.fc_layers = nn.Sequential(*fc_layers)
        else:
            self.fc_layers = None

        self.classifier = nn.Sequential(nn.Linear(hidden_size, vocab_size - 1))

        self.pred_cardinality = pred_cardinality
        if self.pred_cardinality != CardinalityPredictionType.none:
            self.fc_cardinality = nn.Sequential(nn.Linear(hidden_size, nobjects))

    def forward(self, img_features: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        :param img_features: features extract of the image - shape (N, embedding_size, sequence_length)
        """
        feat = self._average_pooling(img_features)
        if self.fc_layers is not None:
            feat = self.fc_layers(feat)
        logits = self.classifier(feat)
        if self.pred_cardinality != CardinalityPredictionType.none:
            return logits, self.fc_cardinality(feat)
        else:
            return logits, None

    @staticmethod
    def _average_pooling(img_features):
        return torch.mean(img_features, dim=-1)
