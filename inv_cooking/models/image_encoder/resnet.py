# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torchvision.models.resnet as resnet

from inv_cooking.config import ImageEncoderConfig
from inv_cooking.models.modules.utils import freeze_fn


class ResnetImageEncoder(nn.Module):
    """
    Extract feature vectors from input images.
    """

    def __init__(
        self,
        embed_size: int,
        config: ImageEncoderConfig,
    ):
        super().__init__()

        # Load the pre-trained resnet encoder
        pretrained_net = resnet.__dict__[config.model](pretrained=config.pretrained)

        # Delete avg pooling and last fc layer
        modules = list(pretrained_net.children())[:-2]
        self.pretrained_net = nn.Sequential(*modules)

        # Adapt the output dimension in case of mismatch
        in_dim = pretrained_net.fc.in_features
        self.last_module = self._build_adaptation_head(in_dim, embed_size, dropout=config.dropout)
        self._freeze_layers(config.freeze)

    @staticmethod
    def _build_adaptation_head(input_size: int, embed_size: int, dropout: float):
        if input_size == embed_size:
            return None
        else:
            return nn.Sequential(
                nn.Conv2d(input_size, embed_size, kernel_size=1, padding=0, bias=False),
                nn.Dropout(dropout),
                nn.BatchNorm2d(embed_size, momentum=0.01),
                nn.ReLU(),
            )

    def _freeze_layers(self, freeze: bool):
        if freeze:
            freeze_fn(self.pretrained_net)

    def forward(self, images: torch.Tensor, return_reshaped_features=True):
        if images is None:
            return None

        features = self.pretrained_net(images)
        if self.last_module is not None:
            features = self.last_module(features)
        if return_reshaped_features:
            return features.reshape(features.size(0), features.size(1), -1)
        else:
            return features
