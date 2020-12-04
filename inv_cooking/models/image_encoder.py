# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet

from inv_cooking.models.modules.utils import freeze_fn


class ImageEncoder(nn.Module):
    """
    Extract feature vectors from input images.
    """

    def __init__(
        self, embed_size, dropout=0.5, model="resnet50", pretrained=True, freeze="none"
    ):
        super().__init__()

        if "resnet" in model or "resnext" in model:
            pretrained_net = resnet.__dict__[model](pretrained=pretrained)
            # delete avg pooling and last fc layer
            modules = list(pretrained_net.children())[:-2]
            self.pretrained_net = nn.Sequential(*modules)
            in_dim = pretrained_net.fc.in_features
        else:
            raise ValueError("Invalid image model {}".format(model))

        if in_dim == embed_size:
            self.last_module = None
        else:
            self.last_module = nn.Sequential(
                nn.Conv2d(in_dim, embed_size, kernel_size=1, padding=0, bias=False),
                nn.Dropout(dropout),
                nn.BatchNorm2d(embed_size, momentum=0.01),
                nn.ReLU(),
            )
        self._freeze_layers(freeze)

    def _freeze_layers(self, freeze: str):
        if freeze == "pretrained":
            freeze_fn(self.pretrained_net)
        elif freeze == "all":
            freeze_fn(self.pretrained_net)
            if self.last_module is not None:
                freeze_fn(self.last_module)

    def forward(self, images: torch.Tensor):
        if images is None:
            return None

        features = self.pretrained_net(images)
        if self.last_module is not None:
            features = self.last_module(features)
        return features.reshape(features.size(0), features.size(1), -1)
