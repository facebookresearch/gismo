# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from inv_cooking.models.modules.utils import freeze_fn


class ImageEncoder(nn.Module):
    def __init__(
        self, embed_size, dropout=0.5, model="resnet50", pretrained=True, freeze="none"
    ):
        """Load the pretrained model and replace top fc layer."""
        super(ImageEncoder, self).__init__()

        pretrained_net = globals()[model](pretrained=pretrained)

        if "resnet" in model or "resnext" in model:
            modules = list(pretrained_net.children())[
                :-2
            ]  # delete avg pooling and last fc layer
        else:
            raise ValueError("Invalid image model {}".format(model))

        self.pretrained_net = nn.Sequential(*modules)
        in_dim = pretrained_net.fc.in_features

        if in_dim == embed_size:
            self.last_module = None
        else:
            self.last_module = nn.Sequential(
                nn.Conv2d(in_dim, embed_size, kernel_size=1, padding=0, bias=False),
                nn.Dropout(dropout),
                nn.BatchNorm2d(embed_size, momentum=0.01),
                nn.ReLU(),
            )

        # eventually freeze image encoder
        if freeze == "pretrained":
            freeze_fn(self.pretrained_net)
        elif freeze == "all":
            freeze_fn(self.pretrained_net)
            if self.last_module is not None:
                freeze_fn(self.last_module)

    def forward(self, images, keep_cnn_gradients=False):
        """Extract feature vectors from input images."""

        if images is None:
            return None

        pretrained_feats = self.pretrained_net(images)

        # Apply last_module to change the number of channels in the encoder output
        if self.last_module is not None:
            features = self.last_module(pretrained_feats)
        else:
            features = pretrained_feats

        # Reshape features
        features = features.view(features.size(0), features.size(1), -1)

        return features