# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from inv_cooking.config import ImageEncoderConfig

from .clip import ClipBasedEncoder
from .resnet import ResnetImageEncoder
from .vit import create_vit_image_encoder


def create_image_encoder(embed_size: int, config: ImageEncoderConfig):
    if not config.with_image_encoder:
        return nn.Identity()
    if "resnet" in config.model or "resnext" in config.model:
        return ResnetImageEncoder(embed_size, config)
    elif "vit" in config.model:
        return create_vit_image_encoder(
            embed_size, config, image_size=448
        )  # TODO - through configuration
    elif config.model.startswith(ClipBasedEncoder.PREFIX):
        return ClipBasedEncoder(embed_size, config)
    else:
        raise ValueError("Invalid image model {}".format(config.model))
