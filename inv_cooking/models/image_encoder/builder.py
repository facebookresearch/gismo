# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from inv_cooking.config import ImageEncoderConfig
from .resnet import ResnetImageEncoder


def create_image_encoder(embed_size: int, config: ImageEncoderConfig):
    if "resnet" in config.model or "resnext" in config.model:
        return ResnetImageEncoder(embed_size, config)
    else:
        raise ValueError("Invalid image model {}".format(config.model))

