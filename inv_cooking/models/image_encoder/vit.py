# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

from typing import NamedTuple
import timm.models.vision_transformer as timm
import torch
import torch.nn as nn

from inv_cooking.config import ImageEncoderConfig


class VitParameters(NamedTuple):
    patch_size: int
    embed_dim: int
    down_sampling: int


_VALID_MODELS = {
    "vit_32_small": VitParameters(patch_size=32, embed_dim=768, down_sampling=1),
    "vit_16_small": VitParameters(patch_size=16, embed_dim=768, down_sampling=4),
}


class VitImageEncoder(timm.VisionTransformer):
    """
    Encodes the image using the encoder part of a ViT (vision transformer).
    The flavor of ViT used is the small one (embedding size of 768).

    Implementation inherited from 'timm' but:
    - without the classification token
    - with the full "feature map" as output
    - with interpolation at the end to reduce the sequence length to 49 (same as ResNet)
    """

    def __init__(self, embed_size: int, config: ImageEncoderConfig, image_size: int = 224):
        assert config.model in _VALID_MODELS, f"Unsupported model {config.model}, use one of {list(_VALID_MODELS.keys())}"
        parameters = _VALID_MODELS[config.model]
        super().__init__(
            img_size=image_size,
            drop_rate=config.dropout,
            patch_size=parameters.patch_size,
            embed_dim=parameters.embed_dim,
        )
        self._remove_classification_token()
        if parameters.down_sampling != 1:
            self.interpolate = nn.Upsample(scale_factor=1 / parameters.down_sampling)
        else:
            self.interpolate = nn.Identity()
        self.adapt_head = self._build_adaptation_head(input_size=768, embed_size=embed_size, dropout=config.dropout)

    def forward(self, x: torch.Tensor):
        """
        :param x: tensor of shape (batch_size, 3, height, width)
        :return shape (batch_size, embedding_size, seq_len)
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = self.interpolate(x)
        return self.adapt_head(x)

    def _remove_classification_token(self):
        # No classification token, so the positional embedding must be changed
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))

    @staticmethod
    def _build_adaptation_head(input_size: int, embed_size: int, dropout: float):
        if input_size == embed_size:
            return nn.Identity()
        return nn.Sequential(
            nn.Conv1d(input_size, embed_size, kernel_size=1, padding=0, bias=False),
            nn.Dropout(dropout),
            nn.BatchNorm1d(embed_size, momentum=0.01),
            nn.ReLU(),
        )
