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


def create_vit_image_encoder(embed_size: int, config: ImageEncoderConfig, image_size: int = 448):
    """
    Create an image encoder based on VIT. Several flavors are available:
    - no classification token: use the full sequence as output
    - pre-trained (1 classification token) => return sequence of size 1
    - multi-classification tokens: return a sequence as long as number of tokens
    """
    if config.n_cls_tokens == 1:
        return OneClassVit(embed_size=embed_size, config=config, image_size=image_size)
    elif config.n_cls_tokens == 0:
        return NoClassVit(embed_size=embed_size, config=config, image_size=image_size)
    else:
        return None


class OneClassVit(nn.Module):
    """
    Simple Wrapper around a potentially pretrained image VIT classifier on imagenet:
    - handles the resizing of the inputs
    - handles the adaptation of the output size
    """

    def __init__(self, embed_size: int, config: ImageEncoderConfig, image_size: int):
        super().__init__()
        if image_size != 384:
            self.interpolate = nn.Upsample(size=(384, 384))
        else:
            self.interpolate = nn.Identity()
        if config.patch_size == 32:
            self.core_vit = timm.vit_base_patch32_384(pretrained=config.pretrained)
        else:
            self.core_vit = timm.vit_base_patch16_384(pretrained=config.pretrained)
        self.core_vit.head = nn.Identity()
        self.adapt_head = self._build_adaptation_head(input_size=768, embed_size=embed_size, dropout=config.dropout)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        :param x: tensor of shape (batch_size, 3, height, width)
        :return shape (batch_size, embedding_size, 1)
        """
        image = self.interpolate(image)
        out = self.core_vit(image)
        out = self.adapt_head(out)
        return out.unsqueeze(-1)

    @staticmethod
    def _build_adaptation_head(input_size: int, embed_size: int, dropout: float):
        if input_size == embed_size:
            return nn.Identity()
        return nn.Sequential(
            nn.Linear(input_size, embed_size, bias=False),
            nn.Dropout(dropout),
            nn.BatchNorm1d(embed_size),
            nn.ReLU(),
        )


class NoClassVit(timm.VisionTransformer):
    """
    Encodes the image using the encoder part of a ViT (vision transformer).
    The flavor of ViT used is the small one (embedding size of 768).

    Implementation inherited from 'timm' but:
    - without the classification token
    - with the full "feature map" as output
    - with interpolation at the end to reduce the sequence length to 49 (same as ResNet)
    """

    def __init__(self, embed_size: int, config: ImageEncoderConfig, image_size: int):
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
            nn.BatchNorm1d(embed_size),
            nn.ReLU(),
        )
