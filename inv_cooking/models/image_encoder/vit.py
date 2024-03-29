# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import timm.models.vision_transformer as timm
import torch
import torch.nn as nn
import math
from inv_cooking.config import ImageEncoderConfig


def create_vit_image_encoder(
    embed_size: int, config: ImageEncoderConfig, image_size: int = 448
):
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
    elif config.n_cls_tokens > 1:
        return MultiClassVit(
            embed_size=embed_size, config=config, image_size=image_size
        )


class BaseVit(nn.Module):
    def __init__(self, config: ImageEncoderConfig, image_size: int):
        super().__init__()
        self._create_image_size_adapter(image_size)
        self._create_vit_trunk(config)

    def _create_image_size_adapter(self, image_size: int):
        if image_size != 384:
            self.interpolate = nn.Upsample(size=(384, 384))
        else:
            self.interpolate = nn.Identity()

    def _create_vit_trunk(self, config: ImageEncoderConfig):
        if config.patch_size == 32:
            self.pretrained_net = timm.vit_base_patch32_384(
                pretrained=config.pretrained
            )
        else:
            self.pretrained_net = timm.vit_base_patch16_384(
                pretrained=config.pretrained
            )
        self.pretrained_net.head = nn.Identity()
        if config.pretrained and config.pretrained_weights:
            state_dict = torch.load(config.pretrained_weights, map_location="cpu")
            self.pretrained_net.load_state_dict(state_dict, strict=False)


class OneClassVit(BaseVit):
    """
    Simple Wrapper around a potentially pretrained image VIT classifier on imagenet:
    - handles the resizing of the inputs
    - handles the adaptation of the output size
    """

    def __init__(self, embed_size: int, config: ImageEncoderConfig, image_size: int):
        super().__init__(config, image_size)
        self.additional_repr_levels = list(config.additional_repr_levels)
        if self.additional_repr_levels:
            self.additional_repr_norms = nn.ModuleList(
                [
                    nn.LayerNorm(768, eps=1e-6)
                    for _ in range(len(self.additional_repr_levels))
                ]
            )
        self.concatenate_repr_levels = config.concatenate_repr_levels
        input_size_multiplier = (
            (1 + len(self.additional_repr_levels))
            if self.concatenate_repr_levels
            else 1
        )
        self.adapt_head = self._build_adaptation_head(
            input_size=768 * input_size_multiplier,
            embed_size=embed_size,
            dropout=config.dropout,
        )

    def forward(
        self, image: torch.Tensor, return_reshaped_features=True
    ) -> torch.Tensor:
        """
        :param image: tensor of shape (batch_size, 3, height, width)
        :param return_reshaped_features: reshape as a sequence of 1 token
        :return shape (batch_size, embedding_size, seq_len)
        """
        image = self.interpolate(image)
        outputs = self.vit_encoding(self.pretrained_net, image)

        # Combine each output either as a single representation, or as a sequence
        if self.concatenate_repr_levels:
            out = torch.cat(outputs, dim=-1).unsqueeze(-1)
        else:
            out = torch.stack(outputs, dim=-1)
        out = self.adapt_head(out)

        # To stay compatible with the ResNet50 image encoder
        if return_reshaped_features:
            return out
        else:
            return out.unsqueeze(-1)

    def vit_encoding(self, vit, x):
        """
        Adaptation of the implementation of the timm.VisionTransformer.forward_feature
        to extract the representation at several level
        """
        batch_size = x.shape[0]
        x = vit.patch_embed(x)

        cls_tokens = vit.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + vit.pos_embed
        x = vit.pos_drop(x)

        outputs = []
        for i, blk in enumerate(vit.blocks):
            x = blk(x)
            if i in self.additional_repr_levels:
                y = self.additional_repr_norms[self.additional_repr_levels.index(i)](x)
                outputs.append(y[:, 0])

        x = vit.norm(x)
        outputs.append(x[:, 0])
        return outputs

    @staticmethod
    def _build_adaptation_head(input_size: int, embed_size: int, dropout: float):
        if input_size == embed_size:
            return nn.Identity()
        return nn.Sequential(
            nn.Conv1d(input_size, embed_size, kernel_size=1, stride=1, bias=False),
            nn.Dropout(dropout),
            nn.BatchNorm1d(embed_size),
            nn.ReLU(),
        )


class MultiClassVit(BaseVit):
    """
    Encodes the image using the encoder part of a ViT (vision transformer).
    The flavor of ViT used is the small one (embedding size of 768).

    Implementation inherited from 'timm' but:
    - several classification tokens
    - several outputs (one by classification token)
    """

    def __init__(self, embed_size: int, config: ImageEncoderConfig, image_size: int):
        super().__init__(config, image_size)
        self.n_cls_tokens = config.n_cls_tokens
        self._switch_classification_tokens()
        self.adapt_head = self._build_adaptation_head(
            input_size=768, embed_size=embed_size, dropout=config.dropout
        )

    def _switch_classification_tokens(self):
        """
        Repeat the class tokens and its position embedding several times and make
        these learn-able so that the VIT can differentiate them during fine-tuning
        """
        new_cls_tokens = self.pretrained_net.cls_token.repeat(1, self.n_cls_tokens, 1)
        self.pretrained_net.cls_token = nn.Parameter(new_cls_tokens)
        cls_token_pos_embed = self.pretrained_net.pos_embed[:, 0, :].repeat(
            1, self.n_cls_tokens, 1
        )
        new_position_embeddings = torch.cat(
            [cls_token_pos_embed, self.pretrained_net.pos_embed[:, 1:, :]], dim=1
        )
        self.pretrained_net.pos_embed = nn.Parameter(new_position_embeddings)

    def forward(self, x: torch.Tensor, return_reshaped_features=True):
        """
        :param x: tensor of shape (batch_size, 3, height, width)
        :return shape (batch_size, embedding_size, seq_len)
        """
        B = x.shape[0]
        x = self.interpolate(x)
        x = self.pretrained_net.patch_embed(x)
        cls_tokens = self.pretrained_net.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pretrained_net.pos_embed
        x = self.pretrained_net.pos_drop(x)
        for blk in self.pretrained_net.blocks:
            x = blk(x)
        x = self.pretrained_net.norm(x)
        x = x.permute(0, 2, 1)
        x = x[:, :, : self.n_cls_tokens]
        x = self.adapt_head(x)
        if return_reshaped_features:
            return x
        else:
            return x.unsqueeze(-1)

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


class NoClassVit(BaseVit):
    """
    Encodes the image using the encoder part of a ViT (vision transformer).
    The flavor of ViT used is the small one (embedding size of 768).

    Implementation inherited from 'timm' but:
    - without the classification token
    - with the full "feature map" as output
    - with interpolation at the end to reduce the sequence length to 49 (same as ResNet)
    """

    def __init__(self, embed_size: int, config: ImageEncoderConfig, image_size: int):
        super().__init__(config, image_size)
        self._remove_classification_token()
        self.adapt_head = self._build_adaptation_head(
            input_size=768,
            embed_size=embed_size,
            dropout=config.dropout,
            pooling_kernel_size=config.pooling_kernel_size,
            pooling_kernel_dim=config.pooling_kernel_dim,
        )

    def forward(self, x: torch.Tensor, return_reshaped_features=True):
        """
        :param x: tensor of shape (batch_size, 3, height, width)
        :return shape (batch_size, embedding_size, seq_len)
        """
        x = self.interpolate(x)
        x = self.pretrained_net.patch_embed(x)
        x = x + self.pretrained_net.pos_embed
        x = self.pretrained_net.pos_drop(x)

        for blk in self.pretrained_net.blocks:
            x = blk(x)
        x = self.pretrained_net.norm(x)
        x = x.permute(0, 2, 1)

        x = self.adapt_head(x)
        if return_reshaped_features:
            return x
        else:
            return x.unsqueeze(-1)

    def _remove_classification_token(self):
        # No classification token, so the positional embedding must be changed
        self.pretrained_net.pos_embed = nn.Parameter(
            self.pretrained_net.pos_embed[:, 1:, :]
        )

    @staticmethod
    def _build_adaptation_head(
        input_size: int, embed_size: int, dropout: float, pooling_kernel_dim: int, pooling_kernel_size: int
    ) -> nn.Module:
        if input_size == embed_size:
            return nn.Identity()

        if pooling_kernel_dim == 1:
            return nn.Sequential(
                nn.Conv1d(
                    input_size,
                    embed_size,
                    kernel_size=pooling_kernel_size,
                    stride=pooling_kernel_size,
                    padding=0,
                    bias=False,
                ),
                nn.Dropout(dropout),
                nn.BatchNorm1d(embed_size),
                nn.ReLU(),
            )

        return WithSequenceToSquare(
            nn.Sequential(
                nn.Conv2d(
                    input_size,
                    embed_size,
                    kernel_size=(pooling_kernel_size, pooling_kernel_size),
                    stride=(pooling_kernel_size, pooling_kernel_size),
                    padding=0,
                    bias=False,
                ),
                nn.Dropout(dropout),
                nn.BatchNorm2d(embed_size),
                nn.ReLU(),
            )
        )


class WithSequenceToSquare(nn.Module):
    def __init__(self, wrapped: nn.Module):
        super().__init__()
        self.wrapped = wrapped
        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channel, seq_len = x.shape
        height = int(math.sqrt(seq_len))
        x = x.reshape((batch_size, channel, height, -1))
        x = self.wrapped(x)
        return self.flatten(x)
