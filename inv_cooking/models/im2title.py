# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Dict, Tuple

import torch
import torch.nn as nn

from inv_cooking.config import (
    ImageEncoderConfig,
    RecipeGeneratorConfig,
    EncoderAttentionType,
)
from inv_cooking.models.image_encoder import create_image_encoder
from inv_cooking.models.modules.transformer_encoder import EncoderTransformer
from inv_cooking.models.recipe_generator import RecipeGenerator


class Im2Title(nn.Module):
    def __init__(
        self,
        image_encoder_config: ImageEncoderConfig,
        embed_size: int,
        title_gen_config: RecipeGeneratorConfig,
        title_vocab_size: int,
        max_title_len: int,
    ):
        super().__init__()
        self.title_vocab_size = title_vocab_size
        self.encoder_attn = title_gen_config.encoder_attn

        self.image_encoder = create_image_encoder(embed_size, image_encoder_config)

        if embed_size != title_gen_config.embed_size:
            self.img_features_transform = nn.Sequential(
                nn.Conv2d(embed_size, title_gen_config.embed_size, kernel_size=1, padding=0, bias=False),
                nn.Dropout(title_gen_config.dropout),
                nn.BatchNorm2d(title_gen_config.embed_size, momentum=0.01),
                nn.ReLU(),
            )
        else:
            self.img_features_transform = nn.Identity()

        if self.encoder_attn == EncoderAttentionType.concat_tf:
            self.transformer_encoder = EncoderTransformer(
                title_gen_config.embed_size,
                dropout=title_gen_config.dropout,
                attention_nheads=title_gen_config.n_att_heads,
                pos_embeddings=False,
                num_layers=title_gen_config.tf_enc_layers,
                learned=False,
                activation=title_gen_config.activation,
            )
            
        # recipe generator
        if self.encoder_attn in [EncoderAttentionType.concat, EncoderAttentionType.concat_tf]:
            num_cross_attn = 1
        elif self.encoder_attn in [EncoderAttentionType.seq_img_first, EncoderAttentionType.seq_ingr_first]:
            num_cross_attn = 2
        else:
            assert False, f"Invalid encoder attention type: {self.encoder_attn}"

        self.recipe_gen = RecipeGenerator(
            title_gen_config, title_vocab_size, max_title_len, num_cross_attn,
        )

    def forward(
        self,
        image: torch.Tensor,
        target_title: torch.Tensor,
        compute_losses: bool = False,
        compute_predictions: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Predict the ingredients and the recipe for the provided image
        :param image: input image from which to predict ingredient and recipe - shape (N, C, H, W)
        :param target_title: target title to predict - shape (N, max_title_len)
        :param compute_losses: whether or not to compute the loss
        :param compute_predictions: whether or not to output the recipe prediction
        """
        losses = {}
        img_features = self.image_encoder(image, return_reshaped_features=False)

        # transform image features and create mask where all features are taken into account
        img_features = self.img_features_transform(img_features)
        img_features = img_features.reshape(img_features.size(0), img_features.size(1), -1)
        img_mask = torch.zeros(img_features.shape[0], img_features.shape[2])
        img_mask = img_mask.bool().to(device=img_features.device)

        # generate recipe and compute losses if necessary
        loss, title_predictions = self.recipe_gen(
            features=img_features,
            masks=img_mask,
            recipe_gt=target_title,
            compute_losses=compute_losses,
            compute_predictions=compute_predictions,
        )

        losses["title_loss"] = loss
        return losses, title_predictions
