# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from inv_cooking.config import (
    ImageEncoderConfig,
    IngredientPredictorConfig,
)
from inv_cooking.config.config import TitleEncoderConfig
from inv_cooking.models.image_encoder import create_image_encoder
from inv_cooking.models.ingredients_predictor import create_ingredient_predictor


class Im2Ingr(nn.Module):
    def __init__(
        self,
        image_encoder_config: ImageEncoderConfig,
        title_encoder_config: TitleEncoderConfig,
        ingr_pred_config: IngredientPredictorConfig,
        title_vocab_size: int,
        ingr_vocab_size: int,
        max_num_ingredients: int,
        ingr_eos_value: int,
    ):
        super().__init__()
        self.ingr_vocab_size = ingr_vocab_size

        self.image_encoder = create_image_encoder(
            ingr_pred_config.embed_size, image_encoder_config
        )

        if title_encoder_config.with_title:
            self.title_encoder = self.create_title_encoder(
                config=title_encoder_config,
                title_vocab_size=title_vocab_size,
                embed_size=ingr_pred_config.embed_size,
            )
        else:
            self.title_encoder = nn.Identity()

        self.ingr_predictor = create_ingredient_predictor(
            ingr_pred_config,
            vocab_size=ingr_vocab_size,
            max_num_ingredients=max_num_ingredients,
            eos_value=ingr_eos_value,
        )

    @staticmethod
    def create_title_encoder(config: TitleEncoderConfig, title_vocab_size: int, embed_size: int):
        return nn.Embedding(num_embeddings=title_vocab_size, embedding_dim=embed_size)

    def forward(
        self,
        image: torch.Tensor,
        title: Optional[torch.Tensor] = None,
        target_ingredients: Optional[torch.Tensor] = None,
        compute_losses: bool = False,
        compute_predictions: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Predict a set of ingredients from an image
        :param image: input image - shape is (N, C, H, W)
        :param target_ingredients: ground truth of ingredients to predict - shape is (N, max_num_ingredients)
        :param compute_losses: whether or not to compute the loss
        :param compute_predictions: whether or not to output the ingredients prediction
        """

        # Compute the representation of the image
        img_features = self.image_encoder(image)  # shape (N, C, seq1)

        # If title is provided, combine the sequences of both representations
        if title is not None:
            title_features = self.title_encoder(title)  # shape (N, seq2, C)
            title_features = title_features.permute((0, 2, 1))  # shape (N, C, seq2)
            features = torch.cat([img_features, title_features], dim=2)
        else:
            features = img_features

        # Then feed the sequence to the ingredient predictor
        return self.ingr_predictor(
            features,
            label_target=target_ingredients,
            compute_losses=compute_losses,
            compute_predictions=compute_predictions,
        )
