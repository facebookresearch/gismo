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
from inv_cooking.models.title_encoder import TitleEncoder


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
            self.title_encoder = TitleEncoder(
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

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        title: Optional[torch.Tensor] = None,
        target_ingredients: Optional[torch.Tensor] = None,
        compute_losses: bool = False,
        compute_predictions: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Predict a set of ingredients from an image
        :param image: input image - shape is (N, C, H, W)
        :param title: input title - shape is (N, seq)
        :param target_ingredients: ground truth of ingredients to predict - shape is (N, max_num_ingredients)
        :param compute_losses: whether or not to compute the loss
        :param compute_predictions: whether or not to output the ingredients prediction
        """

        assert image is not None or title is not None, "Need some input to deduce the ingredients"

        # Compute the representation of the image
        if image is not None:
            img_features = self.image_encoder(image)  # shape (N, C, seq1)
        else:
            img_features = None

        # If title is provided, combine the sequences of both representations
        if title is not None:
            title_features = self.title_encoder(title)  # shape (N, C, seq2)
            if img_features is not None:
                print(img_features.shape, title_features.shape)
                features = torch.cat([img_features, title_features], dim=2)
            else:
                features = title_features
        else:
            features = img_features

        # Then feed the sequence to the ingredient predictor
        return self.ingr_predictor(
            features,
            label_target=target_ingredients,
            compute_losses=compute_losses,
            compute_predictions=compute_predictions,
        )
