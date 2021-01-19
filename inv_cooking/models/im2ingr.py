# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from inv_cooking.config import (
    ImageEncoderConfig,
    IngredientPredictorConfig,
)
from inv_cooking.models.image_encoder import create_image_encoder
from inv_cooking.models.ingredients_predictor import create_ingredient_predictor
from inv_cooking.models.modules.utils import freeze_fn


class Im2Ingr(nn.Module):
    def __init__(
        self,
        image_encoder_config: ImageEncoderConfig,
        ingr_pred_config: IngredientPredictorConfig,
        ingr_vocab_size: int,
        max_num_ingredients: int,
        ingr_eos_value: int,
    ):
        super().__init__()
        self.ingr_vocab_size = ingr_vocab_size

        self.image_encoder = create_image_encoder(
            ingr_pred_config.embed_size, image_encoder_config
        )

        self.ingr_predictor = create_ingredient_predictor(
            ingr_pred_config,
            vocab_size=ingr_vocab_size,
            max_num_ingredients=max_num_ingredients,
            eos_value=ingr_eos_value,
        )

        # load pretrained model from checkpoint
        if ingr_pred_config.load_pretrained_from != "None":
            pretrained_model = torch.load(ingr_pred_config.load_pretrained_from)
            pretrained_image_encoder_dict = {k[len('model.image_encoder.'):]: v for k, v in pretrained_model['state_dict'].items() if 'image_encoder' in k}
            self.image_encoder.load_state_dict(pretrained_image_encoder_dict)
            pretrained_ingr_predictor_dict = {k[len('model.ingr_predictor.'):]: v for k, v in pretrained_model['state_dict'].items() if 'ingr_predictor' in k}
            self.ingr_predictor.load_state_dict(pretrained_ingr_predictor_dict)

        # freeze pretrained model
        if ingr_pred_config.freeze:
            freeze_fn(self.image_encoder)
            freeze_fn(self.ingr_predictor)

    def forward(
        self,
        image: torch.Tensor,
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
        img_features = self.image_encoder(image)
        return self.ingr_predictor(
            img_features,
            label_target=target_ingredients,
            compute_losses=compute_losses,
            compute_predictions=compute_predictions,
        )
