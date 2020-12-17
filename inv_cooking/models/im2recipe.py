# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from inv_cooking.config import (
    ImageEncoderConfig,
    IngredientPredictorConfig,
    RecipeGeneratorConfig,
)
from inv_cooking.models.image_encoder import ImageEncoder
from inv_cooking.models.ingredients_encoder import IngredientsEncoder
from inv_cooking.models.ingredients_predictor import get_ingr_predictor, mask_from_eos
from inv_cooking.models.recipe_generator import RecipeGenerator


class Im2Recipe(nn.Module):
    def __init__(
        self,
        image_encoder_config: ImageEncoderConfig,
        ingr_pred_config: IngredientPredictorConfig,
        recipe_gen_config: RecipeGeneratorConfig,
        ingr_vocab_size: int,
        instr_vocab_size: int,
        max_num_labels: int,
        max_recipe_len: int,
        ingr_eos_value,
    ):
        super().__init__()
        self.ingr_vocab_size = ingr_vocab_size
        self.ingr_eos_value = ingr_eos_value
        self.instr_vocab_size = instr_vocab_size

        if ingr_pred_config.freeze:
            image_encoder_config.freeze = "all"

        self.image_encoder = ImageEncoder(
            ingr_pred_config.embed_size, image_encoder_config
        )
        self.ingr_predictor = get_ingr_predictor(
            ingr_pred_config,
            vocab_size=ingr_vocab_size,
            maxnumlabels=max_num_labels,
            eos_value=ingr_eos_value,
        )
        self.ingr_encoder = IngredientsEncoder(
            recipe_gen_config.embed_size,
            voc_size=ingr_vocab_size,
            dropout=recipe_gen_config.dropout,
            scale_grad=False,
        )
        self.recipe_gen = RecipeGenerator(
            recipe_gen_config, instr_vocab_size, max_recipe_len
        )

    def forward(
        self,
        image: torch.Tensor,
        recipe_gt: torch.Tensor,
        ingr_gt: Optional[torch.Tensor] = None,
        use_ingr_pred: bool = False,
        compute_losses: bool = False,
        compute_predictions: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Predict the ingredients and the recipe for the provided image
        :param image: input image from which to predict ingredient and recipe - shape (N, C, H, W)
        :param recipe_gt: target recipe to predict - shape (N, max_recipe_len)
        :param ingr_gt: target ingredients to predict - shape (N, max_num_labels)
        :param use_ingr_pred: whether or not predict the ingredient or use the ground truth
        :param compute_losses: whether or not to compute the loss
        :param compute_predictions: whether or not to output the recipe prediction
        """
        ingr_predictions = None
        img_features = self.image_encoder(image)

        if use_ingr_pred:
            # predict ingredients (do not use the ground truth)
            ingr_losses, ingr_predictions = self.ingr_predictor(
                img_features,
                label_target=ingr_gt,
                compute_losses=compute_losses,
                compute_predictions=True,
            )
            ingr_features = self.ingr_encoder(ingr_predictions)
            ingr_mask = mask_from_eos(
                ingr_predictions, eos_value=self.ingr_eos_value, mult_before=False
            )
            ingr_mask = ingr_mask.float().unsqueeze(1)
        else:
            # encode ingredients (using ground truth ingredients)
            ingr_features = self.ingr_encoder(ingr_gt)
            ingr_mask = mask_from_eos(
                ingr_gt, eos_value=self.ingr_eos_value, mult_before=False
            )
            ingr_mask = ingr_mask.float().unsqueeze(1)

        # generate recipe and compute losses if necessary
        loss, recipe_predictions = self.recipe_gen(
            img_features=img_features,
            ingr_features=ingr_features,
            ingr_mask=ingr_mask,
            recipe_gt=recipe_gt,
            compute_losses=compute_losses,
            compute_predictions=compute_predictions,
        )

        losses = {"recipe_loss": loss}
        return losses, ingr_predictions, recipe_predictions
