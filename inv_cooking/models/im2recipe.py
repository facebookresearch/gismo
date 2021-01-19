# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from inv_cooking.config import (
    ImageEncoderConfig,
    IngredientPredictorConfig,
    RecipeGeneratorConfig,
)
from inv_cooking.models.image_encoder import create_image_encoder
from inv_cooking.models.ingredients_encoder import IngredientsEncoder
from inv_cooking.models.ingredients_predictor import (
    create_ingredient_predictor,
    mask_from_eos,
)
from inv_cooking.models.recipe_generator import RecipeGenerator

class Im2Recipe(nn.Module):
    def __init__(
        self,
        image_encoder_config: ImageEncoderConfig,
        ingr_pred_config: IngredientPredictorConfig,
        recipe_gen_config: RecipeGeneratorConfig,
        ingr_vocab_size: int,
        instr_vocab_size: int,
        max_num_ingredients: int,
        max_recipe_len: int,
        ingr_eos_value: int,
    ):
        super().__init__()
        self.ingr_vocab_size = ingr_vocab_size
        self.ingr_eos_value = ingr_eos_value
        self.instr_vocab_size = instr_vocab_size

        self.image_encoder = create_image_encoder(
            ingr_pred_config.embed_size, image_encoder_config
        )
        self.ingr_predictor = create_ingredient_predictor(
            ingr_pred_config,
            max_num_ingredients=max_num_ingredients,
            ingr_vocab_size=ingr_vocab_size,
            ingr_eos_value=ingr_eos_value,
        )

        if ingr_pred_config.embed_size != recipe_gen_config.embed_size:
            self.img_features_transform = nn.Sequential(
                nn.Conv2d(ingr_pred_config.embed_size, recipe_gen_config.embed_size, kernel_size=1, padding=0, bias=False),
                nn.Dropout(recipe_gen_config.dropout),
                nn.BatchNorm2d(recipe_gen_config.embed_size, momentum=0.01),
                nn.ReLU(),
            )
        else:
            self.img_features_transform = None
        
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
        target_recipe: torch.Tensor,
        target_ingredients: Optional[torch.Tensor] = None,
        use_ingr_pred: bool = False,
        compute_losses: bool = False,
        compute_predictions: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Predict the ingredients and the recipe for the provided image
        :param image: input image from which to predict ingredient and recipe - shape (N, C, H, W)
        :param target_recipe: target recipe to predict - shape (N, max_recipe_len)
        :param target_ingredients: target ingredients to predict - shape (N, max_num_ingredients)
        :param use_ingr_pred: whether or not predict the ingredient or use the ground truth
        :param compute_losses: whether or not to compute the loss
        :param compute_predictions: whether or not to output the recipe prediction
        """
        ingr_predictions = None
        losses = {}
        img_features = self.image_encoder(image, return_reshaped_features=False)

        if use_ingr_pred:
            # predict ingredients (do not use the ground truth)
            ingr_losses, ingr_predictions = self.ingr_predictor(
                img_features.reshape(img_features.size(0), img_features.size(1), -1),
                label_target=target_ingredients,
                compute_losses=compute_losses,
                compute_predictions=True,
            )
            losses.update(ingr_losses)
            ingr_features = self.ingr_encoder(ingr_predictions)
            ingr_mask = mask_from_eos(
                ingr_predictions, eos_value=self.ingr_eos_value, mult_before=False
            )
            ingr_mask = ingr_mask.float().unsqueeze(1)
        else:
            # encode ingredients (using ground truth ingredients)
            ingr_features = self.ingr_encoder(target_ingredients)
            ingr_mask = mask_from_eos(
                target_ingredients, eos_value=self.ingr_eos_value, mult_before=False
            )
            ingr_mask = ingr_mask.float().unsqueeze(1)

        if self.img_features_transform is not None:
            img_features = self.img_features_transform(img_features)

        # generate recipe and compute losses if necessary
        loss, recipe_predictions = self.recipe_gen(
            img_features=img_features.reshape(img_features.size(0), img_features.size(1), -1),
            ingr_features=ingr_features,
            ingr_mask=ingr_mask,
            recipe_gt=target_recipe,
            compute_losses=compute_losses,
            compute_predictions=compute_predictions,
        )

        losses["recipe_loss"] = loss
        return losses, ingr_predictions, recipe_predictions
