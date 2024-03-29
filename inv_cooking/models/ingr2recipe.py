# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
# Code adapted from https://github.com/facebookresearch/inversecooking
# This source code is licensed under the MIT license found in the
# LICENSE file in https://github.com/facebookresearch/inversecooking

from typing import Dict, Tuple

import torch
import torch.nn as nn

from inv_cooking.config import RecipeGeneratorConfig
from inv_cooking.models.ingredients_predictor import mask_from_eos
from inv_cooking.models.modules.ingredient_embeddings import IngredientEmbeddings
from inv_cooking.models.recipe_generator import RecipeGenerator


class Ingr2Recipe(nn.Module):
    def __init__(
        self,
        recipe_gen_config: RecipeGeneratorConfig,
        ingr_vocab_size: int,
        instr_vocab_size: int,
        max_recipe_len: int,
        ingr_eos_value,
    ):
        super().__init__()
        self.ingr_vocab_size = ingr_vocab_size
        self.ingr_eos_value = ingr_eos_value
        self.instr_vocab_size = instr_vocab_size

        self.ingr_encoder = IngredientEmbeddings(
            recipe_gen_config.embed_size,
            voc_size=ingr_vocab_size,
            dropout=recipe_gen_config.dropout,
            scale_grad=False,
        )
        self.recipe_gen = RecipeGenerator(
            recipe_gen_config,
            instr_vocab_size,
            max_recipe_len,
            num_cross_attn=1,
        )

    def forward(
        self,
        ingredients: torch.Tensor,
        target_recipe: torch.Tensor,
        compute_losses: bool = False,
        compute_predictions: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Transform the ingredients into a recipe
        :param ingredients: the input ingredients to make a recipe from - shape (N, num_ingredient)
        :param target_recipe: the ground truth to reach - shape (N, max_recipe_len)
        :param compute_losses: whether or not to compute the loss (requires recipe ground truth)
        :param compute_predictions: whether or not to output the recipe prediction
        """

        # encode ingredients
        ingr_features = self.ingr_encoder(ingredients)
        ingr_mask = mask_from_eos(
            ingredients, eos_value=self.ingr_eos_value, mult_before=False
        )
        ingr_mask = (1 - ingr_mask).bool()

        # generate recipe and compute losses if necessary
        loss, predictions = self.recipe_gen(
            features=ingr_features,
            masks=ingr_mask,
            recipe_gt=target_recipe,
            compute_losses=compute_losses,
            compute_predictions=compute_predictions,
        )
        losses = {"recipe_loss": loss}
        return losses, predictions
