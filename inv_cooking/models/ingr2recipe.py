# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch.nn as nn
from omegaconf import DictConfig

from inv_cooking.config import RecipeGeneratorConfig, ImageEncoderConfig
from inv_cooking.models.ingredients_encoder import IngredientsEncoder
from inv_cooking.models.ingredients_predictor import get_ingr_predictor, mask_from_eos
from inv_cooking.models.recipe_generator import RecipeGenerator


class Ingr2Recipe(nn.Module):
    def __init__(
        self,
        recipe_gen_config: RecipeGeneratorConfig,
        ingr_vocab_size: int,
        instr_vocab_size: int,
        maxrecipelen: int,
        ingr_eos_value,
    ):
        super().__init__()
        self.ingr_vocab_size = ingr_vocab_size
        self.ingr_eos_value = ingr_eos_value
        self.instr_vocab_size = instr_vocab_size

        self.ingr_encoder = IngredientsEncoder(
            recipe_gen_config.embed_size,
            voc_size=ingr_vocab_size,
            dropout=recipe_gen_config.dropout,
            scale_grad=False,
        )
        self.recipe_gen = RecipeGenerator(
            recipe_gen_config, instr_vocab_size, maxrecipelen
        )

    def forward(
        self,
        recipe_gt,
        ingr_gt,
        compute_losses=False,
        compute_predictions=False,
    ):

        # encode ingredients
        ingr_features = self.ingr_encoder(ingr_gt)
        ingr_mask = mask_from_eos(
            ingr_gt, eos_value=self.ingr_eos_value, mult_before=False
        )
        ingr_mask = ingr_mask.float().unsqueeze(1)

        # generate recipe and compute losses if necessary
        loss, predictions = self.recipe_gen(
            ingr_features=ingr_features,
            ingr_mask=ingr_mask,
            recipe_gt=recipe_gt,
            compute_losses=compute_losses,
            compute_predictions=compute_predictions,
        )
        losses = {"recipe_loss": loss}
        return losses, predictions
