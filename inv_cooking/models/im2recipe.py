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

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from inv_cooking.config import (
    EncoderAttentionType,
    ImageEncoderConfig,
    IngredientPredictorConfig,
    PretrainedConfig,
    RecipeGeneratorConfig,
)
from inv_cooking.models.image_encoder import create_image_encoder
from inv_cooking.models.ingredients_predictor import (
    create_ingredient_predictor,
    mask_from_eos,
)
from inv_cooking.models.modules.ingredient_embeddings import IngredientEmbeddings
from inv_cooking.models.modules.transformer_encoder import EncoderTransformer
from inv_cooking.models.modules.utils import freeze_fn
from inv_cooking.models.recipe_generator import RecipeGenerator
from inv_cooking.utils.checkpointing import (
    list_available_checkpoints,
    select_best_checkpoint,
)


class Im2Recipe(nn.Module):
    def __init__(
        self,
        image_encoder_config: ImageEncoderConfig,
        ingr_pred_config: IngredientPredictorConfig,
        recipe_gen_config: RecipeGeneratorConfig,
        pretrained_im2ingr_config: PretrainedConfig,
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
        self.encoder_attn = recipe_gen_config.encoder_attn

        self.image_encoder = create_image_encoder(
            ingr_pred_config.embed_size, image_encoder_config
        )
        self.ingr_predictor = create_ingredient_predictor(
            ingr_pred_config,
            vocab_size=ingr_vocab_size,
            max_num_ingredients=max_num_ingredients,
            eos_value=ingr_eos_value,
        )

        # load pre-trained model from checkpoint
        im2ingr_path = pretrained_im2ingr_config.load_pretrained_from
        if im2ingr_path != "None":
            self._load_im2ingr_pretrained_model(im2ingr_path)

        # freeze pretrained model
        if pretrained_im2ingr_config.freeze:
            freeze_fn(self.image_encoder)
            freeze_fn(self.ingr_predictor)

        if ingr_pred_config.embed_size != recipe_gen_config.embed_size:
            self.img_features_transform = nn.Sequential(
                nn.Conv2d(
                    ingr_pred_config.embed_size,
                    recipe_gen_config.embed_size,
                    kernel_size=1,
                    padding=0,
                    bias=False,
                ),
                nn.Dropout(recipe_gen_config.dropout),
                nn.BatchNorm2d(recipe_gen_config.embed_size, momentum=0.01),
                nn.ReLU(),
            )
        else:
            self.img_features_transform = None

        self.ingr_encoder = IngredientEmbeddings(
            recipe_gen_config.embed_size,
            voc_size=ingr_vocab_size,
            dropout=recipe_gen_config.dropout,
            scale_grad=False,
        )

        if self.encoder_attn == EncoderAttentionType.concat_tf:
            self.transformer_encoder = EncoderTransformer(
                recipe_gen_config.embed_size,
                dropout=recipe_gen_config.dropout,
                attention_nheads=recipe_gen_config.n_att_heads,
                pos_embeddings=False,
                num_layers=recipe_gen_config.tf_enc_layers,
                learned=False,
                activation=recipe_gen_config.activation,
            )

        # recipe generator
        if self.encoder_attn in [
            EncoderAttentionType.concat,
            EncoderAttentionType.concat_tf,
        ]:
            num_cross_attn = 1
        elif self.encoder_attn in [
            EncoderAttentionType.seq_img_first,
            EncoderAttentionType.seq_ingr_first,
        ]:
            num_cross_attn = 2

        self.recipe_gen = RecipeGenerator(
            recipe_gen_config, instr_vocab_size, max_recipe_len, num_cross_attn,
        )

    def _load_im2ingr_pretrained_model(self, im2ingr_path: str):
        # If provided a directory, find the best checkpoint in that directory
        if os.path.isdir(im2ingr_path):
            all_checkpoints = list_available_checkpoints(im2ingr_path)
            im2ingr_path = select_best_checkpoint(all_checkpoints, metric_mode="max")

        # Load the checkpoint at the chosen path
        print(f"Using im2ingr checkpoint: {im2ingr_path}")
        pretrained_model = torch.load(im2ingr_path)

        # Initializing the image encoder
        pretrained_image_encoder_dict = {
            k[len("model.image_encoder.") :]: v
            for k, v in pretrained_model["state_dict"].items()
            if "image_encoder" in k
        }
        self.image_encoder.load_state_dict(pretrained_image_encoder_dict)

        # Initialize the ingredient predictor
        pretrained_ingr_predictor_dict = {
            k[len("model.ingr_predictor.") :]: v
            for k, v in pretrained_model["state_dict"].items()
            if "ingr_predictor" in k
        }
        self.ingr_predictor.load_state_dict(pretrained_ingr_predictor_dict)

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

        # predict ingredients (do not use the ground truth)
        if use_ingr_pred:
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
            ingr_mask = (1 - ingr_mask).bool()

        # encode ingredients (using ground truth ingredients)
        else:
            ingr_features = self.ingr_encoder(target_ingredients)
            ingr_mask = mask_from_eos(
                target_ingredients, eos_value=self.ingr_eos_value, mult_before=False
            )
            ingr_mask = (1 - ingr_mask).bool()

        # transform image features and create mask
        if self.img_features_transform is not None:
            img_features = self.img_features_transform(img_features)
        img_features = img_features.reshape(
            img_features.size(0), img_features.size(1), -1
        )
        img_mask = torch.zeros(img_features.shape[0], img_features.shape[2]).type_as(
            ingr_mask
        )

        # prepare encoder conditioning for cross attention in recipe tf
        if self.encoder_attn == EncoderAttentionType.concat:
            features = torch.cat((img_features, ingr_features), 2)
            masks = torch.cat((img_mask, ingr_mask), 1)
        elif self.encoder_attn == EncoderAttentionType.seq_img_first:
            features = [img_features, ingr_features]
            masks = [img_mask, ingr_mask]
        elif self.encoder_attn == EncoderAttentionType.seq_ingr_first:
            features = [ingr_features, img_features]
            masks = [ingr_mask, img_mask]
        elif self.encoder_attn == EncoderAttentionType.concat_tf:
            features = self.transformer_encoder(
                features=torch.cat((img_features, ingr_features), 2),
                masks=torch.cat((img_mask, ingr_mask), 1),
            )
            masks = None

        # generate recipe and compute losses if necessary
        loss, recipe_predictions = self.recipe_gen(
            features=features,
            masks=masks,
            recipe_gt=target_recipe,
            compute_losses=compute_losses,
            compute_predictions=compute_predictions,
        )

        losses["recipe_loss"] = loss
        return losses, ingr_predictions, recipe_predictions
