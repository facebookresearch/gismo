# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch.nn as nn

from inv_cooking.models.image_encoder import ImageEncoder
from inv_cooking.models.ingredients_encoder import IngredientsEncoder
from inv_cooking.models.ingredients_predictor import get_ingr_predictor, mask_from_eos
from inv_cooking.models.recipe_generator import RecipeGenerator


class Im2Recipe(nn.Module):
    def __init__(
        self,
        im_args,
        ingrpred_args,
        recipegen_args,
        ingr_vocab_size,
        instr_vocab_size,
        dataset,
        maxnumlabels,
        maxrecipelen,
        ingr_eos_value,
        eps=1e-8,
    ):

        super(Im2Recipe, self).__init__()

        self.ingr_vocab_size = ingr_vocab_size
        self.ingr_pad_value = ingr_vocab_size - 1
        self.ingr_eos_value = ingr_eos_value
        self.instr_vocab_size = instr_vocab_size

        if ingrpred_args.freeze:
            im_args.freeze = "all"

        # image encoder model
        self.image_encoder = ImageEncoder(ingrpred_args.embed_size, **im_args)

        # set predictor model
        self.ingr_predictor = get_ingr_predictor(
            ingrpred_args,
            vocab_size=ingr_vocab_size,
            dataset=dataset,
            maxnumlabels=maxnumlabels,
            eos_value=ingr_eos_value,
        )

        # ingredient encoder model
        self.ingr_encoder = IngredientsEncoder(
            recipegen_args.embed_size,
            num_classes=ingr_vocab_size,
            dropout=recipegen_args.dropout,
            scale_grad=False,
        )

        # recipe generator model
        self.recipe_gen = RecipeGenerator(
            recipegen_args, instr_vocab_size, maxrecipelen
        )

    def forward(
        self,
        img,
        recipe_gt,
        ingr_gt=None,
        use_ingr_pred=False,
        compute_losses=False,
        compute_predictions=False,
    ):

        losses = {}
        ingr_predictions = None
        predictions = None

        img_features = self.image_encoder(img)

        if use_ingr_pred:
            # get ingredients predictions
            ingr_losses, ingr_predictions = self.ingr_predictor(
                img_features,
                label_target=ingr_gt,
                compute_losses=compute_losses,
                compute_predictions=True,
            )

            # encode ingredients
            ingr_features = self.ingr_encoder(ingr_predictions)

            # save ingredient losses losses

            # get ingredients' mask
            ingr_mask = mask_from_eos(
                ingr_predictions, eos_value=self.ingr_eos_value, mult_before=False
            )
            ingr_mask = ingr_mask.float().unsqueeze(1)
        else:
            # encode ingredients (using gt ingredients)
            ingr_features = self.ingr_encoder(ingr_gt)

            # get ingredients' mask
            ingr_mask = mask_from_eos(
                ingr_gt, eos_value=self.ingr_eos_value, mult_before=False
            )
            ingr_mask = ingr_mask.float().unsqueeze(1)

        # generate recipe and compute losses if necessary
        loss, predictions = self.recipe_gen(
            img_features,
            ingr_features,
            ingr_mask,
            recipe_gt,
            compute_losses=compute_losses,
            compute_predictions=compute_predictions,
        )

        losses["recipe_loss"] = loss

        return losses, ingr_predictions, predictions
