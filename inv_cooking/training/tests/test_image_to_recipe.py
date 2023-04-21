# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from inv_cooking.training.image_to_recipe import ImageToRecipe
from inv_cooking.training.tests.utils import _BaseTest


class TestImageToRecipe(_BaseTest):
    @torch.no_grad()
    def test_im2recipe(self):
        module = ImageToRecipe(
            image_encoder_config=self.default_image_encoder_config(),
            ingr_pred_config=self.default_ar_ingredient_predictor_config(),
            recipe_gen_config=self.default_recipe_generator_config(),
            optim_config=self.default_optimization_config(),
            pretrained_im2ingr_config=self.default_pretrained_config(),
            ingr_teachforce_config=self.default_ingr_teachforce_config(),
            max_num_ingredients=self.MAX_NUM_INGREDIENTS,
            max_recipe_len=self.MAX_NUM_INSTRUCTIONS * self.MAX_INSTRUCTION_LENGTH,
            ingr_vocab_size=self.INGR_VOCAB_SIZE,
            instr_vocab_size=self.RECIPE_VOCAB_SIZE,
            ingr_eos_value=self.INGR_EOS_VALUE,
        )

        batch_size = 5
        image = torch.randn(size=(batch_size, 3, 224, 224))
        ingredients = torch.randint(
            low=0,
            high=self.INGR_VOCAB_SIZE,
            size=(batch_size, self.MAX_NUM_INGREDIENTS + 1),
        )
        recipe = torch.randint(
            low=0, high=self.RECIPE_VOCAB_SIZE, size=(batch_size, self.MAX_RECIPE_LEN)
        )

        # Try building an optimizer
        self.assert_all_parameters_used(module)
        optimizers, schedulers = module.configure_optimizers()
        assert len(optimizers) == 1
        assert len(schedulers) == 1

        # Try "train" forward pass
        losses, predictions = module(
            image=image,
            ingredients=ingredients,
            recipe=recipe,
            use_ingr_pred=False,
            compute_losses=True,
            compute_predictions=True,
        )
        assert len(losses) == 1
        assert losses["recipe_loss"].shape == torch.Size([5])
        assert predictions[0] is None, "No ingredient prediction in training"
        assert predictions[1].shape == torch.Size([5, self.MAX_RECIPE_LEN])

        # Try "train" step
        batch = dict(image=image, ingredients=ingredients, recipe=recipe)
        losses = module.training_step(batch=batch, batch_idx=0)
        assert losses["recipe_loss"].shape == torch.Size([5])

        # Try "valid" forward step
        losses, predictions = module(
            image=image,
            ingredients=ingredients,
            recipe=recipe,
            use_ingr_pred=False,
            compute_losses=True,
            compute_predictions=True,
        )
        assert len(losses) == 1
        assert losses["recipe_loss"].shape == torch.Size([5])
        assert predictions[0] is None, "No ingredient prediction in training"
        assert predictions[1].shape == torch.Size([5, self.MAX_RECIPE_LEN])

        # Try "valid" step
        batch = dict(image=image, ingredients=ingredients, recipe=recipe)
        losses = module.validation_step(batch=batch, batch_idx=0)
        assert losses["recipe_loss"].shape == torch.Size([5])
        assert losses["n_samples"] == 5
        assert losses["ingr_gt"].shape == torch.Size([5, self.MAX_NUM_INGREDIENTS + 1])
        assert losses["ingr_pred"] is None

        # Try "test" forward step
        losses, predictions = module(
            image=image,
            ingredients=ingredients,
            recipe=recipe,
            use_ingr_pred=True,
            compute_losses=True,
            compute_predictions=True,
        )
        assert len(losses) == 2
        assert losses["label_loss"].shape == torch.Size([])
        assert losses["recipe_loss"].shape == torch.Size([5])
        assert predictions[0].shape == torch.Size([5, self.MAX_NUM_INGREDIENTS + 1])
        assert predictions[1].shape == torch.Size([5, self.MAX_RECIPE_LEN])

        # Try "test" step
        batch = dict(image=image, ingredients=ingredients, recipe=recipe)
        losses = module.test_step(batch=batch, batch_idx=0)
        assert losses["label_loss"].shape == torch.Size([])
        assert losses["recipe_loss"].shape == torch.Size([5])
        assert losses["n_samples"] == 5
        assert losses["ingr_gt"].shape == torch.Size([5, self.MAX_NUM_INGREDIENTS + 1])
        assert losses["ingr_pred"].shape == torch.Size(
            [5, self.MAX_NUM_INGREDIENTS + 1]
        )
