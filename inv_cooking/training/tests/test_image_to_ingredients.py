# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from inv_cooking.training.image_to_ingredients import ImageToIngredients
from inv_cooking.training.tests.utils import _BaseTest


class TestImageToIngredients(_BaseTest):
    @torch.no_grad()
    def test_im2ingr(self):
        module = ImageToIngredients(
            image_encoder_config=self.default_image_encoder_config(),
            ingr_pred_config=self.default_ar_ingredient_predictor_config(),
            optim_config=self.default_optimization_config(),
            max_num_ingredients=self.MAX_NUM_INGREDIENTS,
            ingr_vocab_size=self.INGR_VOCAB_SIZE,
            ingr_eos_value=self.INGR_EOS_VALUE,
        )

        batch_size = 5
        image = torch.randn(size=(batch_size, 3, 224, 224))
        ingredients = torch.randint(
            low=0,
            high=self.INGR_VOCAB_SIZE,
            size=(batch_size, self.MAX_NUM_INGREDIENTS + 1),
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
            compute_losses=True,
            compute_predictions=True,
        )
        assert len(losses) == 1
        assert losses["label_loss"].shape == torch.Size([])
        assert len(predictions) == 1
        assert predictions[0].shape == torch.Size([5, self.MAX_NUM_INGREDIENTS + 1])

        # Try "train" step
        batch = dict(image=image, ingredients=ingredients)
        losses = module.training_step(batch=batch, batch_idx=0)
        assert len(losses) == 1
        assert losses["label_loss"].shape == torch.Size([])

        # Try "val" step
        losses = module.validation_step(batch=batch, batch_idx=0)
        assert losses["label_loss"].shape == torch.Size([])
        assert losses["n_samples"] == 5
        assert losses["ingr_gt"].shape == torch.Size([5, self.MAX_NUM_INGREDIENTS + 1])
        assert losses["ingr_pred"].shape == torch.Size(
            [5, self.MAX_NUM_INGREDIENTS + 1]
        )

        # Try "test" step
        losses = module.test_step(batch=batch, batch_idx=0)
        assert losses["label_loss"].shape == torch.Size([])
        assert losses["n_samples"] == 5
        assert losses["ingr_gt"].shape == torch.Size([5, self.MAX_NUM_INGREDIENTS + 1])
        assert losses["ingr_pred"].shape == torch.Size(
            [5, self.MAX_NUM_INGREDIENTS + 1]
        )
