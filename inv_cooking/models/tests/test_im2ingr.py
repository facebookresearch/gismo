# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from inv_cooking.models.im2ingr import Im2Ingr
from inv_cooking.models.ingredients_predictor.tests.utils import (
    FakeIngredientPredictorConfig,
)
from inv_cooking.models.tests.utils import FakeConfig


def test_Im2Ingr():
    torch.random.manual_seed(0)
    max_num_ingredients = 10
    vocab_size = 20
    title_vocab_size = 14
    batch_size = 5
    image = torch.randn(size=(batch_size, 3, 224, 224))
    labels = torch.randint(low=0, high=vocab_size - 1, size=(5, max_num_ingredients))

    ingredient_predictors = [
        (
            FakeIngredientPredictorConfig.ff_config(),
            torch.Size([batch_size, max_num_ingredients]),
        ),
        (
            FakeIngredientPredictorConfig.tf_config(),
            torch.Size([batch_size, max_num_ingredients + 1]),
        ),
        (
            FakeIngredientPredictorConfig.lstm_config(),
            torch.Size([batch_size, max_num_ingredients + 1]),
        ),
    ]

    for ingr_predictor, expected_prediction_shape in ingredient_predictors:
        model = Im2Ingr(
            image_encoder_config=FakeConfig.image_encoder_config(),
            ingr_pred_config=ingr_predictor,
            ingr_vocab_size=vocab_size,
            max_num_ingredients=max_num_ingredients,
            ingr_eos_value=vocab_size - 1,
            title_encoder_config=FakeConfig.no_title_encoder_config(),
            title_vocab_size=title_vocab_size,
        )

        losses, predictions = model(
            image=image,
            target_ingredients=labels,
            compute_losses=True,
            compute_predictions=True,
        )

        assert losses["label_loss"] is not None
        assert predictions.shape == expected_prediction_shape
        assert predictions.min() >= 0
        assert predictions.max() <= vocab_size - 1


def test_Im2Ingr_with_title_encoder():
    torch.random.manual_seed(0)
    max_num_ingredients = 10
    vocab_size = 20
    title_vocab_size = 14
    batch_size = 5
    image = torch.randn(size=(batch_size, 3, 224, 224))
    title = torch.randint(low=0, high=title_vocab_size - 1, size=(batch_size, 20))
    labels = torch.randint(low=0, high=vocab_size - 1, size=(5, max_num_ingredients))

    ingredient_predictors = [
        (
            FakeIngredientPredictorConfig.ff_config(),
            torch.Size([batch_size, max_num_ingredients]),
        ),
        (
            FakeIngredientPredictorConfig.tf_config(),
            torch.Size([batch_size, max_num_ingredients + 1]),
        ),
        (
            FakeIngredientPredictorConfig.lstm_config(),
            torch.Size([batch_size, max_num_ingredients + 1]),
        ),
    ]

    for ingr_predictor, expected_prediction_shape in ingredient_predictors:
        model = Im2Ingr(
            image_encoder_config=FakeConfig.image_encoder_config(),
            ingr_pred_config=ingr_predictor,
            ingr_vocab_size=vocab_size,
            max_num_ingredients=max_num_ingredients,
            ingr_eos_value=vocab_size - 1,
            title_encoder_config=FakeConfig.with_title_encoder_config(),
            title_vocab_size=title_vocab_size,
        )

        losses, predictions = model(
            image=image,
            title=title,
            target_ingredients=labels,
            compute_losses=True,
            compute_predictions=True,
        )
        assert losses["label_loss"] is not None
        assert predictions.shape == expected_prediction_shape
        assert predictions.min() >= 0
        assert predictions.max() <= vocab_size - 1

        losses, predictions = model(
            image=None,
            title=title,
            target_ingredients=labels,
            compute_losses=True,
            compute_predictions=True,
        )
        assert losses["label_loss"] is not None
        assert predictions.shape == expected_prediction_shape
        assert predictions.min() >= 0
        assert predictions.max() <= vocab_size - 1
