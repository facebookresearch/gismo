# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

from inv_cooking.config import (
    IngredientPredictorConfig,
    IngredientPredictorFFConfig,
    IngredientPredictorLSTMConfig,
    IngredientPredictorTransformerConfig, IngredientPredictorType,
)
from .predictor import IngredientsPredictor
from .predictor_ar import AutoRegressiveIngredientsPredictor
from .predictor_ff import FeedForwardIngredientsPredictor


def create_ingredient_predictor(
    config: IngredientPredictorConfig,
    vocab_size: int,
    max_num_labels: int,
    eos_value: int,
) -> IngredientsPredictor:
    """
    Create the ingredient predictor based on the configuration
    """
    if config.model == IngredientPredictorType.ff:
        config = cast(IngredientPredictorFFConfig, config)
        return FeedForwardIngredientsPredictor.from_config(config, max_num_labels, vocab_size)
    elif config.model == IngredientPredictorType.lstm:
        config = cast(IngredientPredictorLSTMConfig, config)
        return AutoRegressiveIngredientsPredictor.create_lstm_from_config(config, max_num_labels, vocab_size, eos_value)
    elif config.model == IngredientPredictorType.tf:
        config = cast(IngredientPredictorTransformerConfig, config)
        return AutoRegressiveIngredientsPredictor.create_tf_from_config(config, max_num_labels, vocab_size, eos_value)
