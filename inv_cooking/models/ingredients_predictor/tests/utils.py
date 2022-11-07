# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from inv_cooking.config import (
    CardinalityPredictionType,
    IngredientPredictorCriterion,
    IngredientPredictorFFConfig,
    IngredientPredictorLSTMConfig,
    IngredientPredictorTransformerConfig,
    IngredientPredictorType,
    IngredientPredictorVITConfig,
    SetPredictionType,
)


class FakeIngredientPredictorConfig:
    @staticmethod
    def ff_config() -> IngredientPredictorFFConfig:
        return IngredientPredictorFFConfig(
            model=IngredientPredictorType.ff,
            embed_size=2048,
            cardinality_pred=CardinalityPredictionType.none,
            layers=2,
            dropout=0.0,
            criterion=IngredientPredictorCriterion.bce,
        )

    @staticmethod
    def lstm_config(
        with_set_prediction: SetPredictionType = SetPredictionType.none,
    ) -> IngredientPredictorLSTMConfig:
        return IngredientPredictorLSTMConfig(
            model=IngredientPredictorType.lstm,
            embed_size=2048,
            with_set_prediction=with_set_prediction,
            dropout=0.1,
        )

    @staticmethod
    def tf_config(
        with_set_prediction: SetPredictionType = SetPredictionType.none,
    ) -> IngredientPredictorTransformerConfig:
        return IngredientPredictorTransformerConfig(
            model=IngredientPredictorType.tf,
            layers=0,
            embed_size=2048,
            with_set_prediction=with_set_prediction,
            n_att=8,
            dropout=0.3,
        )

    @staticmethod
    def vit_config(
        with_set_prediction: SetPredictionType = SetPredictionType.none,
    ) -> IngredientPredictorVITConfig:
        return IngredientPredictorVITConfig(
            model=IngredientPredictorType.vit,
            embed_size=2048,
            with_set_prediction=with_set_prediction,
            layers=2,
            dropout=0.1,
        )
