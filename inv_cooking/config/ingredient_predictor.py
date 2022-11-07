# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum

from omegaconf import MISSING


class IngredientPredictorType(Enum):
    ff = 0
    lstm = 1
    tf = 2
    vit = 3


@dataclass
class IngredientPredictorConfig:
    model: IngredientPredictorType = MISSING
    embed_size: int = MISSING


class IngredientPredictorCriterion(Enum):
    bce = 0
    iou = 1
    td = 2


class CardinalityPredictionType(Enum):
    none = 0
    categorical = 1


class SetPredictionType(Enum):
    none = 0
    pooled_bce = 1
    chamfer_l2 = 2
    bipartite = 3
    chamfer_ce = 4
    chamfer_unilateral_ce = 5


@dataclass
class IngredientPredictorFFConfig(IngredientPredictorConfig):
    layers: int = MISSING
    dropout: float = MISSING
    criterion: IngredientPredictorCriterion = MISSING
    cardinality_pred: CardinalityPredictionType = CardinalityPredictionType.none


@dataclass
class IngredientPredictorVITConfig(IngredientPredictorConfig):
    with_set_prediction: SetPredictionType = SetPredictionType.none
    layers: int = MISSING
    dropout: float = MISSING


@dataclass
class IngredientPredictorARConfig(IngredientPredictorConfig):
    with_set_prediction: SetPredictionType = SetPredictionType.none


@dataclass
class IngredientPredictorLSTMConfig(IngredientPredictorARConfig):
    dropout: float = MISSING


@dataclass
class IngredientPredictorTransformerConfig(IngredientPredictorARConfig):
    layers: int = MISSING
    n_att: int = MISSING
    dropout: float = MISSING
    activation: str = MISSING
