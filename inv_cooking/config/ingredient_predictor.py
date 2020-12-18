from dataclasses import dataclass
from enum import Enum

from omegaconf import MISSING


class IngredientPredictorType(Enum):
    ff = 0
    lstm = 1
    tf = 2


@dataclass
class IngredientPredictorConfig:
    model: IngredientPredictorType = MISSING
    embed_size: int = MISSING
    freeze: bool = MISSING
    load_pretrained_from: str = MISSING


class IngredientPredictorCriterion(Enum):
    bce = 0
    iou = 1
    td = 2


class CardinalityPredictionType(Enum):
    none = 0
    categorical = 1


@dataclass
class IngredientPredictorFFConfig(IngredientPredictorConfig):
    layers: int = MISSING
    dropout: float = MISSING
    criterion: IngredientPredictorCriterion = MISSING
    cardinality_pred: CardinalityPredictionType = CardinalityPredictionType.none


@dataclass
class IngredientPredictorARConfig(IngredientPredictorConfig):
    with_set_prediction: bool = MISSING


@dataclass
class IngredientPredictorLSTMConfig(IngredientPredictorARConfig):
    dropout: float = MISSING


@dataclass
class IngredientPredictorTransformerConfig(IngredientPredictorARConfig):
    layers: int = MISSING
    n_att: int = MISSING
    dropout: float = MISSING
