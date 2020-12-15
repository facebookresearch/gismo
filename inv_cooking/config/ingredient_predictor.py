from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class IngredientPredictorConfig:
    model: str = MISSING  # Either "ff", "lstm" or "tf"
    embed_size: int = MISSING
    freeze: bool = MISSING
    load_pretrained_from: str = MISSING
    with_set_prediction: bool = MISSING
    with_shuffle_labels: bool = MISSING
    cardinality_pred: str = "none"  # Either "dc" or "cat" or "none"


@dataclass
class IngredientPredictorFFConfig(IngredientPredictorConfig):
    layers: int = MISSING
    dropout: float = MISSING


@dataclass
class IngredientPredictorLSTMConfig(IngredientPredictorConfig):
    dropout: float = MISSING


@dataclass
class IngredientPredictorTransformerConfig(IngredientPredictorConfig):
    layers: int = MISSING
    n_att: int = MISSING
    dropout: float = MISSING
