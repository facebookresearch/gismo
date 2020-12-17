from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class IngredientPredictorConfig:
    model: str = MISSING  # Either "ff", "lstm" or "tf"
    embed_size: int = MISSING
    freeze: bool = MISSING
    load_pretrained_from: str = MISSING
    with_shuffle_labels: bool = MISSING


@dataclass
class IngredientPredictorFFConfig(IngredientPredictorConfig):
    layers: int = MISSING
    dropout: float = MISSING
    cardinality_pred: str = "none"  # Either "cat" or "none"


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
