from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class IngredientPredictorConfig:
    model: str = MISSING
    embed_size: int = MISSING
    freeze: bool = MISSING
    load_pretrained_from: str = MISSING


@dataclass
class IngredientPredictorFFConfig(IngredientPredictorConfig):
    layers: int = MISSING
    dropout: float = MISSING


@dataclass
class IngredientPredictorLSTMConfig(IngredientPredictorConfig):
    dropout: float = MISSING


@dataclass
class IngredientPredictorTransformerConfig(IngredientPredictorConfig):
    n_att: int = MISSING
    dropout: float = MISSING
