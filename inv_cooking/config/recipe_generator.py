from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class RecipeGeneratorConfig:
    dropout: float = MISSING
    embed_size: int = MISSING
    n_att_heads: int = MISSING
    layers: int = MISSING
    normalize_before: bool = MISSING
