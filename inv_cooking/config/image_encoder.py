from dataclasses import dataclass, field
from typing import List

from omegaconf import MISSING


@dataclass
class ImageEncoderConfig:
    model: str = MISSING
    pretrained: bool = MISSING
    dropout: float = MISSING
    freeze: bool = False
    patch_size: int = MISSING  # Only used for VIT: 16 or 32
    n_cls_tokens: int = MISSING  # Only used for VIT
    additional_repr_levels: List[int] = field(default_factory=list)  # Only used for VIT with one class token
