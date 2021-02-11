from dataclasses import dataclass
from enum import Enum

from omegaconf import MISSING


@dataclass
class ImageEncoderConfig:
    model: str = MISSING
    pretrained: bool = MISSING
    dropout: float = MISSING
    freeze: bool = False
    patch_size: int = MISSING  # Only used for VIT: 16 or 32
    n_cls_tokens: int = MISSING  # Only used for VIT
