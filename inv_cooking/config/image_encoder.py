from dataclasses import dataclass
from enum import Enum

from omegaconf import MISSING


@dataclass
class ImageEncoderConfig:
    model: str = MISSING
    pretrained: bool = MISSING
    dropout: float = MISSING
    freeze: bool = False
