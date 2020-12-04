from dataclasses import dataclass
from enum import Enum

from omegaconf import MISSING


class ImageEncoderFreezeType(Enum):
    none = 0
    all = 1
    pretrained = 2


@dataclass
class ImageEncoderConfig:
    model: str = MISSING
    pretrained: bool = MISSING
    dropout: float = MISSING
    freeze: ImageEncoderFreezeType = ImageEncoderFreezeType.none
