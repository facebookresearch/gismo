from dataclasses import dataclass, field
from typing import List

from omegaconf import MISSING


@dataclass
class ImageEncoderConfig:
    with_image_encoder: bool = True  # Set to false to to title to ingredient
    model: str = MISSING
    pretrained: bool = MISSING
    pretrained_weights: str = ""  # Path to pre-trained weights
    dropout: float = MISSING
    freeze: bool = False
    patch_size: int = MISSING  # Only used for VIT: 16 or 32
    n_cls_tokens: int = MISSING  # Only used for VIT
    additional_repr_levels: List[int] = field(
        default_factory=list
    )  # Only used for VIT with one class token
    concatenate_repr_levels: bool = False  # Only used for VIT with one class token
