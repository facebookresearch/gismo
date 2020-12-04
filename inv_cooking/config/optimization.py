from dataclasses import dataclass, field
from typing import Dict

from omegaconf import MISSING


@dataclass
class OptimizationConfig:
    seed: int = MISSING
    lr: float = MISSING
    scale_lr_pretrained: float = MISSING
    lr_decay_rate: float = MISSING
    lr_decay_every: int = MISSING
    weight_decay: float = MISSING
    max_epochs: int = MISSING
    patience: int = MISSING
    loss_weights: Dict[str, float] = field(default_factory=dict)
