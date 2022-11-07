# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
    sync_batchnorm: bool = False
    loss_weights: Dict[str, float] = field(default_factory=dict)
