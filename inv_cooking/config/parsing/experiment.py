from dataclasses import dataclass, field
from typing import Dict

from inv_cooking.config.optimization import OptimizationConfig


@dataclass
class Experiment:
    comment: str = ""
    optimization: OptimizationConfig = OptimizationConfig()


@dataclass
class Experiments:
    im2ingr: Dict[str, Experiment] = field(default_factory=dict)
    im2recipe: Dict[str, Experiment] = field(default_factory=dict)
    ingr2recipe: Dict[str, Experiment] = field(default_factory=dict)

