from dataclasses import dataclass
from enum import Enum

from omegaconf import MISSING, DictConfig, OmegaConf

from .dataset import DatasetConfig
from .image_encoder import ImageEncoderConfig
from .optimization import OptimizationConfig
from .recipe_generator import RecipeGeneratorConfig
from .slurm import SlurmConfig
from .utils import untyped_config


class TaskType(Enum):
    im2ingr = 0
    im2recipe = 1
    ingr2recipe = 2


class ExecutorType(Enum):
    local = 0
    slurm = 1


@dataclass
class CheckpointConfig:
    dir: str = MISSING


@dataclass
class Config:
    task: TaskType = TaskType.im2ingr
    executor: ExecutorType = ExecutorType.local
    recipe_gen: RecipeGeneratorConfig = RecipeGeneratorConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    dataset: DatasetConfig = DatasetConfig()
    image_encoder: ImageEncoderConfig = ImageEncoderConfig()
    ingr_predictor: DictConfig = untyped_config()
    slurm: SlurmConfig = SlurmConfig()

    @classmethod
    def parse_config(cls, cfg: DictConfig) -> "Config":
        schema = OmegaConf.structured(Config)
        return OmegaConf.merge(schema, cfg)
