from dataclasses import dataclass
from enum import Enum

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .dataset import DatasetConfig
from .image_encoder import ImageEncoderConfig
from .ingredient_predictor import (
    IngredientPredictorConfig,
    IngredientPredictorFFConfig,
    IngredientPredictorLSTMConfig,
    IngredientPredictorTransformerConfig
)
from .optimization import OptimizationConfig
from .recipe_generator import RecipeGeneratorConfig
from .slurm import SlurmConfig


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
    ingr_predictor: IngredientPredictorConfig = MISSING
    slurm: SlurmConfig = SlurmConfig()


cs = ConfigStore.instance()
cs.store(group="ingr_predictor/model", name="ff_bce", node=IngredientPredictorFFConfig, package="ingr_predictor")
cs.store(group="ingr_predictor/model", name="lstmset", node=IngredientPredictorLSTMConfig, package="ingr_predictor")
cs.store(group="ingr_predictor/model", name="tf", node=IngredientPredictorTransformerConfig, package="ingr_predictor")
cs.store(name="config", node=Config)
