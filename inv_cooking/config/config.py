from dataclasses import dataclass
from enum import Enum

from omegaconf import MISSING

from .dataset import DatasetConfig
from .image_encoder import ImageEncoderConfig
from .ingredient_predictor import IngredientPredictorConfig
from .optimization import OptimizationConfig
from .recipe_generator import RecipeGeneratorConfig
from .slurm import SlurmConfig


class TaskType(Enum):
    im2ingr = 0
    im2recipe = 1
    ingr2recipe = 2


@dataclass
class CheckpointConfig:
    log_folder: str = MISSING
    dir: str = MISSING


@dataclass
class Config:
    """
    Configuration fed to the main scheduler of experiment.
    It contains all information necessary to run a single job.
    """
    task: TaskType = TaskType.im2ingr
    name: str = MISSING
    comment: str = MISSING
    recipe_gen: RecipeGeneratorConfig = RecipeGeneratorConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    dataset: DatasetConfig = DatasetConfig()
    image_encoder: ImageEncoderConfig = ImageEncoderConfig()
    ingr_predictor: IngredientPredictorConfig = MISSING
    slurm: SlurmConfig = SlurmConfig()

