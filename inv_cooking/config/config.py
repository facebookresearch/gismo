from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf, DictConfig

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
from .utils import untyped_config


class TaskType(Enum):
    im2ingr = 0
    im2recipe = 1
    ingr2recipe = 2


@dataclass
class CheckpointConfig:
    dir: str = MISSING


@dataclass
class Experiment:
    comment: str = ""
    optimization: OptimizationConfig = OptimizationConfig()


@dataclass
class Experiments:
    im2ingr: Dict[str, Experiment] = field(default_factory=dict)
    im2recipe: Dict[str, Experiment] = field(default_factory=dict)
    ingr2recipe: Dict[str, Experiment] = field(default_factory=dict)


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


@dataclass
class RawConfig:
    """
    The schema of the raw configuration as read by hydra:
    * it contains the full DB of experiment
    * it contains the information needed to select the experiment
    This configuration is then transformed to be the configuration of one experiment.
    """
    task: str = MISSING
    name: str = MISSING
    recipe_gen: RecipeGeneratorConfig = RecipeGeneratorConfig()
    image_encoder: ImageEncoderConfig = ImageEncoderConfig()
    ingr_predictor: DictConfig = untyped_config()
    checkpoint: CheckpointConfig = CheckpointConfig()
    dataset: DatasetConfig = DatasetConfig()
    slurm: SlurmConfig = SlurmConfig()
    experiments: Experiments = Experiments()

    @classmethod
    def to_config(cls, raw_config: 'RawConfig') -> Config:
        experiment_name, experiment = cls._get_experiment(raw_config)
        config = OmegaConf.structured(Config)
        config.task = raw_config.task
        config.name = experiment_name
        config.comment = experiment.comment
        config.slurm = raw_config.slurm
        config.dataset = raw_config.dataset  # TODO - search the right dataset
        config.recipe_gen = raw_config.recipe_gen
        config.image_encoder = raw_config.image_encoder
        config.ingr_predictor = cls._get_ingr_predictor(raw_config.ingr_predictor)
        config.checkpoint = raw_config.checkpoint
        config.optimization = experiment.optimization
        return config

    @staticmethod
    def _get_experiment(raw_config: 'RawConfig') -> Tuple[str, Experiment]:
        experiment_name = raw_config.name or raw_config.task
        experiment = raw_config.experiments[raw_config.task][experiment_name]
        return experiment_name, experiment

    @staticmethod
    def _get_ingr_predictor(ingr_predictor: DictConfig) -> IngredientPredictorConfig:
        model = ingr_predictor.model
        if model == "ff_bce":
            schema = OmegaConf.structured(IngredientPredictorFFConfig)
            return OmegaConf.merge(schema, ingr_predictor)
        elif model == "lstmset":
            schema = OmegaConf.structured(IngredientPredictorLSTMConfig)
            return OmegaConf.merge(schema, ingr_predictor)
        elif model == "tf":
            schema = OmegaConf.structured(IngredientPredictorTransformerConfig)
            return OmegaConf.merge(schema, ingr_predictor)
        else:
            raise ValueError(f"Invalid ingredient predictor model {model}")


cs = ConfigStore.instance()
cs.store(name="config", node=RawConfig)
