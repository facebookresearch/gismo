from dataclasses import dataclass
from typing import Tuple

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from inv_cooking.config.config import CheckpointConfig, Config
from inv_cooking.config.dataset import DatasetConfig
from inv_cooking.config.image_encoder import ImageEncoderConfig
from inv_cooking.config.ingredient_predictor import (
    IngredientPredictorConfig,
    IngredientPredictorFFConfig,
    IngredientPredictorLSTMConfig,
    IngredientPredictorTransformerConfig,
)
from inv_cooking.config.recipe_generator import RecipeGeneratorConfig
from inv_cooking.config.slurm import SlurmConfig
from .experiment import Experiment, Experiments
from .utils import untyped_config


@dataclass
class RawConfig:
    """
    The schema of the raw configuration as read by hydra:
    * it contains the full database of experiment
    * it contains the information needed to select the experiment
    * it is partially untyped to allow for some complex mechanism like inheritance to be used

    This configuration is then transformed to be the configuration of one experiment.
    Types are enforced during this transformation to catch the last remaining possible errors.
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
    def to_config(cls, raw_config: "RawConfig") -> Config:
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
    def _get_experiment(raw_config: "RawConfig") -> Tuple[str, Experiment]:
        experiment_name = raw_config.name or raw_config.task
        experiment = raw_config.experiments[raw_config.task][experiment_name]
        return experiment_name, experiment

    @staticmethod
    def _get_ingr_predictor(ingr_predictor: DictConfig) -> IngredientPredictorConfig:
        model = ingr_predictor.model
        if "ff" in model:
            schema = OmegaConf.structured(IngredientPredictorFFConfig)
            return OmegaConf.merge(schema, ingr_predictor)
        elif "lstm" in model:
            schema = OmegaConf.structured(IngredientPredictorLSTMConfig)
            return OmegaConf.merge(schema, ingr_predictor)
        elif "tf" in model:
            schema = OmegaConf.structured(IngredientPredictorTransformerConfig)
            return OmegaConf.merge(schema, ingr_predictor)
        else:
            raise ValueError(f"Invalid ingredient predictor model {model}")


cs = ConfigStore.instance()
cs.store(name="config", node=RawConfig)
