from dataclasses import dataclass
from typing import List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from inv_cooking.config.config import CheckpointConfig, Config
from inv_cooking.config.dataset import DatasetConfig
from inv_cooking.config.ingredient_predictor import (
    IngredientPredictorConfig,
    IngredientPredictorFFConfig,
    IngredientPredictorLSTMConfig,
    IngredientPredictorTransformerConfig,
)
from inv_cooking.config.slurm import SlurmConfig

from .experiment import Experiment, Experiments, parse_experiments


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
    debug_mode: bool = MISSING
    checkpoint: CheckpointConfig = CheckpointConfig()
    dataset: DatasetConfig = DatasetConfig()
    slurm: SlurmConfig = SlurmConfig()
    experiments: Experiments = Experiments()

    @classmethod
    def to_config(cls, raw_config: "RawConfig") -> List[Config]:
        """
        Read the raw configuration as read by Hydra and expand it as valid
        configurations, each one describing a valid configuration to run
        """
        configs = []
        experiments = cls._get_experiments(raw_config)
        for experiment in experiments:
            config = OmegaConf.structured(Config)
            config.task = raw_config.task
            config.name = experiment.name
            config.comment = experiment.comment
            config.debug_mode = raw_config.debug_mode
            config.slurm = raw_config.slurm
            config.dataset = raw_config.dataset
            config.recipe_gen = experiment.recipe_gen
            config.image_encoder = experiment.image_encoder
            config.ingr_predictor = cls._get_ingr_predictor(experiment.ingr_predictor)
            config.checkpoint = raw_config.checkpoint
            config.optimization = experiment.optimization
            configs.append(config)
        return configs

    @staticmethod
    def _get_experiments(raw_config: "RawConfig") -> List[Experiment]:
        return parse_experiments(
            raw_config.experiments, raw_config.task, raw_config.name or raw_config.task
        )

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
