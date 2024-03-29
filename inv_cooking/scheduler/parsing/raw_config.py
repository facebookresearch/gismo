# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
    IngredientPredictorVITConfig,
)
from inv_cooking.config.slurm import SlurmConfig
from inv_cooking.utils.hydra import merge_with_non_missing

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
    eval_on_test: bool = MISSING
    eval_checkpoint_dir: str = MISSING
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
            config.eval_on_test = raw_config.eval_on_test
            if not OmegaConf.is_missing(experiment, "eval_checkpoint_dir"):
                config.eval_checkpoint_dir = experiment.eval_checkpoint_dir
            else:
                config.eval_checkpoint_dir = raw_config.eval_checkpoint_dir
            config.slurm = raw_config.slurm
            config.dataset = raw_config.dataset
            config.dataset.loading = merge_with_non_missing(
                config.dataset.loading, experiment.loading
            )
            config.recipe_gen = experiment.recipe_gen
            config.image_encoder = experiment.image_encoder
            config.title_encoder = experiment.title_encoder
            config.ingr_predictor = cls._get_ingr_predictor(experiment.ingr_predictor)
            config.checkpoint = raw_config.checkpoint
            config.optimization = experiment.optimization
            config.pretrained_im2ingr = experiment.pretrained_im2ingr
            config.ingr_teachforce = experiment.ingr_teachforce
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
        elif "vit" in model:
            schema = OmegaConf.structured(IngredientPredictorVITConfig)
            return OmegaConf.merge(schema, ingr_predictor)
        else:
            raise ValueError(f"Invalid ingredient predictor model {model}")


cs = ConfigStore.instance()
cs.store(name="config", node=RawConfig)
