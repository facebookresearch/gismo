# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
    im2title = 3
    ingrsubs = 4


@dataclass
class CheckpointConfig:
    log_folder: str = MISSING
    dir: str = MISSING
    tensorboard_folder: str = MISSING


@dataclass
class PretrainedConfig:
    freeze: bool = MISSING
    load_pretrained_from: str = MISSING


class IngredientTeacherForcingFlag(Enum):
    use_predictions = 0
    use_ground_truth = 1
    use_substitutions = 2


@dataclass
class IngredientTeacherForcingConfig:
    train: bool = MISSING
    val: bool = MISSING
    test: IngredientTeacherForcingFlag = MISSING


@dataclass
class TitleEncoderConfig:
    with_title: bool = False
    layers: int = 0
    layer_dim: int = 512


@dataclass
class Config:
    """
    Configuration fed to the main scheduler of experiment.
    It contains all information necessary to run a single job.
    """

    task: TaskType = TaskType.im2ingr
    name: str = MISSING
    comment: str = MISSING
    debug_mode: bool = MISSING
    eval_on_test: bool = MISSING
    eval_checkpoint_dir: str = MISSING  # Only used for eval.py
    recipe_gen: RecipeGeneratorConfig = RecipeGeneratorConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    dataset: DatasetConfig = DatasetConfig()
    image_encoder: ImageEncoderConfig = ImageEncoderConfig()
    title_encoder: TitleEncoderConfig = TitleEncoderConfig()
    ingr_predictor: IngredientPredictorConfig = MISSING
    slurm: SlurmConfig = SlurmConfig()
    pretrained_im2ingr: PretrainedConfig = PretrainedConfig()
    ingr_teachforce: IngredientTeacherForcingConfig = IngredientTeacherForcingConfig()
