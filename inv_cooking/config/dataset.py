from dataclasses import dataclass
from enum import Enum

from omegaconf import MISSING, DictConfig

from .utils import untyped_config


class DatasetName(Enum):
    recipe1m = 0


@dataclass
class DatasetLoadingConfig:
    batch_size: int = MISSING
    num_workers: int = MISSING


@dataclass
class DatasetFilterConfig:
    max_num_images: int = MISSING
    max_num_labels: int = MISSING
    max_num_instructions: int = MISSING
    max_instruction_length: int = MISSING
    max_title_seq_len: int = MISSING


@dataclass
class DatasetConfig:
    name: DatasetName = DatasetName.recipe1m
    path: str = MISSING
    splits_path: str = MISSING
    eval_split: str = MISSING
    image_resize: int = MISSING
    image_crop_size: int = MISSING
    loading: DatasetLoadingConfig = DatasetLoadingConfig()
    filtering: DatasetFilterConfig = DatasetFilterConfig()
    pre_processing: DictConfig = untyped_config()
