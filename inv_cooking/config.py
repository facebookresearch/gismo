from dataclasses import dataclass, field
from enum import Enum

from omegaconf import MISSING, DictConfig, OmegaConf


def untyped_config():
    return field(default_factory=lambda: DictConfig(content=dict()))


class TaskType(Enum):
    im2ingr = 0
    im2recipe = 1
    ingr2recipe = 2


class ExecutorType(Enum):
    local = 0
    slurm = 1


class DatasetName(Enum):
    recipe1m = 0


@dataclass
class DatasetFilterConfig:
    max_num_images: int = MISSING
    max_num_labels: int = MISSING
    max_num_instructions: int = MISSING
    max_instruction_length: int = MISSING


@dataclass
class DatasetConfig:
    name: DatasetName = DatasetName.recipe1m
    path: str = MISSING
    splits_path: str = MISSING
    image_resize: int = MISSING
    image_crop_size: int = MISSING
    filtering: DatasetFilterConfig = DatasetFilterConfig()
    pre_processing: DictConfig = untyped_config()


@dataclass
class RecipeGeneratorConfig:
    dropout: float = MISSING
    embed_size: int = MISSING
    n_att_heads: int = MISSING
    layers: int = MISSING
    normalize_before: bool = MISSING


@dataclass
class SlurmConfig:
    log_folder: str = MISSING
    partition: str = MISSING
    nodes: int = MISSING
    cpus_per_task: int = MISSING
    gpus_per_node: int = MISSING
    mem_by_gpu: int = MISSING
    timeout_min: int = MISSING
    gpu_type: str = MISSING


@dataclass
class Config:
    task: TaskType = TaskType.im2ingr
    executor: ExecutorType = ExecutorType.local
    recipe_gen: RecipeGeneratorConfig = RecipeGeneratorConfig()
    optim: DictConfig = untyped_config()
    checkpoint: DictConfig = untyped_config()
    misc: DictConfig = untyped_config()
    dataset: DatasetConfig = DatasetConfig()
    image_encoder: DictConfig = untyped_config()
    ingr_predictor: DictConfig = untyped_config()
    slurm: SlurmConfig = SlurmConfig()

    @classmethod
    def parse_config(cls, cfg: DictConfig) -> "Config":
        schema = OmegaConf.structured(Config)
        return OmegaConf.merge(schema, cfg)
