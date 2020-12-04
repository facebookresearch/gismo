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
    recipe_gen: DictConfig = untyped_config()
    optim: DictConfig = untyped_config()
    checkpoint: DictConfig = untyped_config()
    preprocessing: DictConfig = untyped_config()
    misc: DictConfig = untyped_config()
    dataset: DictConfig = untyped_config()
    image_encoder: DictConfig = untyped_config()
    ingr_predictor: DictConfig = untyped_config()
    slurm: SlurmConfig = SlurmConfig()

    @classmethod
    def parse_config(cls, cfg: DictConfig):
        schema = OmegaConf.structured(Config)
        return OmegaConf.merge(schema, cfg)
