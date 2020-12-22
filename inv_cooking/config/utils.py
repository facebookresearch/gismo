from dataclasses import field

from omegaconf import DictConfig


def untyped_config():
    return field(default_factory=lambda: DictConfig(content=dict()))
