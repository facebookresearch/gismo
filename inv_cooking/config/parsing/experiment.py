"""
The idea behind the experiment is to have a way to inherit from another configuration
- each experiment is an entry in a DB
- each experiment can be launched just by name
- meta-experiments allow to run a batch of experiments
"""


from dataclasses import dataclass, field
from typing import Dict, Any, List

from omegaconf import OmegaConf, DictConfig

from inv_cooking.config.optimization import OptimizationConfig


@dataclass
class Experiment:
    name: str = ""
    comment: str = ""
    optimization: OptimizationConfig = OptimizationConfig()


@dataclass
class Experiments:
    im2ingr: Dict[str, Any] = field(default_factory=dict)
    im2recipe: Dict[str, Any] = field(default_factory=dict)
    ingr2recipe: Dict[str, Any] = field(default_factory=dict)


def parse_experiment(config: Experiments, task: str, name: str) -> Experiment:
    """
    Read the raw configuration and return the experiment matching the task and the name
    TODO - make it return a list so that we can do hyper-parameter searches
    """
    task_experiments = getattr(config, task)

    # Get all parents
    names = _get_parents_name(task_experiments, name)
    names.append(name)

    # Merge the configuration with those of parents
    OmegaConf.set_struct(task_experiments, False)
    configs = [task_experiments[name] for name in names]
    config = OmegaConf.merge(*configs)
    del config["parent"]

    # Validate the schema
    schema = OmegaConf.structured(Experiment)
    schema.name = name
    return OmegaConf.merge(schema, config)


def _get_parents_name(config: DictConfig, name: str) -> List[str]:
    """
    Return the list of parents of a given experiment, from the oldest parent to the closest parent
    """
    parents = []
    current_name = name
    while "parent" in config[current_name]:
        parent_name = config[current_name]["parent"]
        if parent_name in parents:
            raise ValueError(f"Circular parents for configuration named {name}")
        parents.append(parent_name)
        current_name = parent_name
    return parents[::-1]
