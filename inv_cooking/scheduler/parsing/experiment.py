"""
The idea behind the experiment is to have a way to inherit from another configuration
- each experiment is an entry in a DB
- each experiment can be launched just by name
- meta-experiments allow to run a batch of experiments
"""


from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from omegaconf import DictConfig, OmegaConf

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


def parse_experiments(config: Experiments, task: str, name: str) -> List[Experiment]:
    """
    Read the raw configuration, select the experiment matching the task and the name,
    and expand it as a list of configuration (in case of hyper-parameter searches)
    """
    task_experiments = getattr(config, task, None)
    if task_experiments is None:
        raise ValueError(f"Unknown task {task}")

    # Get all parents
    names = _get_parents_name(task_experiments, name)
    names.append(name)

    # Merge the configuration with those of parents
    OmegaConf.set_struct(task_experiments, False)
    configs = [task_experiments[name] for name in names]
    config = OmegaConf.merge(*configs)
    config.pop("parent", default=None)

    # Now, expand the configuration in case it is a generator configuration
    experiments = []
    configs = _expand_search(config)
    for config in configs:
        schema = OmegaConf.structured(Experiment)
        schema.name = name
        experiments.append(OmegaConf.merge(schema, config))
    return experiments


SEARCH_PREFIX = "SEARCH"


@dataclass
class Search:
    values: List[Any]

    @classmethod
    def parse(cls, s: str) -> "Search":
        s = s[len(SEARCH_PREFIX) :]
        values = OmegaConf.create(s)
        return cls(values=values)


def _expand_search(config: DictConfig) -> List[DictConfig]:
    """
    Expand a configuration-generator (which might include search) into multiple configurations,
    one for each of the configuration to try
    """
    generated_configs = []
    variables = _collect_variable_paths(config)
    assignments = _all_variables_assignments(variables)
    for assignment in assignments:
        new_config = deepcopy(config)
        for path, value in assignment:
            OmegaConf.update(new_config, path, value, merge=False)
        generated_configs.append(new_config)
    return generated_configs


def _collect_variable_paths(config: DictConfig) -> List[Tuple[str, Search]]:
    """
    Look for all the paths of a configuration generator which contains values to search

    Proceeds by a Breadth First Search through the DictConfig, to keep the document order,
    stopping at leaves and search nodes tagged with "SEARCH".

    Search nodes values are parsed, helping with the next phase: expansion.
    """
    variables = []
    to_visit = deque(config.keys())
    while to_visit:
        path = to_visit.popleft()
        node = OmegaConf.select(config, path)
        if OmegaConf.is_dict(node):
            for key in node.keys():
                to_visit.append(path + "." + key)
        elif OmegaConf.is_list(node):
            for i in range(len(node)):
                to_visit.append(path + "." + str(i))
        elif isinstance(node, str) and node.startswith(SEARCH_PREFIX):
            variables.append((path, Search.parse(node)))
    return variables


def _all_variables_assignments(
    variables: List[Tuple[str, Search]]
) -> List[List[Tuple[str, Any]]]:
    """
    Generate all combination of variable assignments for all different paths
    """
    nb_var = len(variables)
    possibles = [[]]
    for i in range(nb_var):
        next_possibles = []
        var_name = variables[i][0]
        for value in variables[i][1].values:
            for p in possibles:
                next_possibles.append(p + [(var_name, value)])
        possibles = next_possibles
    return possibles


def _get_parents_name(config: DictConfig, name: str) -> List[str]:
    """
    Return the list of parents of a given experiment, from the oldest parent to the closest parent
    """
    parents = []
    current_name = name
    while True:
        if current_name not in config:
            raise ValueError(f"Could not find experiment named {current_name}")
        if "parent" not in config[current_name]:
            break

        parent_name = config[current_name]["parent"]
        if parent_name in parents:
            raise ValueError(f"Circular parents for configuration named {name}")
        parents.append(parent_name)
        current_name = parent_name
    return parents[::-1]
