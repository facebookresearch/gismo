# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import submitit
from omegaconf import DictConfig, OmegaConf


def dump_configuration(cfg: DictConfig):
    """
    Write down the configuration in the logs so that we can
    easily compare the experiments
    """
    to_visit = list(cfg.keys())
    while to_visit:
        path = to_visit.pop()
        if OmegaConf.is_missing(cfg, path):
            continue

        node = OmegaConf.select(cfg, path, throw_on_missing=False)
        if OmegaConf.is_dict(node):
            for k in node.keys():
                to_visit.append(path + "." + k)
        elif OmegaConf.is_list(node):
            for i, v in enumerate(node):
                to_visit.append(path + "." + str(i))
        elif node is not None:
            print(f"[HPARAM] {path}: {node}")


def get_log_version() -> Optional[str]:
    """
    Return the version to be used by loggers such as tensorboard
    to identify separate runs of the same experiment:
    - return the SLURM JOB ID when running slurm
    - otherwise return None (and the logger will create an artificial version)
    """
    try:
        env = submitit.JobEnvironment()
        return env.job_id
    except RuntimeError:
        return None
