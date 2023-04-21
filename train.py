# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra

from inv_cooking.scheduler import schedule_jobs, RawConfig
from inv_cooking.scheduler.scheduler import TrainingMode


@hydra.main(config_path="conf", config_name="config")
def main(cfg: RawConfig) -> None:
    """
    Run the distributed training on the selected configuration.

    Example usage:
    `python train.py task=im2recipe name=im2recipe slurm=dev`
    """
    schedule_jobs(cfg, training_mode=TrainingMode.TRAIN)


if __name__ == "__main__":
    main()
