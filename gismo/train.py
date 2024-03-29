# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import hydra
from recommender_gcn import Trainer


@hydra.main(config_path="conf", config_name="config")
def main(cfg) -> None:
    """
    Run the distributed training on the selected configuration.
    """
    print("Config completion...")
    cfg.base_dir = os.path.expanduser(cfg.base_dir)
    cfg.data_path = os.path.expanduser(cfg.data_path)
    print(cfg)

    print("Trainer started...")
    trainer = Trainer()
    trainer.train_recommender_gcn(cfg)


if __name__ == "__main__":
    main()
