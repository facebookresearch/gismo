# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra

from inv_cooking.scheduler import RawConfig
import os


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: RawConfig) -> None:
    print(f"Removing all debug experiments from folder {cfg.checkpoint.tensorboard_folder}...")
    for folder_path, _, _ in os.walk(cfg.checkpoint.tensorboard_folder):
        last_folder = os.path.split(folder_path)[-1]
        if last_folder.startswith("version_"):
            os.system(f"rm -rf {folder_path}")


if __name__ == "__main__":
    main()
