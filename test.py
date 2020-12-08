import os

import hydra
from omegaconf import OmegaConf

from inv_cooking.config import RawConfig


@hydra.main(config_path="conf", config_name="config")
def main(cfg: RawConfig) -> None:
    config = RawConfig.to_config(cfg)
    print(os.getcwd())
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
