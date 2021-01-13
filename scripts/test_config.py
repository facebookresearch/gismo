import hydra
from omegaconf import OmegaConf

from inv_cooking.scheduler import RawConfig


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: RawConfig) -> None:
    """
    Print all job configurations that are generated from a given task and experiment name
    """
    configs = RawConfig.to_config(cfg)
    for i, config in enumerate(configs):
        print("-" * 50)
        print(f"EXPERIMENT {i+1}")
        print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
