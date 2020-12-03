import hydra
from omegaconf import DictConfig

from inv_cooking.main import main as run


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
