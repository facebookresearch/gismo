import hydra
from omegaconf import DictConfig

from inv_cooking.scheduler import schedule_job


@hydra.main(config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    schedule_job(cfg)


if __name__ == "__main__":
    main()
