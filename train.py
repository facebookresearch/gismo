import hydra

from inv_cooking.config import Config
from inv_cooking.scheduler import schedule_job


@hydra.main(config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    schedule_job(cfg)


if __name__ == "__main__":
    main()
