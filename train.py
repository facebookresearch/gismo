import hydra

from inv_cooking.config.parsing.raw_config import RawConfig
from inv_cooking.scheduler import schedule_job


@hydra.main(config_path="conf", config_name="config")
def main(cfg: RawConfig) -> None:
    config = RawConfig.to_config(cfg)
    schedule_job(config)


if __name__ == "__main__":
    main()
