import hydra

from inv_cooking.scheduler import schedule_job, RawConfig


@hydra.main(config_path="conf", config_name="config")
def main(cfg: RawConfig) -> None:
    configs = RawConfig.to_config(cfg)
    for config in configs:
        schedule_job(config)


if __name__ == "__main__":
    main()
