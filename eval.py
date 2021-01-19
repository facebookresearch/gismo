import hydra

from inv_cooking.scheduler import schedule_jobs, RawConfig


@hydra.main(config_path="conf", config_name="config")
def main(cfg: RawConfig) -> None:
    schedule_jobs(RawConfig.to_config(cfg), mode="eval")


if __name__ == "__main__":
    main()
