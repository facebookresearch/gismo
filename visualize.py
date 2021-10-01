import hydra

from inv_cooking.scheduler import schedule_jobs, RawConfig
from inv_cooking.scheduler.scheduler import TrainingMode


@hydra.main(config_path="conf", config_name="config")
def main(cfg: RawConfig) -> None:
    schedule_jobs(cfg, training_mode=TrainingMode.VISUALIZE)


if __name__ == "__main__":
    main()
