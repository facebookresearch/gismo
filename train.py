import hydra

from inv_cooking.scheduler import schedule_jobs, RawConfig


@hydra.main(config_path="conf", config_name="config")
def main(cfg: RawConfig) -> None:
    """
    Run the distributed training on the selected configuration.

    Example usage:
    `python train.py task=im2recipe name=im2recipe slurm=dev`
    """
    schedule_jobs(cfg, training_mode=True)


if __name__ == "__main__":
    main()
