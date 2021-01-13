import hydra

from inv_cooking.scheduler import RawConfig
import os


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: RawConfig) -> None:
    print(f"Removing all experiments from folder {cfg.checkpoint.tensorboard_folder}...")
    os.system(f"rm -rf {cfg.checkpoint.tensorboard_folder}")
    os.makedirs(cfg.checkpoint.tensorboard_folder, exist_ok=True)


if __name__ == "__main__":
    main()
