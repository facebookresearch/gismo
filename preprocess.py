import hydra

from inv_cooking.config import Config
from inv_cooking.datasets.recipe1m.preprocess import run_dataset_pre_processing


@hydra.main(config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    run_dataset_pre_processing(cfg.dataset.path, cfg.dataset.pre_processing)


if __name__ == "__main__":
    main()
