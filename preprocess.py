import hydra

from inv_cooking.config import RawConfig
from inv_cooking.datasets.recipe1m.preprocess import run_dataset_pre_processing


@hydra.main(config_path="conf", config_name="config")
def main(cfg: RawConfig) -> None:
    config = RawConfig.to_config(cfg)
    run_dataset_pre_processing(config.dataset.path, config.dataset.pre_processing)


if __name__ == "__main__":
    main()
