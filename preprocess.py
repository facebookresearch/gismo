import hydra

from inv_cooking.config.parsing.raw_config import RawConfig
from inv_cooking.datasets.recipe1m.preprocess import run_dataset_pre_processing


@hydra.main(config_path="conf", config_name="config")
def main(cfg: RawConfig) -> None:
    configs = RawConfig.to_config(cfg)
    if configs:
        # TODO - the expansion of experiment might not be necessary for pre-processing
        config = configs[0]
        run_dataset_pre_processing(config.dataset.path, config.dataset.pre_processing)


if __name__ == "__main__":
    main()
