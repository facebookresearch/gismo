import hydra
from omegaconf import DictConfig

from inv_cooking.loaders.recipe1m_preprocess import run_dataset_pre_processing


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dataset_pre_processing(cfg.dataset.path, cfg.dataset.pre_processing)


if __name__ == "__main__":
    main()
