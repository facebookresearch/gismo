import hydra

from inv_cooking.scheduler import RawConfig
from inv_cooking.datasets.recipe1m import run_dataset_pre_processing
from inv_cooking.datasets.recipe1m import run_comment_pre_processing


@hydra.main(config_path="conf", config_name="config")
def main(cfg: RawConfig) -> None:
    """
    Preprocess the selected dataset.

    Example usage:
    `python preprocess.py dataset=recipe1m`
    """
    vocab_ingrs, dataset = run_dataset_pre_processing(cfg.dataset.path, cfg.dataset.pre_processing)
    run_comment_pre_processing(cfg.dataset.path, cfg.dataset.pre_processing.save_path, vocab_ingrs, dataset)

if __name__ == "__main__":
    main()
