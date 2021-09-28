import os
import hydra

from inv_cooking.scheduler import RawConfig
from inv_cooking.datasets.recipe1m import run_dataset_pre_processing
from inv_cooking.datasets.recipe1m import run_comment_pre_processing
from inv_cooking.datasets.recipe1m import create_pre_processed_recipesubs_data


@hydra.main(config_path="conf", config_name="config")
def main(cfg: RawConfig) -> None:
    """
    Preprocess the selected dataset.

    Example usage:
    `python preprocess.py dataset=recipe1m`
    """
    vocab_ingrs, dataset = run_dataset_pre_processing(cfg.dataset.path, cfg.dataset.pre_processing)
    # the next 3 lines are pending some updates from Bahare before they can be removed
    substitution_files = ["train_comments_subs.pkl", "val_comments_subs.pkl", "test_comments_subs.pkl"] 
    substitution_files = [os.path.join(cfg.dataset.pre_processing.save_path, file) for file in substitution_files]
    if not all([os.path.exists(file) for file in substitution_files]):
        run_comment_pre_processing(cfg.dataset.path, cfg.dataset.pre_processing.save_path, vocab_ingrs, dataset)
    create_pre_processed_recipesubs_data(cfg.dataset.pre_processing.save_path, dataset)

if __name__ == "__main__":
    main()
