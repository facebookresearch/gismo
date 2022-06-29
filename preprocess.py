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

    print("Start running: run_dataset_pre_processing...")
    vocab_ingrs, dataset = run_dataset_pre_processing(
        cfg.dataset.path,
        cfg.dataset.pre_processing
    )
    print("Done running: run_dataset_pre_processing!")

    print("Start running: run_comment_pre_processing...")
    train_subs, val_subs, test_subs = run_comment_pre_processing(
        cfg.dataset.path,
        os.path.expanduser(cfg.dataset.pre_processing.save_path),
        vocab_ingrs,
        dataset
    )
    print("Done running: run_comment_pre_processing!")

    print("Start running: create_pre_processed_recipesubs_data...")
    create_pre_processed_recipesubs_data(cfg.dataset.pre_processing.save_path, dataset, {"train": train_subs, "val": val_subs, "test": test_subs})
    print("Done running: create_pre_processed_recipesubs_data!")


if __name__ == "__main__":
    main()
