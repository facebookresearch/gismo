import hydra

from inv_cooking.datasets.recipe1m import LoadingOptions, Recipe1MDataModule, Recipe1M
from inv_cooking.scheduler import RawConfig
from inv_cooking.utils.hydra import copy_source_code_to_cwd


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: RawConfig) -> None:
    copy_source_code_to_cwd()
    data_module = load_ingredient_dataset(cfg)
    print("TRAINING SET:")
    check_missing_ingredients(data_module.dataset_train)
    print("VALIDATION SET:")
    check_missing_ingredients(data_module.dataset_val)


def load_ingredient_dataset(cfg):
    loading_options = LoadingOptions(
        with_image=False,
        with_ingredient=True,
        with_ingredient_eos=False,
        with_recipe=False,
    )
    data_module = Recipe1MDataModule(
        dataset_config=cfg.dataset,
        seed=0,
        loading_options=loading_options,
    )
    data_module.prepare_data()
    data_module.setup("fit")
    return data_module


def check_missing_ingredients(dataset: Recipe1M):
    ingr_vocab_size = dataset.get_ingr_vocab_size()
    ingr_vocab_used = set()
    for i in range(len(dataset)):
        ingr_indices = dataset.load_ingredients(i)
        ingr_vocab_used.update(ingr_indices)
    print("- total:", ingr_vocab_size)
    print("- used:", len(ingr_vocab_used))
    print("- missing:", ingr_vocab_size - len(ingr_vocab_used))


if __name__ == "__main__":
    main()
