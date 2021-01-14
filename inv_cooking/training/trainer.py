import os

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from inv_cooking.config import Config, TaskType
from inv_cooking.datasets.recipe1m import LoadingOptions, Recipe1MDataModule
from inv_cooking.models.ingredients_predictor.builder import requires_eos_token
from .image_to_ingredients import ImageToIngredients
from .image_to_recipe import ImageToRecipe
from .ingredient_to_recipe import IngredientToRecipe


def run_training(
    cfg: Config, gpus: int, nodes: int, distributed_mode: str, load_checkpoint: bool,
) -> None:
    seed_everything(cfg.optimization.seed)

    checkpoint_dir = os.path.join(cfg.checkpoint.dir, cfg.task.name + "-" + cfg.name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # data module
    data_module = _load_data_set(cfg)
    data_module.prepare_data()
    data_module.setup("fit")

    # model
    model = _create_model(cfg, data_module)

    # logging
    logger = pl_loggers.TensorBoardLogger(
        cfg.checkpoint.tensorboard_folder, name=cfg.task.name + "-" + cfg.name,
    )

    # check-pointing and early stopping
    monitored_metric = model.get_monitored_metric()
    checkpoint_callback = ModelCheckpoint(
        monitor=monitored_metric.name,
        dirpath=checkpoint_dir,
        filename="best",
        save_last=True,
        mode=monitored_metric.mode,
        save_top_k=1,
    )
    early_stop_callback = EarlyStopping(
        monitor=monitored_metric.name,
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode=monitored_metric.mode,
    )

    # trainer
    trainer = pl.Trainer(
        gpus=gpus,
        num_nodes=nodes,
        accelerator=distributed_mode,
        benchmark=True,  # increases speed for fixed image sizes
        check_val_every_n_epoch=1,
        checkpoint_callback=True,
        max_epochs=cfg.optimization.max_epochs,
        num_sanity_val_steps=0,  # to debug validation without training
        precision=32,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
        ],  # need to overwrite ModelCheckpoint callback? check loader/iterator state
        logger=logger,
        # log_every_n_steps=10,
        # flush_logs_every_n_steps=50,
        resume_from_checkpoint=checkpoint_dir if load_checkpoint else None,
        sync_batchnorm=cfg.optimization.sync_batchnorm,
        # weights_save_path=checkpoint_dir,
        # limit_train_batches=10,
        fast_dev_run=cfg.debug_mode,
        progress_bar_refresh_rate=1 if cfg.slurm.partition == "local" else 0,
        weights_summary=None,  # for it otherwise logs lots of useless information
    )

    trainer.fit(model, datamodule=data_module)

    if cfg.eval_on_test:
        data_module.setup("test")
        trainer.test(datamodule=data_module)


def _load_data_set(cfg):
    return Recipe1MDataModule(
        dataset_config=cfg.dataset,
        seed=cfg.optimization.seed,
        loading_options=_get_loading_options(cfg),
        # checkpoint=None ## TODO: check how this would work
    )


def _get_loading_options(cfg: Config) -> LoadingOptions:
    include_eos = requires_eos_token(cfg.ingr_predictor)
    if cfg.task == TaskType.im2ingr:
        return LoadingOptions(
            with_image=True, with_ingredient=True, with_ingredient_eos=include_eos,
        )
    elif cfg.task == TaskType.im2recipe:
        return LoadingOptions(
            with_image=True,
            with_ingredient=True,
            with_ingredient_eos=include_eos,
            with_recipe=True,
        )
    elif cfg.task == TaskType.ingr2recipe:
        return LoadingOptions(
            with_ingredient=True, with_ingredient_eos=include_eos, with_recipe=True,
        )
    else:
        raise ValueError(f"Unknown task: {cfg.task.name}.")


def _create_model(cfg: Config, data_module: Recipe1MDataModule):
    max_num_ingredients = cfg.dataset.filtering.max_num_labels
    max_recipe_len = (
        cfg.dataset.filtering.max_num_instructions
        * cfg.dataset.filtering.max_instruction_length
    )
    if cfg.task == TaskType.im2ingr:
        return ImageToIngredients(
            cfg.image_encoder,
            cfg.ingr_predictor,
            cfg.optimization,
            max_num_ingredients=max_num_ingredients,
            ingr_vocab_size=data_module.ingr_vocab_size,
            ingr_eos_value=data_module.ingr_eos_value,
        )
    elif cfg.task == TaskType.im2recipe:
        return ImageToRecipe(
            cfg.image_encoder,
            cfg.ingr_predictor,
            cfg.recipe_gen,
            cfg.optimization,
            max_num_ingredients=max_num_ingredients,
            max_recipe_len=max_recipe_len,
            ingr_vocab_size=data_module.ingr_vocab_size,
            instr_vocab_size=data_module.instr_vocab_size,
            ingr_eos_value=data_module.ingr_eos_value,
        )
    elif cfg.task == TaskType.ingr2recipe:
        return IngredientToRecipe(
            cfg.recipe_gen,
            cfg.optimization,
            max_recipe_len=max_recipe_len,
            ingr_vocab_size=data_module.ingr_vocab_size,
            instr_vocab_size=data_module.instr_vocab_size,
            ingr_eos_value=data_module.ingr_eos_value,
        )
    else:
        raise ValueError(f"Unknown task: {cfg.task.name}.")
