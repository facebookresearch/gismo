import os
from typing import Tuple

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from inv_cooking.config import Config, IngredientPredictorType, TaskType
from inv_cooking.datasets.recipe1m import LoadingOptions, Recipe1MDataModule

from .module import LitInverseCooking


def run_training(
    cfg: Config, gpus: int, nodes: int, distributed_mode: str, load_checkpoint: bool,
) -> None:
    seed_everything(cfg.optimization.seed)

    checkpoint_dir = os.path.join(cfg.checkpoint.dir, cfg.task.name + "-" + cfg.name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # data module
    dm = Recipe1MDataModule(
        dataset_config=cfg.dataset,
        seed=cfg.optimization.seed,
        loading_options=_get_loading_options(cfg),
        # checkpoint=None ## TODO: check how this would work
    )
    dm.prepare_data()
    dm.setup("fit")

    # model
    model = LitInverseCooking(
        task=cfg.task,
        image_encoder_config=cfg.image_encoder if "im" in cfg.task.name else None,
        ingr_pred_config=cfg.ingr_predictor if "im" in cfg.task.name else None,
        recipe_gen_config=cfg.recipe_gen if "recipe" in cfg.task.name else None,
        optim_config=cfg.optimization,
        max_num_labels=cfg.dataset.filtering.max_num_labels,
        max_recipe_len=cfg.dataset.filtering.max_num_instructions
        * cfg.dataset.filtering.max_instruction_length,
        ingr_vocab_size=dm.ingr_vocab_size,
        instr_vocab_size=dm.instr_vocab_size,
        ingr_eos_value=dm.ingr_eos_value,
    )

    logger = pl_loggers.TensorBoardLogger(
        os.path.join(cfg.checkpoint.dir, "logs"), name=cfg.task.name + "-" + cfg.name,
    )

    # checkpointing and early stopping
    monitor_metric, best_metric = _get_monitored_metric(cfg)
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath=checkpoint_dir,
        filename="best",
        save_last=True,
        mode=best_metric,
        save_top_k=1,
    )
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode=best_metric,
    )

    # trainer
    is_local = cfg.slurm.partition == "local"
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
        progress_bar_refresh_rate=1 if is_local else 0,
        weights_summary=None,  # for it otherwise logs lots of useless information
    )

    trainer.fit(model, datamodule=dm)

    if cfg.eval_on_test:
        dm.setup("test")
        trainer.test(datamodule=dm)


def _get_loading_options(cfg: Config) -> LoadingOptions:
    include_eos = cfg.ingr_predictor.model != IngredientPredictorType.ff
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
        raise ValueError(f"Unknown task: {cfg.task}.")


def _get_monitored_metric(cfg) -> Tuple[str, str]:
    if cfg.task == TaskType.im2ingr:
        return "val_o_f1", "max"
    elif cfg.task == TaskType.im2recipe:
        return "val_perplexity", "min"
    elif cfg.task == TaskType.ingr2recipe:
        return "val_perplexity", "min"
    else:
        raise ValueError(f"Unknown task: {cfg.task}.")
