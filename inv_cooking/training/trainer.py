# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from inv_cooking.config import Config, TaskType
from inv_cooking.datasets.recipe1m import LoadingOptions, Recipe1MDataModule
from inv_cooking.models.ingredients_predictor.builder import requires_eos_token
from inv_cooking.utils.checkpointing import get_checkpoint_directory
from inv_cooking.utils.logging import dump_configuration, get_log_version

from .image_to_ingredients import ImageToIngredients
from .image_to_recipe import ImageToRecipe
from .image_to_title import ImageToTitle
from .ingredient_to_recipe import IngredientToRecipe


def run_training(
    cfg: Config,
    gpus: int,
    nodes: int,
    distributed_mode: str,
    load_checkpoint: bool,
) -> None:
    seed_everything(cfg.optimization.seed)
    dump_configuration(cfg)

    checkpoint_dir = get_checkpoint_directory(cfg)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint.log_folder, exist_ok=True)
    os.makedirs(cfg.checkpoint.tensorboard_folder, exist_ok=True)

    # data module
    data_module = load_data_set(cfg)
    data_module.prepare_data()
    data_module.setup("fit")

    # model
    model = create_model(cfg, data_module)

    # logging
    if not cfg.debug_mode:
        logger = pl_loggers.TensorBoardLogger(
            save_dir=cfg.checkpoint.tensorboard_folder,
            name=cfg.task.name + "-" + cfg.name,
            version=get_log_version(),
            default_hp_metric=False,
        )
    else:
        logger = False

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
        patience=cfg.optimization.patience,
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
        # track_grad_norm=2,  # Track the L2 norm of the gradients (debugging)
        # terminate_on_nan=True,  # Show when gradients have NAN (debugging)
    )

    trainer.fit(model, datamodule=data_module)

    if cfg.eval_on_test:
        data_module.setup("test")
        trainer.test(datamodule=data_module)


def load_data_set(cfg, with_id: bool = False):
    return Recipe1MDataModule(
        dataset_config=cfg.dataset,
        seed=cfg.optimization.seed,
        loading_options=_get_loading_options(cfg, with_id=with_id),
    )


def _get_loading_options(cfg: Config, with_id: bool = False) -> LoadingOptions:
    include_eos = requires_eos_token(cfg.ingr_predictor)
    with_substitutions = cfg.dataset.ablation.with_substitutions
    if cfg.task == TaskType.im2ingr:
        return LoadingOptions(
            with_id=with_id,
            with_image=cfg.image_encoder.with_image_encoder,
            with_ingredient=True,
            with_ingredient_eos=include_eos,
            with_title=cfg.title_encoder.with_title,
        )
    elif cfg.task == TaskType.im2recipe:
        return LoadingOptions(
            with_id=with_id,
            with_image=True,
            with_ingredient=True,
            with_ingredient_eos=include_eos,
            with_recipe=True,
            with_substitutions=with_substitutions,
        )
    elif cfg.task == TaskType.ingr2recipe:
        return LoadingOptions(
            with_id=with_id,
            with_ingredient=True,
            with_ingredient_eos=include_eos,
            with_recipe=True,
        )
    elif cfg.task == TaskType.im2title:
        return LoadingOptions(
            with_id=with_id,
            with_image=True,
            with_title=True
        )
    elif cfg.task == TaskType.ingrsubs:
        return LoadingOptions(
            with_id=True,
            with_ingredient=True,
            with_ingredient_eos=include_eos,
            with_recipe=True,
            with_title=True,
        )
    else:
        raise ValueError(f"Unknown task: {cfg.task.name}.")


def create_model(cfg: Config, data_module: Recipe1MDataModule):
    max_num_ingredients = cfg.dataset.filtering.max_num_labels
    max_recipe_len = (
        cfg.dataset.filtering.max_num_instructions
        * cfg.dataset.filtering.max_instruction_length
    )
    if cfg.task == TaskType.im2ingr:
        return ImageToIngredients(
            image_encoder_config=cfg.image_encoder,
            title_encoder_config=cfg.title_encoder,
            ingr_pred_config=cfg.ingr_predictor,
            optim_config=cfg.optimization,
            max_num_ingredients=max_num_ingredients,
            title_vocab_size=data_module.title_vocab_size,
            ingr_vocab_size=data_module.ingr_vocab_size,
            ingr_eos_value=data_module.ingr_eos_value,
        )
    elif cfg.task == TaskType.im2recipe:
        return ImageToRecipe(
            cfg.image_encoder,
            cfg.ingr_predictor,
            cfg.recipe_gen,
            cfg.optimization,
            cfg.pretrained_im2ingr,
            cfg.ingr_teachforce,
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
    elif cfg.task == TaskType.im2title:
        return ImageToTitle(
            image_encoder_config=cfg.image_encoder,
            ingr_pred_config=cfg.ingr_predictor,
            title_gen_config=cfg.recipe_gen,
            optim_config=cfg.optimization,
            max_title_len=cfg.dataset.filtering.max_title_seq_len,
            title_vocab_size=data_module.title_vocab_size,
        )
    elif cfg.task == TaskType.ingrsubs:
        raise Exception("TO DO")
    else:
        raise ValueError(f"Unknown task: {cfg.task.name}.")
