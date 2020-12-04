import os

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from inv_cooking.config import Config, TaskType
from inv_cooking.datasets.recipe1m.loader import Recipe1MDataModule
from inv_cooking.inversecooking import LitInverseCooking


def run_training(cfg: Config, gpus: int, nodes: int, distributed_mode: str) -> None:

    # fix seed
    seed_everything(cfg.misc.seed)

    # checkpointing
    checkpoint_dir = os.path.join(
        cfg.checkpoint.dir, str(cfg.task) + "-" + cfg.ingr_predictor.model
    )
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # what flags do we need to activate in the data module
    shuffle_labels = True if "shuffle" in cfg.ingr_predictor.model else False
    include_eos = (
        False
        if "ff" in cfg.ingr_predictor.model and not "shuffle" in cfg.ingr_predictor
        else True
    )

    if cfg.task == TaskType.im2ingr:
        return_img = True
        return_ingr = True
        return_recipe = False
        monitor_metric = "val_o_f1"
        best_metric = "max"
    elif cfg.task == TaskType.im2recipe:
        return_img = True
        return_ingr = True
        return_recipe = True
        monitor_metric = "val_perplexity"
        best_metric = "min"
    elif cfg.task == TaskType.ingr2recipe:
        return_img = False
        return_ingr = True
        return_recipe = True
        monitor_metric = "val_perplexity"
        best_metric = "min"
    else:
        raise ValueError(f"Unknown task: {cfg.task}.")

    # data module
    splits_path = os.path.join(os.getcwd(), cfg.dataset.splits_path)
    dm = Recipe1MDataModule(
        data_dir=cfg.dataset.path,
        splits_path=splits_path,
        maxnumlabels=cfg.dataset.maxnumlabels,
        maxnuminstrs=cfg.dataset.maxnuminstrs,
        maxinstrlength=cfg.dataset.maxinstrlength,
        batch_size=cfg.misc.batch_size,
        num_workers=cfg.misc.num_workers,
        shuffle_labels=shuffle_labels,
        include_eos=include_eos,
        seed=cfg.misc.seed,
        preprocessing=cfg.preprocessing,
        return_img=return_img,
        return_ingr=return_ingr,
        return_recipe=return_recipe
        # checkpoint=None ## TODO: check how this would work
    )
    dm.prepare_data()
    dm.setup("fit")

    # model
    model = LitInverseCooking(
        task=cfg.task,
        im_args=cfg.image_encoder,
        ingrpred_args=cfg.ingr_predictor,
        recipegen_args=cfg.recipe_gen if "recipe" in str(cfg.task) else None,
        optim_args=cfg.optim,
        dataset_name=cfg.dataset.name,
        maxnumlabels=cfg.dataset.maxnumlabels,
        maxrecipelen=cfg.dataset.maxnuminstrs * cfg.dataset.maxinstrlength,
        ingr_vocab_size=dm.ingr_vocab_size,
        instr_vocab_size=dm.instr_vocab_size if "recipe" in str(cfg.task) else None,
        ingr_eos_value=dm.ingr_eos_value,
    )

    # logger
    tb_logger = pl_loggers.TensorBoardLogger(
        os.path.join(cfg.checkpoint.dir, "logs/"),
        name=str(cfg.task) + "-" + cfg.ingr_predictor.model,
    )

    # checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath=checkpoint_dir,
        filename="best",
        save_last=True,
        mode=best_metric,
        save_top_k=1,
    )

    # early stopping
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode=best_metric,
    )

    # trainer
    trainer = pl.Trainer(
        gpus=gpus,
        num_nodes=nodes,
        accelerator=distributed_mode,
        benchmark=True,  # increases speed for fixed image sizes
        check_val_every_n_epoch=1,
        checkpoint_callback=True,
        max_epochs=cfg.optim.max_epochs,
        num_sanity_val_steps=0,  # to debug validation without training
        precision=32,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
        ],  # need to overwrite ModelCheckpoint callback? check loader/iterator state
        logger=tb_logger,
        # log_every_n_steps=10,
        # flush_logs_every_n_steps=50,
        # resume_from_checkpoint=cfg.checkpoint.resume_from,
        # sync_batchnorm=True,
        # weights_save_path=checkpoint_dir,
        # limit_train_batches=10,
        fast_dev_run=True,  # set to true for debugging
    )

    # train
    trainer.fit(model, datamodule=dm)

    # test
    dm.setup("test")
    trainer.test(datamodule=dm)
