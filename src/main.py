import os

import hydra
import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from inversecooking import LitInverseCooking
from loaders.recipe1m_loader import Recipe1MDataModule
from loaders.recipe1m_preprocess import Vocabulary


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:

    seed_everything(cfg.misc.seed)

    # data module
    shuffle_labels = True if 'shuffle' in  cfg.ingr_predictor.model else False
    include_eos = False if 'ff' in  cfg.ingr_predictor.model else True

    dm = Recipe1MDataModule(data_dir=cfg.dataset.path,
                            batch_size=cfg.misc.batch_size,
                            num_workers=cfg.misc.num_workers,
                            shuffle_labels=shuffle_labels,
                            include_eos=include_eos,
                            seed=cfg.misc.seed,
                            preprocessing=cfg.preprocessing,
                            # checkpoint=None ## TODO: check how this would work
                            )
    dm.prepare_data()
    dm.setup('fit')

    # model
    model = LitInverseCooking(cfg.image_encoder,
                              cfg.ingr_predictor,
                              cfg.optim,
                              cfg.dataset.name,
                              cfg.dataset.maxnumlabels,
                              dm.ingr_vocab_size)

    # logger
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(cfg.checkpoint.dir, 'logs/'))

    # checkpointing
    checkpoint_callback = ModelCheckpoint(monitor='val_o_f1',
                                          dirpath=cfg.checkpoint.dir,
                                          filename='im2ingr-'+cfg.ingr_predictor.model+'{epoch:02d}-{val_o_f1:.2f}',  ## TODO adapt
                                          save_last=True,
                                          mode='max')

    # trainer
    trainer = pl.Trainer(
        gpus=2,
        num_nodes=1,
        accelerator='ddp',
        benchmark=True,  # increases speed for fixed image sizes
        check_val_every_n_epoch=1,
        checkpoint_callback=True,
        # log_every_n_steps=10,
        # flush_logs_every_n_steps=50,
        max_epochs=1000,
        num_sanity_val_steps=0,  # to debug validation without training
        precision=32,
        # resume_from_checkpoint=cfg.checkpoint.resume_from,
        sync_batchnorm=False,  ## TODO
        weights_save_path=cfg.checkpoint.dir,
        callbacks=[checkpoint_callback],  # need to overwrite ModelCheckpoint callback
        logger=tb_logger,
        # limit_train_batches=100,
        fast_dev_run=False  # set to true for debugging
    )

    # train
    trainer.fit(model, datamodule=dm)

    # test
    dm.setup('test')
    trainer.test(datamodule=dm)


if __name__ == '__main__':
    main()