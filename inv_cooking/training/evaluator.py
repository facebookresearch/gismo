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
from .trainer import _load_data_set, _create_model


def run_eval(
    cfg: Config, gpus: int, nodes: int, distributed_mode: str,
) -> None:
    seed_everything(cfg.optimization.seed)

    checkpoint_file = os.path.join(cfg.checkpoint.dir, cfg.task.name + "-" + cfg.name, 'best.ckpt')
    if not os.path.exists(checkpoint_file):
        raise Error(f'Checkpoint {checkpoint_dir} does not exist.')

    # data module
    data_module = _load_data_set(cfg)
    data_module.prepare_data()
    data_module.setup("test")

    # model
    # note: it should be possible to directly load the weights when creating the model by using load_from_checkpoint function
    # this function seems to rely on saving hyper-parameters though
    model = _create_model(cfg, data_module)

    # trainer
    trainer = pl.Trainer(
            gpus=gpus,
            num_nodes=nodes,
            accelerator=distributed_mode,
            benchmark=True,  # increases speed for fixed image sizes
            precision=32,
            resume_from_checkpoint=checkpoint_file,
        )

    # test model
    trainer.test(model, datamodule=data_module)