import os
import glob
import torch

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

    checkpoint_dir = os.path.join(cfg.checkpoint.dir, cfg.task.name + "-" + cfg.name)
    all_checkpoints = glob.glob(os.path.join(checkpoint_dir, 'best*.ckpt'))
    if len(all_checkpoints) == 0:
        raise ValueError(f'Checkpoint {checkpoint_dir} does not exist.')

    # data module
    data_module = _load_data_set(cfg)
    data_module.prepare_data()
    data_module.setup("test")

    # model
    # note: it should be possible to directly load the weights when creating the model by using load_from_checkpoint function
    # this function seems to rely on saving hyper-parameters though
    model = _create_model(cfg, data_module)
    monitored_metric = model.get_monitored_metric()

    # check best checkpoint path
    best_scores = torch.zeros(len(all_checkpoints))        
    for i, filename in enumerate(all_checkpoints):
        check = torch.load(filename)
        best_scores[i] = list(check['callbacks'].values())[0]['best_model_score']
        del check
    if monitored_metric.mode == 'min':
        pos = best_scores.argmin()
    else:
        pos = best_scores.argmax()

    print(f'Using checkpoint {all_checkpoints[pos]}')

    # trainer
    trainer = pl.Trainer(
            gpus=gpus,
            num_nodes=nodes,
            accelerator=distributed_mode,
            benchmark=True,  # increases speed for fixed image sizes
            precision=32,
            resume_from_checkpoint=all_checkpoints[pos],
        )

    # test model
    trainer.test(model, datamodule=data_module)