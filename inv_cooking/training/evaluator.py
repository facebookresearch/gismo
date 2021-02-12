import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from inv_cooking.config import Config
from inv_cooking.utils.checkpointing import (
    get_checkpoint_directory,
    list_available_checkpoints,
    select_best_checkpoint,
)

from .trainer import create_model, load_data_set


def run_eval(cfg: Config, gpus: int, nodes: int, distributed_mode: str) -> None:
    seed_everything(cfg.optimization.seed)

    checkpoint_dir = get_checkpoint_directory(cfg)
    all_checkpoints = list_available_checkpoints(checkpoint_dir)
    if len(all_checkpoints) == 0:
        raise ValueError(f"Checkpoint {checkpoint_dir} does not exist.")

    # data module
    data_module = load_data_set(cfg)
    data_module.prepare_data()
    data_module.setup("test")

    # model
    # note: it should be possible to directly load the weights when creating the model by using load_from_checkpoint function
    # this function seems to rely on saving hyper-parameters though
    model = create_model(cfg, data_module)
    monitored_metric = model.get_monitored_metric()

    # check best checkpoint path
    best_checkpoint = select_best_checkpoint(all_checkpoints, monitored_metric.mode)
    print(f"Using checkpoint {best_checkpoint}")

    # trainer
    trainer = pl.Trainer(
        gpus=gpus,
        num_nodes=nodes,
        accelerator=distributed_mode,
        benchmark=True,  # increases speed for fixed image sizes
        precision=32,
        resume_from_checkpoint=best_checkpoint,
    )

    # test model
    trainer.test(model, datamodule=data_module)
