import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything

from inv_cooking.config import Config
from inv_cooking.utils.checkpointing import (
    list_available_checkpoints,
    select_best_checkpoint,
)

from .trainer import create_model, load_data_set


def run_eval(cfg: Config, gpus: int, nodes: int, distributed_mode: str) -> None:
    """
    Evaluate a model using:
    - either the validation set or the test set
    - ground truth as ingredient or not (for im2recipe)
    """

    seed_everything(cfg.optimization.seed)

    checkpoint_dir = cfg.eval_checkpoint_dir
    all_checkpoints = list_available_checkpoints(checkpoint_dir)
    if len(all_checkpoints) == 0:
        raise ValueError(f"Checkpoint {checkpoint_dir} does not exist.")

    # Creating the data module
    data_module = load_data_set(cfg)
    data_module.prepare_data()
    data_module.setup("test")

    # Creating the model
    model = create_model(cfg, data_module)
    monitored_metric = model.get_monitored_metric()

    # Find best checkpoint path
    best_checkpoint = select_best_checkpoint(all_checkpoints, monitored_metric.mode)
    print(f"Using checkpoint {best_checkpoint}")

    # Load the checkpoint
    checkpoint = torch.load(best_checkpoint, map_location="cpu")
    model.on_load_checkpoint(checkpoint)  # callback of our models
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    # Create the training, initializing from the provided checkpoint
    trainer = pl.Trainer(
        gpus=gpus,
        num_nodes=nodes,
        accelerator=distributed_mode,
        benchmark=True,  # increases speed for fixed image sizes
        precision=32,
        progress_bar_refresh_rate=1 if cfg.slurm.partition == "local" else 0,
        # resume_from_checkpoint=best_checkpoint,
    )

    # Run the evaluation on the module
    trainer.test(
        model,
        datamodule=data_module,
        # ckpt_path = best_checkpoint,
    )
