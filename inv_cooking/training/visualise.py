import torch
from pytorch_lightning import seed_everything

from inv_cooking.config import Config
from inv_cooking.training.trainer import create_model, load_data_set
from inv_cooking.utils.checkpointing import (
    list_available_checkpoints,
    select_best_checkpoint,
)
from inv_cooking.utils.visualisation.im2recipe import Im2RecipeVisualiser


def run_visualisation(cfg: Config) -> None:
    seed_everything(cfg.optimization.seed)
    assert cfg.eval_checkpoint_dir, "You need to provide a checkpoint to visualize"

    checkpoint_dir = cfg.eval_checkpoint_dir
    all_checkpoints = list_available_checkpoints(checkpoint_dir)
    if len(all_checkpoints) == 0:
        raise ValueError(
            f"Checkpoint {checkpoint_dir} does not exist or does not contain any checkpoints."
        )

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

    # Dump the model and data_module, so that they can be used to play in a
    # jupyter notebook later
    result = {
        "model": model,
        "data_module": data_module,
    }
    torch.save(result, "model_and_module.torch")

    # Take some example of data and print the corresponding recipes
    visualizer = Im2RecipeVisualiser(model=model, data_module=data_module)
    visualizer.visualize()
