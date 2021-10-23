import glob
import os
from typing import List

import numpy as np
import pytorch_lightning
import torch

from inv_cooking.config import Config


def get_checkpoint_directory(cfg: Config) -> str:
    """
    Return the directory in which the checkpoints of the given config will be stored
    """
    return os.path.join(cfg.checkpoint.dir, cfg.task.name + "-" + cfg.name)


def list_available_checkpoints(checkpoint_dir: str) -> List[str]:
    """
    Get all the lightning checkpoints that are available in the given directory
    """
    return glob.glob(os.path.join(checkpoint_dir, "best*.ckpt"))


def select_best_checkpoint(available_checkpoints: List[str], metric_mode: str) -> str:
    """
    Select the best checkpoint among the available checkpoints paths, by loading them
    and selecting the one with the most advantageous metric
    :param available_checkpoints: list of checkpoint paths
    :param metric_mode: either "min" or "max", how to select the best metric
    :return the path of the best checkpoint
    """
    best_scores = []
    for i, filename in enumerate(available_checkpoints):
        check = torch.load(filename)
        best_score = check["callbacks"][
            pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint
        ]["best_model_score"]
        best_scores.append(best_score.item())
        del check
    best_scores = np.array(best_scores)
    pos = best_scores.argmin() if metric_mode == "min" else best_scores.argmax()
    best_checkpoint = available_checkpoints[pos.item()]
    return best_checkpoint
