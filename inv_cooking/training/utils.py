import abc
from typing import Any, Dict, List, NamedTuple

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from inv_cooking.config import OptimizationConfig


class MonitoredMetric(NamedTuple):
    name: str
    mode: str


class OptimizationGroup(NamedTuple):
    model: nn.Module
    name: str
    pretrained: bool


class _BaseModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_monitored_metric(self) -> MonitoredMetric:
        ...

    def log_training_losses(
        self, losses: Dict[str, torch.Tensor], optim_config: OptimizationConfig
    ) -> torch.Tensor:
        """
        Log all the training losses given as parameters and return the total loss
        """
        total_loss = 0
        for loss_key in losses.keys():
            loss = losses[loss_key].mean()  # Average across GPUs
            self.log_on_progress_bar(loss_key, loss)
            total_loss += loss * optim_config.loss_weights[loss_key]
        self.log_on_progress_bar("train_loss", total_loss)
        return total_loss

    def log_on_progress_bar(self, key: str, value: Any):
        self.log(
            key, value, on_step=True, on_epoch=True, prog_bar=True, logger=True,
        )

    @classmethod
    def create_optimizers(
        cls, optim_groups: List[OptimizationGroup], optim_config: OptimizationConfig,
    ):
        parameter_groups = []
        for optim_group in optim_groups:
            parameter_groups += cls.create_parameter_group(
                module=optim_group.model,
                optim_config=optim_config,
                pretrained=optim_group.pretrained,
                name=optim_group.name,
            )
        return cls.make_adam_optimizer(parameter_groups, optim_config)

    @staticmethod
    def make_adam_optimizer(parameter_groups, optim_config: OptimizationConfig):
        optimizer = torch.optim.Adam(
            parameter_groups,
            lr=optim_config.lr,
            weight_decay=optim_config.weight_decay,
        )
        scheduler = {
            "scheduler": ExponentialLR(optimizer, optim_config.lr_decay_rate),
            "interval": "epoch",
            "frequency": optim_config.lr_decay_every,
        }
        return [optimizer], [scheduler]

    @staticmethod
    def create_parameter_group(
        module: nn.Module,
        optim_config: OptimizationConfig,
        pretrained: bool = False,
        name: str = "",
    ) -> List[Dict[str, Any]]:
        pretrained_lr = optim_config.lr * optim_config.scale_lr_pretrained
        lr = pretrained_lr if pretrained else optim_config.lr
        params = [p for p in module.parameters() if p.requires_grad]
        nb_params = sum([p.numel() for p in params])
        print(f"Number of trainable parameters in {name} is {nb_params}.")
        if nb_params > 0:
            return [{"params": params, "lr": lr}]
        return []
