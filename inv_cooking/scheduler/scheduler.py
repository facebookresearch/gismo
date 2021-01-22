from typing import List

import submitit
import torch

from inv_cooking.config import Config
from inv_cooking.training import run_training, run_eval
from inv_cooking.utils.hydra import copy_source_code_to_cwd


def schedule_jobs(configurations: List[Config], training_mode: bool) -> None:
    copy_source_code_to_cwd()  # Because Hydra create a new running folder
    for config in configurations:
        if config.slurm.partition == "local":
            _schedule_job_locally(config, training_mode)
        else:
            _schedule_job_on_slurm(config, training_mode)


def _schedule_job_locally(cfg: Config, training_mode: bool):
    nb_gpu = torch.cuda.device_count()
    if training_mode:
        run_training(
            cfg, gpus=nb_gpu, nodes=1, distributed_mode="ddp", load_checkpoint=False
        )
    else:
        run_eval(cfg, gpus=nb_gpu, nodes=1, distributed_mode="ddp")


def _schedule_job_on_slurm(cfg: Config, training_mode: bool):
    nb_gpus = cfg.slurm.gpus_per_node
    executor = submitit.AutoExecutor(folder=cfg.checkpoint.log_folder)
    executor.update_parameters(
        name=cfg.name,
        slurm_comment=cfg.comment,
        slurm_partition=cfg.slurm.partition,
        slurm_constraint=cfg.slurm.gpu_type,
        timeout_min=cfg.slurm.timeout_min,
        nodes=cfg.slurm.nodes,
        cpus_per_task=cfg.slurm.cpus_per_task,
        tasks_per_node=nb_gpus,
        gpus_per_node=nb_gpus,
        mem_gb=cfg.slurm.mem_by_gpu * nb_gpus,
    )

    if training_mode:
        trainer = ResumableTrainer(config=cfg, nb_gpu=nb_gpus, nb_node=cfg.slurm.nodes)
        job = executor.submit(
            trainer,
        )
        print(f"Submitted {job.job_id}")
    else:
        raise NotImplementedError


class ResumableTrainer:
    """
    A training function that can be resumed from a checkpoint
    """

    def __init__(
        self, config: Config, nb_gpu: int, nb_node: int, load_checkpoint: bool = False
    ):
        self.config = config
        self.nb_gpu = nb_gpu
        self.nb_node = nb_node
        self.load_checkpoint = load_checkpoint

    def __call__(self):
        run_training(
            self.config,
            self.nb_gpu,
            self.nb_node,
            distributed_mode="ddp",
            load_checkpoint=self.load_checkpoint,
        )

    def checkpoint(self):
        trainer = ResumableTrainer(
            config=self.config,
            nb_gpu=self.nb_gpu,
            nb_node=self.nb_node,
            load_checkpoint=True,
        )
        return submitit.helpers.DelayedSubmission(
            trainer,
        )
