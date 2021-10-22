from enum import Enum, auto

import submitit
from omegaconf import OmegaConf

from inv_cooking.config import Config
from inv_cooking.scheduler.parsing import RawConfig
from inv_cooking.training import run_eval, run_training
from inv_cooking.training.visualise import run_visualisation
from inv_cooking.utils.hydra import copy_source_code_to_cwd
from inv_cooking.utils.slurm import get_job_id

# Allow to use the slurm_job_id in the configuration
OmegaConf.register_resolver("slurm_job_id", get_job_id)


class TrainingMode(Enum):
    TRAIN = auto()
    EVALUATE = auto()
    VISUALIZE = auto()


def schedule_jobs(cfg: RawConfig, training_mode: TrainingMode) -> None:
    # Because Hydra create a new running folder, we copy code and data to this
    # location (to have an isolated working copy). This should be done even in
    # case the process is spawn by Pytorch Lightning in the context of DDP
    # since each process will run in a separate hydra folder
    copy_source_code_to_cwd()

    # Run each of the jobs described by the configuration (there might be more)
    # than one in case of hyper-parameter search
    for config in RawConfig.to_config(cfg):
        if config.slurm.partition == "local":
            schedule_job_locally(config, training_mode)
        else:
            schedule_job_on_slurm(config, training_mode)


def schedule_job(config: Config, training_mode: TrainingMode):
    if config.slurm.partition == "local":
        schedule_job_locally(config, training_mode)
    else:
        schedule_job_on_slurm(config, training_mode)


def schedule_job_locally(cfg: Config, training_mode: TrainingMode):
    nb_gpu = cfg.slurm.gpus_per_node
    if training_mode == TrainingMode.TRAIN:
        run_training(
            cfg, gpus=nb_gpu, nodes=1, distributed_mode="ddp", load_checkpoint=False
        )
    elif training_mode == TrainingMode.EVALUATE:
        run_eval(cfg, gpus=nb_gpu, nodes=1, distributed_mode="ddp")
    elif training_mode == TrainingMode.VISUALIZE:
        run_visualisation(cfg)


def schedule_job_on_slurm(cfg: Config, training_mode: TrainingMode):
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

    if training_mode == TrainingMode.TRAIN:
        trainer = ResumableTrainer(config=cfg, nb_gpu=nb_gpus, nb_node=cfg.slurm.nodes)
        job = executor.submit(trainer)
        print(f"Submitted {job.job_id}")
    elif training_mode == TrainingMode.EVALUATE:
        evaluator = AsyncEvaluation(config=cfg, nb_gpu=nb_gpus, nb_node=cfg.slurm.nodes)
        job = executor.submit(evaluator)
        print(f"Submitted {job.job_id}")
    else:
        raise NotImplementedError


class AsyncEvaluation:
    """
    A way to run an evaluation on SLURM
    """

    def __init__(
        self, config: Config, nb_gpu: int, nb_node: int, load_checkpoint: bool = False
    ):
        self.config = config
        self.nb_gpu = nb_gpu
        self.nb_node = nb_node
        self.load_checkpoint = load_checkpoint

    def __call__(self):
        run_eval(self.config, gpus=self.nb_gpu, nodes=self.nb_node, distributed_mode="ddp")


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
