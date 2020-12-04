import os
import shutil

import hydra
import submitit
from omegaconf import DictConfig

from inv_cooking.config import Config, ExecutorType
from inv_cooking.trainer import run_training


def schedule_job(cfg: DictConfig) -> None:
    cfg = Config.parse_config(cfg)
    _copy_source_code_to_cwd()  # Because Hydra create a new running folder
    if cfg.executor == ExecutorType.local:
        run_training(cfg, gpus=2, nodes=1, distributed_mode="dp")
    elif cfg.executor == ExecutorType.slurm:
        _schedule_job_on_slurm_using_dp(cfg)


def _copy_source_code_to_cwd():
    """
    Copy the important folder to the path in which hydra will be running the code
    * this allows relative path to continue to work
    * this allows submitit to find the main module
    """
    original_path = hydra.utils.get_original_cwd()
    target_path = os.getcwd()
    folders_to_copy = ["inv_cooking", "data"]
    for folder in folders_to_copy:
        src_folder = os.path.join(original_path, folder)
        dst_folder = os.path.join(target_path, folder)
        shutil.copytree(src_folder, dst_folder)
    print(f"Running in folder {target_path}")


def _schedule_job_on_slurm_using_dp(cfg: Config):
    nb_gpus = cfg.slurm.gpus_per_node
    executor = submitit.AutoExecutor(folder=cfg.slurm.log_folder)
    executor.update_parameters(
        name="recipe1m_im2ingr_of1",  # TODO
        slurm_comment="",  # TODO
        slurm_partition=cfg.slurm.partition,
        slurm_constraint=cfg.slurm.gpu_type,
        timeout_min=cfg.slurm.timeout_min,
        nodes=cfg.slurm.nodes,
        cpus_per_task=cfg.slurm.cpus_per_task,
        tasks_per_node=1,
        gpus_per_node=nb_gpus,
        mem_gb=cfg.slurm.mem_by_gpu * nb_gpus,
    )
    job = executor.submit(run_training, cfg, nb_gpus, cfg.slurm.nodes, 'dp')
    print(f"Submitted {job.job_id}")


def _schedule_job_on_slurm_using_ddp(cfg: DictConfig):
    nb_gpus = cfg.slurm.gpus_per_node
    executor = submitit.AutoExecutor(folder=cfg.slurm.log_folder)
    executor.update_parameters(
        name="recipe1m_im2ingr_of1",  # TODO
        slurm_comment="",  # TODO
        slurm_partition=cfg.slurm.partition,
        slurm_constraint=cfg.slurm.gpu_type,
        timeout_min=cfg.slurm.timeout_min,
        nodes=cfg.slurm.nodes,
        cpus_per_task=cfg.slurm.cpus_per_task,
        tasks_per_node=nb_gpus,
        gpus_per_node=nb_gpus,
        mem_gb=cfg.slurm.mem_by_gpu * nb_gpus,
    )
    job = executor.submit(run_training, cfg, nb_gpus, cfg.slurm.nodes, 'ddp')
    print(f"Submitted {job.job_id}")
