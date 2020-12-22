import os
import shutil
from typing import List

import hydra
import submitit

from inv_cooking.config import Config
from inv_cooking.trainer import run_training


def schedule_jobs(configurations: List[Config]) -> None:
    _copy_source_code_to_cwd()  # Because Hydra create a new running folder
    for config in configurations:
        if config.slurm.partition == "local":
            run_training(config, gpus=2, nodes=1, distributed_mode="dp")
        else:
            _schedule_job_on_slurm_using_dp(config)


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
    """
    Run the job whose configuration is given as parameter on SLURM
    """
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
        tasks_per_node=1,
        gpus_per_node=nb_gpus,
        mem_gb=cfg.slurm.mem_by_gpu * nb_gpus,
    )
    job = executor.submit(run_training, cfg, nb_gpus, cfg.slurm.nodes, 'dp')
    print(f"Submitted {job.job_id}")
