import submitit


def get_job_id() -> str:
    """
    Return the current job_id or "local" in case the code is not running on slurm
    """
    try:
        env = submitit.JobEnvironment()
        return env.job_id
    except RuntimeError:
        return "local"
