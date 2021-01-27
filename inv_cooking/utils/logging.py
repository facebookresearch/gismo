from typing import Optional

import submitit


def get_log_version() -> Optional[str]:
    """
    Return the version to be used by loggers such as tensorboard
    to identify separate runs of the same experiment:
    - return the SLURM JOB ID when running slurm
    - otherwise return None (and the logger will create an artificial version)
    """
    try:
        env = submitit.JobEnvironment()
        return env.job_id
    except RuntimeError:
        return None
