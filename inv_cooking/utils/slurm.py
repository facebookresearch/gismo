# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
