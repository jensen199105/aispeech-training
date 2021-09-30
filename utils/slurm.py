"""Make slurm environment easily accessable"""
import os


is_slurm_job = 'SLURM_JOB_ID' in os.environ

def _get_env_int(key, default=1):
    return int(os.environ.get(key, default))

world_size = _get_env_int('SLURM_ARRAY_TASK_COUNT')
job_id = _get_env_int('SLURM_ARRAY_JOB_ID', -1)
# Rank is 0-based while task-id is 1-based
rank = _get_env_int('SLURM_ARRAY_TASK_ID') - _get_env_int('SLURM_ARRAY_TASK_MIN')

task_step = _get_env_int('SLURM_ARRAY_TASK_STEP')
if task_step != 1:
    raise RuntimeError(f'Please submit the slurm job array with step=1.(Current step={task_step}')
