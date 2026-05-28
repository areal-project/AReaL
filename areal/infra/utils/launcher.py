# SPDX-License-Identifier: Apache-2.0

import dataclasses
import enum
import getpass
import os
import pathlib
import sys
import time

from areal.api.alloc_mode import AllocationType, _AllocationMode
from areal.utils import logging, name_resolve, names
from areal.utils.fs import validate_shared_path

logger = logging.getLogger("LauncherUtils")

LOCAL_CACHE_DIR = os.getenv("AREAL_CACHE_DIR", f"/tmp/areal-{getpass.getuser()}")
PYTORCH_KERNEL_CACHE_PATH = (
    f"{LOCAL_CACHE_DIR}/.cache/{getpass.getuser()}/torch/kernels/"
)
VLLM_CACHE_ROOT = f"{LOCAL_CACHE_DIR}/.cache/{getpass.getuser()}/vllm/"
TRITON_CACHE_PATH = f"{LOCAL_CACHE_DIR}/.cache/{getpass.getuser()}/triton/"


def _find_repo_root():
    p = pathlib.Path(__file__).resolve()
    while p != p.parent:
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    return None  # installed environment, no repo checkout


_repo_root = _find_repo_root()
PYTHONPATH = os.pathsep.join(
    filter(
        None,
        [os.getenv("PYTHONPATH"), str(_repo_root) if _repo_root else None],
    )
)
os.makedirs(PYTORCH_KERNEL_CACHE_PATH, exist_ok=True)
os.makedirs(VLLM_CACHE_ROOT, exist_ok=True)
os.makedirs(TRITON_CACHE_PATH, exist_ok=True)
BASE_ENVIRONS = {
    "TOKENIZERS_PARALLELISM": "true",
    "PYTORCH_KERNEL_CACHE_PATH": PYTORCH_KERNEL_CACHE_PATH,
    "TRITON_CACHE_DIR": TRITON_CACHE_PATH,
    "VLLM_CACHE_ROOT": VLLM_CACHE_ROOT,
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
}

# Thread control environment variables for OpenMP/MKL/etc.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# Default thread count when cpus_per_task cannot be determined
THREAD_NUM_DEFAULT_WHEN_UNKNOWN = 8


def get_thread_env_vars(
    cpus_per_task: int | None = None,
    existing_env_vars: dict[str, str] | None = None,
) -> dict[str, str]:
    """Get thread control environment variables.

    Priority (from highest to lowest):
    1. existing_env_vars settings (user-configured via SchedulingSpec.env_vars)
    2. os.environ settings (user pre-set in script/shell)
    3. cpus_per_task dynamically computed value
    4. THREAD_NUM_DEFAULT_WHEN_UNKNOWN fallback value

    Args:
        cpus_per_task: Number of CPU cores allocated per task.
        existing_env_vars: Existing environment variable dict (e.g., SchedulingSpec.env_vars)

    Returns:
        Thread environment variable dict
    """
    if existing_env_vars is None:
        existing_env_vars = {}

    # Determine thread count
    if cpus_per_task is not None and cpus_per_task > 0:
        num_threads = cpus_per_task
    else:
        num_threads = THREAD_NUM_DEFAULT_WHEN_UNKNOWN

    thread_env = {}
    for var in THREAD_ENV_VARS:
        # Priority: existing_env_vars > os.environ > computed value
        if var in existing_env_vars:
            thread_env[var] = existing_env_vars[var]
        elif var in os.environ:
            thread_env[var] = os.environ[var]
        else:
            thread_env[var] = str(num_threads)

    # Log auto-set variables
    newly_set = [
        v for v in THREAD_ENV_VARS if v not in existing_env_vars and v not in os.environ
    ]
    if newly_set:
        logger.info(
            f"Auto-setting thread env vars to {num_threads}: {', '.join(newly_set)}"
        )

    return thread_env


def get_scheduling_spec(config_part):
    """Safely retrieve SchedulingSpec from config.

    Args:
        config_part: Config object (e.g., config.actor or config.rollout)

    Returns:
        SchedulingSpec instance (default if not configured)
    """
    from areal.api.cli_args import SchedulingSpec, to_structured_cfg

    try:
        return to_structured_cfg(config_part.scheduling_spec[0], SchedulingSpec)
    except AttributeError:
        return SchedulingSpec()


def get_env_vars(additional_env_vars: str | None = None) -> dict[str, str]:
    """Returns the environment variables for the cluster.

    This function dynamically captures the current Python path (sys.path) to ensure
    that spawned worker processes can import modules from the same locations as the
    parent process. This is essential for deserializing custom config classes defined
    outside the areal package (e.g., in examples/).
    """
    _additional_env_vars = (
        dict(item.split("=") for item in additional_env_vars.split(","))
        if additional_env_vars
        else dict()
    )

    # Dynamically build PYTHONPATH from current sys.path to ensure workers
    # can import custom modules in controller mode
    all_paths = []
    # The order of paths is important for module resolution.
    # 1. Current working directory.
    all_paths.append(os.getcwd())

    # 2. All paths from sys.path (excluding empty strings and zip files).
    all_paths.extend(p for p in sys.path if p and not p.endswith(".zip"))

    # 3. Original PYTHONPATH for completeness.
    if PYTHONPATH:
        all_paths.extend(p for p in PYTHONPATH.split(os.pathsep) if p)

    # Remove duplicates while preserving order
    pythonpath_parts = list(dict.fromkeys(all_paths))

    dynamic_pythonpath = os.pathsep.join(pythonpath_parts)

    return {**BASE_ENVIRONS, "PYTHONPATH": dynamic_pythonpath, **_additional_env_vars}


class JobState(enum.Enum):
    NOT_FOUND = 0
    PENDING = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4
    CANCELLED = 5

    def active(self):
        return self == self.PENDING or self == self.RUNNING


class JobException(Exception):
    def __init__(self, run_name, worker_type, host, reason: JobState):
        super().__init__(f"Job {run_name}:{worker_type} {reason} at node {host}")
        self.run_name = run_name
        self.worker_type = worker_type
        self.host = host
        self.reason = reason


@dataclasses.dataclass
class JobInfo:
    name: str
    state: JobState
    host: str | None = (
        None  # The host on which the job is/was running. None if the job had not run.
    )
    submit_time: str | None = None
    start_time: str | None = None
    slurm_id: int | None = None  # Slurm only. The Slurm id of the job.


def wait_llm_server_addrs(
    experiment_name: str,
    trial_name: str,
    n_rollout_servers: int = 1,
    timeout: int | None = 1200,
):
    # Get rollout nodes, find the hosts
    name = names.gen_servers(experiment_name, trial_name)
    start = time.perf_counter()
    while True:
        rollout_addrs = name_resolve.get_subtree(name)
        if len(rollout_addrs) >= n_rollout_servers:
            logger.info(
                f"Found {len(rollout_addrs)} rollout servers: {', '.join(rollout_addrs)}"
            )
            break

        time.sleep(1)
        if timeout is not None and time.perf_counter() - start > timeout:
            raise TimeoutError(
                f"Timeout waiting for rollout servers to be ready. "
                f"Expected {n_rollout_servers} servers, found {len(rollout_addrs)}."
            )
    return rollout_addrs


def validate_config_for_launcher(config):
    if not hasattr(config, "actor"):
        raise RuntimeError("`actor` field is required by all yaml configs.")
    allocation_mode = config.allocation_mode
    allocation_mode = _AllocationMode.from_str(allocation_mode)
    if allocation_mode.gen_backend in ["sglang", "vllm"]:
        if not hasattr(config, "rollout"):
            raise RuntimeError(
                "`rollout` field is required when `allocation_mode` "
                "includes inference backends like sglang/vllm. "
                "You should use a yaml config for RL rather than SFT, or "
                "remove the inference part from `allocation_mode`."
            )

    validate_shared_path(config.cluster.fileroot, "cluster.fileroot")
    if config.cluster.name_resolve.type == "nfs":
        validate_shared_path(
            config.cluster.name_resolve.nfs_record_root,
            "name_resolve.nfs_record_root",
        )


def validate_config_for_distributed_launcher(config):
    validate_config_for_launcher(config)
    n_nodes = config.cluster.n_nodes
    n_gpus_per_node = config.cluster.n_gpus_per_node
    allocation_mode = config.allocation_mode
    allocation_mode = _AllocationMode.from_str(allocation_mode)
    if allocation_mode.type_ == AllocationType.DECOUPLED_TRAIN:
        assert (
            allocation_mode.gen.world_size + allocation_mode.train.world_size
            == n_nodes * n_gpus_per_node
        ), (
            f"#GPUs required for allocation mode {allocation_mode.gen.world_size + allocation_mode.train.world_size} "
            f"is not equal to #GPUs in the config {n_nodes * n_gpus_per_node}."
        )
    if allocation_mode.gen_backend == "sglang":
        # Launcher should launch SGLang servers according to allocation mode.
        if allocation_mode.gen.pp_size > 1:
            raise NotImplementedError(
                "Pipeline generation in SGLang is not supported for now."
            )
    elif allocation_mode.gen_backend == "vllm":
        # Launcher should launch vLLM servers according to allocation mode.
        if allocation_mode.gen.pp_size > 1:
            raise NotImplementedError(
                "Pipeline generation in vLLM is not supported for now."
            )
        if allocation_mode.gen.tp_size > config.cluster.n_gpus_per_node:
            raise ValueError("Currently only support vLLM TP size <= #GPUs per node.")
