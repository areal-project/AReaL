# SPDX-License-Identifier: Apache-2.0

import importlib.util
import pathlib
import re
import sys
import time
import warnings
from collections.abc import Callable
from functools import partial

import ray
import ray.exceptions
from ray.runtime_env import RuntimeEnv
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

import areal.utils.logging as logging
from areal.api.alloc_mode import AllocationType, _AllocationMode
from areal.api.cli_args import (
    ClusterSpecConfig,
    InferenceEngineConfig,
    RecoverConfig,
    SGLangConfig,
    parse_cli_args,
    to_structured_cfg,
    vLLMConfig,
)
from areal.infra.platforms import current_platform, is_npu_available
from areal.infra.utils.exp_metadata import save_experiment_metadata
from areal.infra.utils.launcher import (
    BASE_ENVIRONS,
    JobException,
    JobState,
    get_scheduling_spec,
    get_thread_env_vars,
    validate_config_for_distributed_launcher,
    wait_llm_server_addrs,
)
from areal.infra.utils.ray import get_placement_group_master_ip_and_port
from areal.utils import name_resolve, names
from areal.utils.offload import get_tms_env_vars
from areal.utils.recover import check_if_recover

logger = logging.getLogger("RayLauncher")

RAY_WAIT_CHECK_TIME_INTERVAL = 5  # seconds
DEFAULT_MAIN_FUNC_NAME = "main"
RAY_LAUNCHER = None
RECOVER_TIME_INTERVAL = 10  # seconds


def run_func(file_path, function_name, *args, **kwargs):
    # Convert the file path to a module name
    module_name = file_path.replace("/", "_").replace(".", "_")

    # Load the module from file path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise FileNotFoundError(
            f"Cannot load module from file path '{file_path}'. "
            f"Please ensure the file exists and the path is correct."
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Get the function and execute it
    try:
        function = getattr(module, function_name)
    except AttributeError as e:
        raise ValueError(
            f"Function '{function_name}' not found in module '{module_name}'. "
            f"Please ensure the name of the main function in your entry point "
            f"is '{function_name}'."
        ) from e
    return function(*args, **kwargs)


class RayLauncher:
    def __init__(self, experiment_name: str, trial_name: str, fileroot: str):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.fileroot = fileroot

        # job_name to ray future
        self.jobs = {}
        self.placement_groups = {}

    @property
    def run_name(self):
        return f"{self.experiment_name}_{self.trial_name}"

    def submit(
        self,
        job_name: str,
        file_path: str,
        func_name: str,
        args: list[str],  # arguments to pass to the function
        gpus: int,
        cpus: int,
        mem: int,  # MB
        env_vars: dict | None = None,
        placement_group: PlacementGroup | None = None,
        bundle_index: int = -1,
        kwargs: (
            dict[str, str] | None
        ) = None,  # keyword arguments to pass to the function
    ):
        if kwargs is None:
            kwargs = {}
        runtime_env = RuntimeEnv(
            env_vars=env_vars or dict(),
        )
        scheduling_strategy = (
            PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=bundle_index,
                placement_group_capture_child_tasks=True,
            )
            if placement_group is not None
            else "DEFAULT"
        )
        if is_npu_available:
            future = ray.remote(
                num_cpus=cpus,
                resources={"NPU": gpus},
                memory=mem * 1024 * 1024,  # Convert MB to bytes
                runtime_env=runtime_env,
                scheduling_strategy=scheduling_strategy,
            )(run_func).remote(file_path, func_name, *args, **kwargs)
            self.jobs[job_name] = future
        else:
            future = ray.remote(
                num_cpus=cpus,
                num_gpus=gpus,
                memory=mem * 1024 * 1024,  # Convert MB to bytes
                runtime_env=runtime_env,
                scheduling_strategy=scheduling_strategy,
            )(run_func).remote(file_path, func_name, *args, **kwargs)
            self.jobs[job_name] = future
        return future

    def submit_array(
        self,
        job_name: str,
        file_path: str,
        func_name: str,
        count: int,
        nodes: int,
        list_args: list[list],
        gpus_per_task: int,
        cpus_per_task: int,
        mem_per_task: int,  # MB
        list_kwargs: list[dict] | None = None,
        env_vars: dict | None = None,
        env_hook: Callable[[PlacementGroup], list[dict]] | None = None,
    ):
        """Submit an array of jobs to Ray with ray placement groups.

        Note: Here we use `ray.remote` instead of `ray job submit` since `ray job submit`
        does not support placement groups, and can not specify which node to run the job on.
        Therefore we could not know the IP address of jobs for torch distributed initialization.
        """

        if count % nodes != 0:
            raise ValueError(
                f"Count {count} is not divisible by nodes {nodes}. "
                "Please ensure that count is a multiple of nodes."
            )
        if len(list_args) != count:
            raise ValueError(
                f"Length of list_args {len(list_args)} does not match count {count}."
            )
        if list_kwargs is not None:
            if len(list_kwargs) != count:
                raise ValueError(
                    f"Length of list_kwargs {len(list_kwargs)} does not match count {count}."
                )

        tasks_per_node = count // nodes
        gpus_per_node = gpus_per_task * tasks_per_node
        cpus_per_node = cpus_per_task * tasks_per_node
        mem_per_node = mem_per_task * tasks_per_node

        if job_name not in self.placement_groups:
            if is_npu_available:
                device_bundles = [
                    {
                        "CPU": cpus_per_node,
                        "NPU": gpus_per_node,
                        "memory": mem_per_node * 1024 * 1024,  # Convert MB to bytes
                    }
                ] * nodes
            else:
                device_bundles = [
                    {
                        "CPU": cpus_per_node,
                        "GPU": gpus_per_node,
                        "memory": mem_per_node * 1024 * 1024,  # Convert MB to bytes
                    }
                ] * nodes
            placement_group = ray.util.placement_group(
                bundles=device_bundles, strategy="PACK"
            )
            try:
                ray.get(placement_group.ready(), timeout=30)
            except ray.exceptions.GetTimeoutError as e:
                logger.error(
                    "Ray placement group timeout, please check if the resource requirement "
                    "for your experiment exceeds the available resources in the cluster. \n"
                    f"ray.nodes(): {ray.nodes()} \n"
                    f"Placement Group bundles: "
                    f"cpus_per_node={cpus_per_node}, gpus_per_node={gpus_per_node}, "
                    f"mem_per_node={mem_per_node}MB, nodes={nodes}"
                )
                raise e
            self.placement_groups[job_name] = placement_group
        else:
            # Reuse placement group in recover runs
            placement_group = self.placement_groups[job_name]

        if env_hook:
            extra_env_vars = env_hook(placement_group)

        futures = []
        for i in range(count):
            args = list_args[i]
            kwargs = list_kwargs[i] if list_kwargs is not None else {}

            # manage environment variables
            env_vars = env_vars or {}
            if current_platform.device_control_env_var in env_vars:
                logger.warning(
                    f"Setting {current_platform.device_control_env_var} before running ray jobs may result in unexpected behavior."
                )

            node_id = i // tasks_per_node

            if env_hook:
                _env_vars = env_vars.copy()
                _env_vars |= extra_env_vars[i]
            else:
                _env_vars = env_vars

            future = self.submit(
                job_name=f"{job_name}:{i}",
                file_path=file_path,
                func_name=func_name,
                args=args,
                gpus=gpus_per_task,
                cpus=cpus_per_task,
                mem=mem_per_task,
                env_vars=_env_vars,
                placement_group=placement_group,
                bundle_index=node_id,
                kwargs=kwargs,
            )
            futures.append(future)

        return futures

    def stop(self, job_name: str, force: bool = False):
        """Stop a job by name."""
        if job_name in self.jobs:
            future = self.jobs[job_name]
            try:
                ray.cancel(future, force=force)
            except Exception as e:
                logger.error(f"Failed to cancel job {job_name}: {e}")
                return
            self.jobs.pop(job_name, None)
            logger.info(f"Job {job_name} stopped.")
        else:
            logger.warning(f"Job {job_name} not found in running jobs.")

    def stop_all(self, force: bool = False, pattern: str | None = None):
        """Stop all jobs with pattern matched."""
        job_names = list(self.jobs.keys())
        if pattern:
            job_names = [
                job_name for job_name in job_names if re.search(pattern, job_name)
            ]
        for job_name in job_names:
            self.stop(job_name, force=force)
        if pattern:
            logger.info(f'Jobs matching the pattern "{pattern}" stopped')
        else:
            logger.info("All jobs stopped.")
        cur_job_names = self.jobs.keys()
        for job_name in job_names:
            if job_name in cur_job_names:
                self.jobs.pop(job_name)

    def wait(
        self, check_status=(JobState.FAILED,), remove_status=(JobState.COMPLETED,)
    ):
        """Check every RAY_WAIT_CHECK_TIME_INTERVAL seconds for the status of all jobs.
        If a ray job returns, its status changes to JobState.COMPLETED.
        If a ray job failed, its status changes to JobState.FAILED.
        If any job is in check_status, stop all jobs at once.
        If any job is in remove status, remove them from job list.
        Return if all jobs are removed from job list, or some job is in check status.
        """
        for status in list(check_status) + list(remove_status):
            assert status in [
                JobState.COMPLETED,
                JobState.FAILED,
            ], "In RayLauncher.wait, we only check completed or failed jobs."
        logger.info(f"Waiting for {len(self.jobs)} jobs.")
        while self.jobs:
            job_status = {}
            for job_name, future in list(self.jobs.items()):
                try:
                    r = ray.get(future, timeout=0.1)
                    logger.info(f"Job {job_name} completed with result: {r}")
                    job_status[job_name] = JobState.COMPLETED
                except ray.exceptions.RayTaskError as e:
                    logger.error(f"Job {job_name} failed with error: {e}.")
                    job_status[job_name] = JobState.FAILED
                except ray.exceptions.GetTimeoutError:
                    continue

            for job_name, status in job_status.items():
                if status in check_status:
                    logger.info(f"Job {job_name} is {status}, stopping all jobs.")
                    # raise exception to enter recover.
                    # should not changed to stop_all
                    raise JobException(
                        run_name=self.run_name,
                        worker_type=job_name.split(":")[0],
                        host="ray",
                        reason=status,
                    )
                if status in remove_status:
                    logger.info(f"Job {job_name} is {status}, removed.")
                    self.jobs.pop(job_name)

            time.sleep(RAY_WAIT_CHECK_TIME_INTERVAL)


def main():
    ray.init()
    config, _ = parse_cli_args(sys.argv[1:])
    ray_main(config, run_id=0)


def ray_main(config, run_id: int = 0):
    config.recover = to_structured_cfg(config.recover, RecoverConfig)
    config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
    is_recover_run = check_if_recover(config.recover, run_id)
    warnings.warn(
        "SPMD launchers use the deprecated _AllocationMode parser which will be removed. "
        "Bare dimension strings (e.g., 'd4t2') are NO LONGER ACCEPTED. "
        "All allocation strings must include an explicit backend prefix "
        "(e.g., 'fsdp:d4', 'sglang:d4t2'). "
        "Migrate to single-controller mode (scheduler.type=local) with per-engine 'backend' "
        "fields (e.g., actor.backend='fsdp:d4'). See docs/en/reference/alloc_mode.md.",
        FutureWarning,
        stacklevel=2,
    )
    validate_config_for_distributed_launcher(config)

    name_resolve.reconfigure(config.cluster.name_resolve)
    name_resolve.clear_subtree(
        names.trial_root(
            experiment_name=config.experiment_name, trial_name=config.trial_name
        )
    )

    n_nodes = config.cluster.n_nodes
    n_gpus_per_node = config.cluster.n_gpus_per_node

    # To reuse ray placement groups in recover runs.
    global RAY_LAUNCHER
    if RAY_LAUNCHER is None:
        assert run_id == 0
        launcher = RayLauncher(
            experiment_name=config.experiment_name,
            trial_name=config.trial_name,
            fileroot=config.cluster.fileroot,
        )
        RAY_LAUNCHER = launcher
    else:
        launcher = RAY_LAUNCHER

    allocation_mode = config.allocation_mode
    allocation_mode = _AllocationMode.from_str(allocation_mode)

    actor_spec = get_scheduling_spec(config.actor)

    if allocation_mode.gen_backend in ("sglang", "vllm"):
        config.rollout = to_structured_cfg(config.rollout, InferenceEngineConfig)
        rollout_spec = get_scheduling_spec(config.rollout)

    if not is_recover_run:
        metadata_file = save_experiment_metadata(
            config.cluster.fileroot,
            config.experiment_name,
            config.trial_name,
        )
        logger.info(f"Saved experiment metadata to {metadata_file}")

    sglang_addrs = []
    n_sglang_nodes = 0
    vllm_addrs = []
    n_vllm_nodes = 0
    if allocation_mode.gen_backend == "sglang":
        # Launcher should launch SGLang servers according to allocation mode.
        config.sglang = to_structured_cfg(config.sglang, SGLangConfig)
        n_sglang_servers = allocation_mode.gen.dp_size
        n_sglang_nodes = max(1, allocation_mode.gen.world_size // n_gpus_per_node)
        node_group_size = max(1, allocation_mode.gen_instance_size // n_gpus_per_node)
        n_servers_per_node = max(n_sglang_servers // n_sglang_nodes, 1)
        cross_nodes = allocation_mode.gen_instance_size > n_gpus_per_node

        base_seed = config.sglang.random_seed
        sglang_args_list = [
            [
                sys.argv[1:]
                + [f"sglang.random_seed={base_seed + i * n_servers_per_node}"]
            ]
            for i in range(n_sglang_nodes)
        ]
        sglang_entry_point = str(
            pathlib.Path(__file__).resolve().parent.joinpath("sglang_server.py")
        )

        def sglang_env_hook(
            n_tasks: int, task_group_size: int, placement_group: PlacementGroup
        ) -> list[dict]:
            master_addrs = []
            master_ports = []
            for i in range(0, n_tasks, task_group_size):
                host_ip, port = get_placement_group_master_ip_and_port(
                    placement_group, i
                )
                master_addrs.append(host_ip)
                master_ports.append(port)

            env_vars = []
            for i in range(n_tasks):
                env_vars.append(
                    dict(
                        AREAL_SGLANG_MULTI_NODE_RANK=str(i % task_group_size),
                        AREAL_SGLANG_MULTI_NODE_MASTER_ADDR=master_addrs[
                            i // task_group_size
                        ],
                        AREAL_SGLANG_MULTI_NODE_MASTER_PORT=str(
                            master_ports[i // task_group_size]
                        ),
                    )
                )

            return env_vars

        # Launch a task to start all sglang servers in one node.
        # Use full-node CPU allocation for llm_server since one Ray task manages
        # all GPUs on a node. Thread env vars are set per-node (not per-GPU).
        sglang_cpus_per_task = rollout_spec.cpu * n_gpus_per_node
        thread_env = get_thread_env_vars(
            cpus_per_task=sglang_cpus_per_task,
            existing_env_vars=rollout_spec.env_vars,
        )
        launcher.submit_array(
            job_name="llm_server",
            file_path=sglang_entry_point,
            func_name=DEFAULT_MAIN_FUNC_NAME,
            count=n_sglang_nodes,
            nodes=n_sglang_nodes,
            list_args=sglang_args_list,
            gpus_per_task=n_gpus_per_node,
            cpus_per_task=sglang_cpus_per_task,
            mem_per_task=rollout_spec.mem * 1024 * n_gpus_per_node,
            env_vars={**BASE_ENVIRONS, **thread_env, **rollout_spec.env_vars},
            env_hook=(
                partial(sglang_env_hook, n_sglang_nodes, node_group_size)
                if cross_nodes
                else None
            ),
        )
        # Get SGLang server addresses via name_resolve
        try:
            sglang_addrs = wait_llm_server_addrs(
                config.experiment_name,
                config.trial_name,
                n_sglang_servers,
            )
        except (TimeoutError, KeyboardInterrupt) as e:
            launcher.stop_all(
                force=False
            )  # force=False will send KeyboardInterrupt to sglang_server.main() to further clean all sglang-related processes
            raise e
    elif allocation_mode.gen_backend == "vllm":
        config.vllm = to_structured_cfg(config.vllm, vLLMConfig)
        # Launcher should launch vLLM servers according to allocation mode.
        vllm_tp_size = allocation_mode.gen.tp_size
        n_vllm_servers = allocation_mode.gen.dp_size
        n_vllm_nodes = allocation_mode.gen.world_size // n_gpus_per_node

        base_seed = config.vllm.seed
        vllm_args_list = [
            [sys.argv[1:] + [f"vllm.seed={base_seed + i}"]]
            for i in range(n_vllm_servers)
        ]
        vllm_entry_point = str(
            pathlib.Path(__file__).resolve().parent.joinpath("vllm_server.py")
        )
        vllm_cpus_per_task = rollout_spec.cpu * vllm_tp_size
        thread_env = get_thread_env_vars(
            cpus_per_task=vllm_cpus_per_task,
            existing_env_vars=rollout_spec.env_vars,
        )
        launcher.submit_array(
            job_name="llm_server",
            file_path=vllm_entry_point,
            func_name=DEFAULT_MAIN_FUNC_NAME,
            count=n_vllm_servers,
            nodes=n_vllm_nodes,
            list_args=vllm_args_list,
            gpus_per_task=vllm_tp_size,
            cpus_per_task=vllm_cpus_per_task,
            mem_per_task=rollout_spec.mem * 1024 * vllm_tp_size,
            env_vars={**BASE_ENVIRONS, **thread_env, **rollout_spec.env_vars},
        )
        # Get vllm server addresses via name_resolve
        try:
            vllm_addrs = wait_llm_server_addrs(
                config.experiment_name,
                config.trial_name,
                n_vllm_servers,
            )
        except (TimeoutError, KeyboardInterrupt) as e:
            launcher.stop_all(force=True)
            raise e

    if config.get("enable_offload", False):
        tms_env_vars = get_tms_env_vars()
    else:
        tms_env_vars = {}

    trainer_n_nodes = n_nodes - (
        n_sglang_nodes if allocation_mode.gen_backend == "sglang" else n_vllm_nodes
    )
    gpus_per_task = 1
    trainer_entry_point = sys.argv[1]
    n_trainer_processes = trainer_n_nodes * config.cluster.n_gpus_per_node
    trainer_args_list = [[sys.argv[2:]] for _ in range(n_trainer_processes)]
    if allocation_mode.type_ != AllocationType.LLM_SERVER_ONLY:
        llm_addrs = (
            sglang_addrs if allocation_mode.gen_backend == "sglang" else vllm_addrs
        )

        # In ray, we launch trainer in the granularity of processes (1 GPU per process)
        # We amend environment variable similar to torchrun to ensure correct initialization of
        # torch distributed.
        def torch_env_hook(n_tasks: int, placement_group: PlacementGroup) -> list[dict]:
            host_ip, port = get_placement_group_master_ip_and_port(placement_group)
            logger.info(
                f"Amend torch distributed env vars: MASTER_ADDR={host_ip}, PORT={port}"
            )
            env_vars = []
            for i in range(n_tasks):
                # NOTE: Here we only provide environment variables for torch distributed
                # initialization, and LOCAL_RANK for torch.device.
                # Other environment variables automatically set by torchrun are not set, and
                # they should be never accessed in trainer code.
                env_vars.append(
                    {
                        "RANK": str(i),
                        "WORLD_SIZE": str(n_tasks),
                        # Ray will automatically isolate CUDA_VISIBLE_DEVICES for each GPU
                        "LOCAL_RANK": "0",
                        "MASTER_ADDR": str(host_ip),
                        "MASTER_PORT": str(port),
                    }
                )
            return env_vars

        _env_vars = dict(
            AREAL_LLM_SERVER_ADDRS=",".join(llm_addrs),
        )
        if allocation_mode.gen_backend == "sglang":
            # Required by NCCL weight update group.
            _env_vars["NCCL_CUMEM_ENABLE"] = "0"
            _env_vars["NCCL_NVLS_ENABLE"] = "0"

        # Use per-GPU CPU count for thread env vars since Ray spawns individual
        # tasks per GPU, each inheriting these env vars. This differs from
        # llm_server which uses full-node allocation.
        thread_env = get_thread_env_vars(
            cpus_per_task=actor_spec.cpu,
            existing_env_vars=actor_spec.env_vars,
        )
        launcher.submit_array(
            job_name="trainer",
            file_path=trainer_entry_point,
            func_name=DEFAULT_MAIN_FUNC_NAME,
            count=trainer_n_nodes * config.cluster.n_gpus_per_node,
            nodes=trainer_n_nodes,
            list_args=trainer_args_list,
            gpus_per_task=gpus_per_task,
            cpus_per_task=actor_spec.cpu,
            mem_per_task=actor_spec.mem * 1024,
            env_vars={
                **BASE_ENVIRONS,
                **thread_env,
                **actor_spec.env_vars,
                **_env_vars,
                **tms_env_vars,
                "AREAL_SPMD_MODE": "1",
            },
            env_hook=partial(torch_env_hook, trainer_n_nodes * n_gpus_per_node),
        )

    try:
        launcher.wait(check_status=(JobState.COMPLETED, JobState.FAILED))
    except (KeyboardInterrupt, JobException, TimeoutError) as e:
        # The 'force' is passed to ray.cancel(future, force=force).
        # If force=False, a KeyboardInterrupt will be raised in sglang_server.main(),
        # allowing for a more thorough cleanup of all sglang-related processes.
        # This is particularly important when using sglang's dp_attention,
        # as it will leave residual processes that occupy GPU memory.
        launcher.stop_all(force=False, pattern="llm_server")
        # If force=True, the task is immediately killed, triggering the trainer to end the job.
        # Note: For trainer processes, we use force=True because the trainer doesn't
        # handle KeyboardInterrupt properly when force=False.
        launcher.stop_all(force=True, pattern="trainer")
        recover_states = [JobState.FAILED]
        if isinstance(e, JobException):
            recover_this = (
                e.reason in recover_states
                and run_id < config.recover.retries
                and config.recover.mode in ("on", "auto")
            )
            if recover_this:
                time.sleep(RECOVER_TIME_INTERVAL)
                ray_main(config, run_id=run_id + 1)
            else:
                raise e
        else:
            raise e


if __name__ == "__main__":
    # usage: python -m areal.infra.launcher.ray \
    #   <entry_point> --config <config_path> [<additional_args>]
    main()
