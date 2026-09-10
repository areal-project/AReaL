# SPDX-License-Identifier: Apache-2.0

import getpass
import importlib.util
import os
import pathlib
import re
import sys
import threading
import time
import traceback
import warnings
from collections.abc import Callable
from functools import partial

import ray
import ray.exceptions
from ray.actor import ActorHandle
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
from areal.infra.launcher.ray_bootstrap import (
    bootstrap_head,
    bootstrap_worker,
    detect_node_ip,
    detect_node_rank,
    stop_local_ray,
)
from areal.infra.platforms import current_platform
from areal.infra.utils.exp_metadata import save_experiment_metadata
from areal.infra.utils.launcher import (
    BASE_ENVIRONS,
    JobException,
    JobState,
    get_scheduling_spec,
    get_thread_env_vars,
    run_post_exit_hook,
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
LOG_WRITER_PING_TIMEOUT = 30  # seconds
LOG_WRITER_DRAIN_TIMEOUT = 30  # seconds


def _select_trainer_node_count(
    train_world_size: int,
    available_nodes: int,
    n_gpus_per_node: int,
) -> int:
    if train_world_size <= 0:
        raise ValueError(
            f"Training world size must be positive, got {train_world_size}"
        )
    if available_nodes <= 0:
        raise ValueError(
            "No nodes are available for trainer processes: "
            f"available_nodes={available_nodes}"
        )
    if n_gpus_per_node <= 0:
        raise ValueError(f"n_gpus_per_node must be positive, got {n_gpus_per_node}")

    available_gpus = available_nodes * n_gpus_per_node
    if train_world_size > available_gpus:
        raise ValueError(
            f"Training allocation requires {train_world_size} GPUs, but only "
            f"{available_gpus} trainer GPUs are available "
            f"({available_nodes} nodes x {n_gpus_per_node} GPUs)."
        )

    min_nodes = (train_world_size + n_gpus_per_node - 1) // n_gpus_per_node
    max_nodes = min(available_nodes, train_world_size)
    for node_count in range(min_nodes, max_nodes + 1):
        if train_world_size % node_count == 0:
            return node_count

    raise ValueError(
        f"Cannot evenly place {train_world_size} trainer processes on up to "
        f"{available_nodes} nodes with {n_gpus_per_node} GPUs per node. "
        "RayLauncher.submit_array requires an equal number of trainer "
        "processes on each trainer node."
    )


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


@ray.remote(num_cpus=0)
class _LogWriter:
    """Appends log data from all tasks of a job to a single file.

    Funneling every task's output through one actor keeps the merged log file
    consistent: concurrent O_APPEND writes from multiple nodes are not atomic
    on NFS, where `cluster.fileroot` usually lives.
    """

    def __init__(self, log_file: str):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # Unbuffered append so `tail -f` works and recover runs concatenate.
        self._log_f = open(log_file, "ab", buffering=0)

    def write(self, data: bytes):
        self._log_f.write(data)

    def drain(self):
        """Acknowledge that all previously queued writes reached the file."""
        self._log_f.flush()


def run_func_with_file_log(
    log_writer, task_label, file_path, function_name, *args, **kwargs
):
    """Run `run_func` while tee-ing the task's stdout/stderr to a job log file.

    Mimics the per-job log files written by the slurm launcher (sbatch
    ``--output``): the output of all tasks of a job (e.g. every trainer rank)
    is merged into one file, written by the shared `_LogWriter` actor.
    Redirection happens at the file-descriptor level so output from C
    extensions and subprocesses is captured as well. The original stdout is
    preserved via a pump thread, so Ray keeps streaming task output to the
    driver.
    """
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    # Line-buffer python-level stdio so bare `print` reaches the pipe (and
    # thus the log file and ray's driver streaming) in real time.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, OSError, ValueError):
        pass

    def _ship(data: bytes):
        try:
            log_writer.write.remote(data)
        except Exception:
            # Never let log shipping break the task or the console stream.
            pass

    launch_banner = (
        f"==== {task_label} launched at {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n"
    ).encode()
    drain_ref = []

    def _pump():
        console_ok = True
        try:
            # Submit the banner, every output chunk, and the drain acknowledgement
            # from this thread so Ray's per-caller actor FIFO ordering applies.
            _ship(launch_banner)
            while True:
                data = os.read(read_fd, 65536)
                if not data:
                    break
                _ship(data)
                if console_ok:
                    try:
                        os.write(saved_stdout, data)
                    except OSError:
                        # The driver-stream pipe broke. Keep draining the
                        # task's pipe (a full pipe would block the task's
                        # next print forever) and keep shipping to the log
                        # file; only the console copy is lost.
                        console_ok = False
        except OSError:
            pass
        finally:
            try:
                drain_ref.append(log_writer.drain.remote())
            except Exception:
                # Log actor failure must not change the task result.
                pass
            finally:
                os.close(read_fd)

    pump_thread = threading.Thread(target=_pump, daemon=True)
    pump_thread.start()
    try:
        return run_func(file_path, function_name, *args, **kwargs)
    except BaseException:
        # Print while stderr still points at the pipe so the traceback lands
        # in the log file before Ray reports the failure to the driver.
        traceback.print_exc()
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        # EOF reaches the pump thread once all pipe writers are gone; leaked
        # subprocesses may keep it open, so don't block task completion on it.
        pump_thread.join(timeout=5)
        if pump_thread.is_alive():
            logger.warning(
                f"Timed out draining the output pipe for task `{task_label}`; "
                "a child process may still hold it open, so trailing logs may be lost."
            )
        else:
            if drain_ref:
                try:
                    ray.get(
                        drain_ref[0],
                        timeout=LOG_WRITER_DRAIN_TIMEOUT,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to drain merged logs for task `{task_label}`: {e}"
                    )
            os.close(saved_stdout)


class RayLauncher:
    def __init__(self, experiment_name: str, trial_name: str, fileroot: str):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.fileroot = fileroot

        # job_name to ray future
        self.jobs = {}
        self.placement_groups = {}
        # base job name (e.g. "trainer") to its shared _LogWriter actor
        self._log_writers = {}

    @property
    def run_name(self):
        return f"{self.experiment_name}_{self.trial_name}"

    def log_path_of(self, job_name: str) -> str:
        log_path = f"{self.fileroot}/logs/{getpass.getuser()}/{self.experiment_name}/{self.trial_name}"
        os.makedirs(log_path, exist_ok=True)
        return os.path.join(log_path, f"{job_name}.log")

    def _log_writer_of(self, job_name: str):
        # All tasks of a job share one merged log file ("trainer:3" ->
        # trainer.log), matching the per-job logs of the slurm launcher.
        base_name = job_name.split(":")[0]
        writer = self._log_writers.get(base_name)
        if writer is not None and self._log_writer_alive(writer, base_name):
            return writer
        writer = _LogWriter.remote(self.log_path_of(base_name))
        # Surface actor construction errors (e.g. fileroot not mounted on
        # the actor's node) at submit time instead of silently dropping all
        # log output later. A timeout only means slow storage -- fine.
        try:
            ray.get(writer.write.remote(b""), timeout=LOG_WRITER_PING_TIMEOUT)
        except ray.exceptions.GetTimeoutError:
            pass
        self._log_writers[base_name] = writer
        return writer

    def _log_writer_alive(self, writer, base_name: str) -> bool:
        """Ping a cached log writer actor before reusing it.

        A recover run may inherit a handle whose actor died with its node
        (often the very failure that triggered the recover); reusing it
        would silently drop the whole recovered run's logs.
        """
        try:
            ray.get(writer.write.remote(b""), timeout=LOG_WRITER_PING_TIMEOUT)
            return True
        except ray.exceptions.GetTimeoutError:
            # Alive but busy (e.g. NFS stall backlog). Do NOT recreate:
            # the queued log data would be lost with the old actor.
            return True
        except Exception:
            logger.warning(
                f"Log writer actor of job `{base_name}` is gone (node "
                "failure?); recreating it. Log lines buffered in the old "
                "actor may be lost."
            )
            return False

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
        log_writer: ActorHandle | None = None,
    ):
        if kwargs is None:
            kwargs = {}
        if log_writer is None:
            log_writer = self._log_writer_of(job_name)
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
        if current_platform.ray_device_key == "NPU":
            future = ray.remote(
                num_cpus=cpus,
                resources={"NPU": gpus},
                memory=mem * 1024 * 1024,  # Convert MB to bytes
                runtime_env=runtime_env,
                scheduling_strategy=scheduling_strategy,
            )(run_func_with_file_log).remote(
                log_writer, job_name, file_path, func_name, *args, **kwargs
            )
            self.jobs[job_name] = future
        else:
            future = ray.remote(
                num_cpus=cpus,
                num_gpus=gpus,
                memory=mem * 1024 * 1024,  # Convert MB to bytes
                runtime_env=runtime_env,
                scheduling_strategy=scheduling_strategy,
            )(run_func_with_file_log).remote(
                log_writer, job_name, file_path, func_name, *args, **kwargs
            )
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
            if current_platform.ray_device_key == "NPU":
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

        # Resolve and probe the shared writer once for this job submission.
        # Calling through submit() for every rank would serialize startup behind
        # repeated health checks when the writer is busy on slow storage.
        log_writer = self._log_writer_of(job_name)
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
                log_writer=log_writer,
            )
            futures.append(future)

        logger.info(
            f"Submitted {count} Ray tasks for job `{job_name}`. To check the "
            f"merged output of all tasks, run\n\t`tail -f "
            f"{self.log_path_of(job_name)}`."
        )
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
        self,
        check_status=(JobState.FAILED,),
        remove_status=(JobState.COMPLETED,),
        complete_all_worker_types: tuple[str, ...] = (),
    ):
        """Check every RAY_WAIT_CHECK_TIME_INTERVAL seconds for the status of all jobs.
        If a ray job returns, its status changes to JobState.COMPLETED.
        If a ray job failed, its status changes to JobState.FAILED.
        If any job is in check_status, stop all jobs at once. Worker types in
        complete_all_worker_types only report COMPLETED after all of their tasks
        complete.
        If any job is in remove status, remove them from job list.
        Return if all jobs are removed from job list, or some job is in check status.
        """
        for status in list(check_status) + list(remove_status):
            assert status in [
                JobState.COMPLETED,
                JobState.FAILED,
            ], "In RayLauncher.wait, we only check completed or failed jobs."
        logger.info(f"Waiting for {len(self.jobs)} jobs.")
        completed_jobs = set()
        while self.jobs:
            job_status = {}
            for job_name, future in list(self.jobs.items()):
                if job_name in completed_jobs:
                    continue
                try:
                    r = ray.get(future, timeout=0.1)
                    logger.info(f"Job {job_name} completed with result: {r}")
                    job_status[job_name] = JobState.COMPLETED
                except ray.exceptions.RayTaskError as e:
                    logger.error(f"Job {job_name} failed with error: {e}.")
                    job_status[job_name] = JobState.FAILED
                except ray.exceptions.GetTimeoutError:
                    continue

            # A failure must win over a completion observed in the same poll.
            for job_name, status in job_status.items():
                if status == JobState.FAILED and status in check_status:
                    self._raise_job_status(job_name, status)

            # A server completing on its own is abnormal and must also win
            # over an aggregate trainer success in the same poll.
            for job_name, status in job_status.items():
                worker_type = job_name.split(":")[0]
                if (
                    status == JobState.COMPLETED
                    and status in check_status
                    and worker_type not in complete_all_worker_types
                ):
                    self._raise_job_status(job_name, status)

            for job_name, status in job_status.items():
                worker_type = job_name.split(":")[0]
                if (
                    status != JobState.COMPLETED
                    or status not in check_status
                    or worker_type not in complete_all_worker_types
                ):
                    continue
                completed_jobs.add(job_name)
                worker_jobs = [
                    name for name in self.jobs if name.split(":")[0] == worker_type
                ]
                if all(name in completed_jobs for name in worker_jobs):
                    self._raise_job_status(job_name, status)

            for job_name, status in job_status.items():
                if status in remove_status:
                    logger.info(f"Job {job_name} is {status}, removed.")
                    self.jobs.pop(job_name)

            time.sleep(RAY_WAIT_CHECK_TIME_INTERVAL)

    def _raise_job_status(self, job_name: str, status: JobState):
        logger.info(f"Job {job_name} is {status}, stopping all jobs.")
        raise JobException(
            run_name=self.run_name,
            worker_type=job_name.split(":")[0],
            host="ray",
            reason=status,
        )


def main():
    config, _ = parse_cli_args(sys.argv[1:])
    config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
    # `to_structured_cfg` returns a DictConfig, so merged CLI/YAML values do
    # not invoke ClusterSpecConfig.__post_init__. Validate the runtime value
    # explicitly before any local Ray processes are stopped or started.
    ClusterSpecConfig.validate_ray_port(config.cluster.ray_port)
    n_nodes = config.cluster.n_nodes
    n_gpus_per_node = config.cluster.n_gpus_per_node

    if os.environ.get("RAY_ADDRESS"):
        # A Ray cluster is explicitly designated (e.g. manually assembled via
        # `ray start` + RAY_ADDRESS=auto): connect and run the launcher.
        ray.init()
        ray_main(config, run_id=0)
        return

    node_rank = detect_node_rank()
    if n_nodes > 1 and node_rank is not None:
        # Multi-node gang-scheduled platform job (e.g. AIS/PAI), where this
        # same command runs on every node: assemble the Ray cluster first.
        # Rank 0 starts the head and proceeds into the launcher; other ranks
        # join as workers and block until the head shuts down.
        node_ip = detect_node_ip()
        logger.info(
            f"Ray bootstrap: node_rank={node_rank}, node_ip={node_ip}, "
            f"n_nodes={n_nodes}, n_gpus_per_node={n_gpus_per_node}"
        )
        # Clear stale Ray processes left over in reused containers.
        stop_local_ray()
        if node_rank == 0:
            try:
                bootstrap_head(
                    node_ip,
                    n_nodes,
                    n_gpus_per_node,
                    ray_port=config.cluster.ray_port,
                    dashboard_port=config.cluster.ray_dashboard_port,
                    wait_timeout=config.cluster.ray_bootstrap_timeout_seconds,
                    accelerator_resource=current_platform.ray_device_key,
                )
                ray_main(config, run_id=0)
            finally:
                logger.info("Head workload finished, stopping Ray")
                stop_local_ray()
        else:
            bootstrap_worker(
                node_ip,
                n_gpus_per_node,
                ray_port=config.cluster.ray_port,
                wait_timeout=config.cluster.ray_bootstrap_timeout_seconds,
                accelerator_resource=current_platform.ray_device_key,
            )
        return

    if n_nodes > 1:
        # Multi-node run outside a platform job: a pre-assembled Ray cluster
        # must already be running on this node.
        try:
            ray.init(address="auto")
        except ConnectionError as e:
            raise RuntimeError(
                f"cluster.n_nodes={n_nodes} > 1, but no running Ray cluster "
                "was found and no platform node-rank signal "
                "(AREAL_NODE_RANK/RANK/...) exists. Either pre-assemble a Ray "
                "cluster (`ray start --head` / `ray start --address=...`) "
                "before launching, or set AREAL_NODE_RANK and run this "
                "command on every node of the job."
            ) from e
        ray_main(config, run_id=0)
        return

    # Single node: connect to the local Ray instance, or start one.
    ray.init()
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
            run_post_exit_hook(config)
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
            try:
                launcher.stop_all(force=True)
            finally:
                run_post_exit_hook(config)
            raise e

    if config.get("enable_offload", False):
        tms_env_vars = get_tms_env_vars()
    else:
        tms_env_vars = {}

    available_trainer_nodes = n_nodes - (
        n_sglang_nodes if allocation_mode.gen_backend == "sglang" else n_vllm_nodes
    )
    gpus_per_task = 1
    trainer_entry_point = sys.argv[1]
    if allocation_mode.type_ != AllocationType.LLM_SERVER_ONLY:
        train_strategy = allocation_mode.train
        if train_strategy is None:
            raise ValueError(
                "Trainer launch requested, but allocation_mode has no training "
                f"allocation: {config.allocation_mode}"
            )
        n_trainer_processes = train_strategy.world_size
        trainer_n_nodes = _select_trainer_node_count(
            n_trainer_processes,
            available_trainer_nodes,
            config.cluster.n_gpus_per_node,
        )
        trainer_args_list = [[sys.argv[2:]] for _ in range(n_trainer_processes)]
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
        if (
            any(a.backend == "megatron" for a in allocation_mode.allocations)
            and config.actor.megatron.use_deterministic_algorithms
        ):
            # TransformerEngine snapshots this env var at import or attention
            # module construction depending on version; exporting it before
            # the trainer process starts is safe for all of them.
            _env_vars["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"

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
            count=n_trainer_processes,
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
            env_hook=partial(torch_env_hook, n_trainer_processes),
        )

    try:
        launcher.wait(
            check_status=(JobState.COMPLETED, JobState.FAILED),
            complete_all_worker_types=("trainer",),
        )
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
        run_post_exit_hook(config)
        if (
            isinstance(e, JobException)
            and e.reason == JobState.COMPLETED
            and e.worker_type == "trainer"
        ):
            # A trainer task finishing means the experiment is over: the
            # remaining jobs were torn down above, so exit cleanly instead
            # of surfacing a traceback and a non-zero exit code for a
            # successful run. (An llm_server completing on its own is NOT
            # normal and still falls through to the raise below.)
            logger.info("Trainer completed; experiment finished successfully.")
            return
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
    #
    # Works in three setups with the same command:
    # 1. Single node (cluster.n_nodes=1): starts a local Ray instance.
    # 2. Pre-assembled Ray cluster (`ray start` beforehand or RAY_ADDRESS
    #    set): connects to it; run this command once, e.g. on the head node.
    # 3. Multi-node platform job (AIS/PAI etc., node rank detectable from
    #    env): use this command as the job command on EVERY node; rank 0
    #    bootstraps the Ray head, other ranks join as workers. See
    #    areal/infra/launcher/ray_bootstrap.py for the env knobs.
    main()
