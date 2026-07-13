# SPDX-License-Identifier: Apache-2.0

import asyncio
import getpass
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import orjson
import requests

from areal.api import Job, Scheduler, Worker
from areal.api.cli_args import (
    BaseExperimentConfig,
    NameResolveConfig,
    SchedulingSpec,
    SchedulingStrategyType,
)
from areal.infra.platforms import current_platform
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.infra.scheduler.exceptions import (
    EngineCallError,
    EngineCreationError,
    EngineImportError,
    GPUAllocationError,
    PortAllocationError,
    RPCConnectionError,
    SchedulerError,
    WorkerConfigurationError,
    WorkerCreationError,
    WorkerFailedError,
    WorkerNotFoundError,
    WorkerTimeoutError,
)
from areal.infra.utils.concurrent import run_async_task
from areal.infra.utils.http import get_default_connector
from areal.infra.utils.launcher import (
    get_env_vars,
    get_thread_env_vars,
)
from areal.infra.utils.proc import kill_process_tree, run_with_streaming_logs
from areal.utils import logging, name_resolve, names
from areal.utils.fs import validate_shared_path
from areal.utils.network import (
    find_free_ports,
    format_hostport,
    gethostip,
)
from areal.utils.offload import get_tms_env_vars

logger = logging.getLogger("LocalScheduler")

_MAX_STARTUP_PORT_CONFLICT_RETRIES = 3


@dataclass
class WorkerInfo:
    worker: Worker
    process: subprocess.Popen | None  # None for forked workers (managed by parent)
    role: str
    gpu_devices: list[int]
    created_at: float
    log_file: str
    env_vars: dict[str, str] = field(default_factory=dict)


def _get_device_count_safely() -> int | None:
    """
    Safely get device count without initializing CUDA context.
    """
    gpu_types = ["nvidia", "davinci"]
    try:
        if os.path.exists("/dev"):
            for gpu_type in gpu_types:
                devices = [
                    f
                    for f in os.listdir("/dev")
                    if f.startswith(gpu_type) and f[len(gpu_type) :].isdigit()
                ]
                if devices:
                    return len(devices)
    except (OSError, ValueError):
        return None


class LocalScheduler(Scheduler):
    """Local scheduler that manages worker subprocesses on a single GPU node.

    This scheduler spawns worker processes running RPC servers and manages their lifecycle.
    It supports different worker types through a unified interface with dynamic port allocation,
    round-robin GPU assignment, process health monitoring, and graceful cleanup.
    """

    def __init__(
        self,
        gpu_devices: list[int] | None = None,
        log_dir: str | None = None,
        startup_timeout: float = 30.0,
        health_check_interval: float = 1.0,
        *,
        experiment_name: str | None = None,
        trial_name: str | None = None,
        fileroot: str | None = None,
        enable_tms_offload: bool | None = None,
        name_resolve_type: str = "nfs",
        nfs_record_root: str = "/tmp/areal/name_resolve",
        etcd3_addr: str = "localhost:2379",
        exp_config: BaseExperimentConfig | None = None,
    ):
        self.gpu_devices = gpu_devices or self._detect_gpus()

        # Resolve experiment/trial names (exp_config overwrites direct params)
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.fileroot = fileroot
        self.enable_tms_offload = bool(enable_tms_offload)
        if exp_config is not None:
            self.experiment_name = exp_config.experiment_name
            self.trial_name = exp_config.trial_name
            self.fileroot = exp_config.cluster.fileroot
            self.enable_tms_offload = exp_config.enable_offload

        # name_resolve config (exp_config overwrites direct params)
        self.name_resolve_config = NameResolveConfig(
            type=name_resolve_type,
            nfs_record_root=nfs_record_root,
            etcd3_addr=etcd3_addr,
        )
        if exp_config is not None:
            self.name_resolve_config = exp_config.cluster.name_resolve

        if self.fileroot:
            validate_shared_path(self.fileroot, "cluster.fileroot")
        if self.name_resolve_config.type == "nfs":
            validate_shared_path(
                self.name_resolve_config.nfs_record_root,
                "name_resolve.nfs_record_root",
            )

        # Reconfigure name_resolve and clear old entries
        if self.experiment_name and self.trial_name:
            name_resolve.reconfigure(self.name_resolve_config)
            name_resolve.clear_subtree(
                names.trial_root(self.experiment_name, self.trial_name)
            )

        if log_dir is not None:
            self.log_dir = Path(log_dir)
        else:
            assert self.experiment_name is not None
            assert self.trial_name is not None
            assert self.fileroot is not None
            self.log_dir = (
                Path(self.fileroot)
                / "logs"
                / getpass.getuser()
                / self.experiment_name
                / self.trial_name
            )
        self.exp_config = exp_config

        self.startup_timeout = startup_timeout
        self.health_check_interval = health_check_interval

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._workers: dict[str, list[WorkerInfo]] = {}
        self._gpu_counter = 0
        self._allocated_ports = set()

        # Colocation tracking: colocated roles reuse workers from target role
        self._colocated_roles: dict[str, str] = {}  # colocated_role -> target_role

        logger.info(
            f"LocalScheduler initialized with GPU devices: {self.gpu_devices}, "
            f"log directory: {self.log_dir}"
        )

    @property
    def n_gpus_per_node(self) -> int:
        return len(self.gpu_devices)

    def _detect_gpus(self) -> list[int]:
        cuda_visible = os.environ.get(current_platform.device_control_env_var)
        if current_platform.device_control_env_var and cuda_visible:
            try:
                return [int(x) for x in cuda_visible.split(",")]
            except ValueError:
                logger.warning(
                    f"Invalid {current_platform.device_control_env_var}: {cuda_visible}, using default [0]"
                )
                return [0]
        cnt = _get_device_count_safely()
        if cnt is None:
            return [0]
        return list(range(cnt))

    def _allocate_gpus(self, num_gpus: int) -> list[int]:
        if num_gpus == 0:
            return []

        if num_gpus > len(self.gpu_devices):
            raise GPUAllocationError(
                f"Requested {num_gpus} GPUs but only {len(self.gpu_devices)} available"
            )

        allocated = []
        for _ in range(num_gpus):
            gpu_id = self.gpu_devices[self._gpu_counter % len(self.gpu_devices)]
            allocated.append(gpu_id)
            self._gpu_counter += 1

        return allocated

    def _get_colocated_gpus(self, target_role: str, worker_idx: int) -> list[int]:
        if target_role not in self._workers:
            raise WorkerNotFoundError(
                f"Cannot colocate with role '{target_role}' - role not found"
            )

        target_workers = self._workers[target_role]
        if worker_idx >= len(target_workers):
            raise ValueError(
                f"Cannot colocate with {target_role}/{worker_idx} - only {len(target_workers)} workers exist"
            )

        return target_workers[worker_idx].gpu_devices

    def _allocate_ports(self, count: int) -> list[int]:
        # Workers are on the same node, so we can directly allocate ports in the scheduler
        try:
            ports = find_free_ports(count, exclude_ports=set(self._allocated_ports))
            self._allocated_ports.update(ports)
            return ports
        except ValueError as e:
            raise PortAllocationError(str(e)) from e

    def _release_ports(self, ports: list[int]) -> None:
        for port in ports:
            self._allocated_ports.discard(port)

    @staticmethod
    def _is_port_conflict_error(details: str) -> bool:
        lowered = details.lower()
        return (
            "address already in use" in lowered
            or "errno 98" in lowered
            or "errno 48" in lowered
            or ("port " in lowered and "is in use by another program" in lowered)
        )

    def _prepare_worker_specs(
        self, role: str, num_workers: int, schedulings: list[SchedulingSpec] | None
    ) -> list[SchedulingSpec]:
        if not schedulings:
            return [
                SchedulingSpec(
                    cpu=1,
                    mem=1024,
                    gpu=0,
                    port_count=2,
                    cmd="python -m areal.infra.rpc.rpc_server",
                )
            ] * num_workers

        if len(schedulings) == 1:
            return [schedulings[0]] * num_workers

        if len(schedulings) == num_workers:
            return schedulings

        raise WorkerCreationError(
            role,
            "Invalid configuration",
            f"schedulings length ({len(schedulings)}) must be 1 or equal to replicas ({num_workers})",
        )

    @staticmethod
    async def _wait_for_fork_ready(
        session: aiohttp.ClientSession,
        host: str,
        port: int,
        timeout: float = 60,
    ) -> bool:
        url = f"http://{format_hostport(host, port)}/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        return True
            except (TimeoutError, aiohttp.ClientError):
                pass
            await asyncio.sleep(0.5)
        return False

    async def _fork_single_worker(
        self,
        session: aiohttp.ClientSession,
        role: str,
        idx: int,
        target_wi: WorkerInfo,
        target_role: str,
        command: str | None = None,
    ) -> WorkerInfo:
        """Fork a single worker asynchronously.

        Parameters
        ----------
        command : str, optional
            Custom module path to run instead of the default rpc_server.
        """
        worker_id = f"{role}/{idx}"
        guard_url = f"http://{format_hostport(target_wi.worker.ip, int(target_wi.worker.worker_ports[0]))}"

        try:
            # 1. Allocate a port on the target guard
            async with session.post(
                f"{guard_url}/alloc_ports",
                json={"count": 1},
            ) as alloc_resp:
                if alloc_resp.status != 200:
                    error_text = await alloc_resp.text()
                    raise WorkerCreationError(
                        role,
                        f"Port allocation failed for worker {idx}",
                        f"HTTP {alloc_resp.status}: {error_text}",
                    )
                alloc_data = await alloc_resp.json()
                forked_host = alloc_data["host"]
                forked_port = alloc_data["ports"][0]

            # 2. Build the full raw command
            module_path = command or "areal.infra.rpc.rpc_server"
            raw_cmd = [
                sys.executable,
                "-m",
                module_path,
                "--host",
                "0.0.0.0",
                "--port",
                str(forked_port),
                "--experiment-name",
                str(self.experiment_name),
                "--trial-name",
                str(self.trial_name),
                "--role",
                role,
                "--worker-index",
                str(idx),
            ]
            if self.name_resolve_config.type:
                raw_cmd.extend(["--name-resolve-type", self.name_resolve_config.type])
            if self.name_resolve_config.nfs_record_root:
                raw_cmd.extend(
                    ["--nfs-record-root", self.name_resolve_config.nfs_record_root]
                )
            if self.name_resolve_config.etcd3_addr:
                raw_cmd.extend(["--etcd3-addr", self.name_resolve_config.etcd3_addr])
            if self.fileroot:
                raw_cmd.extend(["--fileroot", str(self.fileroot)])

            # 3. Fork via raw_cmd
            payload = {
                "role": role,
                "worker_index": idx,
                "raw_cmd": raw_cmd,
            }
            async with session.post(
                f"{guard_url}/fork",
                json=payload,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise WorkerCreationError(
                        role,
                        f"Fork failed for worker {idx}",
                        f"HTTP {response.status}: {error_text}",
                    )

                result = await response.json()

                if result.get("status") != "success":
                    raise WorkerCreationError(
                        role,
                        f"Fork failed for worker {idx}",
                        result.get("error", "Unknown error"),
                    )

                forked_pid = result.get("pid")

            # 4. Wait for the forked worker to become ready
            if not await self._wait_for_fork_ready(session, forked_host, forked_port):
                # Clean up the forked worker on the guard
                try:
                    async with session.post(
                        f"{guard_url}/kill_forked_worker",
                        json={"role": role, "worker_index": idx},
                    ):
                        pass
                except Exception:
                    pass
                raise WorkerCreationError(
                    role,
                    f"Forked worker {idx} failed to become ready",
                    f"Readiness timeout at {forked_host}:{forked_port}",
                )

            logger.info(
                f"Forked worker {worker_id} created at {forked_host}:{forked_port} "
                f"(pid={forked_pid}) from {target_role}/{idx}"
            )

        except aiohttp.ClientError as e:
            raise WorkerCreationError(
                role,
                f"Failed to fork worker {idx} from {target_role}/{idx}",
                str(e),
            ) from e

        worker = Worker(
            id=worker_id,
            ip=forked_host,
            worker_ports=[str(forked_port)],
            engine_ports=[],
        )
        port_cnt = len(self._workers[target_role][0].worker.worker_ports)
        if port_cnt > 1:
            worker.worker_ports += self._allocate_ports(port_cnt - 1)

        return WorkerInfo(
            worker=worker,
            process=None,  # Managed by parent worker
            role=role,
            gpu_devices=target_wi.gpu_devices,  # Inherited from target
            created_at=time.time(),
            log_file=str(self.log_dir / f"{role}.log"),
            env_vars=target_wi.env_vars.copy(),  # Inherited from target
        )

    async def _kill_forked_worker(
        self,
        session: aiohttp.ClientSession,
        role: str,
        idx: int,
        target_wi: WorkerInfo,
    ) -> None:
        """Kill a single forked worker via its parent's RPC server."""
        target_url = f"http://{format_hostport(target_wi.worker.ip, int(target_wi.worker.worker_ports[0]))}/kill_forked_worker"

        try:
            payload = {"role": role, "worker_index": idx}
            async with session.post(
                target_url,
                json=payload,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.warning(
                        f"Failed to kill forked worker {role}/{idx}: "
                        f"HTTP {response.status}: {error_text}"
                    )
                else:
                    result = await response.json()
                    logger.info(
                        result.get("message", f"Killed forked worker {role}/{idx}")
                    )
        except Exception as e:
            logger.warning(f"Exception killing forked worker {role}/{idx}: {e}")

    async def _cleanup_forked_workers_async(
        self,
        role: str,
        target_role: str,
        workers: list[WorkerInfo],
    ) -> None:
        """Cleanup forked workers by calling kill endpoint on parent workers."""
        target_workers = self._workers.get(target_role, [])
        if not target_workers:
            logger.warning(
                f"Cannot cleanup forked workers: target role '{target_role}' not found"
            )
            return

        timeout = aiohttp.ClientTimeout(total=30.0)
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=get_default_connector(),
        ) as session:
            tasks = []
            for worker_info in workers:
                worker_index = int(worker_info.worker.id.split("/")[-1])
                if worker_index < len(target_workers):
                    tasks.append(
                        self._kill_forked_worker(
                            session, role, worker_index, target_workers[worker_index]
                        )
                    )
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _create_forked_workers_async(
        self,
        role: str,
        target_role: str,
        target_workers: list[WorkerInfo],
        command: str | None = None,
    ) -> list[str]:
        """Create forked workers concurrently using async requests.

        Parameters
        ----------
        command : str, optional
            Custom module path to run instead of the default rpc_server.
            If specified, the forked processes run this module.
        """
        timeout = aiohttp.ClientTimeout(total=120.0)
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=get_default_connector(),
        ) as session:
            # Launch all fork requests concurrently with exception handling
            tasks = [
                self._fork_single_worker(
                    session, role, idx, target_wi, target_role, command
                )
                for idx, target_wi in enumerate(target_workers)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successful workers from failures
        workers = []
        failed_indices = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                failed_indices.append(idx)
                logger.error(
                    f"Failed to fork worker {role}/{idx} from {target_role}/{idx}: {result}"
                )
            else:
                workers.append(result)

        # If any fork failed, cleanup successful workers and raise
        if failed_indices:
            if workers:
                logger.warning(
                    f"Cleaning up {len(workers)} successfully forked workers due to partial failure"
                )
                # Kill the forked processes via parent RPC servers
                try:
                    await self._cleanup_forked_workers_async(role, target_role, workers)
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup forked workers: {cleanup_error}")

            raise WorkerCreationError(
                role,
                f"Failed to fork {len(failed_indices)} out of {len(target_workers)} workers",
                f"Failed indices: {failed_indices}",
            )

        self._workers[role] = list(workers)
        self._colocated_roles[role] = target_role
        worker_ids = [w.worker.id for w in workers]

        logger.info(
            f"Role '{role}' forked from '{target_role}': "
            f"created {len(workers)} new worker processes"
        )

        # Configure forked workers if exp_config is available
        if self.exp_config is not None:
            for worker_rank, worker_info in enumerate(workers):
                self._configure_worker(worker_info, worker_rank)

        return worker_ids

    def fork_workers(
        self,
        role: str,
        target_role: str,
        command: str | None = None,
    ) -> list[str]:
        """Fork new worker processes from existing workers.

        Creates new worker processes by forking from existing workers of the target role.
        The forked workers are colocated on the same nodes as their target workers.

        Parameters
        ----------
        role : str
            Role name for the new forked workers (e.g., "proxy")
        target_role : str
            Role of existing workers to fork from (e.g., "rollout")
        command : str, optional
            Custom module path to run instead of the default rpc_server.
            If specified, the forked process runs this module.

        Returns
        -------
        list[str]
            List of worker IDs created (e.g., ["proxy/0", "proxy/1"])
        """
        if target_role not in self._workers:
            raise WorkerNotFoundError(f"Target role '{target_role}' not found for fork")
        target_workers = self._workers[target_role]

        try:
            return run_async_task(
                self._create_forked_workers_async,
                role,
                target_role,
                target_workers,
                command,
            )
        except Exception:
            # Cleanup on failure
            if role in self._workers:
                del self._workers[role]
            if role in self._colocated_roles:
                del self._colocated_roles[role]
            raise

    def create_workers(self, job: Job, *args, **kwargs) -> list[str]:
        """Create worker subprocesses.

        Parameters
        ----------
        job : Job
            Job configuration with role, replicas, tasks, and scheduling strategy
        *args
            Additional arguments passed to worker command
        **kwargs
            Additional keyword arguments

        Returns
        -------
        list[str]
            List of worker IDs created (e.g., ["rollout/0", "rollout/1"])

        Raises
        ------
        WorkerCreationError
            If worker creation fails
        GPUAllocationError
            If GPU allocation fails
        PortAllocationError
            If port allocation fails
        """
        role = job.role
        if role in self._workers:
            raise WorkerCreationError(
                role,
                "Worker group already exists",
                f"Use delete_workers('{role}') first to remove existing workers",
            )

        num_workers = job.replicas
        if num_workers == 0:
            raise WorkerCreationError(
                role, "Invalid configuration", "replicas must be greater than 0"
            )

        schedulings = self._prepare_worker_specs(role, num_workers, job.tasks)

        strategy = job.scheduling_strategy
        strategy_type = SchedulingStrategyType(strategy.type)
        colocate_role = strategy.target
        logger.info(
            f"Creating {num_workers} workers for role '{role}' "
            f"(strategy: {strategy_type}, colocate_with: {colocate_role})"
        )

        # Handle colocation: reuse existing workers from target role
        if strategy_type == SchedulingStrategyType.colocation:
            if not colocate_role:
                raise WorkerCreationError(
                    role,
                    "Invalid strategy",
                    "Colocation strategy requires target role to be specified",
                )
            if colocate_role not in self._workers:
                raise WorkerNotFoundError(
                    f"Cannot colocate with role '{colocate_role}' - role not found"
                )

            target_workers = self._workers[colocate_role]
            if num_workers != len(target_workers):
                raise WorkerCreationError(
                    role,
                    "Replica count mismatch",
                    f"Colocated role must have same replica count as target "
                    f"({num_workers} != {len(target_workers)})",
                )

            # Check if fork mode is enabled
            if strategy.fork:
                # Fork mode: spawn new processes on same GPUs via /fork endpoint
                worker_ids = self.fork_workers(role, colocate_role)
            else:
                # Reuse existing workers - no new processes spawned
                worker_ids = [w.worker.id for w in target_workers]
            self._colocated_roles[role] = colocate_role

            logger.info(
                f"Role '{role}' colocated with '{colocate_role}': "
                f"reusing workers {worker_ids}"
            )
            return worker_ids

        if strategy_type != SchedulingStrategyType.separation:
            raise ValueError(f"Unknown scheduling strategy type: {strategy_type}")
        # Non-colocated: spawn new worker processes
        workers = []
        worker_ids = []
        try:
            for idx in range(num_workers):
                worker_id = f"{role}/{idx}"
                scheduling = schedulings[idx]

                try:
                    gpu_devices = self._allocate_gpus(scheduling.gpu)
                    logger.debug(f"Worker {worker_id} allocated GPUs {gpu_devices}")
                except (
                    GPUAllocationError,
                    PortAllocationError,
                    WorkerNotFoundError,
                    ValueError,
                ) as e:
                    self._cleanup_workers(workers)
                    raise WorkerCreationError(
                        role, f"Resource allocation failed for worker {idx}", str(e)
                    ) from e

                env = get_env_vars(
                    ",".join([f"{k}={v}" for k, v in scheduling.env_vars.items()]),
                )
                if current_platform.device_control_env_var:
                    env[current_platform.device_control_env_var] = ",".join(
                        map(str, gpu_devices)
                    )

                thread_env = get_thread_env_vars(
                    cpus_per_task=scheduling.cpu,
                    existing_env_vars=scheduling.env_vars,
                )
                env.update(thread_env)

                if self.enable_tms_offload:
                    env.update(get_tms_env_vars())

                if scheduling.env_vars:
                    env.update(scheduling.env_vars)

                log_file = self.log_dir / f"{role}.log"
                merged_log = self.log_dir / "merged.log"

                if not scheduling.cmd:
                    self._cleanup_workers(workers)
                    raise WorkerCreationError(
                        role,
                        f"SchedulingSpec.cmd is required but not set for worker {worker_id}",
                        "Specify 'python -m areal.infra.rpc.rpc_server' in your config.",
                    )

                if "--port" in scheduling.cmd:
                    raise WorkerCreationError(
                        role,
                        "Custom command should not include --port argument",
                        "The scheduler automatically allocates and provides the port.",
                    )
                cmd_prefix = shlex.split(scheduling.cmd)
                cmd_suffix = [
                    "--experiment-name",
                    str(self.experiment_name),
                    "--trial-name",
                    str(self.trial_name),
                    "--role",
                    role,
                    "--worker-index",
                    str(idx),
                    "--name-resolve-type",
                    self.name_resolve_config.type,
                    "--nfs-record-root",
                    self.name_resolve_config.nfs_record_root,
                    "--etcd3-addr",
                    self.name_resolve_config.etcd3_addr,
                    "--fileroot",
                    str(self.fileroot),
                ]

                process = None
                ports = []
                for attempt in range(1, _MAX_STARTUP_PORT_CONFLICT_RETRIES + 1):
                    try:
                        ports = self._allocate_ports(scheduling.port_count)
                    except (
                        PortAllocationError,
                        WorkerNotFoundError,
                        ValueError,
                    ) as e:
                        self._cleanup_workers(workers)
                        raise WorkerCreationError(
                            role,
                            f"Resource allocation failed for worker {idx}",
                            str(e),
                        ) from e

                    cmd = [*cmd_prefix, "--port", str(ports[0]), *cmd_suffix]

                    logger.info(
                        "Starting worker %s (attempt %s/%s): %s",
                        worker_id,
                        attempt,
                        _MAX_STARTUP_PORT_CONFLICT_RETRIES,
                        " ".join(cmd),
                    )
                    if cmd[0].startswith("python"):
                        cmd[0] = sys.executable

                    try:
                        process = run_with_streaming_logs(
                            cmd,
                            log_file,
                            merged_log,
                            role,
                            env_vars_in_cmd=env,
                        )
                    except Exception as e:
                        self._release_ports(ports)
                        self._cleanup_workers(workers)
                        raise WorkerCreationError(
                            role,
                            f"Failed to spawn subprocess for worker {idx}",
                            str(e),
                        ) from e

                    time.sleep(0.1)
                    if process.poll() is None:
                        break

                    stderr = self._read_log_tail(str(log_file))
                    self._release_ports(ports)

                    if self._is_port_conflict_error(stderr):
                        logger.warning(
                            "Worker %s hit port conflict on startup attempt %s/%s; retrying with new ports.",
                            worker_id,
                            attempt,
                            _MAX_STARTUP_PORT_CONFLICT_RETRIES,
                        )
                        if attempt < _MAX_STARTUP_PORT_CONFLICT_RETRIES:
                            time.sleep(0.1 * attempt)
                            continue

                    self._cleanup_workers(workers)
                    raise WorkerCreationError(
                        role,
                        f"Worker {worker_id} exited immediately with code {process.returncode}",
                        stderr,
                    )
                else:
                    self._cleanup_workers(workers)
                    raise WorkerCreationError(
                        role,
                        f"Worker {worker_id} failed to start after {_MAX_STARTUP_PORT_CONFLICT_RETRIES} attempts",
                        self._read_log_tail(str(log_file)),
                    )

                worker = Worker(
                    id=worker_id,
                    ip=gethostip(),
                    worker_ports=[str(p) for p in ports],
                    engine_ports=[],
                )

                worker_info = WorkerInfo(
                    worker=worker,
                    process=process,
                    role=role,
                    gpu_devices=gpu_devices,
                    created_at=time.time(),
                    log_file=str(log_file),
                    env_vars=env,
                )

                workers.append(worker_info)
                worker_ids.append(worker_id)
                logger.info(
                    f"Worker {worker_id} started (PID: {process.pid}, "
                    f"GPUs: {gpu_devices}, ports: {ports})"
                )

            self._workers[role] = workers

            logger.info(
                f"Successfully created {len(workers)} workers for role '{role}'"
            )

        except Exception as e:
            self._cleanup_workers(workers)
            if isinstance(e, SchedulerError):
                raise
            raise WorkerCreationError(role, "Unexpected error", str(e)) from e

        if self.exp_config is not None:
            for worker_rank, worker_info in enumerate(workers):
                self._configure_worker(worker_info, worker_rank)

        return worker_ids

    def get_workers(self, role: str, timeout: float | None = None) -> list[Worker]:
        """Get workers and wait for them to be ready.

        Parameters
        ----------
        role : str
            Worker role name
        timeout : float, optional
            Maximum time to wait for workers to be ready (None = use default)

        Returns
        -------
        list[Worker]
            List of Worker objects

        Raises
        ------
        WorkerNotFoundError
            If role doesn't exist
        WorkerFailedError
            If any worker process failed
        WorkerTimeoutError
            If timeout exceeded waiting for workers
        """
        # Handle colocated/forked roles
        if role in self._colocated_roles:
            # Forked roles have their own workers in _workers
            if role not in self._workers:
                # Colocated roles delegate to target role's workers
                target_role = self._colocated_roles[role]
                logger.debug(
                    f"Role '{role}' is colocated with '{target_role}', "
                    "returning target role's workers"
                )
                return self.get_workers(target_role, timeout)
            # Forked roles fall through to normal worker handling below

        if role not in self._workers:
            raise WorkerNotFoundError(role)

        workers = self._workers[role]
        timeout = timeout if timeout is not None else self.startup_timeout

        self._check_worker_health(role)

        start_time = time.time()
        ready_workers = set()

        while len(ready_workers) < len(workers):
            if time.time() - start_time > timeout:
                raise WorkerTimeoutError(
                    role,
                    timeout,
                )

            for worker_info in workers:
                if worker_info.worker.id in ready_workers:
                    continue

                # Forked workers have process=None - skip process check for them
                if (
                    worker_info.process is not None
                    and worker_info.process.poll() is not None
                ):
                    stderr = self._read_log_tail(worker_info.log_file)
                    raise WorkerFailedError(
                        worker_info.worker.id,
                        worker_info.process.returncode,
                        stderr,
                    )

                if self._is_worker_ready(worker_info):
                    ready_workers.add(worker_info.worker.id)
                    logger.debug(f"Worker {worker_info.worker.id} is ready")

            if len(ready_workers) < len(workers):
                time.sleep(self.health_check_interval)

        logger.info(f"All {len(workers)} workers for role '{role}' are ready")
        return [w.worker for w in workers]

    def _is_worker_ready(self, worker_info: WorkerInfo) -> bool:
        port = int(worker_info.worker.worker_ports[0])
        url = f"http://{format_hostport(worker_info.worker.ip, port)}/health"

        try:
            response = requests.get(url, timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def _configure_worker(self, worker_info: WorkerInfo, worker_rank: int):
        while not self._is_worker_ready(worker_info):
            time.sleep(0.1)

        worker_id = worker_info.worker.id
        port = int(worker_info.worker.worker_ports[0])
        url = f"http://{format_hostport(worker_info.worker.ip, port)}/configure"

        try:
            response = requests.post(
                url,
                data=orjson.dumps(
                    serialize_value(
                        dict(
                            config=self.exp_config,
                            role=worker_info.role,
                            rank=worker_rank,
                        )
                    )
                ),
                headers={"Content-Type": "application/json"},
                timeout=300.0,
            )

            if response.status_code == 200:
                logger.info(f"Configuration successfully on worker '{worker_id}'")
                return
            elif response.status_code == 400:
                error_detail = response.json().get("error", "Unknown error")
                raise WorkerConfigurationError(worker_id, error_detail, str(400))
            elif response.status_code == 500:
                error_detail = response.json().get("error", "Unknown error")
                raise WorkerConfigurationError(worker_id, error_detail, str(500))
            else:
                raise WorkerConfigurationError(
                    worker_id,
                    f"Unexpected status code: {response.status_code}",
                    str(response.status_code),
                )

        except requests.exceptions.ConnectionError as e:
            if (
                worker_info.process is not None
                and worker_info.process.poll() is not None
            ):
                stderr = self._read_log_tail(worker_info.log_file)
                raise WorkerFailedError(
                    worker_id, worker_info.process.returncode, stderr
                ) from e
            raise RPCConnectionError(
                worker_id, worker_info.worker.ip, port, str(e)
            ) from e

        except requests.exceptions.Timeout as e:
            raise WorkerConfigurationError(worker_id, f"Request timed out: {e}") from e

        except WorkerConfigurationError:
            raise

        except Exception as e:
            raise WorkerConfigurationError(
                worker_id, f"Unexpected error: {str(e)}"
            ) from e

    def _check_worker_health(self, role: str):
        if role not in self._workers:
            return

        for worker_info in self._workers[role]:
            # Forked workers have process=None - skip process check for them
            if worker_info.process is None:
                continue
            returncode = worker_info.process.poll()
            if returncode is not None:
                stderr = self._read_log_tail(worker_info.log_file)
                raise WorkerFailedError(
                    worker_info.worker.id,
                    returncode,
                    stderr,
                )

    def delete_workers(self, role: str | None = None, reverse_order: bool = False):
        """Delete workers and clean up resources.

        Parameters
        ----------
        role : str, optional
            Specific worker role to delete, or None to delete all
        reverse_order : bool, optional
            If True, terminate workers in reverse rank order so that rank-0
            (owner of the global TCPStore) is signalled last. See
            ``Scheduler.delete_workers`` for background.
        """
        if role is None:
            # Delete colocated roles first (they don't own processes)
            colocated_roles = list(self._colocated_roles.keys())
            for r in colocated_roles:
                self.delete_workers(r, reverse_order=reverse_order)
            # Then delete actual worker roles
            roles = list(self._workers.keys())
            for r in roles:
                self.delete_workers(r, reverse_order=reverse_order)
            return

        # Handle colocated/forked role
        if role in self._colocated_roles:
            # Forked roles have their own workers that need port cleanup
            if role in self._workers:
                logger.info(f"Removing forked role '{role}' (managed by parent worker)")
                workers = self._workers[role]
                if reverse_order:
                    workers = list(reversed(workers))
                self._cleanup_workers(
                    workers
                )  # Release ports, but process=None skips kill
                del self._workers[role]
            else:
                # Colocated roles don't have their own workers
                logger.info(f"Removing colocated role '{role}' mapping")
            del self._colocated_roles[role]
            return

        if role not in self._workers:
            logger.warning(f"Worker role '{role}' not found, skipping deletion")
            return

        workers = self._workers[role]
        logger.info(f"Deleting {len(workers)} workers for role '{role}'")

        if reverse_order:
            workers = list(reversed(workers))
        self._cleanup_workers(workers)

        del self._workers[role]

        logger.info(f"Successfully deleted workers for role '{role}'")

    def _cleanup_workers(self, workers: list[WorkerInfo]):
        """Tear down a batch of workers with coordinated teardown semantics.

        The previous implementation iterated ``workers`` serially and called
        ``kill_process_tree(..., timeout=3, graceful=True)`` on each one.
        Because that helper blocks for up to ``timeout`` seconds between
        SIGTERM and the fallback SIGKILL, a 4-rank job could spend ~12 s
        killing workers one-by-one. During that window only a single rank
        was executing its ``engine.destroy()`` path, so the CPU barrier
        added in ``FSDPEngine.destroy()`` could never actually synchronise
        -- every rank timed out on its barrier and the NCCL teardown race
        that produced ``TCPStore.recvValue failed`` / HeartbeatMonitor
        warnings was not fixed.

        The corrected behaviour is:

        1. Release port allocations synchronously (cheap, no I/O).
        2. Send SIGTERM to every worker in the order provided by the
           caller, with no blocking waits in between. ``delete_workers``
           passes the list in reverse rank order when
           ``reverse_order=True``, which preserves the "rank-0 signalled
           last" guarantee while keeping the dispatch window in the
           millisecond range.
        3. Wait for every worker to exit in parallel using one thread per
           worker. Each thread re-uses ``kill_process_tree`` so the
           existing SIGTERM -> wait -> SIGKILL escalation is preserved
           per-worker; we just no longer serialise the waits.

        With this change every rank enters ``engine.destroy()`` within the
        same small window, the CPU ``dist.barrier`` inside can actually
        rendezvous, and the NCCL / TCPStore teardown becomes race-free.
        """
        import threading

        # Phase 1: always release ports, regardless of whether the worker
        # owns a process (forked workers have ``process is None``).
        live_workers: list[WorkerInfo] = []
        for worker_info in workers:
            try:
                for port_str in worker_info.worker.worker_ports:
                    self._allocated_ports.discard(int(port_str))
            except Exception as e:
                logger.error(
                    f"Error releasing ports for worker {worker_info.worker.id}: {e}",
                    exc_info=True,
                )
            if worker_info.process is not None:
                live_workers.append(worker_info)
            else:
                logger.debug(f"Cleaned up worker {worker_info.worker.id}")

        if not live_workers:
            return

        # Phase 2: dispatch SIGTERM to every worker concurrently via
        # background threads so that all ranks reach their teardown
        # barrier within the same window. The list order is preserved as
        # thread start order: when the caller requests reverse_order,
        # rank-0 is the last thread to be started, which keeps the
        # "rank-0 dies last" property while staying non-blocking.
        def _finalize(worker_info: WorkerInfo) -> None:
            try:
                kill_process_tree(worker_info.process.pid, timeout=3, graceful=True)
                logger.debug(f"Cleaned up worker {worker_info.worker.id}")
            except Exception as e:
                logger.error(
                    f"Error cleaning up worker {worker_info.worker.id}: {e}",
                    exc_info=True,
                )

        threads: list[threading.Thread] = []
        for worker_info in live_workers:
            t = threading.Thread(
                target=_finalize,
                args=(worker_info,),
                name=f"cleanup-{worker_info.worker.id}",
                daemon=True,
            )
            t.start()
            threads.append(t)

        # Phase 3: wait for every cleanup thread. Each ``kill_process_tree``
        # call internally waits up to ``timeout=3`` seconds for graceful
        # shutdown and then SIGKILLs stragglers, so a small safety margin
        # on ``join`` is sufficient.
        join_timeout = 10.0
        for t in threads:
            t.join(timeout=join_timeout)
            if t.is_alive():
                logger.warning(
                    f"Cleanup thread {t.name} did not finish within "
                    f"{join_timeout}s; leaving it as daemon."
                )

    def _read_log_tail(self, log_file: str, lines: int = 50) -> str:
        try:
            with open(log_file) as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:])
        except Exception as e:
            return f"[Could not read log file: {e}]"

    async def set_worker_env(self, worker_id: str, env: dict[str, str]) -> None:
        """Set environment variables on a worker before engine creation."""
        worker_info = self._verify_worker_alive(worker_id)
        if not env:
            return

        payload = {"env": env}
        port = int(worker_info.worker.worker_ports[0])
        url = f"http://{format_hostport(worker_info.worker.ip, port)}/set_env"

        try:
            timeout = aiohttp.ClientTimeout(total=30.0)
            async with aiohttp.ClientSession(
                timeout=timeout,
                connector=get_default_connector(),
            ) as session:
                async with session.post(
                    url,
                    data=orjson.dumps(payload),
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        return
                    detail = (await response.json()).get("error", "Unknown error")
                    raise SchedulerError(
                        worker_id,
                        f"Failed to set env on worker (status={response.status}): {detail}",
                    )
        except (aiohttp.ClientConnectionError, aiohttp.ClientConnectorError) as e:
            raise RPCConnectionError(
                worker_id, worker_info.worker.ip, port, str(e)
            ) from e
        except TimeoutError as e:
            raise SchedulerError(worker_id, f"set_env timed out: {e}") from e

    async def create_engine(
        self,
        worker_id: str,
        engine: str,
        engine_name: str | None = None,
        *args,
        **kwargs,
    ) -> Any:
        """Create an engine instance on a remote worker.

        The engine parameter is a string import path (e.g., "areal.engine.fsdp_engine.FSDPPPOActor")
        that will be dynamically imported and instantiated on the worker.

        Parameters
        ----------
        worker_id : str
            Worker ID in format "role/index"
        engine : str
            Import path to the engine class (e.g., "areal.engine.fsdp_engine.FSDPPPOActor")
        engine_name : str, optional
            Unique name for this engine instance. Defaults to worker_id.
        *args
            Initialization arguments
        **kwargs
            Initialization keyword arguments

        Returns
        -------
        Any
            Result from engine initialization

        Raises
        ------
        WorkerNotFoundError
            If worker doesn't exist
        WorkerFailedError
            If worker process has failed
        EngineCreationError
            If engine creation fails
        """
        # Verify worker exists and is alive
        worker_info = self._verify_worker_alive(worker_id)

        # Default engine_name to worker_id for backward compatibility
        if engine_name is None:
            engine_name = worker_id

        # Validate engine is a string import path
        if not isinstance(engine, str):
            raise EngineCreationError(
                worker_id,
                f"Engine must be a string import path, got {type(engine)}",
            )

        # Build JSON payload with serialized args and kwargs
        payload = {
            "engine": engine,
            "engine_name": engine_name,
            "init_args": serialize_value(list(args)),
            "init_kwargs": serialize_value(kwargs),
        }

        # Send HTTP request to create engine
        port = int(worker_info.worker.worker_ports[0])
        url = f"http://{format_hostport(worker_info.worker.ip, port)}/create_engine"

        try:
            logger.debug(
                f"Creating engine '{engine_name}' (class: {engine}) on worker '{worker_id}'"
            )

            timeout = aiohttp.ClientTimeout(total=300.0)
            async with aiohttp.ClientSession(
                timeout=timeout,
                read_bufsize=1024 * 1024 * 10,
                connector=get_default_connector(),
            ) as session:
                async with session.post(
                    url,
                    data=orjson.dumps(payload),
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.debug(
                            f"Engine '{engine_name}' created successfully on worker '{worker_id}'"
                        )
                        return result.get("result")
                    elif response.status == 400:
                        # Import error or bad request
                        error_detail = (await response.json()).get(
                            "error", "Unknown error"
                        )
                        if "Failed to import" in error_detail:
                            raise EngineImportError(engine, error_detail)
                        else:
                            raise EngineCreationError(worker_id, error_detail, 400)
                    elif response.status == 500:
                        # Engine initialization failed
                        error_detail = (await response.json()).get(
                            "error", "Unknown error"
                        )
                        raise EngineCreationError(worker_id, error_detail, 500)
                    else:
                        raise EngineCreationError(
                            worker_id,
                            f"Unexpected status code: {response.status}",
                            response.status,
                        )

        except (aiohttp.ClientConnectionError, aiohttp.ClientConnectorError) as e:
            if (
                worker_info.process is not None
                and worker_info.process.poll() is not None
            ):
                stderr = self._read_log_tail(worker_info.log_file)
                raise WorkerFailedError(
                    worker_id, worker_info.process.returncode, stderr
                ) from e
            raise RPCConnectionError(
                worker_id, worker_info.worker.ip, port, str(e)
            ) from e

        except TimeoutError as e:
            raise EngineCreationError(worker_id, f"Request timed out: {e}") from e

        except (EngineCreationError, EngineImportError, RPCConnectionError):
            raise

        except Exception as e:
            raise EngineCreationError(worker_id, f"Unexpected error: {str(e)}") from e

    def call_engine(
        self,
        worker_id: str,
        method: str,
        engine_name: str | None = None,
        *args,
        rpc_meta: dict[str, Any] | None = None,
        http_timeout: float = 7200.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs,
    ) -> Any:
        """Call a method on an engine.

        Parameters
        ----------
        worker_id : str
            Worker ID in format "role/index"
        method : str
            Method name to call
        engine_name : str, optional
            Name of the engine to call. Defaults to worker_id.
        *args
            Method arguments
        max_retries : int, optional
            Maximum number of retry attempts, by default 3
        retry_delay : float, optional
            Initial delay between retries (exponential backoff), by default 1.0
        **kwargs
            Method keyword arguments

        Returns
        -------
        Any
            Result from method call

        Raises
        ------
        WorkerNotFoundError
            If worker doesn't exist
        WorkerFailedError
            If worker process has failed
        EngineCallError
            If method call fails
        """
        # Get worker info (initial verification)
        worker_info = self._find_worker_by_id(worker_id)
        if worker_info is None:
            raise WorkerNotFoundError(worker_id)

        # Default engine_name to worker_id for backward compatibility
        if engine_name is None:
            engine_name = worker_id

        # Serialize args and kwargs (convert tensors to SerializedTensor dicts)
        serialized_args = serialize_value(list(args))
        serialized_kwargs = serialize_value(kwargs)

        # Build JSON payload
        payload = {
            "method": method,
            "engine_name": engine_name,
            "args": serialized_args,
            "kwargs": serialized_kwargs,
            "rpc_meta": rpc_meta,
        }

        # Retry logic with exponential backoff
        port = int(worker_info.worker.worker_ports[0])
        url = f"http://{format_hostport(worker_info.worker.ip, port)}/call"
        last_error = None

        for attempt in range(1, max_retries + 1):
            # Check worker health before each attempt (forked workers have process=None)
            if (
                worker_info.process is not None
                and worker_info.process.poll() is not None
            ):
                stderr = self._read_log_tail(worker_info.log_file)
                raise WorkerFailedError(
                    worker_id,
                    worker_info.process.returncode,
                    stderr,
                )

            try:
                logger.debug(
                    f"Calling method '{method}' on worker '{worker_id}' (attempt {attempt})"
                )

                response = requests.post(
                    url,
                    json=payload,
                    timeout=http_timeout,
                )

                result, should_retry, error_msg = self._handle_call_response(
                    response, worker_id, method, attempt
                )
                if not should_retry:
                    if attempt > 1:
                        logger.debug(
                            f"Method '{method}' succeeded on worker '{worker_id}' "
                            f"after {attempt} attempts"
                        )
                    return result
                last_error = error_msg

            except Exception as e:
                last_error = self._handle_call_exception(e, worker_info, worker_id)

            # Retry with exponential backoff
            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Method '{method}' failed on worker '{worker_id}' "
                    f"(attempt {attempt}/{max_retries}): {last_error}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        # All retries exhausted
        raise EngineCallError(
            worker_id,
            method,
            last_error or "Max retries exceeded",
            attempt=max_retries,
        )

    async def async_call_engine(
        self,
        worker_id: str,
        method: str,
        engine_name: str | None = None,
        *args,
        rpc_meta: dict[str, Any] | None = None,
        http_timeout: float = 7200.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs,
    ) -> Any:
        """Async version of call_engine for calling engine methods asynchronously.

        Parameters
        ----------
        worker_id : str
            Worker ID in format "role/index"
        method : str
            Method name to call
        engine_name : str, optional
            Name of the engine to call. Defaults to worker_id.
        *args
            Method arguments
        max_retries : int, optional
            Maximum number of retry attempts, by default 3
        retry_delay : float, optional
            Initial delay between retries (exponential backoff), by default 1.0
        **kwargs
            Method keyword arguments

        Returns
        -------
        Any
            Result from method call

        Raises
        ------
        WorkerNotFoundError
            If worker doesn't exist
        WorkerFailedError
            If worker process has failed
        EngineCallError
            If method call fails
        """
        # Get worker info (initial verification)
        worker_info = self._find_worker_by_id(worker_id)
        if worker_info is None:
            raise WorkerNotFoundError(worker_id)

        # Default engine_name to worker_id for backward compatibility
        if engine_name is None:
            engine_name = worker_id

        # Route to different endpoint based on method
        port = int(worker_info.worker.worker_ports[0])
        # Standard engine method call
        url = f"http://{format_hostport(worker_info.worker.ip, port)}/call"
        # Serialize args and kwargs
        serialized_args = serialize_value(list(args))
        serialized_kwargs = serialize_value(kwargs)
        payload = {
            "method": method,
            "engine_name": engine_name,
            "args": serialized_args,
            "kwargs": serialized_kwargs,
            "rpc_meta": rpc_meta,
        }

        last_error = None

        for attempt in range(1, max_retries + 1):
            # Check worker health before each attempt (forked workers have process=None)
            if (
                worker_info.process is not None
                and worker_info.process.poll() is not None
            ):
                stderr = self._read_log_tail(worker_info.log_file)
                raise WorkerFailedError(
                    worker_id,
                    worker_info.process.returncode,
                    stderr,
                )

            try:
                logger.debug(
                    f"Async calling method '{method}' on worker '{worker_id}' (attempt {attempt})"
                )

                timeo = aiohttp.ClientTimeout(
                    total=http_timeout, sock_connect=http_timeout, connect=http_timeout
                )
                async with aiohttp.ClientSession(
                    timeout=timeo,
                    read_bufsize=1024 * 1024 * 10,
                    connector=get_default_connector(),
                ) as session:
                    async with session.post(
                        url,
                        json=payload,
                        timeout=timeo,
                    ) as response:
                        # Handle response inline since aiohttp json() is async
                        if response.status == 200:
                            result_data = (await response.json()).get("result")
                            deserialized_result = deserialize_value(result_data)
                            if attempt > 1:
                                logger.debug(
                                    f"Method '{method}' succeeded on worker '{worker_id}' "
                                    f"after {attempt} attempts"
                                )
                            return deserialized_result
                        elif response.status == 400:
                            # Bad request (e.g., method doesn't exist) - don't retry
                            error_detail = (await response.json()).get(
                                "error", "Unknown error"
                            )
                            raise EngineCallError(
                                worker_id, method, error_detail, attempt
                            )
                        elif response.status == 500:
                            # Engine method failed - don't retry
                            error_detail = (await response.json()).get(
                                "error", "Unknown error"
                            )
                            raise EngineCallError(
                                worker_id, method, error_detail, attempt
                            )
                        elif response.status == 503:
                            # Service unavailable - retry
                            last_error = "Service unavailable"
                        else:
                            # Other errors - retry
                            response_text = await response.text()
                            last_error = f"HTTP {response.status}: {response_text}"

            except (aiohttp.ClientConnectionError, aiohttp.ClientConnectorError) as e:
                # Check if worker died (forked workers have process=None)
                if (
                    worker_info.process is not None
                    and worker_info.process.poll() is not None
                ):
                    stderr = self._read_log_tail(worker_info.log_file)
                    raise WorkerFailedError(
                        worker_id,
                        worker_info.process.returncode,
                        stderr,
                    ) from e
                last_error = f"Connection error: {e}"
            except TimeoutError as e:
                last_error = f"Timeout: {e}"
            except EngineCallError:
                raise
            except Exception as e:
                last_error = f"Unexpected error: {e}"

            # Retry with exponential backoff
            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Method '{method}' failed on worker '{worker_id}' "
                    f"(attempt {attempt}/{max_retries}): {last_error}. "
                    f"Retrying in {delay:.1f}s..."
                )

                await asyncio.sleep(delay)

        # All retries exhausted
        raise EngineCallError(
            worker_id,
            method,
            last_error or "Max retries exceeded",
            attempt=max_retries,
        )

    def _find_worker_by_id(self, worker_id: str) -> WorkerInfo | None:
        for workers in self._workers.values():
            for worker_info in workers:
                if worker_info.worker.id == worker_id:
                    return worker_info
        return None

    def _verify_worker_alive(self, worker_id: str) -> WorkerInfo:
        worker_info = self._find_worker_by_id(worker_id)
        if worker_info is None:
            raise WorkerNotFoundError(worker_id)

        # Check if process has exited (forked workers have process=None)
        if worker_info.process is not None and worker_info.process.poll() is not None:
            stderr = self._read_log_tail(worker_info.log_file)
            raise WorkerFailedError(
                worker_id,
                worker_info.process.returncode,
                stderr,
            )

        return worker_info

    def _handle_call_response(
        self, response, worker_id: str, method: str, attempt: int
    ):
        if response.status_code == 200:
            result = response.json().get("result")
            # Deserialize result (convert SerializedTensor dicts back to tensors)
            deserialized_result = deserialize_value(result)
            return deserialized_result, False, None
        elif response.status_code == 400:
            # Bad request (e.g., method doesn't exist) - don't retry
            error_detail = response.json().get("error", "Unknown error")
            raise EngineCallError(worker_id, method, error_detail, attempt)
        elif response.status_code == 500:
            # Engine method failed - don't retry
            error_detail = response.json().get("error", "Unknown error")
            raise EngineCallError(worker_id, method, error_detail, attempt)
        elif response.status_code == 503:
            # Service unavailable - retry
            return None, True, "Service unavailable"
        else:
            # Other errors - retry
            return None, True, f"HTTP {response.status_code}: {response.text}"

    def _handle_call_exception(
        self, e: Exception, worker_info: WorkerInfo, worker_id: str
    ) -> str:
        if isinstance(e, requests.exceptions.ConnectionError):
            # Check if worker died (forked workers have process=None)
            if (
                worker_info.process is not None
                and worker_info.process.poll() is not None
            ):
                stderr = self._read_log_tail(worker_info.log_file)
                raise WorkerFailedError(
                    worker_id,
                    worker_info.process.returncode,
                    stderr,
                ) from e
            return f"Connection error: {e}"
        elif isinstance(e, requests.exceptions.Timeout):
            return f"Timeout: {e}"
        elif isinstance(e, EngineCallError):
            raise
        else:
            return f"Unexpected error: {e}"

    def __del__(self):
        try:
            self.delete_workers()
        except Exception:
            pass
