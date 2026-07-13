# SPDX-License-Identifier: Apache-2.0

import asyncio
import getpass
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
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
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.infra.scheduler.exceptions import (
    EngineCallError,
    EngineCreationError,
    EngineImportError,
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
    JobState,
    get_env_vars,
    get_thread_env_vars,
)
from areal.infra.utils.proc import build_streaming_log_cmd
from areal.infra.utils.slurm import (
    cancel_jobs,
    parse_slurm_nodelist,
    query_jobs,
)
from areal.utils import logging, name_resolve, names
from areal.utils.fs import validate_shared_path
from areal.utils.network import format_hostport, split_hostport
from areal.utils.offload import get_tms_env_vars

logger = logging.getLogger("SlurmScheduler")


@dataclass
class SlurmWorkerInfo:
    """Slurm worker information."""

    worker: Worker
    role: str
    slurm_job_id: int  # -1 for forked workers (managed by parent)
    task_index: int
    discovered: bool = False
    spec: SchedulingSpec | None = None
    node: str | None = None


class SlurmScheduler(Scheduler):
    def __init__(
        self,
        n_gpus_per_node: int = 8,
        experiment_name: str | None = None,
        trial_name: str | None = None,
        fileroot: str | None = None,
        cluster_name: str | None = None,
        container_type: str = "apptainer",
        container_mounts: str | None = "/storage:/storage",
        srun_additional_args: str = "--unbuffered --mpi=pmi2 -K --chdir $PWD",
        startup_timeout: float = 300.0,
        health_check_interval: float = 5.0,
        enable_tms_offload: bool | None = None,
        name_resolve_type: str = "nfs",
        nfs_record_root: str = "/tmp/areal/name_resolve",
        etcd3_addr: str = "localhost:2379",
        exp_config: BaseExperimentConfig | None = None,
    ):
        # Get n_gpus_per_node from parameter or config
        self._n_gpus_per_node = n_gpus_per_node
        if exp_config is not None:
            self._n_gpus_per_node = exp_config.cluster.n_gpus_per_node

        # Get other params from config if provided
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.fileroot = fileroot
        self.enable_tms_offload = bool(enable_tms_offload)
        self.cluster_name = cluster_name
        if exp_config is not None:
            self.experiment_name = exp_config.experiment_name
            self.trial_name = exp_config.trial_name
            self.fileroot = exp_config.cluster.fileroot
            self.cluster_name = exp_config.cluster.cluster_name
            self.enable_tms_offload = exp_config.enable_offload
        if self.experiment_name is None or self.trial_name is None:
            raise ValueError("experiment_name and trial_name must be provided")

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

        self.container_type = container_type
        self.container_mounts = container_mounts
        self.srun_additional_args = srun_additional_args
        self.startup_timeout = startup_timeout
        self.health_check_interval = health_check_interval
        self.exp_config = exp_config

        # Internal state
        self._workers: dict[str, list[SlurmWorkerInfo]] = {}
        self._jobs: dict[str, int] = {}  # role -> slurm_job_id
        self._job_status_cache: dict[
            int, tuple[JobState, float]
        ] = {}  # job_id -> (state, timestamp)
        self._status_cache_ttl = 5.0  # Cache status for 5 seconds

        # Colocation tracking: colocated roles reuse workers from target role
        # For forked roles, they also track target but have their own workers in _workers
        self._colocated_roles: dict[str, str] = {}  # colocated_role -> target_role

        logger.info(
            f"Initialized SlurmScheduler: exp={self.experiment_name}, "
            f"trial={self.trial_name}, fileroot={self.fileroot}, "
            f"n_gpus_per_node={self.n_gpus_per_node}"
        )

    @property
    def n_gpus_per_node(self) -> int:
        return self._n_gpus_per_node

    def _slurm_name(self, job_name: str) -> str:
        return f"{self.experiment_name}_{self.trial_name}:{job_name}"

    def _log_path_of(self, role: str) -> str:
        log_path = (
            Path(self.fileroot)
            / "logs"
            / getpass.getuser()
            / self.experiment_name
            / self.trial_name
        )
        log_path.mkdir(parents=True, exist_ok=True)
        return str(log_path / f"{role}.log")

    def _merged_log_path(self) -> str:
        log_path = (
            Path(self.fileroot)
            / "logs"
            / getpass.getuser()
            / self.experiment_name
            / self.trial_name
        )
        log_path.mkdir(parents=True, exist_ok=True)
        return str(log_path / "merged.log")

    def _sbatch_path_of(self, role: str) -> str:
        sbatch_path = (
            Path(self.fileroot)
            / "logs"
            / getpass.getuser()
            / self.experiment_name
            / self.trial_name
        )
        sbatch_path.mkdir(parents=True, exist_ok=True)
        return str(sbatch_path / f"{role}.sh")

    def _read_log_tail(self, role: str, lines: int = 50) -> str:
        try:
            with open(self._log_path_of(role)) as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:])
        except Exception as e:
            return f"[Could not read log file: {e}]"

    def _find_worker_by_id(self, worker_id: str) -> SlurmWorkerInfo | None:
        """Find worker by ID across all roles."""
        for workers in self._workers.values():
            for worker_info in workers:
                if worker_info.worker.id == worker_id:
                    return worker_info
        return None

    def _check_job_status(self, role: str) -> None:
        """Check Slurm job status and raise if failed/cancelled."""
        # For colocated/forked roles, check the target role's job status instead
        if role in self._colocated_roles:
            target_role = self._colocated_roles[role]
            return self._check_job_status(target_role)

        if role not in self._jobs:
            raise WorkerNotFoundError(f"Role '{role}' not found")

        job_id = self._jobs[role]

        # Check cache first
        current_time = time.time()
        if job_id in self._job_status_cache:
            cached_state, cached_time = self._job_status_cache[job_id]
            if current_time - cached_time < self._status_cache_ttl:
                if cached_state in [JobState.FAILED, JobState.CANCELLED]:
                    logs = self._read_log_tail(role)
                    raise WorkerFailedError(
                        f"{role}/*", -1, f"Job {job_id} {cached_state}. Logs:\n{logs}"
                    )
                return

        try:
            job_infos = query_jobs(slurm_ids=[job_id])
            if not job_infos:
                logs = self._read_log_tail(role)
                raise WorkerFailedError(
                    f"{role}/*", -1, f"Job {job_id} not in queue. Logs:\n{logs}"
                )

            state = job_infos[0].state
            self._job_status_cache[job_id] = (state, current_time)

            if state in [JobState.FAILED, JobState.CANCELLED]:
                logs = self._read_log_tail(role)
                raise WorkerFailedError(
                    f"{role}/*", -1, f"Job {job_id} {state}. Logs:\n{logs}"
                )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to query job status: {e}")

    def _verify_worker_alive(self, worker_id: str) -> SlurmWorkerInfo:
        """Verify worker exists and job is running."""
        worker_info = self._find_worker_by_id(worker_id)
        if worker_info is None:
            raise WorkerNotFoundError(worker_id)

        # Check Slurm job status
        self._check_job_status(worker_info.role)

        return worker_info

    def _wait_worker_ready(self, worker_info: SlurmWorkerInfo, timeout: int = 60):
        tik = time.time()
        while time.time() - tik < timeout:
            if self._is_worker_ready(worker_info):
                return
            time.sleep(1)

    def _is_worker_ready(self, worker_info: SlurmWorkerInfo) -> bool:
        """Check if worker is ready via health endpoint."""
        if not worker_info.discovered:
            return False

        port = int(worker_info.worker.worker_ports[0])
        url = f"http://{format_hostport(worker_info.worker.ip, port)}/health"

        try:
            response = requests.get(url, timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def _configure_worker(self, worker_info: SlurmWorkerInfo, worker_rank: int) -> None:
        # Wait for worker to be ready
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
                logger.info(f"Configuration successful on worker '{worker_id}'")
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
            self._check_job_status(worker_info.role)
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

    def _discover_worker_network(self, role: str) -> None:
        if role not in self._workers:
            raise WorkerNotFoundError(f"Role '{role}' is not created yet")

        # Apply discoveries to worker infos
        for worker_info in self._workers[role]:
            if worker_info.discovered:
                continue
            task_index = worker_info.task_index
            key = names.worker_discovery(
                self.experiment_name, self.trial_name, role, str(task_index)
            )
            try:
                addr = name_resolve.get(key)
            except name_resolve.NameEntryNotFoundError:
                continue
            ip, port = split_hostport(addr)
            worker_info.worker.ip = ip
            worker_ports = [str(port)]
            worker_info.worker.worker_ports = worker_ports
            worker_info.discovered = True

            self._wait_worker_ready(worker_info)

            # Allocate new ports from the worker
            if worker_info.spec.port_count > 1:
                resp = requests.post(
                    f"http://{format_hostport(ip, port)}/alloc_ports",
                    json=dict(count=worker_info.spec.port_count - 1),
                )
                resp.raise_for_status()
                worker_ports += list(map(str, resp.json()["ports"]))

            logger.debug(f"Discovered {worker_info.worker.id} at {addr}")

    def _prepare_worker_specs(
        self, role: str, num_workers: int, schedulings: list[SchedulingSpec] | None
    ) -> list[SchedulingSpec]:
        """Prepare scheduling specs for workers."""
        if schedulings is None or len(schedulings) == 0:
            raise ValueError(f"No scheduling specs provided for role '{role}'")

        # Amend environment variables
        for sch in schedulings:
            # AReaL env var forwarding
            if self.enable_tms_offload:
                sch.env_vars.update(get_tms_env_vars())
            sch.env_vars.update(get_env_vars())
            thread_env = get_thread_env_vars(
                cpus_per_task=sch.cpu,
                existing_env_vars=sch.env_vars,
            )
            sch.env_vars.update(thread_env)

        if len(schedulings) == 1:
            # Expand single spec to all workers
            return [schedulings[0]] * num_workers
        elif len(schedulings) == num_workers:
            return list(schedulings)
        else:
            raise ValueError(
                f"Number of scheduling specs ({len(schedulings)}) must be 1 or match "
                f"number of workers ({num_workers})"
            )

    def _get_colocation_nodes(self, target_role: str, replicas: int) -> tuple[int, str]:
        """Get node allocation for colocation strategy."""
        if target_role not in self._jobs:
            raise WorkerNotFoundError(
                f"Target role '{target_role}' not found for colocation"
            )
        target_workers = self._workers[target_role]
        if replicas != len(target_workers):
            raise SchedulerError(
                f"Colocated target role {target_role} should "
                f"have the same number of replicas: target {target_workers} != {replicas}"
            )

        # Query Slurm for target job's nodelist
        job_id = self._jobs[target_role]
        try:
            job_infos = query_jobs(slurm_ids=[job_id])
            if not job_infos:
                raise WorkerCreationError(
                    target_role, f"Target job {job_id} not found in queue"
                )

            nodelist = job_infos[0].host  # NodeList from squeue
            nodes = len(parse_slurm_nodelist(nodelist))

            return nodes, nodelist
        except subprocess.CalledProcessError as e:
            raise WorkerCreationError(target_role, f"Failed to query target job: {e}")

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
        target_wi: SlurmWorkerInfo,
        target_role: str,
        command: str | None = None,
    ) -> SlurmWorkerInfo:
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
            async with session.post(
                f"http://{format_hostport(forked_host, forked_port)}/alloc_ports",
                json=dict(count=port_cnt - 1),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise WorkerCreationError(
                        role,
                        f"Fork failed for worker {idx}",
                        f"HTTP {response.status}: {error_text}",
                    )
                new_ports = (await response.json())["ports"]
                worker.worker_ports += list(map(str, new_ports))

        return SlurmWorkerInfo(
            worker=worker,
            role=role,
            slurm_job_id=-1,  # Not a separate Slurm job
            task_index=idx,
            discovered=True,  # Already discovered during fork
            spec=target_wi.spec,  # Inherit from target
            node=target_wi.node,  # Same node as target
        )

    async def _kill_forked_worker(
        self,
        session: aiohttp.ClientSession,
        role: str,
        idx: int,
        target_wi: SlurmWorkerInfo,
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
        workers: list[SlurmWorkerInfo],
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
        target_workers: list[SlurmWorkerInfo],
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

    def _generate_sbatch_script(
        self,
        role: str,
        replicas: int,
        nodes: int,
        total_gpus: int,
        cpus_per_task: int,
        mem_per_task: int,
        schedulings: list[SchedulingSpec],
        nodelist: str | None,
        exclude: str | None,
    ) -> str:
        """Generate sbatch script for worker job with single srun command."""
        ntasks_per_node = replicas // nodes if nodes > 0 else replicas
        spec = schedulings[0]  # Use first spec for global settings

        if total_gpus % self.n_gpus_per_node != 0:
            raise ValueError(
                "Slurm only supports allocating entire nodes. "
                f"Requesting {total_gpus} GPUs but each node has {self.n_gpus_per_node}."
            )

        # Build SBATCH directives
        sbatch_options = [
            f"--job-name={self._slurm_name(role)}",
            # Note: output handled via tee in script for merged log support
            "--output=/dev/null",
            "--no-requeue",
            f"--nodes={nodes}",
            f"--ntasks-per-node={ntasks_per_node}",
            f"--cpus-per-task={cpus_per_task}",
            f"--mem={mem_per_task * ntasks_per_node}M",
        ]
        if total_gpus > 0:
            sbatch_options.append(f"--gres=gpu:{self.n_gpus_per_node}")
        if nodelist:
            sbatch_options.append(f"--nodelist={nodelist}")
        if exclude:
            sbatch_options.append(f"--exclude={exclude}")

        sbatch_options_str = "\n".join([f"#SBATCH {opt}" for opt in sbatch_options])

        # Calculate resources
        mem_per_cpu = (
            mem_per_task // cpus_per_task if cpus_per_task > 0 else mem_per_task
        )

        # Build RPC command (port will be auto-assigned by server)
        rpc_cmd = spec.cmd or "python -m areal.infra.rpc.rpc_server"
        rpc_cmd_flags = [
            "--experiment-name",
            self.experiment_name,
            "--trial-name",
            self.trial_name,
            "--role",
            role,
            "--name-resolve-type",
            self.name_resolve_config.type,
            "--nfs-record-root",
            self.name_resolve_config.nfs_record_root,
            "--etcd3-addr",
            self.name_resolve_config.etcd3_addr,
        ]
        if self.fileroot:
            rpc_cmd_flags.extend(["--fileroot", str(self.fileroot)])
        rpc_cmd = " ".join([rpc_cmd] + rpc_cmd_flags)

        # Build environment variables (common to all workers)
        env_vars_dict = spec.env_vars.copy() if spec.env_vars else {}

        bash_cmds = (spec.additional_bash_cmds or []).copy()

        # Set CUDA_VISIBLE_DEVICES based on SLURM_LOCALID before any Python imports.
        # This MUST happen before Python starts, otherwise CUDA runtime ignores the
        # env var change once it's initialized.
        # We use bash commands instead of env_vars_dict because SLURM_LOCALID is only
        # available at runtime and each task needs a different value.
        if total_gpus > 0:
            gpus_per_task = spec.gpu
            if gpus_per_task == 1:
                cuda_setup_cmd = (
                    f"export CUDA_VISIBLE_DEVICES=$((SLURM_LOCALID * {gpus_per_task}))"
                )
            else:
                cuda_setup_cmd = (
                    f"export CUDA_VISIBLE_DEVICES=$(seq -s, $((SLURM_LOCALID * {gpus_per_task})) "
                    f"$((SLURM_LOCALID * {gpus_per_task} + {gpus_per_task} - 1)))"
                )
            # Also set ASCEND_RT_VISIBLE_DEVICES for Ascend NPU compatibility
            ascend_setup_cmd = "export ASCEND_RT_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
            bash_cmds.insert(0, cuda_setup_cmd)
            bash_cmds.insert(1, ascend_setup_cmd)

        bash_cmds.append(rpc_cmd)
        bash_cmds_str = ";\n".join(bash_cmds)
        cmd = f"bash -c {shlex.quote(bash_cmds_str)}"

        # Build final command and export string
        if self.container_type == "apptainer":
            # For apptainer, pass env vars to singularity
            env_string = " ".join(f"--env {k}={v}" for k, v in env_vars_dict.items())
            final_cmd = "singularity exec --no-home --writable-tmpfs --nv"
            if self.container_mounts:
                final_cmd += f" --bind {self.container_mounts}"
            final_cmd += f" {env_string}"
            final_cmd += f" {spec.image}"
            final_cmd += f" {cmd}"
        else:  # native
            final_cmd = cmd

        srun_flags = [
            f"--nodes={nodes}",
            f"--ntasks={replicas}",
            f"--cpus-per-task={cpus_per_task}",
            f"--mem-per-cpu={mem_per_cpu}M",
        ]
        if total_gpus > 0:
            srun_flags.append(f"--gres=gpu:{self.n_gpus_per_node}")

        # Log files and prefix for merged log
        role_log = self._log_path_of(role)
        merged_log = self._merged_log_path()

        # Build srun command with streaming log pipeline
        srun_cmd = (
            f"srun {self.srun_additional_args} {' '.join(srun_flags)} {final_cmd}"
        )
        log_pipeline = build_streaming_log_cmd(srun_cmd, role_log, merged_log, role)

        # Complete sbatch script with single srun command
        sbatch_script = f"""#!/bin/bash
{sbatch_options_str}

# Single srun command launches all workers
# Output goes to role log (no prefix) and merged log (with prefix)
# stdbuf -oL ensures line-buffered streaming for both outputs
{log_pipeline}
"""
        return sbatch_script

    def create_workers(self, job: Job, *args, **kwargs) -> list[str]:
        """Create workers via Slurm job array submission.

        Parameters
        ----------
        job : Job
            Job specification with replicas, tasks, and scheduling strategy

        Returns
        -------
        list[str]
            List of worker IDs created

        Raises
        ------
        WorkerCreationError
            If worker creation fails
        """
        role = job.role
        replicas = job.replicas
        if ":" in role:
            raise ValueError("Invalid worker name.")
        num_workers = job.replicas

        # Validation
        if role in self._workers:
            raise WorkerCreationError(role, f"Role '{role}' already exists")
        if num_workers <= 0:
            raise WorkerCreationError(
                role, "Invalid configuration", "replicas must be greater than 0"
            )

        # Prepare scheduling specs
        schedulings = self._prepare_worker_specs(role, num_workers, job.tasks)

        strategy = job.scheduling_strategy
        strategy_type = SchedulingStrategyType(strategy.type)
        colocate_role = strategy.target
        logger.info(
            f"Creating {num_workers} workers for role '{role}' "
            f"(strategy: {strategy_type}, colocate_with: {colocate_role})"
        )

        # Determine node allocation and handle colocation
        if strategy_type == SchedulingStrategyType.colocation:
            colocate_role = strategy.target
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
                # Fork mode: spawn new processes on same nodes via /fork endpoint
                return self.fork_workers(role, colocate_role)

            # Reuse existing workers - no new Slurm job submitted
            worker_ids = [w.worker.id for w in target_workers]
            self._colocated_roles[role] = colocate_role

            logger.info(
                f"Role '{role}' colocated with '{colocate_role}': "
                f"reusing workers {worker_ids}"
            )
            return worker_ids

        if strategy_type != SchedulingStrategyType.separation:
            raise ValueError(f"Unknown scheduling strategy type: {strategy_type}")
        # Non-colocated: calculate nodes needed and submit new Slurm job
        spec = schedulings[0]
        total_gpus = spec.gpu * replicas
        nodes = max(1, (total_gpus + self.n_gpus_per_node - 1) // self.n_gpus_per_node)
        nodelist = spec.nodelist

        # Calculate resource requirements
        n_gpus_per_node = min(
            self.n_gpus_per_node, (spec.gpu * replicas + nodes - 1) // nodes
        )
        cpus_per_task = spec.cpu
        mem_per_task = spec.mem * 1024  # Convert GB to MB

        logger.info(
            f"Creating {replicas} workers for role '{role}': "
            f"nodes={nodes}, gpus_per_node={n_gpus_per_node}, "
            f"cpus={cpus_per_task}, mem={mem_per_task}MB"
        )

        # Generate sbatch script
        sbatch_script = self._generate_sbatch_script(
            role=role,
            replicas=replicas,
            nodes=nodes,
            total_gpus=spec.gpu * replicas,
            cpus_per_task=cpus_per_task,
            mem_per_task=mem_per_task,
            schedulings=schedulings,
            nodelist=nodelist,
            exclude=spec.exclude,
        )

        # Write and submit sbatch script
        sbatch_path = self._sbatch_path_of(role)
        with open(sbatch_path, "w") as f:
            f.write(sbatch_script)

        try:
            output = (
                subprocess.check_output(["sbatch", sbatch_path]).decode("utf-8").strip()
            )
            logger.info(f"Submitted job for role '{role}': {output}")
        except subprocess.CalledProcessError as e:
            raise WorkerCreationError(
                role, "sbatch submission failed", f"Error: {e}\nScript: {sbatch_path}"
            )

        # Parse job ID
        match = re.search(r"Submitted batch job (\d+)", output)
        if not match:
            raise WorkerCreationError(
                role, "Failed to parse job ID from sbatch output", f"Output: {output}"
            )
        slurm_job_id = int(match.group(1))

        # Initialize worker tracking
        workers = []
        worker_ids = []
        for idx in range(replicas):
            worker_id = f"{role}/{idx}"
            worker = Worker(
                id=worker_id,
                ip="",  # Will be discovered
                worker_ports=[],  # Will be discovered
                engine_ports=[],
            )
            worker_spec = (
                schedulings[idx] if len(schedulings) == replicas else schedulings[0]
            )
            worker_info = SlurmWorkerInfo(
                worker=worker,
                role=role,
                slurm_job_id=slurm_job_id,
                task_index=idx,
                discovered=False,
                spec=worker_spec,
            )
            workers.append(worker_info)
            worker_ids.append(worker_id)

        self._workers[role] = workers
        self._jobs[role] = slurm_job_id

        logger.info(
            f"Created {replicas} workers for role '{role}' with job ID {slurm_job_id}"
        )
        return worker_ids

    def get_workers(self, role: str, timeout: float | None = None) -> list[Worker]:
        """Wait for workers to be ready and return their information.

        Parameters
        ----------
        role : str
            Role name to query
        timeout : float, optional
            Maximum wait time in seconds

        Returns
        -------
        list[Worker]
            List of ready workers

        Raises
        ------
        WorkerNotFoundError
            If role doesn't exist
        WorkerTimeoutError
            If timeout exceeded
        WorkerFailedError
            If workers failed
        """
        # Handle colocated/forked roles
        if role in self._colocated_roles:
            # Forked roles have their own workers in _workers
            if role in self._workers:
                workers = self._workers[role]
                # Forked workers are already discovered and configured during creation
                # Just verify they're still healthy
                for worker_info in workers:
                    if not self._is_worker_ready(worker_info):
                        raise WorkerFailedError(
                            worker_info.worker.id, -1, "Forked worker not responding"
                        )
                logger.info(
                    f"All {len(workers)} forked workers ready for role '{role}'"
                )
                return [w.worker for w in workers]
            else:
                # Colocated roles delegate to target role's workers
                target_role = self._colocated_roles[role]
                logger.debug(
                    f"Role '{role}' is colocated with '{target_role}', "
                    "returning target role's workers"
                )
                return self.get_workers(target_role, timeout)

        if role not in self._workers:
            raise WorkerNotFoundError(f"Role '{role}' not found")

        workers = self._workers[role]
        timeout = timeout if timeout is not None else self.startup_timeout
        start_time = time.time()
        pending_logged = False

        logger.info(
            f"Waiting for {len(workers)} workers of role '{role}' to be ready..."
        )

        while time.time() - start_time < timeout:
            # Check job status
            try:
                self._check_job_status(role)
            except WorkerFailedError:
                raise

            # Log if job is pending
            job_id = self._jobs[role]
            if job_id in self._job_status_cache:
                state, _ = self._job_status_cache[job_id]
                if state == JobState.PENDING and not pending_logged:
                    logger.info(
                        f"Job {job_id} for role '{role}' is PENDING in queue..."
                    )
                    pending_logged = True

            if any(not w.discovered for w in workers):
                self._discover_worker_network(role)

            # Wait for all to be discovered
            discovered_count = sum(1 for w in workers if w.discovered)
            if discovered_count < len(workers):
                if discovered_count > 0:
                    logger.debug(
                        f"Discovered {discovered_count}/{len(workers)} workers"
                    )
                time.sleep(self.health_check_interval)
                continue

            # Health check all workers
            ready_workers = []

            for worker_info in workers:
                if self._is_worker_ready(worker_info):
                    ready_workers.append(worker_info)

            # All ready
            if len(ready_workers) == len(workers):
                logger.info(f"All {len(workers)} workers ready for role '{role}'")

                # Configure workers if exp_config is available
                if self.exp_config is not None:
                    for worker_rank, worker_info in enumerate(workers):
                        self._configure_worker(worker_info, worker_rank)

                return [w.worker for w in workers]

            # Log progress
            if ready_workers:
                logger.debug(f"{len(ready_workers)}/{len(workers)} workers ready")

            time.sleep(self.health_check_interval)

        raise WorkerTimeoutError(role, timeout)

    def _destroy_engines_on_workers(
        self, workers: list[SlurmWorkerInfo], timeout: float = 30.0
    ) -> None:
        """Call ``engine.destroy()`` on every worker via HTTP before killing jobs.

        All calls are dispatched concurrently so that the engine-side CPU
        barrier (``dist.barrier`` + ``dist.destroy_process_group``) can
        complete across all ranks.  A bounded *timeout* prevents indefinite
        blocking when a worker is already unreachable.
        """
        if not workers:
            return

        async def _destroy_all():
            destroy_timeout = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(
                timeout=destroy_timeout,
                connector=get_default_connector(),
            ) as session:
                tasks = []
                for wi in workers:
                    port = int(wi.worker.worker_ports[0])
                    url = f"http://{format_hostport(wi.worker.ip, port)}/call"
                    payload = {
                        "method": "destroy",
                        "engine_name": wi.worker.id,
                        "args": serialize_value([]),
                        "kwargs": serialize_value({}),
                        "rpc_meta": None,
                    }
                    tasks.append(
                        session.post(
                            url,
                            data=orjson.dumps(payload),
                            headers={"Content-Type": "application/json"},
                        )
                    )
                results = await asyncio.gather(
                    *[self._safe_destroy_request(t) for t in tasks],
                    return_exceptions=True,
                )
                for wi, res in zip(workers, results):
                    if isinstance(res, BaseException):
                        logger.warning(
                            f"engine.destroy() on {wi.worker.id} failed: "
                            f"{type(res).__name__}: {res}"
                        )

        try:
            run_async_task(_destroy_all)
        except Exception as e:
            logger.warning(f"Failed to destroy engines before cancel: {e}")

    @staticmethod
    async def _safe_destroy_request(coro):
        """Await an aiohttp context-manager response, suppressing errors."""
        try:
            async with coro as resp:
                await resp.read()
        except Exception as e:
            raise RuntimeError(str(e)) from e

    def delete_workers(self, role: str | None = None, reverse_order: bool = False):
        """Delete workers and cancel Slurm jobs.

        Teardown follows a two-phase protocol analogous to the Ray and Local
        schedulers:

        1. **Engine destroy** – call ``engine.destroy()`` on every worker via
           HTTP concurrently.  This runs the engine-side CPU barrier and
           ``dist.destroy_process_group`` so that NCCL communicators and the
           TCPStore are shut down cleanly on all ranks.
        2. **Job cancel** – ``scancel`` the Slurm job.  At this point process
           groups are already torn down, so killing the processes will not
           produce spurious ``TCPStore.recvValue failed`` warnings.

        Parameters
        ----------
        role : str, optional
            Role to delete. If None, deletes all roles.
        reverse_order : bool, optional
            Accepted for API compatibility with other schedulers but ignored
            here: Slurm tears down the entire job step atomically via
            ``scancel``, so per-rank ordering cannot be enforced.
        """
        del reverse_order  # unused, see docstring
        if role is None:
            # Delete colocated/forked roles first (they don't own Slurm jobs)
            colocated_roles = list(self._colocated_roles.keys())
            for r in colocated_roles:
                self.delete_workers(r)
            # Then delete actual worker roles
            for r in list(self._workers.keys()):
                self.delete_workers(r)
            return

        # Handle colocated/forked role
        if role in self._colocated_roles:
            # Forked roles have their own workers that need cleanup
            if role in self._workers:
                logger.info(f"Removing forked role '{role}' (managed by parent worker)")
                del self._workers[role]
            else:
                logger.info(f"Removing colocated role '{role}' mapping")
            del self._colocated_roles[role]
            return

        if role not in self._workers:
            logger.warning(f"Role '{role}' not found, skipping deletion")
            return

        job_id = self._jobs.get(role)
        if job_id is None:
            # Role exists in _workers but not in _jobs - shouldn't happen for regular roles
            logger.warning(f"Role '{role}' has no job ID, cleaning up workers only")
            del self._workers[role]
            return

        workers = self._workers[role]
        logger.info(
            f"Deleting {len(workers)} workers for role '{role}' (job ID {job_id})"
        )

        # Phase 1: destroy engines so that the CPU barrier and
        # dist.destroy_process_group complete on every rank.
        self._destroy_engines_on_workers(workers)

        # Phase 2: cancel the Slurm job. Process groups are already torn
        # down, so scancel will not cause TCPStore race conditions.
        try:
            cancel_jobs(slurm_ids=[job_id], signal="SIGTERM")
            time.sleep(2)  # Give time for graceful shutdown

            # Check if still running, force kill if needed
            try:
                job_infos = query_jobs(slurm_ids=[job_id])
                if job_infos and job_infos[0].state == JobState.RUNNING:
                    logger.warning(f"Job {job_id} still running, force killing")
                    cancel_jobs(slurm_ids=[job_id], signal="SIGKILL")
            except subprocess.CalledProcessError:
                pass  # Job already gone
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")

        # Clean up internal state
        del self._workers[role]
        del self._jobs[role]
        if job_id in self._job_status_cache:
            del self._job_status_cache[job_id]

        logger.info(f"Successfully deleted workers for role '{role}'")

    async def set_worker_env(self, worker_id: str, env: dict[str, str]) -> None:
        """Set environment variables on a worker before engine creation.

        Parameters
        ----------
        worker_id : str
            Worker ID in format "role/index"
        env : dict[str, str]
            Environment variables to set
        """
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
            self._check_job_status(worker_info.role)
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

        Parameters
        ----------
        worker_id : str
            Worker ID in format "role/index"
        engine : str
            Import path to engine class
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
            If worker has failed
        EngineCreationError
            If engine creation fails
        """
        worker_info = self._verify_worker_alive(worker_id)

        # Default engine_name to worker_id for backward compatibility
        if engine_name is None:
            engine_name = worker_id

        if not isinstance(engine, str):
            raise EngineCreationError(
                worker_id,
                f"Engine must be a string import path, got {type(engine)}",
            )

        payload = {
            "engine": engine,
            "engine_name": engine_name,
            "init_args": serialize_value(list(args)),
            "init_kwargs": serialize_value(kwargs),
        }

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
                            f"Engine created successfully on worker '{worker_id}'"
                        )
                        return result.get("result")
                    elif response.status == 400:
                        error_detail = (await response.json()).get(
                            "error", "Unknown error"
                        )
                        if "Failed to import" in error_detail:
                            raise EngineImportError(engine, error_detail)
                        else:
                            raise EngineCreationError(worker_id, error_detail, 400)
                    elif response.status == 500:
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
            self._check_job_status(worker_info.role)
            raise RPCConnectionError(
                worker_id, worker_info.worker.ip, port, str(e)
            ) from e

        except TimeoutError as e:
            raise EngineCreationError(
                worker_id, f"Engine creation timed out: {e}"
            ) from e

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
        """Call a method on an engine instance (synchronous).

        Parameters
        ----------
        worker_id : str
            Worker ID in format "role/index"
        method : str
            Name of method to call
        engine_name : str, optional
            Name of the engine to call. Defaults to worker_id.
        *args
            Method arguments
        http_timeout : float, default=7200.0
            HTTP request timeout in seconds
        max_retries : int, default=3
            Maximum retry attempts
        retry_delay : float, default=1.0
            Initial retry delay in seconds
        **kwargs
            Method keyword arguments

        Returns
        -------
        Any
            Result from engine method call

        Raises
        ------
        WorkerNotFoundError
            If worker doesn't exist
        WorkerFailedError
            If worker has failed
        EngineCallError
            If method call fails
        """
        worker_info = self._find_worker_by_id(worker_id)
        if worker_info is None:
            raise WorkerNotFoundError(worker_id)

        # Default engine_name to worker_id for backward compatibility
        if engine_name is None:
            engine_name = worker_id

        serialized_args = serialize_value(list(args))
        serialized_kwargs = serialize_value(kwargs)
        payload = {
            "method": method,
            "engine_name": engine_name,
            "args": serialized_args,
            "kwargs": serialized_kwargs,
            "rpc_meta": rpc_meta,
        }

        port = int(worker_info.worker.worker_ports[0])
        url = f"http://{format_hostport(worker_info.worker.ip, port)}/call"
        last_error = None

        for attempt in range(1, max_retries + 1):
            # Check job status before each attempt
            try:
                self._check_job_status(worker_info.role)
            except WorkerFailedError:
                raise

            try:
                response = requests.post(url, json=payload, timeout=http_timeout)

                if response.status_code == 200:
                    result = response.json()
                    return deserialize_value(result.get("result"))
                elif response.status_code == 500:
                    error_detail = response.json().get("error", "Unknown error")
                    # Check if retryable
                    if attempt < max_retries and "timeout" in error_detail.lower():
                        last_error = f"Engine method timeout: {error_detail}"
                        logger.warning(
                            f"Retryable error on attempt {attempt}/{max_retries}: {last_error}"
                        )
                    else:
                        raise EngineCallError(
                            worker_id, method, error_detail, attempt=attempt
                        )
                elif response.status_code == 503:
                    # Service unavailable - retryable
                    last_error = "Service unavailable (503)"
                    logger.warning(
                        f"Worker temporarily unavailable, retry {attempt}/{max_retries}"
                    )
                else:
                    error_detail = response.json().get("error", "Unknown error")
                    raise EngineCallError(
                        worker_id,
                        method,
                        f"HTTP {response.status_code}: {error_detail}",
                        attempt=attempt,
                    )

            except requests.exceptions.Timeout as e:
                last_error = f"Request timeout: {e}"
                logger.warning(f"Request timeout on attempt {attempt}/{max_retries}")
            except requests.exceptions.ConnectionError as e:
                self._check_job_status(worker_info.role)
                last_error = f"Connection error: {e}"
                logger.warning(f"Connection error on attempt {attempt}/{max_retries}")
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                logger.warning(
                    f"Unexpected error on attempt {attempt}/{max_retries}: {e}"
                )

            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.info(
                    f"Retrying in {delay:.1f}s (attempt {attempt}/{max_retries})"
                )
                time.sleep(delay)

        raise EngineCallError(
            worker_id, method, last_error or "Max retries exceeded", attempt=max_retries
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
        """Call a method on an engine instance (asynchronous).

        Parameters
        ----------
        worker_id : str
            Worker ID in format "role/index"
        method : str
            Name of method to call
        engine_name : str, optional
            Name of the engine to call. Defaults to worker_id.
        *args
            Method arguments
        http_timeout : float, default=7200.0
            HTTP request timeout in seconds
        max_retries : int, default=3
            Maximum retry attempts
        retry_delay : float, default=1.0
            Initial retry delay in seconds
        **kwargs
            Method keyword arguments

        Returns
        -------
        Any
            Result from engine method call

        Raises
        ------
        WorkerNotFoundError
            If worker doesn't exist
        WorkerFailedError
            If worker has failed
        EngineCallError
            If method call fails
        """
        worker_info = self._find_worker_by_id(worker_id)
        if worker_info is None:
            raise WorkerNotFoundError(worker_id)

        # Default engine_name to worker_id for backward compatibility
        if engine_name is None:
            engine_name = worker_id

        serialized_args = serialize_value(list(args))
        serialized_kwargs = serialize_value(kwargs)
        payload = {
            "method": method,
            "engine_name": engine_name,
            "args": serialized_args,
            "kwargs": serialized_kwargs,
            "rpc_meta": rpc_meta,
        }

        port = int(worker_info.worker.worker_ports[0])
        url = f"http://{format_hostport(worker_info.worker.ip, port)}/call"
        last_error = None

        for attempt in range(1, max_retries + 1):
            # Check job status before each attempt
            try:
                self._check_job_status(worker_info.role)
            except WorkerFailedError:
                raise

            try:
                timeout = aiohttp.ClientTimeout(total=http_timeout)
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
                            return deserialize_value(result.get("result"))
                        elif response.status == 500:
                            error_detail = (await response.json()).get(
                                "error", "Unknown error"
                            )
                            if (
                                attempt < max_retries
                                and "timeout" in error_detail.lower()
                            ):
                                last_error = f"Engine method timeout: {error_detail}"
                                logger.warning(
                                    f"Retryable error on attempt {attempt}/{max_retries}: {last_error}"
                                )
                            else:
                                raise EngineCallError(
                                    worker_id, method, error_detail, attempt=attempt
                                )
                        elif response.status == 503:
                            last_error = "Service unavailable (503)"
                            logger.warning(
                                f"Worker temporarily unavailable, retry {attempt}/{max_retries}"
                            )
                        else:
                            error_detail = (await response.json()).get(
                                "error", "Unknown error"
                            )
                            raise EngineCallError(
                                worker_id,
                                method,
                                f"HTTP {response.status}: {error_detail}",
                                attempt=attempt,
                            )

            except TimeoutError as e:
                last_error = f"Request timeout: {e}"
                logger.warning(f"Request timeout on attempt {attempt}/{max_retries}")
            except (aiohttp.ClientConnectionError, aiohttp.ClientConnectorError) as e:
                self._check_job_status(worker_info.role)
                last_error = f"Connection error: {e}"
                logger.warning(f"Connection error on attempt {attempt}/{max_retries}")
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                logger.warning(
                    f"Unexpected error on attempt {attempt}/{max_retries}: {e}"
                )

            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.info(
                    f"Retrying in {delay:.1f}s (attempt {attempt}/{max_retries})"
                )
                await asyncio.sleep(delay)

        raise EngineCallError(
            worker_id, method, last_error or "Max retries exceeded", attempt=max_retries
        )
