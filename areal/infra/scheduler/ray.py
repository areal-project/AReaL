# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import getpass
import os
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import orjson
import ray
import ray.exceptions
import requests
from ray.actor import ActorHandle
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from areal.api import Job, Scheduler, Worker
from areal.api.alloc_mode import ModelAllocation
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
    get_env_vars,
    get_thread_env_vars,
)
from areal.infra.utils.proc import kill_process_tree, run_with_streaming_logs
from areal.infra.utils.ray import create_resource_spec, ray_resource_type
from areal.utils import logging, name_resolve, names
from areal.utils.fs import validate_shared_path
from areal.utils.network import (
    format_hostport,
    gethostip,
    split_hostport,
)
from areal.utils.offload import get_tms_env_vars

logger = logging.getLogger("RayScheduler")

DEVICE_CONTROL_ENV_VARS = {
    "GPU": "CUDA_VISIBLE_DEVICES",
    "NPU": "ASCEND_RT_VISIBLE_DEVICES",
}


def _read_log_tail(log_file: str, lines: int = 50) -> str:
    try:
        with open(log_file) as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])
    except Exception as e:
        return f"[Could not read log file: {e}]"


class RayBackendWorkerProcessManager:
    def __init__(
        self,
        role: str,
        log_file: str,
        merged_log: str,
        device_control_env_var: str,
        env_vars: dict[str, str],
        visible_devices: list[str],
        host: str,
    ):
        self.role = role
        self.log_file = log_file
        self.merged_log = merged_log
        self.device_control_env_var = device_control_env_var
        self.env_vars = env_vars
        self.visible_devices = visible_devices
        self.host = host
        self.backend_worker_processes: dict[tuple[tuple[str, str], str, int], Any] = {}

    def start_backend_worker(
        self,
        group_key: tuple[str, str],
        backend: str,
        server_args: dict[str, Any],
    ) -> dict[str, Any]:
        server_args = server_args.copy()
        base_env = os.environ.copy()
        base_env.update(self.env_vars)
        if backend == "sglang":
            from areal.api.cli_args import SGLangConfig
            from areal.engine.sglang_remote import SGLangBackend

            cmd = SGLangConfig.build_cmd_from_args(server_args)
            env = SGLangBackend.build_server_env(base_env)
        elif backend == "vllm":
            from areal.api.cli_args import vLLMConfig
            from areal.engine.vllm_remote import VLLMBackend

            cmd = vLLMConfig.build_cmd_from_args(server_args)
            env = VLLMBackend.build_server_env(base_env)
        else:
            raise RuntimeError(f"Unsupported multi-node inference backend: {backend}")

        node_rank = int(server_args.get("node_rank", 0))
        process_key = (group_key, backend, node_rank)
        old_process = self.backend_worker_processes.pop(process_key, None)
        if old_process is not None and old_process.poll() is None:
            kill_process_tree(old_process.pid, timeout=5, graceful=True)

        if self.visible_devices:
            env[self.device_control_env_var] = ",".join(self.visible_devices)
        process = run_with_streaming_logs(
            cmd,
            self.log_file,
            self.merged_log,
            self.role,
            env=env,
        )
        time.sleep(0.1)
        if process.poll() is not None:
            raise RuntimeError(
                f"{backend} server node {server_args.get('node_rank')} exited "
                f"immediately with code {process.returncode}\n{_read_log_tail(self.log_file)}"
            )
        self.backend_worker_processes[process_key] = process
        return {"host": self.host, "pid": process.pid, "key": process_key}

    def drain_worker_processes(
        self, group_key: tuple[str, str] | None = None
    ) -> list[tuple[str, Any]]:
        processes = []
        for process_key in list(self.backend_worker_processes):
            if group_key is not None and process_key[0] != group_key:
                continue
            process = self.backend_worker_processes.pop(process_key)
            processes.append((f"backend worker {process_key}", process))
        return processes


@ray.remote
class RayWorkerProcessLauncher:
    def __init__(
        self,
        role: str,
        log_file: str,
        merged_log: str,
        ray_device_resource: str,
        device_control_env_var: str,
        env_vars: dict[str, str] | None = None,
    ):
        self.role = role
        self.log_file = log_file
        self.merged_log = merged_log
        self.ray_device_resource = ray_device_resource
        self.device_control_env_var = device_control_env_var
        self.env_vars = env_vars or {}
        self.host = gethostip()
        self.worker_processes: dict[str, Any] = {}
        self.visible_devices = self._get_visible_devices()
        self.backend_process_manager: RayBackendWorkerProcessManager | None = None

    def _get_visible_devices(self) -> list[str]:
        devices = []
        try:
            ids = ray.get_runtime_context().get_accelerator_ids()
            device = self.ray_device_resource
            if device in ids:
                devices = [str(x) for x in ids[device]]
        except Exception:
            pass

        if not devices:
            visible = os.environ.get(self.device_control_env_var)
            if visible:
                devices = [x for x in visible.split(",") if x != ""]

        return sorted(devices, key=int)

    def get_node_info(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "visible_devices": self.visible_devices,
            "node_id": ray.get_runtime_context().get_node_id(),
        }

    def _build_worker_spec(
        self,
        role: str,
        worker_index: int,
        gpu_devices: list[str],
        cmd: str | None,
        experiment_name: str,
        trial_name: str,
        name_resolve_type: str,
        nfs_record_root: str,
        etcd3_addr: str,
        fileroot: str | None,
    ) -> dict[str, Any]:
        worker_id = f"{role}/{worker_index}"
        if not cmd:
            cmd = "python -m areal.infra.rpc.rpc_server"

        # Build RPC command (port will be auto-assigned by server)
        rpc_cmd = shlex.split(cmd)
        if any(token == "--port" or token.startswith("--port=") for token in rpc_cmd):
            raise RuntimeError(
                "Custom command should not include --port argument. "
                "The scheduler automatically allocates and provides the port."
            )

        if rpc_cmd[0].startswith("python"):
            rpc_cmd[0] = sys.executable
        rpc_cmd_flags = [
            "--host",
            "0.0.0.0",
            "--port",
            "0",
            "--experiment-name",
            str(experiment_name),
            "--trial-name",
            str(trial_name),
            "--role",
            role,
            "--worker-index",
            str(worker_index),
            "--name-resolve-type",
            name_resolve_type,
            "--nfs-record-root",
            nfs_record_root,
            "--etcd3-addr",
            etcd3_addr,
        ]
        if fileroot:
            rpc_cmd_flags.extend(["--fileroot", str(fileroot)])
        rpc_cmd = [*rpc_cmd, *rpc_cmd_flags]

        env = os.environ.copy()
        env.update(self.env_vars)
        env[self.device_control_env_var] = ",".join(gpu_devices)

        return {
            "worker_id": worker_id,
            "role": role,
            "gpu_devices": gpu_devices,
            "cmd": rpc_cmd,
            "env": env,
        }

    def start_workers(self, worker_specs: list[dict[str, Any]]) -> None:
        workers = [self._build_worker_spec(**spec) for spec in worker_specs]
        started = []
        try:
            for worker in workers:
                logger.info(
                    "Starting Ray worker %s on %s devices=%s ports=%s: %s",
                    worker["worker_id"],
                    self.host,
                    worker["gpu_devices"],
                    "auto",
                    " ".join(worker["cmd"]),
                )
                worker["process"] = run_with_streaming_logs(
                    worker["cmd"],
                    self.log_file,
                    self.merged_log,
                    worker["role"],
                    env=worker["env"],
                )
                started.append(worker)

            # Give every spawned process a short window to fail fast before returning.
            # Unlike per-worker RPC, all processes in this batch are spawned first.
            time.sleep(0.1)
            failures = [
                worker for worker in started if worker["process"].poll() is not None
            ]
            if failures:
                failed_workers = ", ".join(
                    f"{worker['worker_id']}={worker['process'].returncode}"
                    for worker in failures
                )
                raise RuntimeError(
                    f"Workers exited immediately: {failed_workers}\n"
                    f"{_read_log_tail(self.log_file)}"
                )

            for worker in started:
                process = worker["process"]
                self.worker_processes[worker["worker_id"]] = process
        except Exception:
            for worker in started:
                process = worker.get("process")
                if process is not None and process.poll() is None:
                    try:
                        kill_process_tree(process.pid, timeout=5, graceful=True)
                    except Exception:
                        logger.warning(
                            "Failed to stop partially started Ray worker",
                            exc_info=True,
                        )
            raise

    def worker_statuses(self, worker_ids: list[str]) -> dict[str, dict[str, Any]]:
        statuses = {}
        for worker_id in worker_ids:
            process = self.worker_processes.get(worker_id)
            if process is None:
                statuses[worker_id] = {"exists": False, "returncode": None}
            else:
                statuses[worker_id] = {"exists": True, "returncode": process.poll()}
        return statuses

    def start_backend_worker(
        self, group_key: tuple[str, str], backend: str, server_args: dict[str, Any]
    ) -> dict[str, Any]:
        if self.backend_process_manager is None:
            self.backend_process_manager = RayBackendWorkerProcessManager(
                self.role,
                self.log_file,
                self.merged_log,
                self.device_control_env_var,
                self.env_vars,
                self.visible_devices,
                self.host,
            )
        return self.backend_process_manager.start_backend_worker(
            group_key, backend, server_args
        )

    def _stop_processes(self, processes: list[tuple[str, Any]]) -> None:
        async def stop_process(name: str, process: Any) -> None:
            try:
                if process.poll() is None:
                    await asyncio.to_thread(
                        kill_process_tree, process.pid, timeout=5, graceful=True
                    )
            except Exception:
                logger.warning("Failed to stop Ray-managed %s", name, exc_info=True)

        if not processes:
            return

        async def stop_processes() -> None:
            await asyncio.gather(
                *(stop_process(name, process) for name, process in processes)
            )

        run_async_task(stop_processes)

    def stop_backend_worker_group(self, group_key: tuple[str, str]) -> None:
        if self.backend_process_manager is None:
            return

        self._stop_processes(
            self.backend_process_manager.drain_worker_processes(group_key)
        )

    def stop_all_processes(self) -> None:
        try:
            processes = []
            if self.backend_process_manager is not None:
                processes.extend(self.backend_process_manager.drain_worker_processes())
            processes.extend(
                (f"worker process {worker_id}", process)
                for worker_id, process in self.worker_processes.items()
            )
            self._stop_processes(processes)
            self.worker_processes.clear()
        except Exception:
            logger.warning("Failed to stop Ray launcher processes", exc_info=True)

    def __ray_shutdown__(self) -> None:
        self.stop_all_processes()


@dataclass
class RayWorkerInfo:
    """Ray worker information."""

    worker: Worker
    role: str
    task_index: int
    launchers: list[ActorHandle] = field(default_factory=list)
    spec: SchedulingSpec | None = None


class RayMultiNodeRolloutCoordinator:
    def __init__(
        self,
        exp_config: BaseExperimentConfig | None,
        startup_timeout: float,
    ):
        self.startup_timeout = startup_timeout
        rollout_config = getattr(exp_config, "rollout", None)
        self._rollout_inference_backend = (
            ModelAllocation.from_str(rollout_config.backend, name="rollout").backend
            if rollout_config is not None
            else None
        )

    async def _build_multi_node_server_args(
        self,
        worker_info: RayWorkerInfo,
        backend: str,
        server_args: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        port = int(worker_info.worker.worker_ports[0])
        timeout = aiohttp.ClientTimeout(total=self.startup_timeout)
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=get_default_connector(),
        ) as session:
            async with session.post(
                f"http://{format_hostport(worker_info.worker.ip, port)}/alloc_ports",
                json=dict(count=1),
            ) as resp:
                resp.raise_for_status()
                host_ports = await resp.json()
        master_host = host_ports["host"]
        master_port = host_ports["ports"][0]
        n_nodes = len(worker_info.launchers)

        head_args = copy.deepcopy(server_args)
        worker_args = []
        if backend == "sglang":
            dist_init_addr = f"{master_host}:{master_port}"
            head_args.update(nnodes=n_nodes, node_rank=0, dist_init_addr=dist_init_addr)
            for node_rank in range(1, n_nodes):
                args = copy.deepcopy(server_args)
                args.pop("host", None)
                args.pop("port", None)
                args.update(
                    nnodes=n_nodes, node_rank=node_rank, dist_init_addr=dist_init_addr
                )
                worker_args.append(args)
        elif backend == "vllm":
            head_args.update(
                nnodes=n_nodes,
                node_rank=0,
                master_addr=master_host,
                master_port=str(master_port),
            )
            for node_rank in range(1, n_nodes):
                args = copy.deepcopy(server_args)
                args.pop("host", None)
                args.pop("port", None)
                args.update(
                    nnodes=n_nodes,
                    node_rank=node_rank,
                    master_addr=master_host,
                    master_port=str(master_port),
                    headless=True,
                )
                worker_args.append(args)
        else:
            raise EngineCallError(
                worker_info.worker.id,
                "launch_server",
                f"Unsupported multi-node inference backend: {backend}",
            )

        return head_args, worker_args

    async def _stop_backend_workers_on_launchers(
        self, backend_worker_group_key: tuple[str, str], launchers: list[ActorHandle]
    ) -> None:
        if not launchers:
            return
        try:
            refs = [
                launcher.stop_backend_worker_group.remote(backend_worker_group_key)
                for launcher in launchers
            ]
            await asyncio.wait_for(asyncio.gather(*refs), timeout=30)
        except Exception:
            logger.warning(
                "Failed to stop Ray-managed multi-node backend workers",
                exc_info=True,
            )

    async def prepare_launch_server(
        self,
        worker_info: RayWorkerInfo,
        engine_name: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        server_args = kwargs.get("server_args")
        if not isinstance(server_args, dict) or len(worker_info.launchers) <= 1:
            return kwargs
        if worker_info.role != "rollout" or self._rollout_inference_backend is None:
            raise EngineCallError(
                worker_info.worker.id,
                "launch_server",
                "Multi-node inference launch is only supported for rollout workers "
                "with rollout.backend set to sglang or vllm.",
            )
        backend = self._rollout_inference_backend

        head_args, worker_args = await self._build_multi_node_server_args(
            worker_info, backend, server_args
        )
        backend_worker_group_key = (worker_info.worker.id, engine_name)
        refs = [
            launcher.start_backend_worker.remote(
                backend_worker_group_key, backend, args
            )
            for launcher, args in zip(
                worker_info.launchers[1:], worker_args, strict=True
            )
        ]
        try:
            await asyncio.wait_for(asyncio.gather(*refs), timeout=self.startup_timeout)
        except BaseException:
            await self._stop_backend_workers_on_launchers(
                backend_worker_group_key, worker_info.launchers[1:]
            )
            raise
        return {**kwargs, "server_args": head_args}

    async def stop_backend_workers(
        self, worker_info: RayWorkerInfo, engine_name: str
    ) -> None:
        backend_worker_group_key = (worker_info.worker.id, engine_name)
        await self._stop_backend_workers_on_launchers(
            backend_worker_group_key, worker_info.launchers[1:]
        )


class RayScheduler(Scheduler):
    def __init__(
        self,
        n_gpus_per_node: int = 8,
        experiment_name: str | None = None,
        trial_name: str | None = None,
        fileroot: str | None = None,
        startup_timeout: float = 300.0,
        health_check_interval: float = 5.0,
        enable_tms_offload: bool | None = None,
        name_resolve_type: str = "nfs",
        nfs_record_root: str = "/tmp/areal/name_resolve",
        etcd3_addr: str = "localhost:2379",
        exp_config: BaseExperimentConfig | None = None,
    ):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Get n_gpus_per_node from parameter or config
        self._n_gpus_per_node = n_gpus_per_node
        if exp_config is not None:
            self._n_gpus_per_node = exp_config.cluster.n_gpus_per_node

        # Get other params from config if provided
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.fileroot = fileroot
        self.enable_tms_offload = bool(enable_tms_offload)
        if exp_config is not None:
            self.experiment_name = exp_config.experiment_name
            self.trial_name = exp_config.trial_name
            self.fileroot = exp_config.cluster.fileroot
            self.enable_tms_offload = exp_config.enable_offload
        if self.experiment_name is None or self.trial_name is None:
            raise ValueError("experiment_name and trial_name must be provided")

        self.ray_device_resource = ray_resource_type()
        if self.ray_device_resource not in DEVICE_CONTROL_ENV_VARS:
            raise RuntimeError(
                f"RayScheduler does not support {self.ray_device_resource}-only clusters"
            )
        self.device_control_env_var = DEVICE_CONTROL_ENV_VARS[self.ray_device_resource]

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

        self.startup_timeout = startup_timeout
        self.health_check_interval = health_check_interval
        self.exp_config = exp_config

        # Internal state
        self._workers: dict[str, list[RayWorkerInfo]] = {}
        self._launchers: dict[
            str, list[ActorHandle]
        ] = {}  # role -> Ray launcher actors
        self._placement_groups: dict[str, Any] = {}
        self._multi_node_rollout = RayMultiNodeRolloutCoordinator(
            exp_config, startup_timeout
        )

        # Colocation tracking: colocated roles reuse workers from target role
        # For forked roles, they also track target but have their own workers in _workers
        self._colocated_roles: dict[str, str] = {}  # colocated_role -> target_role

        logger.info(
            f"Initialized RayScheduler: exp={self.experiment_name}, "
            f"trial={self.trial_name}, fileroot={self.fileroot}, "
            f"n_gpus_per_node={self.n_gpus_per_node}"
        )

    @property
    def n_gpus_per_node(self) -> int:
        return self._n_gpus_per_node

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

    def _read_log_tail(self, role: str, lines: int = 50) -> str:
        return _read_log_tail(self._log_path_of(role), lines=lines)

    def _find_worker_by_id(self, worker_id: str) -> RayWorkerInfo | None:
        """Find worker by ID across all roles."""
        for workers in self._workers.values():
            for worker_info in workers:
                if worker_info.worker.id == worker_id:
                    return worker_info
        return None

    def _stop_launchers(self, role: str, timeout: float) -> None:
        refs = []
        for launcher in self._launchers.get(role, []):
            try:
                refs.append(launcher.stop_all_processes.remote())
            except Exception as e:
                logger.error(f"Error submitting Ray launcher stop for role {role}: {e}")

        for ref in refs:
            try:
                ray.get(ref, timeout=timeout)
            except Exception as e:
                logger.error(f"Error stopping Ray launcher for role {role}: {e}")

    def _check_worker_process_status(self, role: str) -> None:
        """Check Ray worker process status and raise if failed."""
        # For colocated/forked roles, check the target role's process status instead
        if role in self._colocated_roles:
            target_role = self._colocated_roles[role]
            return self._check_worker_process_status(target_role)

        if role not in self._launchers:
            raise WorkerNotFoundError(f"Role '{role}' not found")

        workers_by_launcher: list[tuple[ActorHandle, list[RayWorkerInfo]]] = []
        for worker_info in self._workers.get(role, []):
            if not worker_info.launchers:
                continue
            head_launcher = worker_info.launchers[0]
            for launcher, worker_infos in workers_by_launcher:
                if launcher is head_launcher:
                    worker_infos.append(worker_info)
                    break
            else:
                workers_by_launcher.append((head_launcher, [worker_info]))

        refs = []
        for launcher, worker_infos in workers_by_launcher:
            refs.append(
                (
                    worker_infos,
                    launcher.worker_statuses.remote(
                        [worker_info.worker.id for worker_info in worker_infos]
                    ),
                )
            )

        for worker_infos, ref in refs:
            try:
                statuses = ray.get(ref, timeout=2)
            except ray.exceptions.GetTimeoutError:
                logger.debug(
                    "Timed out querying Ray worker process status for role %s; "
                    "status is unknown",
                    role,
                )
                continue
            except ray.exceptions.RayActorError as e:
                worker_info = worker_infos[0]
                logs = self._read_log_tail(role)
                raise WorkerFailedError(
                    worker_info.worker.id,
                    -1,
                    f"Ray worker process launcher failed: {e}. Logs:\n{logs}",
                ) from e

            for worker_info in worker_infos:
                status = statuses.get(worker_info.worker.id, {})
                if not status.get("exists"):
                    logs = self._read_log_tail(role)
                    raise WorkerFailedError(
                        worker_info.worker.id,
                        -1,
                        f"Ray worker process is missing from launcher. Logs:\n{logs}",
                    )
                if status.get("returncode") is not None:
                    logs = self._read_log_tail(role)
                    raise WorkerFailedError(
                        worker_info.worker.id,
                        status["returncode"],
                        logs,
                    )

    def _verify_worker_alive(self, worker_id: str) -> RayWorkerInfo:
        """Verify worker exists and job is running."""
        worker_info = self._find_worker_by_id(worker_id)
        if worker_info is None:
            raise WorkerNotFoundError(worker_id)

        # Check Ray worker process status
        self._check_worker_process_status(worker_info.role)

        return worker_info

    def _wait_worker_ready(self, worker_info: RayWorkerInfo, timeout: int = 60):
        tik = time.time()
        while time.time() - tik < timeout:
            if self._is_worker_ready(worker_info):
                return
            time.sleep(1)

    def _is_worker_ready(self, worker_info: RayWorkerInfo) -> bool:
        """Check if worker is ready via health endpoint."""
        if not worker_info.worker.worker_ports:
            return False

        port = int(worker_info.worker.worker_ports[0])
        url = f"http://{format_hostport(worker_info.worker.ip, port)}/health"

        try:
            response = requests.get(url, timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    async def _configure_worker(
        self,
        session: aiohttp.ClientSession,
        worker_info: RayWorkerInfo,
        worker_rank: int,
    ) -> None:
        worker_id = worker_info.worker.id
        port = int(worker_info.worker.worker_ports[0])
        url = f"http://{format_hostport(worker_info.worker.ip, port)}/configure"

        try:
            async with session.post(
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
                timeout=aiohttp.ClientTimeout(total=300.0),
            ) as response:
                if response.status == 200:
                    logger.info(f"Configuration successful on worker '{worker_id}'")
                    return
                elif response.status == 400:
                    error_detail = (await response.json()).get("error", "Unknown error")
                    raise WorkerConfigurationError(worker_id, error_detail, str(400))
                elif response.status == 500:
                    error_detail = (await response.json()).get("error", "Unknown error")
                    raise WorkerConfigurationError(worker_id, error_detail, str(500))
                else:
                    raise WorkerConfigurationError(
                        worker_id,
                        f"Unexpected status code: {response.status}",
                        str(response.status),
                    )

        except (aiohttp.ClientConnectionError, aiohttp.ClientConnectorError) as e:
            self._check_worker_process_status(worker_info.role)
            raise RPCConnectionError(
                worker_id, worker_info.worker.ip, port, str(e)
            ) from e

        except TimeoutError as e:
            raise WorkerConfigurationError(worker_id, f"Request timed out: {e}") from e

        except WorkerConfigurationError:
            raise

        except Exception as e:
            raise WorkerConfigurationError(
                worker_id, f"Unexpected error: {str(e)}"
            ) from e

    async def _configure_workers(self, workers: list[RayWorkerInfo]) -> None:
        """Configure workers concurrently and wait for all responses."""
        if not workers:
            return

        logger.info(f"Configuring {len(workers)} workers concurrently")
        timeout = aiohttp.ClientTimeout(total=300.0)
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=get_default_connector(),
        ) as session:
            tasks = [
                self._configure_worker(session, worker_info, worker_rank)
                for worker_rank, worker_info in enumerate(workers)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, BaseException):
                raise result

    def _discover_worker_network(self, role: str) -> None:
        if role not in self._workers:
            raise WorkerNotFoundError(f"Role '{role}' is not created yet")

        # Apply discoveries to worker infos
        for worker_info in self._workers[role]:
            if worker_info.worker.worker_ports:
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
            if sch.additional_bash_cmds:
                raise ValueError(
                    "RayScheduler does not support SchedulingSpec.additional_bash_cmds. "
                    "Use SchedulingSpec.env_vars for Ray worker environment setup."
                )
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
        target_wi: RayWorkerInfo,
        target_role: str,
        command: str | None = None,
    ) -> RayWorkerInfo:
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

        return RayWorkerInfo(
            worker=worker,
            role=role,
            task_index=idx,
            launchers=target_wi.launchers,
            spec=target_wi.spec,  # Inherit from target
        )

    async def _kill_forked_worker(
        self,
        session: aiohttp.ClientSession,
        role: str,
        idx: int,
        target_wi: RayWorkerInfo,
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
        workers: list[RayWorkerInfo],
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
        target_workers: list[RayWorkerInfo],
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
            await self._configure_workers(workers)

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

    def _create_placement_group(
        self, role: str, bundles: list[dict[str, Any]], timeout: float
    ) -> Any:
        """Generate Ray placement group for worker job with bundle-per-node layout."""
        pg = placement_group(bundles=bundles, strategy="PACK")
        try:
            ready_ref = pg.ready()
            tik = time.time()
            while True:
                elapsed = time.time() - tik
                remaining = timeout - elapsed
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for placement group for role '{role}'"
                    )

                if ray.wait(
                    [ready_ref], timeout=min(self.health_check_interval, remaining)
                )[0]:
                    ray.get(ready_ref, timeout=0)
                    break

                elapsed = time.time() - tik
                logger.info(
                    "Waiting for Ray placement group for role '%s': "
                    "elapsed=%.0fs remaining=%.0fs bundles=%s "
                    "available_resources=%s cluster_resources=%s",
                    role,
                    elapsed,
                    max(0.0, timeout - elapsed),
                    bundles,
                    ray.available_resources(),
                    ray.cluster_resources(),
                )
        except TimeoutError as e:
            remove_placement_group(pg)
            logger.error(
                "Ray placement group timeout, please check if the resource requirement "
                "for your experiment exceeds the available resources in the cluster. \n"
                f"ray.available_resources(): {ray.available_resources()} \n"
                f"ray.cluster_resources(): {ray.cluster_resources()} \n"
                f"ray.nodes(): {ray.nodes()} \n"
                f"Placement Group bundles: {bundles}"
            )
            raise WorkerCreationError(
                role,
                "Ray placement group timeout",
                f"Placement Group bundles: {bundles}",
            ) from e
        except Exception as e:
            remove_placement_group(pg)
            raise WorkerCreationError(
                role,
                "Ray placement group creation failed",
                f"{type(e).__name__}: {e}",
            ) from e
        return pg

    def _build_node_plan(
        self, replicas: int, spec: SchedulingSpec
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        if spec.gpu <= 0:
            bundles = [
                {"CPU": spec.cpu * replicas, "memory": spec.mem * replicas * 1024**3}
            ]
            return bundles, [dict(bundle_index=0, node_rank=0, workers=replicas)], 1

        if spec.gpu > self.n_gpus_per_node:
            if spec.gpu % self.n_gpus_per_node != 0:
                raise ValueError(
                    "Ray multi-node instances must use an integer number of nodes. "
                    f"Requesting {spec.gpu} GPUs but each node has {self.n_gpus_per_node}."
                )
            nodes_per_worker = spec.gpu // self.n_gpus_per_node
            if spec.cpu % nodes_per_worker != 0 or spec.mem % nodes_per_worker != 0:
                raise ValueError(
                    "Ray multi-node instances must evenly split CPU and memory "
                    "across nodes. "
                    f"Requesting cpu={spec.cpu}, mem={spec.mem} across "
                    f"{nodes_per_worker} nodes."
                )
            per_node_cpu = spec.cpu // nodes_per_worker
            per_node_mem = spec.mem // nodes_per_worker
            bundles = []
            plan = []
            for worker_idx in range(replicas):
                for node_rank in range(nodes_per_worker):
                    bundle_index = len(bundles)
                    bundles.append(
                        {
                            "CPU": per_node_cpu,
                            self.ray_device_resource: float(self.n_gpus_per_node),
                            "memory": per_node_mem * 1024**3,
                        }
                    )
                    plan.append(
                        dict(
                            bundle_index=bundle_index,
                            worker_idx=worker_idx,
                            node_rank=node_rank,
                            workers=1 if node_rank == 0 else 0,
                        )
                    )
            return bundles, plan, nodes_per_worker

        total_gpus = spec.gpu * replicas
        bundles = []
        plan = []
        remaining_workers = replicas
        while remaining_workers > 0:
            workers_on_node = min(remaining_workers, self.n_gpus_per_node // spec.gpu)
            gpus_on_node = workers_on_node * spec.gpu
            bundle_index = len(bundles)
            bundles.append(
                {
                    "CPU": spec.cpu * workers_on_node,
                    self.ray_device_resource: float(gpus_on_node),
                    "memory": spec.mem * workers_on_node * 1024**3,
                }
            )
            plan.append(
                dict(
                    bundle_index=bundle_index,
                    node_rank=0,
                    workers=workers_on_node,
                    gpus_on_node=gpus_on_node,
                )
            )
            remaining_workers -= workers_on_node

        if (
            total_gpus >= self.n_gpus_per_node
            and total_gpus % self.n_gpus_per_node != 0
        ):
            logger.warning(
                "Ray role uses a partial final node: total_gpus=%s, n_gpus_per_node=%s. "
                "This is allowed for Ray but differs from Slurm's full-node-only policy.",
                total_gpus,
                self.n_gpus_per_node,
            )
        return bundles, plan, 1

    def create_workers(self, job: Job, *args, **kwargs) -> list[str]:
        """Create workers via Ray placement group creation.

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

            # Reuse existing workers - no new Ray launchers created
            worker_ids = [w.worker.id for w in target_workers]
            self._colocated_roles[role] = colocate_role

            logger.info(
                f"Role '{role}' colocated with '{colocate_role}': "
                f"reusing workers {worker_ids}"
            )
            return worker_ids

        if strategy_type != SchedulingStrategyType.separation:
            raise ValueError(f"Unknown scheduling strategy type: {strategy_type}")
        # Non-colocated: calculate nodes needed and create new Ray launchers
        spec = schedulings[0]
        total_gpus = spec.gpu * replicas

        # Calculate resource requirements
        nodes = max(1, (total_gpus + self.n_gpus_per_node - 1) // self.n_gpus_per_node)
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

        launchers = []
        try:
            bundles, plan, nodes_per_worker = self._build_node_plan(replicas, spec)
            pg = self._create_placement_group(role, bundles, self.startup_timeout)
            self._placement_groups[role] = pg

            for item in plan:
                bundle = bundles[item["bundle_index"]]
                gpu_count = int(bundle.get(self.ray_device_resource, 0))
                cpu_count = int(bundle.get("CPU", 0))
                mem_gb = max(1, int(bundle.get("memory", 0) // 1024**3))
                options = create_resource_spec(
                    self.ray_device_resource, cpu_count, gpu_count, mem_gb * 1024**3
                )
                options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=item["bundle_index"],
                    placement_group_capture_child_tasks=True,
                )
                launcher = RayWorkerProcessLauncher.options(**options).remote(
                    role,
                    self._log_path_of(role),
                    self._merged_log_path(),
                    self.ray_device_resource,
                    self.device_control_env_var,
                    spec.env_vars,
                )
                launchers.append((item, launcher))

            node_infos = ray.get(
                [launcher.get_node_info.remote() for _, launcher in launchers],
                timeout=self.startup_timeout,
            )
            ordered = sorted(
                zip(launchers, node_infos, strict=True),
                key=lambda x: (
                    x[1]["host"],
                    min([int(v) for v in x[1]["visible_devices"]] or [0]),
                    x[0][0]["bundle_index"],
                ),
            )
            self._launchers[role] = [launcher for (_, launcher), _ in ordered]

            workers = []
            start_refs = []

            def build_worker_spec(
                worker_idx: int, gpu_devices: list[str], worker_spec: SchedulingSpec
            ) -> dict[str, Any]:
                return dict(
                    role=role,
                    worker_index=worker_idx,
                    gpu_devices=gpu_devices,
                    cmd=worker_spec.cmd,
                    experiment_name=self.experiment_name,
                    trial_name=self.trial_name,
                    name_resolve_type=self.name_resolve_config.type,
                    nfs_record_root=self.name_resolve_config.nfs_record_root,
                    etcd3_addr=self.name_resolve_config.etcd3_addr,
                    fileroot=self.fileroot,
                )

            def build_worker_info(
                worker_idx: int,
                worker_launchers: list[ActorHandle],
                worker_spec: SchedulingSpec,
            ) -> RayWorkerInfo:
                worker_id = f"{role}/{worker_idx}"
                worker = Worker(
                    id=worker_id,
                    ip="",  # Will be discovered
                    worker_ports=[],  # Will be discovered
                    engine_ports=[],
                )
                return RayWorkerInfo(
                    worker=worker,
                    role=role,
                    launchers=worker_launchers,
                    task_index=worker_idx,
                    spec=worker_spec,
                )

            if nodes_per_worker > 1:
                nodes_by_worker: dict[int, list[tuple[int, Any, list[str]]]] = {}
                for (item, launcher), info in ordered:
                    nodes_by_worker.setdefault(item["worker_idx"], []).append(
                        (item["node_rank"], launcher, info["visible_devices"])
                    )
                for worker_idx in range(replicas):
                    node_group = sorted(nodes_by_worker[worker_idx], key=lambda x: x[0])
                    _, head_launcher, head_visible_devices = node_group[0]
                    worker_spec = schedulings[worker_idx]
                    start_refs.append(
                        head_launcher.start_workers.remote(
                            [
                                build_worker_spec(
                                    worker_idx,
                                    head_visible_devices,
                                    worker_spec,
                                )
                            ]
                        )
                    )
                    workers.append(
                        build_worker_info(
                            worker_idx,
                            [launcher for _, launcher, _ in node_group],
                            worker_spec,
                        )
                    )

            else:
                next_worker_idx = 0
                for (item, launcher), info in ordered:
                    visible = info["visible_devices"]
                    workers_on_node = item["workers"]
                    batch = []
                    for local_idx in range(workers_on_node):
                        worker_idx = next_worker_idx
                        next_worker_idx += 1
                        start = local_idx * max(1, spec.gpu)
                        end = start + max(1, spec.gpu)
                        gpu_devices = visible[start:end] if spec.gpu > 0 else []
                        worker_spec = schedulings[worker_idx]
                        batch.append(
                            build_worker_spec(worker_idx, gpu_devices, worker_spec)
                        )
                        workers.append(
                            build_worker_info(worker_idx, [launcher], worker_spec)
                        )
                    if batch:
                        start_refs.append(launcher.start_workers.remote(batch))

            ray.get(
                start_refs,
                timeout=self.startup_timeout,
            )

            self._workers[role] = workers
            worker_ids = [worker_info.worker.id for worker_info in workers]

            logger.info(f"Created {replicas} workers for role '{role}' with Ray")
        except Exception as e:
            if role in self._launchers:
                self._stop_launchers(role, timeout=10)
                del self._launchers[role]
            if role in self._placement_groups:
                remove_placement_group(self._placement_groups[role])
                del self._placement_groups[role]
            if isinstance(e, WorkerCreationError):
                raise
            logs = self._read_log_tail(role)
            raise WorkerCreationError(
                role,
                "Ray worker creation failed",
                f"{type(e).__name__}: {e}\nLogs:\n{logs}",
            ) from e

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
                # Forked workers already have known endpoints and are configured during creation.
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

        logger.info(
            f"Waiting for {len(workers)} workers of role '{role}' to be ready..."
        )

        while time.time() - start_time < timeout:
            # Check job status
            try:
                self._check_worker_process_status(role)
            except WorkerFailedError:
                raise

            if any(not w.worker.worker_ports for w in workers):
                self._discover_worker_network(role)

            # Wait for all to be discovered
            discovered_count = sum(1 for w in workers if w.worker.worker_ports)
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
                    run_async_task(self._configure_workers, workers)

                return [w.worker for w in workers]

            # Log progress
            if ready_workers:
                logger.debug(f"{len(ready_workers)}/{len(workers)} workers ready")

            time.sleep(self.health_check_interval)

        raise WorkerTimeoutError(role, timeout)

    def _destroy_engines_on_workers(
        self, workers: list[RayWorkerInfo], timeout: float = 30.0
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
        """Delete workers and stop Ray launchers.

        Teardown follows a two-phase protocol analogous to the Ray and Local
        schedulers:

        1. **Engine destroy** – call ``engine.destroy()`` on every worker via
           HTTP concurrently.  This runs the engine-side CPU barrier and
           ``dist.destroy_process_group`` so that NCCL communicators and the
           TCPStore are shut down cleanly on all ranks.
        2. **Launcher stop** – stop the Ray-managed launcher actors.  At this
           point process groups are already torn down, so killing the
           processes will not produce spurious ``TCPStore.recvValue failed``
           warnings.

        Parameters
        ----------
        role : str, optional
            Role to delete. If None, deletes all roles.
        reverse_order : bool, optional
            Accepted for API compatibility with other schedulers but ignored
            here: Ray launchers tear down worker processes by launcher group,
            so per-rank ordering cannot be enforced globally.
        """
        del reverse_order  # unused, see docstring
        if role is None:
            # Delete colocated/forked roles first (they don't own Ray launchers)
            colocated_roles = list(self._colocated_roles.keys())
            for r in colocated_roles:
                self.delete_workers(r)
            # Then delete actual worker roles
            for r in list(self._workers.keys()):
                self.delete_workers(r)
            return

        # Handle colocated/forked role
        if role in self._colocated_roles:
            target_role = self._colocated_roles[role]
            # Forked roles have their own workers that need cleanup
            if role in self._workers:
                logger.info(f"Removing forked role '{role}' (managed by parent worker)")
                try:
                    run_async_task(
                        self._cleanup_forked_workers_async,
                        role,
                        target_role,
                        self._workers[role],
                    )
                except Exception as e:
                    logger.warning(f"Failed to cleanup forked role '{role}': {e}")
                del self._workers[role]
            else:
                logger.info(f"Removing colocated role '{role}' mapping")
            del self._colocated_roles[role]
            return

        if role not in self._workers:
            logger.warning(f"Role '{role}' not found, skipping deletion")
            return

        workers = self._workers[role]
        logger.info(f"Deleting {len(workers)} workers for role '{role}'")

        # Phase 1: destroy engines so that the CPU barrier and
        # dist.destroy_process_group complete on every rank.
        self._destroy_engines_on_workers(workers)

        # Phase 2: stop the Ray launchers. Process groups are already torn
        # down, so stopping actors will not cause TCPStore race conditions.
        self._stop_launchers(role, timeout=30)
        for launcher in self._launchers.get(role, []):
            try:
                ray.kill(launcher, no_restart=True)
            except Exception:
                pass

        if role in self._placement_groups:
            try:
                remove_placement_group(self._placement_groups[role])
            except Exception as e:
                logger.warning(f"Failed to remove placement group for role {role}: {e}")

        # Clean up internal state
        del self._workers[role]
        self._launchers.pop(role, None)
        self._placement_groups.pop(role, None)

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
            self._check_worker_process_status(worker_info.role)
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
            self._check_worker_process_status(worker_info.role)
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
        """Call a method on an engine instance (synchronous)."""
        return run_async_task(
            self.async_call_engine,
            worker_id,
            method,
            engine_name,
            *args,
            rpc_meta=rpc_meta,
            http_timeout=http_timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            **kwargs,
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

        if method == "launch_server" and len(worker_info.launchers) > 1:
            kwargs = await self._multi_node_rollout.prepare_launch_server(
                worker_info, engine_name, kwargs
            )
            try:
                return await self._async_call_engine_rpc(
                    worker_info,
                    worker_id,
                    method,
                    engine_name,
                    args,
                    kwargs,
                    rpc_meta,
                    http_timeout,
                    max_retries,
                    retry_delay,
                )
            except BaseException:
                await self._multi_node_rollout.stop_backend_workers(
                    worker_info, engine_name
                )
                raise

        try:
            return await self._async_call_engine_rpc(
                worker_info,
                worker_id,
                method,
                engine_name,
                args,
                kwargs,
                rpc_meta,
                http_timeout,
                max_retries,
                retry_delay,
            )
        finally:
            if (
                method in ("destroy", "teardown_server")
                and len(worker_info.launchers) > 1
            ):
                await self._multi_node_rollout.stop_backend_workers(
                    worker_info, engine_name
                )

    async def _async_call_engine_rpc(
        self,
        worker_info: RayWorkerInfo,
        worker_id: str,
        method: str,
        engine_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        rpc_meta: dict[str, Any] | None,
        http_timeout: float,
        max_retries: int,
        retry_delay: float,
    ) -> Any:
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
                self._check_worker_process_status(worker_info.role)
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
                self._check_worker_process_status(worker_info.role)
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

    def __del__(self):
        try:
            self.delete_workers()
        except Exception:
            pass
