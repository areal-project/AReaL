# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

import aiohttp
import orjson
import ray
import ray.exceptions
import requests
from ray.runtime_env import RuntimeEnv
from ray.util.placement_group import (
    PlacementGroup,
    remove_placement_group,
)
from ray.util.scheduling_strategies import (
    NodeAffinitySchedulingStrategy,
    PlacementGroupSchedulingStrategy,
)

from areal.api import Job, Scheduler, Worker
from areal.api.cli_args import (
    BaseExperimentConfig,
    SchedulingSpec,
    SchedulingStrategyType,
)
from areal.infra.rpc.ray_http_worker_manager import RayHTTPWorkerManager
from areal.infra.rpc.ray_rpc_server import RayRPCServer
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
from areal.infra.utils.http import get_default_connector
from areal.infra.utils.launcher import get_env_vars, get_thread_env_vars
from areal.infra.utils.ray import get_placement_group_master_ip_and_port
from areal.infra.utils.ray_placement_group import (
    DeferredDeviceRayPlacementStrategy,
    RayPlacementStrategy,
    SeparatedRayPlacementStrategy,
    SharedRayPlacementStrategy,
    ray_resource_type,
)
from areal.utils import logging
from areal.utils.network import format_hostport
from areal.utils.offload import get_tms_env_vars

logger = logging.getLogger("RayScheduler")


@dataclass
class RayWorkerInfo:
    """RayScheduler worker metadata.

    For ``ray_rpc`` workers, ``actor`` is the RayRPCServer that serves engine
    control calls directly over Ray actor RPC.

    For ``http_server`` workers, ``actor`` is the RayHTTPWorkerManager that
    manages the HTTP subprocess lifecycle; ``worker.ip`` and
    ``worker.worker_ports[0]`` are the actual ProxyRolloutServer endpoint
    used for scheduler control calls and agent traffic.
    """

    worker: Worker
    actor: ray.actor.ActorHandle
    role: str
    placement_group: PlacementGroup
    bundle_index: int | None
    created_at: float
    env_vars: dict[str, str] = field(default_factory=dict)
    worker_kind: Literal["ray_rpc", "http_server"] = "ray_rpc"
    target_worker_id: str | None = None
    target_node_id: str | None = None


class RayScheduler(Scheduler):
    def __init__(
        self,
        startup_timeout: float = 30.0,
        *,
        exp_config: BaseExperimentConfig | None = None,
        n_gpus_per_node: int = 8,
    ):
        self.exp_config = exp_config
        self._n_gpus_per_node = n_gpus_per_node
        self.startup_timeout = startup_timeout
        self.enable_tms_offload = False
        if exp_config is not None:
            self.enable_tms_offload = exp_config.enable_offload
            self._n_gpus_per_node = exp_config.cluster.n_gpus_per_node

        self._workers: dict[str, list[RayWorkerInfo]] = defaultdict(list)
        self._worker_info_by_id: dict[str, RayWorkerInfo] = {}
        self._placement_groups: list[PlacementGroup] = []

        # Colocation tracking: colocated roles reuse workers from target role
        self._colocated_roles: dict[str, str] = {}  # colocated_role -> target_role

    @property
    def n_gpus_per_node(self) -> int:
        return self._n_gpus_per_node

    def _prepare_worker_specs(
        self, role: str, num_workers: int, schedulings: list[SchedulingSpec] | None
    ) -> list[SchedulingSpec]:
        if not schedulings:
            raise WorkerCreationError(
                role, "Invalid configuration", "Tasks SchedulingSpec must be provided"
            )
        if len(schedulings) == 1:
            return [schedulings[0]] * num_workers

        if len(schedulings) == num_workers:
            return schedulings

        raise WorkerCreationError(
            role,
            "Invalid Configuration",
            f"schedulings length ({len(schedulings)}) must be 1 or equal to replicas ({num_workers})",
        )

    def _ping_workers(self, role: str, timeout: float | None = None):
        worker_info_list = self._workers[role]
        timeout = timeout if timeout is not None else self.startup_timeout
        http_workers = [
            wi for wi in worker_info_list if wi.worker_kind == "http_server"
        ]
        for wi in http_workers:
            self._ping_http_worker(wi, timeout)

        ray_workers = [wi for wi in worker_info_list if wi.worker_kind == "ray_rpc"]
        refs = [wi.actor.ping.remote() for wi in ray_workers]
        ref_to_worker = {ref: wi for wi, ref in zip(ray_workers, refs)}

        pending = refs
        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=timeout)
            # ray.wait timed out
            if len(ready) == 0:
                raise WorkerTimeoutError(role, timeout)

            ref = ready[0]

            try:
                # get to determine if this is a failed actor
                ray.get(ref)
            except ray.exceptions.GetTimeoutError:
                failed_worker = ref_to_worker[ref]
                raise WorkerTimeoutError(failed_worker.worker.id, timeout)
            except ray.exceptions.RayActorError:
                failed_worker = ref_to_worker[ref]
                raise WorkerFailedError(failed_worker.worker.id, -1)

    def _ping_http_worker(self, wi: RayWorkerInfo, timeout: float) -> None:
        """Validate manager/target fate-sharing for managed HTTP workers."""
        try:
            if wi.target_worker_id is None or wi.target_node_id is None:
                raise RuntimeError("HTTP worker is missing target worker metadata")
            target_wi = self._worker_info_by_id.get(wi.target_worker_id)
            if target_wi is None:
                raise RuntimeError(f"Target worker {wi.target_worker_id} not found")

            ray.get(target_wi.actor.ping.remote(), timeout=timeout)
            target_node_id = ray.get(
                target_wi.actor.get_node_id.remote(), timeout=timeout
            )
            if target_node_id != wi.target_node_id:
                raise RuntimeError(
                    f"Target worker moved from node {wi.target_node_id} to "
                    f"{target_node_id}"
                )

            launcher_node_id = ray.get(wi.actor.get_node_id.remote(), timeout=timeout)
            if launcher_node_id != wi.target_node_id:
                raise RuntimeError(
                    f"HTTP launcher moved from node {wi.target_node_id} to "
                    f"{launcher_node_id}"
                )
            ray.get(wi.actor.ping.remote(), timeout=timeout)
        except ray.exceptions.GetTimeoutError as e:
            self._fail_http_worker(wi, f"health check timed out: {e}")
        except Exception as e:
            self._fail_http_worker(wi, str(e))

    def _fail_http_worker(self, wi: RayWorkerInfo, reason: str) -> None:
        logger.warning("HTTP worker %s failed: %s", wi.worker.id, reason)
        self._cleanup_forked_workers([wi])
        self._worker_info_by_id.pop(wi.worker.id, None)
        workers = self._workers.get(wi.role)
        if workers is not None:
            self._workers[wi.role] = [w for w in workers if w.worker.id != wi.worker.id]
        raise WorkerFailedError(wi.worker.id, -1, reason)

    def _build_env_vars(self, spec: SchedulingSpec) -> dict[str, str]:
        """Helper to build environment variables for a worker."""
        additional_envs_str = None
        if spec.env_vars:
            additional_envs_str = ",".join(f"{k}={v}" for k, v in spec.env_vars.items())
        env = get_env_vars(additional_envs_str)
        if self.enable_tms_offload:
            env.update(get_tms_env_vars())
        thread_env = get_thread_env_vars(
            cpus_per_task=spec.cpu,
            existing_env_vars=spec.env_vars,
        )
        env.update(thread_env)
        return env

    def _get_placement_strategy(
        self, schedulings: list[SchedulingSpec]
    ) -> RayPlacementStrategy:
        placement_strategies = [spec.ray_placement_strategy for spec in schedulings]

        if not all(ps == placement_strategies[0] for ps in placement_strategies):
            raise RuntimeError(
                f"Not every placement strategy in scheduling spec is the same: {placement_strategies}"
            )

        mode = placement_strategies[0]

        strategy_map = {
            "deferred": DeferredDeviceRayPlacementStrategy,
            "separate": SeparatedRayPlacementStrategy,
            "shared": SharedRayPlacementStrategy,
        }
        if mode in strategy_map:
            return strategy_map[mode]()
        else:
            raise RuntimeError(f"Ray scheduling mode {mode} is not supported")

    def _create_ray_workers(
        self, role: str, schedulings: list[SchedulingSpec]
    ) -> tuple[list[RayWorkerInfo], list[str]]:
        """Create Ray workers with individual placement groups per worker.

        Each worker gets its own placement group with exclusive GPU access.
        This ensures proper GPU isolation and supports forked workers sharing
        the same PG/GPU.
        """
        worker_info_list: list[RayWorkerInfo] = []
        worker_ids: list[str] = []

        placement_strategy = self._get_placement_strategy(schedulings)
        placement_groups = placement_strategy.create_placement_group(
            role,
            schedulings,
            self.exp_config.cluster.n_gpus_per_node,
            timeout=self.startup_timeout,
        )

        master_ip, master_port = get_placement_group_master_ip_and_port(
            placement_groups[0], placement_group_bundle_index=0
        )

        for idx, spec in enumerate(schedulings):
            options, pg_scheduling_strategy = placement_strategy.actor_resources(spec)
            worker_id = f"{role}/{idx}"
            env = self._build_env_vars(spec)
            actor = RayRPCServer.options(
                **options,
                name=worker_id,
                runtime_env=RuntimeEnv(env_vars=env),
                scheduling_strategy=pg_scheduling_strategy,
            ).remote()

            # 0 needed to pad the list as the trainer takes index 1 for ports
            worker_ports = ["0", str(master_port)]
            worker = Worker(
                id=worker_id, ip=master_ip, worker_ports=worker_ports, engine_ports=[]
            )

            wi = RayWorkerInfo(
                worker=worker,
                actor=actor,
                role=role,
                placement_group=pg_scheduling_strategy.placement_group,
                bundle_index=pg_scheduling_strategy.placement_group_bundle_index,
                created_at=time.time(),
                env_vars=env,
            )
            worker_info_list.append(wi)
            worker_ids.append(worker_id)

        return worker_info_list, worker_ids

    def _create_forked_workers_internal(
        self,
        role: str,
        target_role: str,
        target_workers: list[RayWorkerInfo],
        schedulings,
    ) -> list[str]:
        """Create forked workers on same placement groups as target workers.

        Since each target worker has its own PG with bundle_index=0, forked workers
        share the exact same GPU by using the same PG and bundle_index=0.

        Main workers use num_gpus=0.9, leaving 0.1 for forked workers.
        Using num_gpus=0.01 allows up to 10 forked workers per target if needed.

        Parameters
        ----------
        role : str
            Role name for the forked workers
        target_role : str
            Target role to fork from
        target_workers : list[RayWorkerInfo]
            List of target worker infos to fork from
        schedulings : list[SchedulingSpec]
            Scheduling specs for the forked workers

        Returns
        -------
        list[str]
            List of forked worker IDs
        """

        worker_info_list: list[RayWorkerInfo] = []
        worker_ids: list[str] = []

        for idx, (target_wi, spec) in enumerate(zip(target_workers, schedulings)):
            worker_id = f"{role}/{idx}"

            # Reuse placement group from target worker
            pg = target_wi.placement_group
            bundle_idx = target_wi.bundle_index  # Should always be 0 now

            # Build scheduling strategy for same placement group
            strategy_kwargs: dict[str, Any] = {
                "placement_group": pg,
                "placement_group_capture_child_tasks": True,
                "placement_group_bundle_index": bundle_idx,  # Same as target (0)
            }

            # Use 0.01 GPU to share with target worker (which uses 0.9)
            # This allows multiple forked workers per target if needed
            device = ray_resource_type()
            additional_options = {}
            if spec.gpu > 0:
                if spec.gpu > 1:
                    raise NotImplementedError(
                        "Colocation of multi-GPU workers together is not supported by Ray"
                    )
                if device == "GPU":
                    additional_options = dict(num_gpus=0.01)
                else:
                    additional_options = {"resources": {device: 0.01}}
            actor = RayRPCServer.options(
                **additional_options,
                num_cpus=0,  # Minimal CPU allocation for forked actor
                name=worker_id,
                runtime_env=RuntimeEnv(env_vars=target_wi.env_vars),
                scheduling_strategy=PlacementGroupSchedulingStrategy(**strategy_kwargs),
            ).remote()

            # Build Worker object with same IP/ports as target
            worker_ports = ray.get(
                target_wi.actor.alloc_ports.remote(
                    count=len(target_wi.worker.worker_ports)
                )
            )

            worker = Worker(
                id=worker_id,
                ip=target_wi.worker.ip,
                worker_ports=worker_ports,
                engine_ports=[],
            )

            wi = RayWorkerInfo(
                worker=worker,
                actor=actor,
                role=role,
                placement_group=pg,  # Same PG as target
                bundle_index=bundle_idx,
                created_at=time.time(),
                env_vars=target_wi.env_vars.copy(),
            )
            worker_info_list.append(wi)
            worker_ids.append(worker_id)

        # Register forked workers
        self._workers[role] = worker_info_list
        for wi in worker_info_list:
            self._worker_info_by_id[wi.worker.id] = wi

        # Ping forked workers to ensure they're ready
        self._ping_workers(role, self.startup_timeout)

        # Configure if exp_config available
        if self.exp_config is not None:
            for rank, wi in enumerate(worker_info_list):
                try:
                    wi.actor.configure.remote(self.exp_config, wi.role, rank)
                except Exception as e:
                    logger.error(
                        f"Configure failed on forked worker {wi.worker.id}: {e}",
                        exc_info=True,
                    )
                    self._cleanup_forked_workers(worker_info_list)
                    raise WorkerCreationError(
                        role, "Forked worker configuration failed", str(e)
                    )

        logger.info(
            f"Role '{role}' forked from '{target_role}': "
            f"created {len(worker_ids)} new actors on same placement groups"
        )

        return worker_ids

    def _create_managed_http_workers(
        self,
        role: str,
        target_role: str,
        target_workers: list[RayWorkerInfo],
        command: str,
    ) -> list[str]:
        """Create HTTP workers managed by Ray actors colocated with targets.

        This is the RayScheduler equivalent of LocalScheduler's parent guard
        ``/fork`` path: it creates a small Ray manager actor on the target
        Ray node, and that manager owns the HTTP worker subprocess lifecycle.
        The manager is not a generic RayRPC worker and does not proxy engine
        control calls; those calls go directly to the HTTP worker endpoint.
        """
        if self.exp_config is None:
            raise WorkerCreationError(
                role,
                "Missing experiment config",
                "Ray HTTP workers require exp_config to pass experiment/trial, "
                "name_resolve, and fileroot settings to the launched server.",
            )

        worker_info_list: list[RayWorkerInfo] = []
        worker_ids: list[str] = []
        launched_manager_actor: ray.actor.ActorHandle | None = None

        try:
            name_resolve_config = self.exp_config.cluster.name_resolve
            for idx, target_wi in enumerate(target_workers):
                launched_manager_actor = None
                worker_id = f"{role}/{idx}"
                target_node_id = ray.get(
                    target_wi.actor.get_node_id.remote(), timeout=self.startup_timeout
                )
                worker_ports = ray.get(
                    target_wi.actor.alloc_ports.remote(count=1),
                    timeout=self.startup_timeout,
                )
                port = int(worker_ports[0])

                manager_actor = RayHTTPWorkerManager.options(
                    num_cpus=0,
                    max_restarts=0,
                    max_task_retries=0,
                    name=worker_id,
                    runtime_env=RuntimeEnv(env_vars=target_wi.env_vars.copy()),
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=target_node_id,
                        soft=False,
                    ),
                ).remote()
                launched_manager_actor = manager_actor

                manager_node_id = ray.get(
                    manager_actor.get_node_id.remote(), timeout=self.startup_timeout
                )
                if manager_node_id != target_node_id:
                    raise WorkerCreationError(
                        role,
                        "HTTP manager placement mismatch",
                        f"worker={worker_id}, target_node={target_node_id}, "
                        f"manager_node={manager_node_id}",
                    )

                launch_info = ray.get(
                    manager_actor.launch.remote(
                        module=command,
                        host="0.0.0.0",
                        port=port,
                        experiment_name=str(self.exp_config.experiment_name),
                        trial_name=str(self.exp_config.trial_name),
                        role=role,
                        worker_index=idx,
                        name_resolve_type=name_resolve_config.type,
                        nfs_record_root=name_resolve_config.nfs_record_root,
                        etcd3_addr=name_resolve_config.etcd3_addr,
                        fileroot=str(self.exp_config.cluster.fileroot),
                        env=target_wi.env_vars.copy(),
                        startup_timeout=self.startup_timeout,
                    ),
                    timeout=self.startup_timeout + 30.0,
                )

                http_worker = Worker(
                    id=worker_id,
                    ip=str(launch_info["host"]),
                    worker_ports=[str(launch_info["port"])],
                    engine_ports=[],
                )
                wi = RayWorkerInfo(
                    worker=http_worker,
                    actor=manager_actor,
                    role=role,
                    placement_group=target_wi.placement_group,
                    bundle_index=target_wi.bundle_index,
                    created_at=time.time(),
                    env_vars=target_wi.env_vars.copy(),
                    worker_kind="http_server",
                    target_worker_id=target_wi.worker.id,
                    target_node_id=target_node_id,
                )
                worker_info_list.append(wi)
                worker_ids.append(worker_id)
                launched_manager_actor = None

            self._workers[role] = worker_info_list
            for wi in worker_info_list:
                self._worker_info_by_id[wi.worker.id] = wi

            self._ping_workers(role, self.startup_timeout)

            for rank, wi in enumerate(worker_info_list):
                self._configure_http_worker(wi, rank)

        except Exception as e:
            if launched_manager_actor is not None:
                try:
                    ray.get(launched_manager_actor.destroy.remote(), timeout=10.0)
                except Exception:
                    try:
                        ray.kill(launched_manager_actor, no_restart=True)
                    except Exception:
                        pass
            self._cleanup_forked_workers(worker_info_list)
            self._workers.pop(role, None)
            for worker_id in worker_ids:
                self._worker_info_by_id.pop(worker_id, None)
            if isinstance(e, SchedulerError):
                raise
            raise WorkerCreationError(
                role, "HTTP worker creation failed", str(e)
            ) from e

        logger.info(
            f"Role '{role}' forked from '{target_role}': "
            f"created {len(worker_ids)} HTTP workers with Ray managers"
        )
        return worker_ids

    def _http_worker_url(self, wi: RayWorkerInfo, endpoint: str) -> str:
        port = int(wi.worker.worker_ports[0])
        return f"http://{format_hostport(wi.worker.ip, port)}{endpoint}"

    def _extract_response_error(self, data: Any) -> str:
        if isinstance(data, dict):
            detail = data.get("error") or data.get("detail")
            if detail is not None:
                return str(detail)
        return "Unknown error"

    async def _read_aiohttp_error(self, response: aiohttp.ClientResponse) -> str:
        try:
            return self._extract_response_error(await response.json())
        except Exception:
            return await response.text()

    def _read_requests_error(self, response: requests.Response) -> str:
        try:
            return self._extract_response_error(response.json())
        except Exception:
            return response.text

    def _configure_http_worker(self, wi: RayWorkerInfo, worker_rank: int) -> None:
        if self.exp_config is None:
            return
        worker_id = wi.worker.id
        url = self._http_worker_url(wi, "/configure")
        try:
            response = requests.post(
                url,
                data=orjson.dumps(
                    serialize_value(
                        dict(
                            config=self.exp_config,
                            role=wi.role,
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
            raise WorkerConfigurationError(
                worker_id,
                self._read_requests_error(response),
                str(response.status_code),
            )
        except requests.exceptions.ConnectionError as e:
            port = int(wi.worker.worker_ports[0])
            raise RPCConnectionError(worker_id, wi.worker.ip, port, str(e)) from e
        except requests.exceptions.Timeout as e:
            raise WorkerConfigurationError(worker_id, f"Request timed out: {e}") from e

    async def _set_http_worker_env(
        self, wi: RayWorkerInfo, env: dict[str, str]
    ) -> None:
        """Set environment variables on a real HTTP worker endpoint.

        ``wi.actor`` is only the RayHTTPWorkerManager lifecycle owner. Scheduler
        control requests must go directly to ``wi.worker.ip:worker_ports[0]``.
        """
        worker_id = wi.worker.id
        port = int(wi.worker.worker_ports[0])
        url = self._http_worker_url(wi, "/set_env")
        try:
            timeout = aiohttp.ClientTimeout(total=30.0)
            async with aiohttp.ClientSession(
                timeout=timeout,
                connector=get_default_connector(),
            ) as session:
                async with session.post(
                    url,
                    data=orjson.dumps({"env": env}),
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        wi.env_vars.update(env)
                        return
                    detail = await self._read_aiohttp_error(response)
                    raise SchedulerError(
                        f"Failed to set env on worker {worker_id}: "
                        f"HTTP {response.status}: {detail}"
                    )
        except (aiohttp.ClientConnectionError, aiohttp.ClientConnectorError) as e:
            try:
                self._ping_http_worker(wi, 5.0)
            except WorkerFailedError:
                raise
            raise RPCConnectionError(worker_id, wi.worker.ip, port, str(e)) from e
        except TimeoutError as e:
            raise SchedulerError(f"set_env timed out on worker {worker_id}: {e}") from e

    async def _create_engine_on_http_worker(
        self,
        wi: RayWorkerInfo,
        engine: str,
        engine_name: str | None,
        *args,
        **kwargs,
    ) -> Any:
        """Create an engine inside the real HTTP worker endpoint."""
        worker_id = wi.worker.id
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
        port = int(wi.worker.worker_ports[0])
        url = self._http_worker_url(wi, "/create_engine")
        try:
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
                        return result.get("result")
                    error_detail = await self._read_aiohttp_error(response)
                    if response.status == 400 and "Failed to import" in error_detail:
                        raise EngineImportError(engine, error_detail)
                    raise EngineCreationError(worker_id, error_detail, response.status)
        except (aiohttp.ClientConnectionError, aiohttp.ClientConnectorError) as e:
            try:
                self._ping_http_worker(wi, 5.0)
            except WorkerFailedError:
                raise
            raise RPCConnectionError(worker_id, wi.worker.ip, port, str(e)) from e
        except TimeoutError as e:
            raise EngineCreationError(worker_id, f"Request timed out: {e}") from e

    def _call_http_worker_engine(
        self,
        wi: RayWorkerInfo,
        method: str,
        engine_name: str | None,
        *args,
        rpc_meta: dict[str, Any] | None = None,
        http_timeout: float = 7200.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs,
    ) -> Any:
        """Call an engine method through the real HTTP worker endpoint."""
        worker_id = wi.worker.id
        if engine_name is None:
            engine_name = worker_id
        payload = {
            "method": method,
            "engine_name": engine_name,
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
            "rpc_meta": rpc_meta,
        }
        url = self._http_worker_url(wi, "/call")
        last_error: str | None = None

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=http_timeout)
                if response.status_code == 200:
                    result = response.json()
                    if attempt > 1:
                        logger.info(
                            f"Method '{method}' on '{worker_id}' "
                            f"succeeded after {attempt} attempts"
                        )
                    return deserialize_value(result.get("result"))

                error_detail = self._read_requests_error(response)
                if response.status_code == 503:
                    last_error = "Service unavailable (503)"
                elif (
                    response.status_code == 500
                    and attempt < max_retries
                    and "timeout" in error_detail.lower()
                ):
                    last_error = f"Engine method timeout: {error_detail}"
                else:
                    raise EngineCallError(
                        worker_id,
                        method,
                        f"HTTP {response.status_code}: {error_detail}",
                        attempt=attempt,
                    )
            except requests.exceptions.Timeout as e:
                last_error = f"Request timeout: {e}"
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {e}"
                try:
                    self._ping_http_worker(wi, min(http_timeout, 5.0))
                except WorkerFailedError:
                    raise
            except EngineCallError:
                raise
            except Exception as e:
                last_error = f"Unexpected error: {e}"

            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Method '{method}' failed on worker '{worker_id}' "
                    f"(attempt {attempt}/{max_retries}): {last_error}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        raise EngineCallError(
            worker_id,
            method,
            last_error or "Max retries exceeded",
            attempt=max_retries,
        )

    async def _async_call_http_worker_engine(
        self,
        wi: RayWorkerInfo,
        method: str,
        engine_name: str | None,
        *args,
        rpc_meta: dict[str, Any] | None = None,
        http_timeout: float = 7200.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs,
    ) -> Any:
        """Async-call an engine method through the real HTTP worker endpoint."""
        worker_id = wi.worker.id
        if engine_name is None:
            engine_name = worker_id
        payload = {
            "method": method,
            "engine_name": engine_name,
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
            "rpc_meta": rpc_meta,
        }
        url = self._http_worker_url(wi, "/call")
        last_error: str | None = None

        for attempt in range(1, max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(
                    total=http_timeout,
                    sock_connect=http_timeout,
                    connect=http_timeout,
                )
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
                            if attempt > 1:
                                logger.info(
                                    f"Method '{method}' on '{worker_id}' "
                                    f"succeeded after {attempt} attempts"
                                )
                            return deserialize_value(result.get("result"))

                        error_detail = await self._read_aiohttp_error(response)
                        if response.status == 503:
                            last_error = "Service unavailable (503)"
                        elif (
                            response.status == 500
                            and attempt < max_retries
                            and "timeout" in error_detail.lower()
                        ):
                            last_error = f"Engine method timeout: {error_detail}"
                        else:
                            raise EngineCallError(
                                worker_id,
                                method,
                                f"HTTP {response.status}: {error_detail}",
                                attempt=attempt,
                            )
            except (aiohttp.ClientConnectionError, aiohttp.ClientConnectorError) as e:
                last_error = f"Connection error: {e}"
                try:
                    self._ping_http_worker(wi, min(http_timeout, 5.0))
                except WorkerFailedError:
                    raise
            except TimeoutError as e:
                last_error = f"Request timeout: {e}"
            except EngineCallError:
                raise
            except Exception as e:
                last_error = f"Unexpected error: {e}"

            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Method '{method}' failed on worker '{worker_id}' "
                    f"(attempt {attempt}/{max_retries}): {last_error}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

        raise EngineCallError(
            worker_id,
            method,
            last_error or "Max retries exceeded",
            attempt=max_retries,
        )

    def _cleanup_forked_workers(self, workers: list[RayWorkerInfo]):
        """Clean up forked workers without removing placement groups.

        Unlike _cleanup_workers, this doesn't remove placement groups since
        forked workers share placement groups with target workers.

        Teardown is done in two phases so that peer ranks can finish their
        pre-destroy CPU barrier inside ``engine.destroy()`` before any actor
        process is forcibly killed:

        1. Dispatch ``actor.destroy.remote()`` on every actor concurrently
           and collect the ObjectRefs (fire but *don't* forget).
        2. ``ray.wait`` on all of them with a bounded timeout so that all
           ranks return together. Only then do we drop references / kill
           stragglers.
        """
        # Phase 1: concurrently dispatch destroy on all actors.
        destroy_refs: list[tuple[RayWorkerInfo, Any]] = []
        for wi in workers:
            try:
                ref = wi.actor.destroy.remote()
                destroy_refs.append((wi, ref))
            except Exception:
                logger.warning(
                    f"Could not dispatch destroy on forked actor {wi.actor}, "
                    f"force killing actor"
                )
                ray.kill(wi.actor, no_restart=True)

        # Phase 2: wait for all destroys to finish (bounded). This lets the
        # engine-side pre-destroy CPU barrier complete on every rank before
        # we release references.
        if destroy_refs:
            refs = [r for _, r in destroy_refs]
            try:
                ray.wait(refs, num_returns=len(refs), timeout=30.0)
            except Exception as e:
                logger.warning(f"ray.wait on forked destroy refs failed: {e}")

            # Surface per-actor failures; force-kill any that did not finish.
            for wi, ref in destroy_refs:
                try:
                    ray.get(ref, timeout=0)
                except ray.exceptions.GetTimeoutError:
                    logger.warning(
                        f"Forked actor {wi.actor} did not finish destroy in time, "
                        f"force killing"
                    )
                    try:
                        ray.kill(wi.actor, no_restart=True)
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(
                        f"Forked actor {wi.actor} destroy raised "
                        f"{type(e).__name__}: {e}"
                    )

        for wi in workers:
            # Remove from worker_info_by_id
            self._worker_info_by_id.pop(wi.worker.id, None)

    def create_workers(self, job: Job, *args, **kwargs) -> list[str]:
        """
        Create worker actors.

        Parameters
        --------
        job: Job
            Job configuration with role, replicas, tasks, scheduling strategy
        *args
            Additional arguments (UNUSED)
        **kwargs
            Additional keyword arguments (UNUSED)

        Returns
        --------
        list[str]
            List of worker IDs created (e.g., ["rollout/0", "rollout/1])

        Raises
        --------
        WorkerCreationError
            If worker creation fails
        """
        role = job.role
        if role in self._workers or role in self._colocated_roles:
            raise WorkerCreationError(
                role,
                "Worker group already exists",
                f"Use delete_workers('{role}') first to remove existing workers.",
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
                # Fork mode: spawn new actors on same placement groups
                worker_ids = self._create_forked_workers_internal(
                    role, colocate_role, target_workers, schedulings
                )
                self._colocated_roles[role] = colocate_role
                return worker_ids

            # Reuse existing workers - no new actors spawned
            worker_ids = [w.worker.id for w in target_workers]
            self._colocated_roles[role] = colocate_role

            logger.info(
                f"Role '{role}' colocated with '{colocate_role}': "
                f"reusing workers {worker_ids}"
            )
            return worker_ids

        if strategy_type != SchedulingStrategyType.separation:
            raise ValueError(f"Unknown scheduling strategy type: {strategy_type}")
        # Non-colocated: spawn new worker actors
        worker_info_list, worker_ids = self._create_ray_workers(role, schedulings)

        self._workers[role].extend(worker_info_list)

        for wi in worker_info_list:
            self._worker_info_by_id[wi.worker.id] = wi

        self._ping_workers(role, self.startup_timeout)

        if self.exp_config is not None:
            for rank, wi in enumerate(worker_info_list):
                try:
                    wi.actor.configure.remote(self.exp_config, wi.role, rank)
                except Exception as e:
                    logger.error(
                        f"Configure failed on worker {wi.worker.id}: {e}", exc_info=True
                    )
                    self._cleanup_workers(worker_info_list)
                    raise WorkerCreationError(
                        role, "Worker configuration failed", str(e)
                    )

        return worker_ids

    def get_workers(self, role: str, timeout: float | None = None) -> list[Worker]:
        # Check if this is a colocated role
        if role in self._colocated_roles:
            # If forked role (has its own workers), use those
            if role in self._workers:
                worker_info_list = self._workers[role]
                self._ping_workers(role, timeout)
                return [wi.worker for wi in worker_info_list]
            # Otherwise delegate to target role
            target_role = self._colocated_roles[role]
            return self.get_workers(target_role, timeout)

        if role not in self._workers:
            raise WorkerNotFoundError(role)

        worker_info_list = self._workers[role]

        self._ping_workers(role, timeout)

        return [wi.worker for wi in worker_info_list]

    def delete_workers(self, role: str | None = None, reverse_order: bool = False):
        """
        Delete workers and clean up resources

        Parameters
        --------
        role: str, optional
            Specific worker role to delete, or None to delete all
        reverse_order: bool, optional
            If True, iterate workers in reverse rank order when issuing
            ``actor.destroy.remote()`` so that rank-0 is signalled last.
            Note: Ray kills are asynchronous, so ordering here is best-effort.
        """
        if role is None:
            # Delete colocated roles first (they're just mappings)
            colocated_roles = list(self._colocated_roles.keys())
            for r in colocated_roles:
                self.delete_workers(r, reverse_order=reverse_order)
            # Then delete actual worker roles
            roles = list(self._workers.keys())
            for r in roles:
                self.delete_workers(r, reverse_order=reverse_order)
            return

        # Handle colocated role
        if role in self._colocated_roles:
            # Check if this is a forked role (has its own workers)
            if role in self._workers:
                # Forked role: clean up the spawned actors (but not placement groups)
                workers = self._workers[role]
                logger.info(
                    f"Cleaning up {len(workers)} forked actors for role '{role}'"
                )
                if reverse_order:
                    workers = list(reversed(workers))
                self._cleanup_forked_workers(workers)
                del self._workers[role]
            else:
                logger.info(f"Removing colocated role '{role}' mapping")
            # Remove colocated mapping
            del self._colocated_roles[role]
            return

        child_roles = [
            child_role
            for child_role, target in list(self._colocated_roles.items())
            if target == role
        ]
        for child_role in child_roles:
            self.delete_workers(child_role, reverse_order=reverse_order)

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

    def fork_workers(
        self,
        role: str,
        target_role: str,
        command: str | None = None,
    ) -> list[str]:
        """Fork new workers from existing workers.

        Without ``command`` this creates RayRPCServer actors colocated by
        placement group. With ``command`` this creates manager actors bound to
        the target actor's Ray node and launches the requested HTTP module as a
        managed subprocess.
        """
        if role in self._workers or role in self._colocated_roles:
            raise WorkerCreationError(
                role,
                "Worker group already exists",
                f"Use delete_workers('{role}') first to remove existing workers.",
            )

        if target_role not in self._workers:
            raise WorkerNotFoundError(f"Target role '{target_role}' not found for fork")
        target_workers = self._workers[target_role]

        if command is not None:
            worker_ids = self._create_managed_http_workers(
                role, target_role, target_workers, command
            )
            self._colocated_roles[role] = target_role
            return worker_ids

        schedulings = []
        for target_wi in target_workers:
            # Use minimal resources for forked workers
            schedulings.append(SchedulingSpec(cpu=0, mem=0, gpu=1, port_count=1))

        worker_ids = self._create_forked_workers_internal(
            role, target_role, target_workers, schedulings
        )
        self._colocated_roles[role] = target_role
        return worker_ids

    def _cleanup_workers(self, workers: list[RayWorkerInfo]):
        """Tear down actors and their placement groups in three phases.

        The ordering matters for distributed teardown correctness:

        1. Dispatch ``actor.destroy.remote()`` on every actor concurrently
           and collect the ObjectRefs. ``destroy`` on the worker side runs
           the engine's pre-destroy CPU barrier + ``dist.destroy_process_group``,
           which requires all peer ranks to still be alive.
        2. ``ray.wait`` on all destroy refs with a bounded timeout so that
           every rank finishes the barrier together. Without this, rank-0
           (TCPStore owner) may be torn down first and cause a noisy
           ``TCPStore.recvValue failed`` on other ranks.
        3. Only after the barrier phase, remove the placement groups. PG
           removal hard-kills any still-alive actor process, so it must
           come last.
        """
        # Phase 1: concurrently dispatch destroy on all actors.
        destroy_refs: list[tuple[RayWorkerInfo, Any]] = []
        for wi in workers:
            try:
                ref = wi.actor.destroy.remote()
                destroy_refs.append((wi, ref))
            except Exception:
                try:
                    wi.actor.__ray_terminate__.remote()
                except Exception:
                    logger.warning(
                        f"Could not destroy remote actor {wi.actor}, "
                        f"force killing actor"
                    )
                    ray.kill(wi.actor, no_restart=True)

        # Phase 2: wait for destroys to finish so the engine-side CPU
        # barrier has a chance to complete on every rank.
        if destroy_refs:
            ref_to_wi = {id(r): wi for wi, r in destroy_refs}
            refs = [r for _, r in destroy_refs]

            ready_refs, remaining_refs = ray.wait(
                refs, num_returns=len(refs), timeout=30.0
            )

            # Completed: check whether destroy raised an exception.
            for ref in ready_refs:
                wi = ref_to_wi[id(ref)]
                try:
                    ray.get(ref)
                except Exception as e:
                    logger.warning(
                        f"Actor {wi.actor} destroy raised {type(e).__name__}: {e}"
                    )

            # Timed-out: force kill actors that did not finish in time.
            for ref in remaining_refs:
                wi = ref_to_wi[id(ref)]
                logger.warning(
                    f"Actor {wi.actor} did not finish destroy in 30s, force killing"
                )
                try:
                    ray.kill(wi.actor, no_restart=True)
                except Exception:
                    pass

        # Phase 3: collect unique placement groups and remove them.
        # This step hard-kills any actor still using the PG, so it MUST
        # come after the barrier phase above.
        unique_pgs = {wi.placement_group for wi in workers}
        for pg in unique_pgs:
            try:
                remove_placement_group(pg)
            except Exception:
                logger.warning(f"Could not remove placement group {pg}")
            if pg in self._placement_groups:
                self._placement_groups.remove(pg)

        for wi in workers:
            self._worker_info_by_id.pop(wi.worker.id, None)

    def _get_worker_info_by_id(self, worker_id: str) -> RayWorkerInfo | None:
        return self._worker_info_by_id.get(worker_id, None)

    async def set_worker_env(self, worker_id: str, env: dict[str, str]) -> None:
        wi = self._get_worker_info_by_id(worker_id)
        if wi is None:
            raise WorkerNotFoundError(worker_id)
        if not env:
            return

        if wi.worker_kind == "http_server":
            return await self._set_http_worker_env(wi, env)

        await wi.actor.set_env.remote(env)
        wi.env_vars.update(env)

    async def create_engine(
        self,
        worker_id: str,
        engine: str,
        engine_name: str | None = None,
        *args,
        **kwargs,
    ) -> Any:
        wi = self._get_worker_info_by_id(worker_id)
        if wi is None:
            raise WorkerNotFoundError(worker_id)

        if wi.worker_kind == "http_server":
            return await self._create_engine_on_http_worker(
                wi, engine, engine_name, *args, **kwargs
            )

        if not isinstance(engine, str):
            raise WorkerCreationError(
                worker_id, f"Engine must be a string import path, got {type(engine)}"
            )
        # Pass engine_name to support multiple engines per worker (colocation)
        await wi.actor.create_engine.remote(
            engine, *args, engine_name=engine_name, **kwargs
        )

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
        wi = self._get_worker_info_by_id(worker_id)
        if wi is None:
            raise WorkerNotFoundError(worker_id)

        if wi.worker_kind == "http_server":
            return self._call_http_worker_engine(
                wi,
                method,
                engine_name,
                *args,
                rpc_meta=rpc_meta,
                http_timeout=http_timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                **kwargs,
            )

        last_error: str | None = None

        for attempt in range(1, max_retries + 1):
            try:
                # Pass engine_name to support multiple engines per worker (colocation)
                ref = wi.actor.call.remote(
                    method,
                    *args,
                    engine_name=engine_name,
                    rpc_meta=rpc_meta,
                    **kwargs,
                )
                result = ray.get(ref, timeout=http_timeout)
                if attempt > 1:
                    logger.info(
                        f"Method '{method}' on '{worker_id}' "
                        f"succeeded after {attempt} attempts"
                    )
                return result
            except ray.exceptions.GetTimeoutError as e:
                last_error = f"Timeout: {e}"
            except ray.exceptions.RayActorError as e:
                raise WorkerFailedError(worker_id, -1, str(e)) from e
            except ray.exceptions.RayTaskError as e:
                raise EngineCallError(worker_id, method, str(e), attempt) from e
            except EngineCallError:
                raise
            except Exception as e:
                last_error = f"Ray call failed: {e}"

            # Retry with exponential backoff
            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Method '{method}' failed on worker '{worker_id}' "
                    f"(attempt {attempt}/{max_retries}): {last_error}. "
                    f"Retrying in {delay:.1f}s..."
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
        wi = self._get_worker_info_by_id(worker_id)
        if wi is None:
            raise WorkerNotFoundError(worker_id)

        if wi.worker_kind == "http_server":
            return await self._async_call_http_worker_engine(
                wi,
                method,
                engine_name,
                *args,
                rpc_meta=rpc_meta,
                http_timeout=http_timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                **kwargs,
            )

        last_error: str | None = None

        for attempt in range(1, max_retries + 1):
            try:
                # Pass engine_name to support multiple engines per worker (colocation)
                ref = wi.actor.call.remote(
                    method,
                    *args,
                    engine_name=engine_name,
                    rpc_meta=rpc_meta,
                    **kwargs,
                )
                result = await ref
                if attempt > 1:
                    logger.info(
                        f"Method '{method}' on '{worker_id}' "
                        f"succeeded after {attempt} attempts"
                    )
                return result
            except ray.exceptions.GetTimeoutError as e:
                last_error = f"Timeout: {e}"
            except ray.exceptions.RayActorError as e:
                raise WorkerFailedError(worker_id, -1, str(e)) from e
            except ray.exceptions.RayTaskError as e:
                raise EngineCallError(worker_id, method, str(e), attempt) from e
            except EngineCallError:
                raise
            except Exception as e:
                last_error = f"Ray async call failed: {e}"

            # Retry with exponential backoff
            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Method '{method}' failed on worker '{worker_id}' "
                    f"(attempt {attempt}/{max_retries}): {last_error}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

        raise EngineCallError(
            worker_id, method, last_error or "Max retries exceeded", attempt=max_retries
        )

    def __del__(self):
        # delete in case delete_workers is not called from controllers
        # explicit shutdown is by directly calling delete_workers
        try:
            self.delete_workers()
        except Exception:
            pass
