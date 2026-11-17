# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import concurrent.futures
import sys
import threading
import time
import traceback
from copy import deepcopy
from threading import Lock
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import aiohttp

from areal.infra.utils.concurrent import get_executor, run_async_task
from areal.infra.utils.http import create_httpx_client
from areal.utils import logging
from areal.utils.network import format_hostport, gethostip

if TYPE_CHECKING:
    from areal.api import ParallelStrategy, TrainEngine
    from areal.api.cli_args import TrainEngineConfig
    from areal.api.io_struct import FinetuneSpec, WeightUpdateMeta
    from areal.api.scheduler_api import Scheduler, Worker

logger = logging.getLogger("GatewayTrainController")

_MAX_CLEAR_STEP_FAILURES = 2


class GatewayTrainController:
    _GUARD_SUFFIX = "-guard"

    def __init__(
        self,
        train_engine: type[TrainEngine] | str,
        config: TrainEngineConfig,
        scheduler: Scheduler,
    ) -> None:
        from areal.api.alloc_mode import ModelAllocation

        self.train_engine = train_engine
        self.scheduler = scheduler
        self.config = config
        if config.gpu_devices is not None:
            from areal.infra.scheduler.local import LocalScheduler

            if not isinstance(scheduler, LocalScheduler):
                raise ValueError("gpu_devices requires LocalScheduler")
        self.train_alloc = ModelAllocation.from_str(config.backend)
        self.api_key: str | None = None
        self._gateway_addr: str = ""
        self._router_addr: str = ""
        self._model_addr: str = ""
        self._worker_addrs: list[str] = []
        self._worker_pool_addrs: list[str] = []
        self._worker_pool_devices: dict[int, str] = {}
        self._engine_initialize_kwargs: dict[str, Any] = {}
        self._engine_rendezvous_owner: tuple[str, str, int] | None = None
        self._training_held = False
        self._engine_rebuild_ready = True
        self._guard_addrs: list[str] = []
        self._forked_services: list[tuple[str, str, int]] = []
        self._service_roles: list[str] = []
        self._role: str = ""
        self._parallel_strategy = self.train_alloc.parallel
        self._own_process_group = False
        self.rollout: Any | None = None
        self._weight_update_ctrl: Any | None = None
        self._weight_update_meta: WeightUpdateMeta | None = None
        self._weight_update_connected = False
        self._weight_update_port_owner: tuple[str, str, int] | None = None
        self._reconfiguration_lock = Lock()
        self._reconfiguration_phase = "idle"

        # Version management
        self._version_lock = Lock()
        self._version = 0

        # Shared HTTP client (lazy, per-event-loop)
        self._async_client: Any | None = None
        self._async_client_loop: asyncio.AbstractEventLoop | None = None
        self._pending_clear_shards: dict[str, dict[Any, int]] = {}
        self._clear_shards_lock = Lock()
        self._clear_batches_lock = Lock()

        # Pipelined initialization state
        self._init_future: concurrent.futures.Future | None = None
        self._init_lock = threading.Lock()
        self._workers_ready = threading.Event()
        self._shutdown_requested = threading.Event()

    # -- Initialize --------------------------------------------------------

    def initialize(
        self,
        role: str,
        ft_spec: FinetuneSpec | None = None,
        *,
        wait: bool = False,
        **kwargs: Any,
    ) -> concurrent.futures.Future | None:
        if self._init_future is not None:
            raise RuntimeError(
                "initialize() called while a previous initialization is in progress"
            )

        self._role = role

        self._workers_ready.clear()
        self._shutdown_requested.clear()
        self._init_future = get_executor("ctrl_init").submit(
            self._guarded_bg_initialize, role, ft_spec, **kwargs
        )

        ready_timeout = self.config.workers_ready_timeout
        if not self._workers_ready.wait(timeout=ready_timeout):
            raise TimeoutError(f"Worker creation timed out after {ready_timeout}s")
        if self._init_future.done():
            self._init_future.result()

        if wait:
            self._ensure_initialized()
            return None
        return self._init_future

    def _guarded_bg_initialize(self, *args: Any, **kwargs: Any) -> None:
        """Ensure _workers_ready is signaled even if _bg_initialize fails."""
        try:
            self._bg_initialize(*args, **kwargs)
        except BaseException:
            self._workers_ready.set()
            raise

    def _bg_initialize(
        self, role: str, ft_spec: FinetuneSpec | None = None, **kwargs: Any
    ) -> None:
        run_async_task(self._async_initialize, role, ft_spec, **kwargs)
        if self._shutdown_requested.is_set():
            return
        logger.info(
            "GatewayTrainController initialized (role=%s, api_key=%s, gateway=%s)",
            role,
            self.api_key,
            self._gateway_addr,
        )

    def _ensure_initialized(self) -> None:
        if self._init_future is None:
            return
        with self._init_lock:
            future = self._init_future
            if future is None:
                return
            future.result(timeout=self.config.setup_timeout)
            self._init_future = None

    async def _get_async_client(self):
        current_loop = asyncio.get_running_loop()
        if self._async_client is None or self._async_client_loop is not current_loop:
            old = self._async_client
            self._async_client = create_httpx_client(timeout=self.config.setup_timeout)
            self._async_client_loop = current_loop
            if old is not None:
                try:
                    await old.aclose()
                except Exception:
                    pass
        return self._async_client

    async def _async_initialize(
        self,
        role: str,
        ft_spec: FinetuneSpec | None = None,
        **kwargs: Any,
    ) -> None:
        from dataclasses import asdict

        from areal.api.cli_args import SchedulingSpec
        from areal.api.scheduler_api import Job

        cfg = self.config

        world_size = self.train_alloc.parallel.world_size
        pool_devices = kwargs.pop("worker_gpu_devices", None)
        pool_size = world_size
        if pool_devices is not None:
            from areal.infra.scheduler.local import LocalScheduler

            if not isinstance(self.scheduler, LocalScheduler):
                raise ValueError("worker_gpu_devices requires LocalScheduler")
            if not isinstance(pool_devices, list) or not pool_devices:
                raise ValueError("worker_gpu_devices must be a non-empty GPU list")
            active = (
                cfg.gpu_devices
                if cfg.gpu_devices is not None
                else pool_devices[:world_size]
            )
            if (
                len(pool_devices) < world_size
                or any(type(device) is not int for device in pool_devices)
                or len(set(pool_devices)) != len(pool_devices)
                or not set(pool_devices).issubset(self.scheduler.gpu_devices)
                or len(active) != world_size
                or not set(active).issubset(pool_devices)
                or any(type(device) is not int for device in active)
                or len(set(active)) != world_size
                or len(cfg.scheduling_spec) != 1
                or cfg.scheduling_spec[0].gpu != 1
            ):
                raise ValueError(
                    "Worker GPU pool must contain the active single-GPU slots"
                )
            pool_size = len(pool_devices)
            cfg.gpu_devices = list(active)
        self._engine_initialize_kwargs = {
            "addr": kwargs.get("addr"),
            "ft_spec": ft_spec,
        }

        try:
            # ==============================================================
            # Step 0: Create world_size guards via scheduler (one per GPU rank)
            # ==============================================================
            # Each guard is allocated a GPU by the scheduler (like TrainController
            # workers). Forked workers inherit the guard's GPU environment.
            if len(cfg.scheduling_spec) != 1:
                raise ValueError(
                    "GatewayTrainController (controller v2) requires exactly "
                    "one scheduling_spec. Legacy 2-spec worker/engine layouts "
                    "are only supported by TrainController (controller v1)."
                )

            guard_spec = SchedulingSpec(**asdict(cfg.scheduling_spec[0]))
            guard_spec.cmd = "python -m areal.v2.training_service.guard"

            guard_role = f"{role}{self._GUARD_SUFFIX}"
            guard_job = Job(
                replicas=pool_size,
                tasks=[guard_spec],
                scheduling_strategy=cfg.scheduling_strategy,
                role=guard_role,
            )
            placement = {}
            if pool_devices is not None:
                placement["gpu_devices"] = pool_devices
            elif cfg.gpu_devices is not None:
                placement["gpu_devices"] = cfg.gpu_devices
            await asyncio.to_thread(
                self.scheduler.create_workers, job=guard_job, **placement
            )
            self._service_roles.append(guard_role)
            guard_workers = await asyncio.to_thread(
                self.scheduler.get_workers,
                role=guard_role,
                timeout=int(self.config.setup_timeout),
            )
            logger.info("Guards ready: %s", [w.id for w in guard_workers])

            self._workers_ready.set()

            if self._shutdown_requested.is_set():
                return

            # ==============================================================
            # Step 1: Allocate master addr/port for NCCL rendezvous
            # ==============================================================
            guard_addr_0 = f"http://{format_hostport(guard_workers[0].ip, int(guard_workers[0].worker_ports[0]))}"
            master_addr = guard_workers[0].ip

            # Persist guard addresses so connect_engine() can allocate
            # ports later (e.g. for the weight-update NCCL group).
            def _guard_addr(worker: Worker) -> str:
                return (
                    f"http://{format_hostport(worker.ip, int(worker.worker_ports[0]))}"
                )

            self._guard_addrs = [_guard_addr(w) for w in guard_workers]

            client = await self._get_async_client()
            self._engine_rendezvous_owner = (guard_addr_0, f"{role}-engine", 0)
            resp = await client.post(
                f"{guard_addr_0}/alloc_ports",
                json={"count": 1, "role": f"{role}-engine", "worker_index": 0},
                timeout=30.0,
            )
            resp.raise_for_status()
            master_port = resp.json()["ports"][0]

            # ==============================================================
            # Step 1.5: Set NCCL env on each guard so forked workers inherit it
            # ==============================================================

            await self._async_set_guards_env(
                guard_workers,
                _guard_addr,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port,
            )

            # ==============================================================
            # Step 2: Fork one train worker per guard
            # ==============================================================
            async def _fork_worker(rank: int) -> str:
                guard = _guard_addr(guard_workers[rank])
                worker_cmd = [
                    sys.executable,
                    "-m",
                    "areal.v2.training_service.worker",
                    "--admin-api-key",
                    cfg.admin_api_key,
                    "--log-level",
                    cfg.log_level,
                ]

                host, port = await self._async_fork_on_guard(
                    guard_addr=guard,
                    role="train-worker",
                    worker_index=rank,
                    raw_cmd=worker_cmd,
                )
                return f"http://{format_hostport(host, port)}"

            self._worker_pool_addrs = list(
                await asyncio.gather(*[_fork_worker(rank) for rank in range(pool_size)])
            )
            self._worker_addrs = list(self._worker_pool_addrs)
            from areal.infra.scheduler.local import LocalScheduler

            if isinstance(self.scheduler, LocalScheduler) and guard_spec.gpu == 1:
                self._worker_pool_devices = {
                    device: self._worker_pool_addrs[index]
                    for index in range(pool_size)
                    for device in self.scheduler._get_colocated_gpus(guard_role, index)
                }
            if pool_devices is not None:
                self._worker_addrs = [
                    self._worker_pool_devices[device] for device in cfg.gpu_devices
                ]
                await self._async_configure_worker_ranks(
                    self._worker_addrs, master_addr, master_port
                )
            logger.info("Workers: %s", self._worker_addrs)

            # ==============================================================
            # Step 3: Create engines on all workers (coordinated NCCL init)
            # ==============================================================
            await self._async_initialize_engines(self.train_alloc, self.config)

            if self._shutdown_requested.is_set():
                return

            # ==============================================================
            # Step 4: Fork Router on guard 0
            # ==============================================================
            router_cmd = [
                sys.executable,
                "-m",
                "areal.v2.training_service.router",
                "--admin-api-key",
                cfg.admin_api_key,
                "--log-level",
                cfg.log_level,
            ]
            router_host, router_port = await self._async_fork_on_guard(
                guard_addr=guard_addr_0,
                role="router",
                worker_index=0,
                raw_cmd=router_cmd,
            )
            self._router_addr = f"http://{format_hostport(router_host, router_port)}"
            logger.info("Router: %s", self._router_addr)

            if self._shutdown_requested.is_set():
                return

            # ==============================================================
            # Step 5: Fork Data Proxy on a guard
            # ==============================================================
            data_proxy_cmd = [
                sys.executable,
                "-m",
                "areal.v2.training_service.data_proxy",
                "--worker-addrs",
                ",".join(self._worker_addrs),
                "--request-timeout",
                str(cfg.request_timeout),
                "--admin-api-key",
                cfg.admin_api_key,
                "--log-level",
                cfg.log_level,
            ]
            dp_host, dp_port = await self._async_fork_on_guard(
                guard_addr=guard_addr_0,
                role="data-proxy",
                worker_index=0,
                raw_cmd=data_proxy_cmd,
            )
            self._model_addr = f"http://{format_hostport(dp_host, dp_port)}"
            logger.info("Model endpoint: %s", self._model_addr)

            if self._shutdown_requested.is_set():
                return

            # ==============================================================
            # Step 6: Fork Gateway on guard 0
            # ==============================================================
            gw_cmd = [
                sys.executable,
                "-m",
                "areal.v2.training_service.gateway",
                "--admin-api-key",
                cfg.admin_api_key,
                "--router-addr",
                self._router_addr,
                "--forward-timeout",
                str(cfg.request_timeout),
                "--log-level",
                cfg.log_level,
            ]
            gw_host, gw_port = await self._async_fork_on_guard(
                guard_addr=guard_addr_0,
                role="gateway",
                worker_index=0,
                raw_cmd=gw_cmd,
            )
            self._gateway_addr = f"http://{format_hostport(gw_host, gw_port)}"
            logger.info("Gateway: %s", self._gateway_addr)

            # ==============================================================
            # Step 7: Register data proxy with API key in router
            # ==============================================================
            self.api_key = f"ak-{role}-{uuid4().hex[:12]}"
            await self._register_in_router(
                self._router_addr, self._model_addr, self.api_key
            )
            logger.info("Model registered with api_key=%s", self.api_key)
        except Exception:
            logger.error(
                "GatewayTrainController initialization failed, rolling back",
                exc_info=True,
            )
            self._cleanup_runtime_state()
            raise

    # -- Engine creation ---------------------------------------------------

    async def _async_configure_worker_ranks(
        self, addresses: list[str], master_addr: str, master_port: int
    ) -> None:
        client = await self._get_async_client()

        async def configure(rank: int, address: str) -> None:
            response = await client.post(
                f"{address}/configure_runtime",
                json={
                    "rank": rank,
                    "world_size": len(addresses),
                    "master_addr": master_addr,
                    "master_port": master_port,
                },
                headers={"Authorization": f"Bearer {self.config.admin_api_key}"},
            )
            response.raise_for_status()

        await asyncio.gather(
            *(configure(rank, address) for rank, address in enumerate(addresses))
        )

    async def _async_initialize_engines(self, alloc: Any, config: Any) -> None:
        if isinstance(self.train_engine, str):
            engine_class = self.train_engine
        else:
            engine_class = (
                f"{self.train_engine.__module__}.{self.train_engine.__name__}"
            )
        await asyncio.gather(
            *[
                self._create_engine_on_worker(
                    worker_addr=addr,
                    engine_class=engine_class,
                    init_args=[],
                    init_kwargs={"config": config},
                )
                for addr in self._worker_addrs
            ]
        )
        logger.info("Engines created on all workers")

        pg_kwargs = {"parallel_strategy": alloc.parallel}
        await asyncio.gather(
            *[
                self._call_worker_engine_endpoint(
                    addr,
                    "/create_process_group",
                    args=[],
                    kwargs=pg_kwargs,
                    timeout=self.config.setup_timeout,
                )
                for addr in self._worker_addrs
            ]
        )

        await asyncio.gather(
            *[
                self._call_worker_engine_endpoint(
                    addr,
                    "/initialize",
                    args=[],
                    kwargs={
                        "addr": self._engine_initialize_kwargs.get("addr"),
                        "ft_spec": self._engine_initialize_kwargs["ft_spec"],
                    },
                    timeout=self.config.setup_timeout,
                )
                for addr in self._worker_addrs
            ]
        )
        logger.info("Engines initialized on all workers")

    async def _async_set_guards_env(
        self,
        guard_workers: list[Worker],
        guard_addr_fn: Any,
        *,
        world_size: int,
        master_addr: str,
        master_port: int,
    ) -> None:
        client = await self._get_async_client()

        async def _set_env(rank: int) -> None:
            addr = guard_addr_fn(guard_workers[rank])
            env = {
                "RANK": str(rank),
                "LOCAL_RANK": "0",
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
            }
            resp = await client.post(f"{addr}/set_env", json={"env": env}, timeout=30.0)
            resp.raise_for_status()

        await asyncio.gather(*[_set_env(rank) for rank in range(len(guard_workers))])
        logger.info("NCCL env set on %d guards", len(guard_workers))

    async def _create_engine_on_worker(
        self,
        worker_addr: str,
        engine_class: str,
        init_args: list[Any],
        init_kwargs: dict[str, Any],
    ) -> None:
        from areal.infra.rpc.serialization import serialize_value

        payload = {
            "engine_class": engine_class,
            "init_args": serialize_value(init_args),
            "init_kwargs": serialize_value(init_kwargs),
        }
        client = await self._get_async_client()
        resp = await client.post(
            f"{worker_addr}/create_engine",
            json=payload,
            timeout=self.config.setup_timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Engine creation failed on {worker_addr}: {resp.text}")

    async def _call_worker_engine_endpoint(
        self,
        worker_addr: str,
        path: str,
        *,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        timeout: float,
    ) -> Any:
        from areal.infra.rpc.serialization import deserialize_value, serialize_value

        payload = {
            "args": serialize_value(args or []),
            "kwargs": serialize_value(kwargs or {}),
        }
        client = await self._get_async_client()
        resp = await client.post(f"{worker_addr}{path}", json=payload, timeout=timeout)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Worker endpoint call failed on {worker_addr}{path}: {resp.text}"
            )
        data = resp.json()
        return deserialize_value(data.get("result"))

    # -- Router registration -----------------------------------------------

    async def _register_in_router(
        self, router_addr: str, model_addr: str, api_key: str
    ) -> None:
        client = await self._get_async_client()
        resp = await client.post(
            f"{router_addr}/register",
            json={
                "model_addr": model_addr,
                "api_key": api_key,
                "name": self._role,
            },
            headers={"Authorization": f"Bearer {self.config.admin_api_key}"},
            timeout=10.0,
        )
        resp.raise_for_status()

    # -- Guard fork helpers ------------------------------------------------

    def _fork_on_guard(
        self,
        guard_addr: str,
        role: str,
        worker_index: int,
        raw_cmd: list[str],
        env: dict[str, str] | None = None,
        health_path: str = "/health",
    ) -> tuple[str, int]:
        import requests

        resp = requests.post(
            f"{guard_addr}/alloc_ports",
            json={"count": 1, "role": role, "worker_index": worker_index},
            timeout=30,
        )
        resp.raise_for_status()
        port_data = resp.json()
        host = port_data["host"]
        port = port_data["ports"][0]

        cmd = list(raw_cmd) + ["--host", host, "--port", str(port)]

        fork_payload: dict[str, Any] = {
            "role": role,
            "worker_index": worker_index,
            "raw_cmd": cmd,
        }
        if env:
            fork_payload["env"] = env

        try:
            resp = requests.post(f"{guard_addr}/fork", json=fork_payload, timeout=30)
            resp.raise_for_status()
            self._forked_services.append((guard_addr, role, worker_index))
            addr = f"http://{format_hostport(host, port)}"
            self._wait_for_service(f"{addr}{health_path}", role)
        except BaseException:
            for endpoint in ("kill_forked_worker", "release_ports"):
                try:
                    requests.post(
                        f"{guard_addr}/{endpoint}",
                        json={"role": role, "worker_index": worker_index},
                        timeout=10,
                    )
                except Exception:
                    pass
            raise

        return host, port

    async def _async_fork_on_guard(
        self,
        guard_addr: str,
        role: str,
        worker_index: int,
        raw_cmd: list[str],
        env: dict[str, str] | None = None,
        health_path: str = "/health",
    ) -> tuple[str, int]:
        client = await self._get_async_client()
        resp = await client.post(
            f"{guard_addr}/alloc_ports",
            json={"count": 1, "role": role, "worker_index": worker_index},
            timeout=30.0,
        )
        resp.raise_for_status()
        port_data = resp.json()
        host = port_data["host"]
        port = port_data["ports"][0]

        cmd = list(raw_cmd) + ["--host", host, "--port", str(port)]
        fork_payload: dict[str, Any] = {
            "role": role,
            "worker_index": worker_index,
            "raw_cmd": cmd,
        }
        if env:
            fork_payload["env"] = env

        try:
            resp = await client.post(
                f"{guard_addr}/fork", json=fork_payload, timeout=30.0
            )
            resp.raise_for_status()
            self._forked_services.append((guard_addr, role, worker_index))
            addr = f"http://{format_hostport(host, port)}"
            await self._async_wait_for_service(f"{addr}{health_path}", role)
        except BaseException:
            for endpoint in ("kill_forked_worker", "release_ports"):
                try:
                    await client.post(
                        f"{guard_addr}/{endpoint}",
                        json={"role": role, "worker_index": worker_index},
                        timeout=10.0,
                    )
                except Exception:
                    pass
            raise

        return host, port

    def _kill_forked_service(
        self, guard_addr: str, role: str, worker_index: int
    ) -> None:
        import requests

        try:
            resp = requests.post(
                f"{guard_addr}/kill_forked_worker",
                json={"role": role, "worker_index": worker_index},
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info("Killed forked service %s/%d", role, worker_index)
            else:
                logger.warning(
                    "Failed to kill %s/%d: %s", role, worker_index, resp.text
                )
        except Exception as exc:
            logger.error("Error killing %s/%d: %s", role, worker_index, exc)

    # -- Health checks -----------------------------------------------------

    def _wait_for_service(
        self, url: str, name: str, timeout: float | None = None
    ) -> None:
        import requests as _requests

        timeout = timeout or self.config.setup_timeout
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = _requests.get(url, timeout=2)
                if resp.status_code == 200:
                    logger.info("%s is ready at %s", name, url)
                    return
            except _requests.RequestException:
                pass
            time.sleep(0.1)
        raise TimeoutError(f"{name} not healthy at {url} within {timeout}s")

    async def _async_wait_for_service(
        self, url: str, name: str, timeout: float | None = None
    ) -> None:
        timeout = timeout or self.config.setup_timeout
        deadline = time.monotonic() + timeout
        client = await self._get_async_client()
        while time.monotonic() < deadline:
            try:
                resp = await client.get(url, timeout=2.0)
                if resp.status_code == 200:
                    logger.info("%s is ready at %s", name, url)
                    return
            except Exception:
                pass
            await asyncio.sleep(0.1)
        raise TimeoutError(f"{name} not healthy at {url} within {timeout}s")

    # -- Gateway HTTP helpers (duck-type TrainController interface) ---------

    async def _async_held_state_post(self, path: str, payload: Any) -> Any:
        """Restore state through active worker endpoints while compute stays held."""
        if not self._engine_rebuild_ready or not self._worker_addrs:
            raise RuntimeError("Complete Engine initialization before accessing state")
        client = await self._get_async_client()
        responses = await asyncio.gather(
            *(
                client.post(
                    f"{address}{path}",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.config.admin_api_key}"},
                )
                for address in self._worker_addrs
            ),
            return_exceptions=True,
        )
        for response in responses:
            if isinstance(response, BaseException):
                raise response
            response.raise_for_status()
        return responses[0].json()

    def _gateway_post(self, path: str, payload: Any = None) -> Any:
        import requests

        self._ensure_initialized()
        if self._training_held and path in {"/save", "/load", "/set_version"}:
            return run_async_task(self._async_held_state_post, path, payload)
        url = f"{self._gateway_addr}{path}"
        resp = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.config.request_timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Gateway {path} returned {resp.status_code}: {resp.text}"
            )
        return resp.json()

    def _gateway_get(self, path: str) -> Any:
        import requests

        self._ensure_initialized()
        url = f"{self._gateway_addr}{path}"
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.config.request_timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Gateway {path} returned {resp.status_code}: {resp.text}"
            )
        return resp.json()

    def _gateway_post_result(self, path: str, payload: Any = None) -> Any:
        from areal.infra.rpc.serialization import deserialize_value

        data = self._gateway_post(path, payload)
        if not isinstance(data, dict) or "result" not in data:
            raise RuntimeError(f"Gateway {path} response missing 'result': {data!r}")
        return deserialize_value(data["result"])

    def _gateway_get_result(self, path: str) -> Any:
        from areal.infra.rpc.serialization import deserialize_value

        data = self._gateway_get(path)
        if not isinstance(data, dict) or "result" not in data:
            raise RuntimeError(f"Gateway {path} response missing 'result': {data!r}")
        return deserialize_value(data["result"])

    # -- TrainController duck-type interface --------------------------------

    @staticmethod
    def _require_list_batch(input_: Any, method_name: str) -> list[dict[str, Any]]:
        if not isinstance(input_, list):
            raise TypeError(
                f"{method_name} expects `input_` as list[dict[str, Any]] for training-service dispatch; "
                f"got {type(input_).__name__}."
            )
        return input_

    def train_batch(
        self,
        input_: list[dict[str, Any]] | None = None,
        loss_fn: Any = None,
        loss_weight_fn: Any = None,
    ) -> Any:
        from areal.infra.rpc.serialization import serialize_value

        if input_ is None:
            raise TypeError("train_batch expects non-None list[dict[str, Any]] input.")
        batch = self._require_list_batch(input_, "train_batch")

        payload = {
            "args": serialize_value([batch]),
            "kwargs": serialize_value(
                {"loss_fn": loss_fn, "loss_weight_fn": loss_weight_fn}
            ),
        }
        return self._gateway_post_result("/train_batch", payload)

    def forward_batch(
        self, input_: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> Any:
        from areal.infra.rpc.serialization import serialize_value

        if input_ is None:
            raise TypeError(
                "forward_batch expects non-None list[dict[str, Any]] input."
            )
        batch = self._require_list_batch(input_, "forward_batch")

        payload = {
            "args": serialize_value([batch]),
            "kwargs": serialize_value(kwargs),
        }
        return self._gateway_post_result("/forward_batch", payload)

    def eval_batch(
        self,
        input_: list[dict[str, Any]] | None = None,
        loss_fn: Any = None,
        loss_weight_fn: Any = None,
    ) -> Any:
        from areal.infra.rpc.serialization import serialize_value

        if input_ is None:
            raise TypeError("eval_batch expects non-None list[dict[str, Any]] input.")
        batch = self._require_list_batch(input_, "eval_batch")

        payload = {
            "args": serialize_value([batch]),
            "kwargs": serialize_value(
                {"loss_fn": loss_fn, "loss_weight_fn": loss_weight_fn}
            ),
        }
        return self._gateway_post_result("/eval_batch", payload)

    def train(self, mode: bool = True) -> GatewayTrainController:
        from areal.infra.rpc.serialization import serialize_value

        self._gateway_post(
            "/train",
            {
                "args": serialize_value([mode]),
                "kwargs": serialize_value({}),
            },
        )
        return self

    def eval(self) -> GatewayTrainController:
        self._gateway_post("/eval")
        return self

    def set_version(self, version: int) -> None:
        from areal.infra.rpc.serialization import serialize_value

        self._gateway_post(
            "/set_version",
            {
                "args": serialize_value([version]),
                "kwargs": serialize_value({}),
            },
        )
        with self._version_lock:
            self._version = version

    def get_version(self) -> int:
        with self._version_lock:
            return self._version

    def save(self, meta: Any) -> None:
        from areal.infra.rpc.serialization import serialize_value

        self._gateway_post(
            "/save",
            {
                "args": serialize_value([meta]),
                "kwargs": serialize_value({}),
            },
        )

    def load(self, meta: Any) -> None:
        from areal.infra.rpc.serialization import serialize_value

        self._gateway_post(
            "/load",
            {
                "args": serialize_value([meta]),
                "kwargs": serialize_value({}),
            },
        )

    def offload(self) -> None:
        self._gateway_post("/offload")

    def onload(self) -> None:
        self._gateway_post("/onload")

    def step_lr_scheduler(self) -> None:
        self._gateway_post("/step_lr_scheduler")

    def optimizer_zero_grad(self) -> None:
        self._gateway_post("/optimizer_zero_grad")

    def optimizer_step(self) -> Any:
        return self._gateway_post_result("/optimizer_step")

    def export_stats(self) -> dict[str, Any]:
        from areal.utils import stats_tracker

        stats = stats_tracker.export_all()
        stats.update(self._gateway_get_result("/export_stats"))
        return stats

    def get_device_stats(self) -> Any:
        from areal.infra.rpc.serialization import serialize_value

        payload = {
            "args": serialize_value([]),
            "kwargs": serialize_value({}),
        }
        return self._gateway_post_result("/get_device_stats", payload)

    def config_perf_tracer(self, config: Any, role: str) -> None:
        self._ensure_initialized()

        async def _call() -> None:
            tasks = [
                self._call_worker_engine_endpoint(
                    addr,
                    "/config_perf_tracer",
                    args=[],
                    kwargs={"config": config, "rank": rank, "role": role},
                    timeout=self.config.request_timeout,
                )
                for rank, addr in enumerate(self._worker_addrs)
            ]
            await asyncio.gather(*tasks)

        run_async_task(_call)

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        from areal.infra.rpc.serialization import serialize_value

        payload = {
            "args": serialize_value([]),
            "kwargs": serialize_value({"step": step, "force": force}),
        }
        self._gateway_post("/save_perf_tracer", payload)

    def clear_batches(self, *targets: Any) -> None:
        """Clear worker buffers and retry failed storage deletes across one step."""
        with self._clear_batches_lock:
            self._clear_batches_locked(*targets)

    def _clear_batches_locked(self, *targets: Any) -> None:
        from areal.infra.rpc.rtensor import RTensor
        from areal.infra.rpc.serialization import serialize_value

        # Step 1: HTTP DELETE to storage nodes to evict _storage entries
        # (mirrors TrainController._async_clear_batches)
        shards_by_node = RTensor.collect_shards(targets)
        with self._clear_shards_lock:
            for addr, sids in shards_by_node.items():
                pending = self._pending_clear_shards.setdefault(addr, {})
                for sid in sids:
                    pending.setdefault(sid, 0)
            clear_requests = [
                (addr, list(pending))
                for addr, pending in self._pending_clear_shards.items()
                if pending
            ]
        exhausted: dict[str, list[Any]] = {}
        if clear_requests:

            async def _clear_storage():
                results = await asyncio.gather(
                    *[RTensor.clear_node(addr, sids) for addr, sids in clear_requests],
                    return_exceptions=True,
                )
                fatal_error = next(
                    (
                        result
                        for result in results
                        if isinstance(result, BaseException)
                        and not isinstance(result, Exception)
                    ),
                    None,
                )
                if fatal_error is not None:
                    raise fatal_error

                for (addr, sids), result in zip(clear_requests, results, strict=True):
                    if isinstance(result, Exception):
                        with self._clear_shards_lock:
                            pending = self._pending_clear_shards.get(addr, {})
                            exhausted_sids = []
                            for sid in sids:
                                if sid not in pending:
                                    continue
                                failures = min(
                                    pending[sid] + 1, _MAX_CLEAR_STEP_FAILURES
                                )
                                pending[sid] = failures
                                if failures >= _MAX_CLEAR_STEP_FAILURES:
                                    exhausted_sids.append(sid)
                            pending_count = sum(
                                failures < _MAX_CLEAR_STEP_FAILURES
                                for failures in pending.values()
                            )
                        if exhausted_sids:
                            exhausted[addr] = exhausted_sids
                        logger.warning(
                            "Failed to clear %d RTensor shards on storage node "
                            "%s: %s: %s (%d pending, %d retries exhausted)",
                            len(sids),
                            addr,
                            type(result).__name__,
                            result,
                            pending_count,
                            len(exhausted_sids),
                        )
                        continue
                    with self._clear_shards_lock:
                        pending = self._pending_clear_shards.get(addr)
                        if pending is not None:
                            for sid in sids:
                                pending.pop(sid, None)
                            if not pending:
                                self._pending_clear_shards.pop(addr, None)
                    if isinstance(result, dict):
                        logger.debug(
                            "Cleared RTensor shards on storage node %s "
                            "(requested=%d, cleared=%s, remaining_tensors=%s, "
                            "remaining_bytes=%s)",
                            addr,
                            len(sids),
                            result.get("cleared_count", "unknown"),
                            result.get("num_tensors", "unknown"),
                            result.get("total_bytes", "unknown"),
                        )

            run_async_task(_clear_storage)

        # Step 2: Drain _fetch_buffer on workers via engine.clear_batches(shard_ids)
        shard_ids = [sid for _, sids in clear_requests for sid in sids]
        if not shard_ids:
            return
        payload = {
            "args": serialize_value([shard_ids]),
            "kwargs": serialize_value({}),
        }
        self._gateway_post("/clear_batches", payload)

        # Keep retry exhaustion pending until every worker has accepted the
        # buffer drain. A gateway failure then leaves enough state for the next
        # call to retry or report the remote storage leak.
        with self._clear_shards_lock:
            for addr, sids in exhausted.items():
                pending = self._pending_clear_shards.get(addr)
                if pending is None:
                    continue
                for sid in sids:
                    pending.pop(sid, None)
                if not pending:
                    self._pending_clear_shards.pop(addr, None)

        if exhausted:
            summary = ", ".join(
                f"{addr}: {len(sids)}" for addr, sids in sorted(exhausted.items())
            )
            raise RuntimeError(
                "RTensor storage cleanup failed across two clear_batches calls "
                f"({summary})"
            )

    def current_data_parallel_head(self) -> int:
        return 0

    @property
    def context_and_model_parallel_group(self):
        return self.cpu_group

    @property
    def parallel_strategy(self):
        return self._parallel_strategy

    @property
    def data_parallel_world_size(self) -> int:
        return 1

    @property
    def data_parallel_rank(self) -> int:
        return 0

    # -- Properties (duck-type compat) -------------------------------------

    @property
    def cpu_group(self):
        return None

    @property
    def train_worker_urls(self) -> list[str]:
        return list(self._worker_addrs)

    # -- RL parity methods (connect_engine / update_weights / batch) --------

    async def _async_training_control(
        self, endpoint: str, payload: dict[str, Any]
    ) -> None:
        client = await self._get_async_client()
        response = await client.post(
            f"{self._model_addr}{endpoint}",
            json=payload,
            headers={"Authorization": f"Bearer {self.config.admin_api_key}"},
        )
        response.raise_for_status()

    def hold_training(self) -> None:
        """Hold new frontend calls and wait for current worker RPCs to finish."""
        self._ensure_initialized()
        if not self._model_addr:
            raise RuntimeError("Initialize the training frontend before holding it")
        self._training_held = True
        run_async_task(self._async_training_control, "/hold_training", {})

    def release_training(self) -> None:
        """Release calls after state restoration and any required AWEX publication."""
        self._ensure_initialized()
        if not self._engine_rebuild_ready:
            raise RuntimeError(
                "Complete Engine initialization and topology binding first"
            )
        if self.rollout is not None and (
            not self._weight_update_connected
            or self.rollout.get_version() != self.get_version()
        ):
            raise RuntimeError(
                "Reconnect and publish the matching rollout version first"
            )
        run_async_task(self._async_training_control, "/release_training", {})
        self._training_held = False

    def rebuild_engines(self, backend: str, gpu_devices: list[int]) -> None:
        """Replace Engines in the fixed worker pool, returning held with fresh state."""
        with self._reconfiguration_lock, self._clear_batches_lock:
            alloc = self._validate_engine_rebuild(backend, gpu_devices)
            self._rebuild_engines_locked(alloc, gpu_devices)

    def _validate_engine_rebuild(self, backend: str, gpu_devices: list[int]):
        from areal.api.alloc_mode import ModelAllocation

        self._ensure_initialized()
        alloc = ModelAllocation.from_str(backend)
        if (
            not self._worker_pool_devices
            or self.train_alloc.backend != "megatron"
            or alloc.backend != "megatron"
        ):
            raise ValueError(
                "Engine rebuilding requires a local single-GPU Megatron worker pool"
            )
        if (
            not isinstance(gpu_devices, list)
            or len(gpu_devices) != alloc.parallel.world_size
            or any(type(device) is not int for device in gpu_devices)
            or len(set(gpu_devices)) != len(gpu_devices)
            or not set(gpu_devices).issubset(self._worker_pool_devices)
        ):
            raise ValueError(
                "Target GPUs must match the layout and original worker pool"
            )
        if self._pending_clear_shards:
            raise RuntimeError(
                "Finish pending batch storage cleanup before rebuilding Engines"
            )
        return alloc

    def _rebuild_engines_locked(self, alloc: Any, gpu_devices: list[int]) -> None:
        self.hold_training()
        self._engine_rebuild_ready = False
        try:
            self._reconfiguration_phase = "disconnecting-actor"
            self.disconnect_engine()
            self._reconfiguration_phase = "rebuilding-engines"
            run_async_task(self._async_rebuild_engines, alloc, list(gpu_devices))
            self._reconfiguration_phase = "actor-held"
        except BaseException:
            logger.exception(
                "Actor Engine rebuild failed during %s", self._reconfiguration_phase
            )
            raise

    async def _async_release_engine_rendezvous(self) -> None:
        if self._engine_rendezvous_owner is not None:
            guard, role, index = self._engine_rendezvous_owner
            client = await self._get_async_client()
            response = await client.post(
                f"{guard}/release_ports", json={"role": role, "worker_index": index}
            )
            response.raise_for_status()
            self._engine_rendezvous_owner = None

    async def _async_rebuild_engines(self, alloc: Any, gpu_devices: list[int]) -> None:
        await self._async_release_engines()
        await self._async_create_engines(alloc, gpu_devices)

    async def _async_release_engines(self) -> None:
        self._engine_rebuild_ready = False
        client = await self._get_async_client()

        async def destroy(address: str) -> None:
            response = await client.post(f"{address}/destroy_engine", json={})
            response.raise_for_status()

        results = await asyncio.gather(
            *(destroy(address) for address in self._worker_addrs),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        await self._async_release_engine_rendezvous()

    async def _async_create_engines(self, alloc: Any, gpu_devices: list[int]) -> None:
        client = await self._get_async_client()
        guard = self._guard_addrs[0]
        owner = (guard, f"{self._role}-engine", 0)
        self._engine_rendezvous_owner = owner
        response = await client.post(
            f"{guard}/alloc_ports",
            json={"count": 1, "role": owner[1], "worker_index": owner[2]},
        )
        response.raise_for_status()
        rendezvous = response.json()
        self._worker_addrs = [
            self._worker_pool_devices[device] for device in gpu_devices
        ]
        await self._async_configure_worker_ranks(
            self._worker_addrs, rendezvous["host"], rendezvous["ports"][0]
        )
        config = deepcopy(self.config)
        config.backend, config.gpu_devices = str(alloc), gpu_devices
        await self._async_initialize_engines(alloc, config)
        await self._async_training_control(
            "/configure_workers", {"worker_addrs": self._worker_addrs}
        )
        self.train_alloc, self._parallel_strategy, self.config = (
            alloc,
            alloc.parallel,
            config,
        )
        with self._version_lock:
            self._version = 0
        self._engine_rebuild_ready = True

    def connect_engine(self, rollout: Any, meta: Any) -> None:
        self._ensure_initialized()
        import requests

        from areal.v2.inference_service.controller.controller import (
            RolloutControllerV2,
        )
        from areal.v2.weight_update.controller.config import (
            WeightUpdateControllerConfig,
        )
        from areal.v2.weight_update.controller.controller import (
            WeightUpdateController,
        )

        if not isinstance(rollout, RolloutControllerV2):
            raise TypeError(
                f"GatewayTrainController requires RolloutControllerV2, "
                f"got {type(rollout).__name__}. "
                f"Ensure _version='v2' is set on InferenceEngineConfig."
            )

        if meta.type not in ("awex", "disk"):
            raise ValueError(
                f"GatewayTrainController supports 'awex' or 'disk' weight "
                f"updates, got '{meta.type}'"
            )
        if self._weight_update_connected:
            raise RuntimeError(
                "Disconnect the current weight-transfer pair before reconnecting"
            )
        self.rollout = rollout

        ctrl = self._weight_update_ctrl
        if ctrl is None:
            ctrl = WeightUpdateController(
                WeightUpdateControllerConfig(
                    # Bind gateway to this node's outbound IP so cross-host
                    # train/inf workers can reach the kv_store_url the gateway
                    # publishes. Default "127.0.0.1" only works single-machine.
                    host=gethostip(),
                    admin_api_key=self.config.admin_api_key,
                    log_level=self.config.log_level,
                    request_timeout=self.config.request_timeout,
                    init_timeout_s=self.config.request_timeout,
                    update_timeout_s=self.config.request_timeout,
                )
            )
            self._weight_update_ctrl = ctrl
            try:
                ctrl.initialize()
            except BaseException:
                ctrl.destroy()
                self._weight_update_ctrl = None
                raise

        inference_urls: list[str] = rollout.inference_worker_urls
        if not inference_urls:
            raise RuntimeError("No inference servers available for weight transfer")
        pair_name = f"{self._role}-rollout"
        self._weight_update_meta = meta
        self._weight_update_connected = True

        if meta.type == "awex":
            # NCCL rendezvous master must live on the rank-0 process's node.
            # awex assigns rank 0 to inference[0], so allocate on the inference
            # rank-0 guard rather than a train guard.
            inf_guard_addrs = rollout.inference_guard_addrs
            owner_role = f"{pair_name}-awex"
            self._weight_update_port_owner = (inf_guard_addrs[0], owner_role, 0)
            resp = requests.post(
                f"{inf_guard_addrs[0]}/alloc_ports",
                json={"count": 1, "role": owner_role, "worker_index": 0},
                timeout=30,
            )
            resp.raise_for_status()
            port_data = resp.json()
            ctrl.connect(
                pair_name=pair_name,
                train_worker_urls=self._worker_addrs,
                inference_worker_urls=inference_urls,
                mode="awex",
                nccl_master_addr=port_data["host"],
                nccl_master_port=port_data["ports"][0],
            )
        else:  # disk
            ctrl.connect(
                pair_name=pair_name,
                train_worker_urls=self._worker_addrs,
                inference_worker_urls=inference_urls,
                mode="disk",
                save_path=meta.path or "",
                use_lora=meta.use_lora,
                lora_name=meta.lora_name,
                lora_keep_versions=meta.lora_keep_versions,
            )
        logger.info(
            "WeightUpdateController connected (pair=%s, train=%d, inf=%d)",
            pair_name,
            len(self._worker_addrs),
            len(inference_urls),
        )

    def _release_weight_update_port(self) -> None:
        import requests

        if self._weight_update_port_owner is not None:
            if self.rollout is not None and self.rollout._destroyed:
                # The rollout Guard has already released all of its reservations.
                self._weight_update_port_owner = None
                return
            guard, role, index = self._weight_update_port_owner
            response = requests.post(
                f"{guard}/release_ports",
                json={"role": role, "worker_index": index},
                timeout=30,
            )
            response.raise_for_status()
            self._weight_update_port_owner = None

    def disconnect_engine(self) -> None:
        """Release weight-transfer groups while retaining the training engine."""
        self._ensure_initialized()
        if self._weight_update_connected:
            if self._weight_update_meta.type == "awex":
                urls = self._worker_addrs + self.rollout.inference_worker_urls

                async def teardown() -> None:
                    client = await self._get_async_client()
                    responses = await asyncio.gather(
                        *(client.post(f"{url}/awex/teardown", json={}) for url in urls),
                        return_exceptions=True,
                    )
                    for response in responses:
                        if isinstance(response, BaseException):
                            raise response
                        response.raise_for_status()

                run_async_task(teardown)
            self._weight_update_ctrl.disconnect()
            self._weight_update_connected = False
        self._release_weight_update_port()

    def update_weights(self, meta: Any, *, resume_generation: bool = True) -> None:
        if self._weight_update_ctrl is None or self.rollout is None:
            raise RuntimeError(
                "connect_engine() must be called before update_weights()"
            )
        self.rollout.pause_generation()
        if resume_generation:
            assert meta.version is not None and meta.version > 0, (
                f"meta.version must be a positive integer, got {meta.version}"
            )
        elif meta.version is None or meta.version < 0:
            raise ValueError("meta.version must be a non-negative integer")
        result = self._weight_update_ctrl.update_weights(version=meta.version)
        # Publish the version before resuming generation. Requests unblocked by
        # continue_generation() are served by the new weights, so they must be
        # stamped with the new version; otherwise every segment resumed during
        # the version broadcast is recorded as pre-update policy and downstream
        # importance ratios pick the wrong behavior policy for it.
        try:
            if not resume_generation and (
                result.status != "ok" or result.version != meta.version
            ):
                raise RuntimeError(
                    f"Weight publication failed for version {meta.version}: {result.error}"
                )
            self.rollout.set_version(meta.version)
            if not resume_generation:
                self.rollout.verify_proxy_versions(meta.version)
        finally:
            if resume_generation:
                self.rollout.continue_generation()
        logger.info(
            "Weight update v%d completed (%s, %.0fms)",
            meta.version,
            result.status,
            result.duration_ms,
        )

    def prepare_batch(
        self,
        dataloader: Any,
        workflow: Any,
        workflow_kwargs: dict[str, Any],
        should_accept_fn: str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
        min_usable_group_size: int = 1,
        should_accept_kwargs: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self.rollout is None:
            raise RuntimeError("connect_engine() must be called before prepare_batch()")
        return self.rollout.prepare_batch(
            dataloader=dataloader,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            should_accept_kwargs=should_accept_kwargs,
            group_size=group_size,
            dynamic_bs=dynamic_bs,
            reward_normalization=reward_normalization,
            drop_incomplete_group=drop_incomplete_group,
            min_usable_group_size=min_usable_group_size,
        )

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: Any,
        workflow_kwargs: dict[str, Any],
        should_accept_fn: str | None = None,
        group_size: int = 1,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
        min_usable_group_size: int = 1,
        should_accept_kwargs: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self.rollout is None:
            raise RuntimeError("connect_engine() must be called before rollout_batch()")
        return self.rollout.rollout_batch(
            data=data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            should_accept_kwargs=should_accept_kwargs,
            group_size=group_size,
            reward_normalization=reward_normalization,
            drop_incomplete_group=drop_incomplete_group,
            min_usable_group_size=min_usable_group_size,
        )

    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        self._parallel_strategy = parallel_strategy
        import torch.distributed as dist

        from areal.utils.network import find_free_ports

        if not dist.is_initialized():
            port = find_free_ports(1)[0]
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://localhost:{port}",
                rank=0,
                world_size=1,
            )
            self._own_process_group = True

    def is_data_parallel_head(self) -> bool:
        return True

    # -- Destroy -----------------------------------------------------------

    def _graceful_shutdown_workers(self) -> None:
        """Destroy engines on all training workers before killing processes.

        ``dist.destroy_process_group()`` is a local operation
        (``ncclCommAbort`` + HeartbeatMonitor join), but rank-0 hosts the
        TCPStore server.  All workers must stop their HeartbeatMonitor
        before any process exits, otherwise surviving ranks get a
        ``recvValue failed`` warning from the now-dead TCPStore.
        """
        if not self._worker_addrs:
            return

        async def _shutdown_all() -> None:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                tasks = []
                for addr in self._worker_addrs:
                    tasks.append(_shutdown_one(session, addr))
                await asyncio.gather(*tasks, return_exceptions=True)

        async def _shutdown_one(session: aiohttp.ClientSession, addr: str) -> None:
            try:
                async with session.post(f"{addr}/awex/teardown") as resp:
                    resp.raise_for_status()
            except Exception as e:
                logger.warning(
                    "Graceful shutdown: failed to call /awex/teardown on %s: %s",
                    addr,
                    e,
                )
            try:
                async with session.post(f"{addr}/destroy_engine", json={}) as resp:
                    resp.raise_for_status()
            except Exception as e:
                logger.warning(
                    "Graceful shutdown: failed to call /destroy_engine on %s: %s",
                    addr,
                    e,
                )

        run_async_task(_shutdown_all)
        logger.info("All training worker engines destroyed gracefully")

    def _cleanup_runtime_state(self) -> None:
        if self._router_addr and self._model_addr:
            try:
                import requests

                requests.post(
                    f"{self._router_addr}/unregister",
                    json={"model_addr": self._model_addr},
                    headers={"Authorization": f"Bearer {self.config.admin_api_key}"},
                    timeout=10,
                )
            except Exception:
                logger.error("Failed to unregister model: %s", traceback.format_exc())

        if self._weight_update_ctrl is not None:
            try:
                self._weight_update_ctrl.destroy()
            except Exception:
                logger.error("Failed to destroy weight-update gateway", exc_info=True)
            self._weight_update_ctrl = None
            self._weight_update_connected = False
        self._graceful_shutdown_workers()
        try:
            run_async_task(self._async_release_engine_rendezvous)
        except Exception:
            logger.warning("Failed to release Engine rendezvous port", exc_info=True)
        try:
            self._release_weight_update_port()
        except Exception:
            logger.warning(
                "Failed to release weight-update rendezvous port", exc_info=True
            )
        self._weight_update_meta = None

        for guard_addr, role, worker_index in reversed(self._forked_services):
            try:
                self._kill_forked_service(guard_addr, role, worker_index)
            except Exception:
                logger.error(
                    "Error killing %s/%d: %s",
                    role,
                    worker_index,
                    traceback.format_exc(),
                )
        self._forked_services.clear()

        for role in reversed(self._service_roles):
            try:
                self.scheduler.delete_workers(role=role)
                logger.info("Workers deleted for role: %s", role)
            except Exception:
                logger.error(
                    "Error deleting workers for %s: %s", role, traceback.format_exc()
                )
        self._service_roles.clear()
        self._worker_addrs.clear()
        self._worker_pool_addrs.clear()
        self._worker_pool_devices.clear()
        self._engine_initialize_kwargs.clear()
        self._training_held = False
        self._engine_rebuild_ready = True
        self._router_addr = ""
        self._gateway_addr = ""
        self._model_addr = ""
        self.api_key = None

        if self._async_client is not None:
            try:
                run_async_task(self._async_client.aclose)
            except Exception:
                pass
            self._async_client = None
            self._async_client_loop = None

        import torch.distributed as dist

        if self._own_process_group:
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                logger.error(
                    "Failed to destroy process group: %s", traceback.format_exc()
                )
            finally:
                self._own_process_group = False

    def destroy(self) -> None:
        self._shutdown_requested.set()
        future = self._init_future
        self._init_future = None
        if future is not None:
            future.cancel()

        self._cleanup_runtime_state()
