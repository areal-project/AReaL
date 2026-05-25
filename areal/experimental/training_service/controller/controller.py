# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import concurrent.futures
import sys
import threading
import time
import traceback
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import aiohttp

from areal.infra.utils.concurrent import get_executor, run_async_task
from areal.infra.utils.http import create_httpx_client
from areal.utils import logging
from areal.utils.network import format_hostport

if TYPE_CHECKING:
    from areal.api import ParallelStrategy, TrainEngine
    from areal.api.cli_args import TrainEngineConfig
    from areal.api.io_struct import FinetuneSpec
    from areal.api.scheduler_api import Scheduler, Worker

logger = logging.getLogger("GatewayTrainController")


class GatewayTrainController:
    _GUARD_SUFFIX = "-guard"

    # TODO(agent): Controller v2 is not yet a drop-in replacement for
    # TrainController on PPO/GRPO paths. Add parity for connect_engine,
    # prepare_batch/rollout_batch, and update_weights (plus the matching
    # gateway/data-proxy/worker endpoints), or keep RL controllers on v1.

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
        self.train_alloc = ModelAllocation.from_str(config.backend)
        self.api_key: str | None = None
        self._gateway_addr: str = ""
        self._router_addr: str = ""
        self._model_addr: str = ""
        self._worker_addrs: list[str] = []
        self._forked_services: list[tuple[str, str, int]] = []
        self._service_roles: list[str] = []
        self._role: str = ""
        self._parallel_strategy = self.train_alloc.parallel
        self._own_process_group = False
        self._async_client: Any | None = None
        self._async_client_loop: asyncio.AbstractEventLoop | None = None

        # Pipelined initialization state
        self._init_future: concurrent.futures.Future | None = None
        self._init_lock = threading.Lock()
        self._workers_ready = threading.Event()
        self._shutdown_requested = threading.Event()

    _WORKERS_READY_TIMEOUT: float = 30.0

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

        if not self._workers_ready.wait(timeout=self._WORKERS_READY_TIMEOUT):
            raise TimeoutError(
                f"Worker creation timed out after {self._WORKERS_READY_TIMEOUT}s"
            )
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
            guard_spec.cmd = "python -m areal.experimental.training_service.guard"

            guard_role = f"{role}{self._GUARD_SUFFIX}"
            guard_job = Job(
                replicas=world_size,
                tasks=[guard_spec],
                scheduling_strategy=cfg.scheduling_strategy,
                role=guard_role,
            )
            await asyncio.to_thread(self.scheduler.create_workers, job=guard_job)
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

            client = await self._get_async_client()
            resp = await client.post(
                f"{guard_addr_0}/alloc_ports", json={"count": 1}, timeout=30.0
            )
            resp.raise_for_status()
            master_port = resp.json()["ports"][0]

            # ==============================================================
            # Step 1.5: Set NCCL env on each guard so forked workers inherit it
            # ==============================================================
            def _guard_addr(worker: Worker) -> str:
                return (
                    f"http://{format_hostport(worker.ip, int(worker.worker_ports[0]))}"
                )

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
                    "areal.experimental.training_service.worker",
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

            self._worker_addrs = list(
                await asyncio.gather(
                    *[_fork_worker(rank) for rank in range(world_size)]
                )
            )
            logger.info("Workers: %s", self._worker_addrs)

            # ==============================================================
            # Step 3: Create engines on all workers (coordinated NCCL init)
            # ==============================================================
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
                        init_kwargs={"config": self.config},
                    )
                    for addr in self._worker_addrs
                ]
            )
            logger.info("Engines created on all workers")

            pg_kwargs = {"parallel_strategy": self._parallel_strategy}
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
                            "addr": kwargs.get("addr"),
                            "ft_spec": ft_spec,
                        },
                        timeout=self.config.setup_timeout,
                    )
                    for addr in self._worker_addrs
                ]
            )
            logger.info("Engines initialized on all workers")

            if self._shutdown_requested.is_set():
                return

            # ==============================================================
            # Step 4: Fork Router on guard 0
            # ==============================================================
            router_cmd = [
                sys.executable,
                "-m",
                "areal.experimental.training_service.router",
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
                "areal.experimental.training_service.data_proxy",
                "--worker-addrs",
                ",".join(self._worker_addrs),
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
                "areal.experimental.training_service.gateway",
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

        resp = requests.post(f"{guard_addr}/alloc_ports", json={"count": 1}, timeout=30)
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

        resp = requests.post(f"{guard_addr}/fork", json=fork_payload, timeout=30)
        resp.raise_for_status()

        self._forked_services.append((guard_addr, role, worker_index))

        addr = f"http://{format_hostport(host, port)}"
        self._wait_for_service(f"{addr}{health_path}", role)

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
            f"{guard_addr}/alloc_ports", json={"count": 1}, timeout=30.0
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

        resp = await client.post(f"{guard_addr}/fork", json=fork_payload, timeout=30.0)
        resp.raise_for_status()

        self._forked_services.append((guard_addr, role, worker_index))

        addr = f"http://{format_hostport(host, port)}"
        await self._async_wait_for_service(f"{addr}{health_path}", role)

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

    def _gateway_post(self, path: str, payload: Any = None) -> Any:
        import requests

        self._ensure_initialized()
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

    def get_version(self) -> int:
        return int(self._gateway_get_result("/get_version"))

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
        from areal.infra.rpc.serialization import serialize_value

        payload = {
            "args": serialize_value([]),
            "kwargs": serialize_value({"config": config, "role": role}),
        }
        self._gateway_post("/config_perf_tracer", payload)

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        from areal.infra.rpc.serialization import serialize_value

        payload = {
            "args": serialize_value([]),
            "kwargs": serialize_value({"step": step, "force": force}),
        }
        self._gateway_post("/save_perf_tracer", payload)

    def clear_batches(self, *targets: Any) -> None:
        from areal.infra.rpc.serialization import serialize_value

        payload = {
            "args": serialize_value(list(targets)),
            "kwargs": serialize_value({}),
        }
        self._gateway_post("/clear_batches", payload)

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

        self._graceful_shutdown_workers()

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
