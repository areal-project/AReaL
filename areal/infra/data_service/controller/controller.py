# SPDX-License-Identifier: Apache-2.0

"""DataController — orchestrator for the distributed data loading service.

Manages the full lifecycle: create RPCGuard workers → fork DataWorkers,
Router, Gateway → register datasets → serve batches → shutdown.

Follows the same patterns as ``RolloutControllerV2``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import sys
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import aiohttp

from areal.api.scheduler_api import Job
from areal.infra.data_service.controller.config import DataServiceConfig
from areal.utils import logging
from areal.utils.network import format_hostport

if TYPE_CHECKING:
    from areal.api.scheduler_api import Scheduler, Worker

logger = logging.getLogger("DataController")


class DataController:
    """Controller for the distributed data loading service.

    API follows ``TrainController`` / ``RolloutControllerV2`` patterns:
    ``__init__(config, scheduler)`` then ``initialize(role, ...)``.
    """

    _GUARD_SUFFIX = "-data"
    _ADMIN_API_KEY = os.environ.get("AREAL_ADMIN_KEY", "areal-data-admin")

    def __init__(
        self,
        config: DataServiceConfig,
        scheduler: Scheduler,
    ) -> None:
        self.config = config
        self.scheduler = scheduler

        self.workers: list[Worker] = []
        self._worker_role: str = ""

        self._gateway_addr: str = ""
        self._router_addr: str = ""
        self._worker_addrs: list[str] = []

        self._service_roles: list[str] = []
        self._forked_services: list[tuple[str, str, int]] = []

        self._admin_api_key: str = self._ADMIN_API_KEY

        self._datasets: dict[str, dict[str, Any]] = {}

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
        num_dataset_workers: int = 1,
        *,
        wait: bool = False,
        **kwargs: Any,
    ) -> concurrent.futures.Future | None:
        from areal.infra.utils.concurrent import get_executor

        if self._init_future is not None:
            raise RuntimeError(
                "initialize() called while a previous initialization is in progress"
            )

        self._worker_role = role

        self._workers_ready.clear()
        self._shutdown_requested.clear()
        self._init_future = get_executor("ctrl_init").submit(
            self._guarded_bg_initialize, num_dataset_workers, **kwargs
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
        self,
        num_dataset_workers: int,
        **kwargs: Any,
    ) -> None:
        from areal.infra.utils.concurrent import run_async_task

        run_async_task(self._async_initialize, num_dataset_workers, **kwargs)
        if self._shutdown_requested.is_set():
            return
        logger.info("DataController initialized with %d workers", num_dataset_workers)

    def _ensure_initialized(self) -> None:
        if self._init_future is None:
            return
        with self._init_lock:
            future = self._init_future
            if future is None:
                return
            future.result(timeout=self.config.setup_timeout)
            self._init_future = None

    async def _async_initialize(
        self,
        num_dataset_workers: int,
        **kwargs: Any,
    ) -> None:
        cfg = self.config
        spec = cfg.scheduling_spec
        if spec is None:
            raise ValueError(
                "DataServiceConfig.scheduling_spec must be set to launch data service workers"
            )

        # Use sys.executable as the interpreter; don't mutate cfg.scheduling_spec
        cmd = spec.cmd
        if not cmd:
            raise ValueError(
                "DataServiceConfig.scheduling_spec.cmd must be set to launch RPC guards"
            )
        parts = cmd.split("-m", 1)
        if len(parts) == 2:
            module = parts[1].strip()
            guard_cmd = f"{sys.executable} -m {module}"
        else:
            guard_cmd = f"{sys.executable} {cmd}"
        guard_spec = replace(spec, cmd=guard_cmd)

        guard_role = f"{self._worker_role}{self._GUARD_SUFFIX}"

        guard_job = Job(
            replicas=num_dataset_workers,
            tasks=[guard_spec for _ in range(num_dataset_workers)],
            scheduling_strategy=cfg.scheduling_strategy,
            role=guard_role,
        )

        self.scheduler.create_workers(job=guard_job)
        self._service_roles.append(guard_role)
        guard_workers = self.scheduler.get_workers(role=guard_role)
        self.workers = guard_workers
        logger.info("RPCGuard workers ready: %s", [w.id for w in guard_workers])

        self._workers_ready.set()

        if self._shutdown_requested.is_set():
            return

        guard_addrs = [
            f"http://{format_hostport(w.ip, int(w.worker_ports[0]))}"
            for w in guard_workers
        ]
        guard_addr_0 = guard_addrs[0]

        try:
            async with aiohttp.ClientSession(trust_env=False) as session:
                # Wave 1: Fork all DataWorkers + Router in parallel
                worker_tasks = [
                    self._async_fork_on_guard(
                        session,
                        guard_addrs[rank],
                        "data-worker",
                        rank,
                        [
                            sys.executable,
                            "-m",
                            "areal.infra.data_service.worker",
                            "--rank",
                            str(rank),
                            "--world-size",
                            str(num_dataset_workers),
                            "--dataloader-num-workers",
                            str(cfg.dataloader_num_workers),
                            "--seed",
                            str(cfg.seed),
                        ],
                    )
                    for rank in range(num_dataset_workers)
                ]
                router_task = self._async_fork_on_guard(
                    session,
                    guard_addr_0,
                    "data-router",
                    0,
                    [
                        sys.executable,
                        "-m",
                        "areal.infra.data_service.router",
                        "--admin-api-key",
                        self._admin_api_key,
                    ],
                )

                results = await asyncio.gather(*worker_tasks, router_task)

                for host, port in results[:-1]:
                    self._worker_addrs.append(f"http://{format_hostport(host, port)}")
                router_host, router_port = results[-1]
                self._router_addr = (
                    f"http://{format_hostport(router_host, router_port)}"
                )
                logger.info("DataWorkers: %s", self._worker_addrs)
                logger.info("Router: %s", self._router_addr)

                if self._shutdown_requested.is_set():
                    return

                # Wave 2: Fork Gateway + Register workers with Router
                async def _register_workers() -> None:
                    async def _register_one(worker_addr: str) -> None:
                        async with session.post(
                            f"{self._router_addr}/register",
                            json={"worker_addr": worker_addr},
                            headers={"Authorization": f"Bearer {self._admin_api_key}"},
                            timeout=aiohttp.ClientTimeout(total=5),
                        ) as resp:
                            resp.raise_for_status()
                        logger.info("Registered DataWorker %s in router", worker_addr)

                    results = await asyncio.gather(
                        *[_register_one(addr) for addr in self._worker_addrs],
                        return_exceptions=True,
                    )
                    errors = [r for r in results if isinstance(r, BaseException)]
                    if errors:
                        raise errors[0]

                gw_result, _ = await asyncio.gather(
                    self._async_fork_on_guard(
                        session,
                        guard_addr_0,
                        "data-gateway",
                        0,
                        [
                            sys.executable,
                            "-m",
                            "areal.infra.data_service.gateway",
                            "--admin-api-key",
                            self._admin_api_key,
                            "--router-addr",
                            self._router_addr,
                            "--forward-timeout",
                            str(cfg.setup_timeout),
                        ],
                    ),
                    _register_workers(),
                )
                gw_host, gw_port = gw_result
                self._gateway_addr = f"http://{format_hostport(gw_host, gw_port)}"
                logger.info("Gateway: %s", self._gateway_addr)
        except Exception:
            # Rollback: kill forked services and delete scheduler workers
            logger.error(
                "DataController initialization failed, rolling back",
                exc_info=True,
            )
            # Kill forked services concurrently
            if self._forked_services:
                await self._async_kill_forked_services(
                    list(reversed(self._forked_services))
                )
                self._forked_services.clear()
            for role in reversed(self._service_roles):
                try:
                    self.scheduler.delete_workers(role=role)
                except Exception:
                    pass
            self._service_roles.clear()
            self.workers.clear()
            self._worker_addrs.clear()
            self._router_addr = ""
            self._gateway_addr = ""
            raise

    # -- Register / Unregister Datasets ------------------------------------

    def register_dataset(
        self,
        dataset_id: str,
        dataset_path: str,
        dataset_type: str,
        dataset_kwargs: dict[str, Any] | None = None,
        tokenizer_or_processor_path: str = "",
        split: str = "train",
        shuffle: bool = True,
        drop_last: bool = True,
        max_length: int | None = None,
    ) -> dict[str, Any]:
        """Register a dataset with the service.

        POST /v1/datasets/register on Gateway.
        """
        self._ensure_initialized()

        payload = {
            "dataset_id": dataset_id,
            "dataset_path": dataset_path,
            "dataset_type": dataset_type,
            "split": split,
            "tokenizer_or_processor_path": tokenizer_or_processor_path,
            "max_length": max_length,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "dataset_kwargs": dataset_kwargs or {},
        }

        from areal.infra.utils.concurrent import run_async_task

        data = run_async_task(
            self._async_gateway_post,
            "/v1/datasets/register",
            self._admin_api_key,
            payload,
            self.config.setup_timeout,
        )

        total_samples = data["dataset_size"]

        self._datasets[data["api_key"]] = {
            "dataset_id": data["dataset_id"],
            "total_samples": total_samples,
            "drop_last": drop_last,
        }
        logger.info(
            "Registered dataset %s: total_samples=%d, workers=%d",
            dataset_id,
            total_samples,
            data["num_workers"],
        )
        return {
            "api_key": data["api_key"],
            "dataset_id": data["dataset_id"],
            "dataset_size": total_samples,
            "total_samples": total_samples,
            "num_workers": data["num_workers"],
        }

    def unregister_dataset(self, dataset_id: str) -> None:
        """Unregister a dataset from the service."""
        self._ensure_initialized()
        from areal.infra.utils.concurrent import run_async_task

        run_async_task(
            self._async_gateway_post,
            "/v1/datasets/unregister",
            self._admin_api_key,
            {"dataset_id": dataset_id},
            30,
        )

        to_remove = [
            k for k, v in self._datasets.items() if v["dataset_id"] == dataset_id
        ]
        for k in to_remove:
            del self._datasets[k]

        logger.info("Unregistered dataset %s", dataset_id)

    # -- Batch cleanup -----------------------------------------------------

    def clear_batches(self) -> None:
        """Clear batch caches and tensor stores on all data workers.

        Called by trainers after each training step, alongside
        ``actor.clear_batches()``, to free memory held by the data
        service instead of relying on TTL-based eviction.
        """
        self._ensure_initialized()
        if not self._worker_addrs:
            return
        from areal.infra.utils.concurrent import run_async_task

        run_async_task(self._async_clear_batches)

    async def _async_clear_batches(self) -> None:
        async def _clear_one(session: aiohttp.ClientSession, addr: str) -> None:
            try:
                async with session.delete(
                    f"{addr}/data/clear",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    resp.raise_for_status()
            except Exception:
                logger.debug("Failed to clear batches on %s", addr)

        async with aiohttp.ClientSession(trust_env=False) as session:
            await asyncio.gather(
                *(_clear_one(session, addr) for addr in self._worker_addrs),
                return_exceptions=True,
            )

    # -- Destroy -----------------------------------------------------------

    def destroy(self) -> None:
        """Shutdown service: unload all datasets, kill services, delete workers."""
        self._shutdown_requested.set()
        future = self._init_future
        self._init_future = None
        if future is not None:
            future.cancel()

        from areal.infra.utils.concurrent import run_async_task

        if self._gateway_addr:
            try:
                run_async_task(
                    self._async_gateway_post,
                    "/v1/shutdown",
                    self._admin_api_key,
                    {},
                    5,
                )
            except Exception as exc:
                logger.debug(
                    "Gateway shutdown request failed (expected during teardown): %s",
                    exc,
                )

        if self._forked_services:
            run_async_task(
                self._async_kill_forked_services,
                list(reversed(self._forked_services)),
            )
        self._forked_services.clear()

        for role in reversed(self._service_roles):
            try:
                self.scheduler.delete_workers(role=role)
                logger.info("Workers deleted for role: %s", role)
            except Exception as exc:
                logger.debug("Could not delete workers for role %s: %s", role, exc)

        self._service_roles.clear()
        self.workers.clear()
        self._worker_addrs.clear()
        self._router_addr = ""
        self._gateway_addr = ""
        self._datasets.clear()

    # -- Internal HTTP helpers (async) -------------------------------------

    async def _async_fork_on_guard(
        self,
        session: Any,
        guard_addr: str,
        role: str,
        worker_index: int,
        raw_cmd: list[str],
        health_path: str = "/health",
    ) -> tuple[str, int]:
        async with session.post(
            f"{guard_addr}/alloc_ports",
            json={"count": 1},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            alloc_data = await resp.json()
        host = alloc_data["host"]
        port = alloc_data["ports"][0]

        cmd = list(raw_cmd) + ["--host", host, "--port", str(port)]

        async with session.post(
            f"{guard_addr}/fork",
            json={
                "role": role,
                "worker_index": worker_index,
                "raw_cmd": cmd,
            },
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()

        self._forked_services.append((guard_addr, role, worker_index))

        addr = f"http://{format_hostport(host, port)}"
        await self._async_wait_for_service(session, f"{addr}{health_path}", role)

        return host, port

    async def _async_wait_for_service(
        self,
        session: Any,
        url: str,
        name: str,
        timeout: float | None = None,
    ) -> None:
        timeout_val = timeout or self.config.setup_timeout
        deadline = time.monotonic() + timeout_val
        while time.monotonic() < deadline:
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        logger.info("%s is ready at %s", name, url)
                        return
            except Exception:
                pass
            await asyncio.sleep(0.1)
        raise TimeoutError(
            f"{name} did not become healthy at {url} within {timeout_val}s"
        )

    async def _async_kill_forked_services(
        self, services: list[tuple[str, str, int]]
    ) -> None:
        async def _kill_one(
            session: aiohttp.ClientSession,
            guard_addr: str,
            role: str,
            worker_index: int,
        ) -> None:
            try:
                async with session.post(
                    f"{guard_addr}/kill_forked_worker",
                    json={"role": role, "worker_index": worker_index},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        logger.info("Killed forked service %s/%d", role, worker_index)
                    else:
                        text = await resp.text()
                        logger.warning(
                            "Failed to kill %s/%d: HTTP %d: %s",
                            role,
                            worker_index,
                            resp.status,
                            text,
                        )
            except Exception as exc:
                logger.error(
                    "Error killing forked service %s/%d: %s",
                    role,
                    worker_index,
                    exc,
                )

        async with aiohttp.ClientSession(trust_env=False) as session:
            await asyncio.gather(
                *(_kill_one(session, *svc) for svc in services),
                return_exceptions=True,
            )

    def _gateway_post(
        self,
        endpoint: str,
        api_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        from areal.infra.utils.concurrent import run_async_task

        return run_async_task(self._async_gateway_post, endpoint, api_key, payload)

    async def _async_gateway_post(
        self,
        endpoint: str,
        api_key: str,
        payload: dict[str, Any],
        timeout: float = 60,
    ) -> dict[str, Any]:
        url = f"{self._gateway_addr}{endpoint}"
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout), trust_env=False
            ) as session:
                async with session.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                ) as resp:
                    if resp.status >= 400:
                        text = await resp.text()
                        raise RuntimeError(
                            f"Gateway {endpoint} returned {resp.status}: {text}"
                        )
                    return await resp.json()
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"Failed to POST {endpoint}: {exc}") from exc
