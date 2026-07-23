# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import shutil
import threading
import traceback
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any

from flask import Flask, jsonify, request
from torchdata.stateful_dataloader import StatefulDataLoader
from werkzeug.serving import make_server

from areal.api import (
    InferenceEngine,
    Job,
    LocalInfServerInfo,
    ModelRequest,
    ModelResponse,
    ParamSpec,
    RolloutWorkflow,
    Scheduler,
    WeightUpdateMeta,
    Worker,
    WorkflowLike,
)
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    InferenceEngineConfig,
    PerfTracerConfig,
    SchedulingSpec,
    SchedulingStrategyType,
)
from areal.infra.rpc.serialization import deserialize_value
from areal.infra.utils.concurrent import run_async_task
from areal.utils import logging, perf_tracer
from areal.utils.data import cycle_dataloader
from areal.utils.dynamic_import import import_from_string
from areal.utils.network import find_free_ports, format_hostport, gethostip
from areal.utils.perf_tracer import trace_perf

from ..staleness_manager import StalenessManager
from ..workflow_executor import BatchTaskDispatcher, TaskIdGenerator

logger = logging.getLogger("RolloutController")


# NOTE: remote task input has a slightly different
# type annotation, which disallows workflow object or types
@dataclass
class _RemoteRolloutTaskInput:
    task_id: int
    data: dict[str, Any]
    workflow: str | None
    workflow_kwargs: dict[str, Any]
    should_accept_fn: str | None
    is_eval: bool = False
    group_size: int = 1
    proxy_addr: str | None = None


@dataclass
class _RemoteRolloutResult:
    task_id: int
    trajectory: dict[str, Any]


class RolloutController:
    def __init__(
        self,
        inf_engine: type[InferenceEngine],
        config: InferenceEngineConfig,
        scheduler: Scheduler,
    ):
        self.inf_engine = inf_engine
        self.config = config
        self.scheduler = scheduler

        # Parse allocation from config.backend
        self.rollout_alloc = ModelAllocation.from_str(config.backend)

        # Worker management
        self.workers: list[Worker] = []  # List of Worker objects from scheduler
        self.server_infos: list[LocalInfServerInfo] = []
        self._worker_role: str
        self._gpus_per_server: int = 1

        # Round-robin scheduling
        self._current_worker_idx = 0

        # State
        self._version_lock = Lock()
        self._version = 0

        self._task_id_generator = TaskIdGenerator()

        # Use provided staleness manager or create a default one
        # The manager will be properly initialized in initialize()
        self._staleness_manager: StalenessManager | None = None

        # Dispatcher will be initialized in initialize() after staleness_manager is ready
        self._dispatcher: (
            BatchTaskDispatcher[_RemoteRolloutTaskInput, _RemoteRolloutResult] | None
        ) = None

        # HTTP callback server
        self._callback_app: Flask | None = None
        self._callback_server = None
        self._callback_server_thread: threading.Thread | None = None
        self._callback_port: int | None = None
        self._callback_host: str | None = None
        self._callback_loop: asyncio.AbstractEventLoop | None = None
        self._callback_loop_ready = threading.Event()

        # Task completion futures
        self._pending_futures: dict[int, asyncio.Future] = {}
        self._futures_lock = threading.Lock()

        # Proxy worker management (for AgentWorkflow support)
        self.proxy_workers: list[Worker] = []
        self.proxy_addrs: list[str] = []
        self._proxy_started = False

        # Proxy gateway server (for online/external access)
        self._proxy_gateway_app = None
        self._proxy_gateway_server = None
        self._proxy_gateway_thread: threading.Thread | None = None
        self._proxy_gateway_port: int | None = None
        self._proxy_gateway_host: str | None = None

    @property
    def _proxy_role(self) -> str:
        """Generate a unique proxy role name based on the worker role.

        This avoids collisions when multiple controllers (e.g., rollout and
        eval-rollout) each fork proxy workers into the same scheduler.
        """
        if not hasattr(self, "_worker_role"):
            raise RuntimeError(
                "Cannot access _proxy_role before initialize() is called"
            )
        return f"proxy-{self._worker_role}"

    def _proxy_engine_name(self, rank: int) -> str:
        """Generate engine name for a proxy worker rank."""
        return f"{self._proxy_role}/{rank}"

    def _engine_name(self, rank: int) -> str:
        """Generate engine name for a worker rank.

        Engine names follow the "role/index" format (e.g., "rollout/0", "rollout/1").
        """
        return f"{self._worker_role}/{rank}"

    def initialize(
        self,
        role: str,
        server_args: dict[str, Any] | None = None,
        server_infos: list[LocalInfServerInfo] | None = None,
        *args,
        **kwargs,
    ):
        # Get scheduling config from kwargs or use defaults
        # Schedule inference engines in the granularity of instance sizes,
        # usually TP x PP.
        self._worker_role = role

        instance_size = (
            self.rollout_alloc.parallel.tp_size * self.rollout_alloc.parallel.pp_size
        )
        dp_size = self.rollout_alloc.parallel.dp_size
        self._gpus_per_server = instance_size

        # The first element of `self.config.scheduling_spec` is the resource spec
        # of workers, aka the RPC server process. Since a worker exactly matches
        # to a single engine instance in the local environment, we can dirrectly
        # use the spec of engines  as the spec of workers here. Engine scheduling
        # specs are ignored.
        sch_spec = SchedulingSpec(**asdict(self.config.scheduling_spec[0]))
        sch_spec.cpu *= instance_size
        sch_spec.mem *= instance_size
        if sch_spec.gpu > 0:
            sch_spec.gpu = instance_size

        job = Job(
            replicas=dp_size,
            tasks=[sch_spec for _ in range(dp_size)],
            scheduling_strategy=self.config.scheduling_strategy,
            role=self._worker_role,
        )

        # Call async scheduler methods synchronously
        run_async_task(
            self._async_initialize, job, server_args, server_infos, *args, **kwargs
        )

        # Initialize staleness manager for global capacity control
        max_concurrent_rollouts = (
            self.config.max_concurrent_rollouts or self.config.consumer_batch_size
        )
        consumer_batch_size = self.config.consumer_batch_size
        self._staleness_manager = StalenessManager(
            version_provider=self,
            max_concurrent_rollouts=max_concurrent_rollouts,
            consumer_batch_size=consumer_batch_size,
            max_staleness=self.config.max_head_offpolicyness,
        )

        # Create and initialize the dispatcher
        qsize = self.config.queue_size or max_concurrent_rollouts * 16
        self._dispatcher = BatchTaskDispatcher[
            _RemoteRolloutTaskInput, _RemoteRolloutResult
        ](
            max_queue_size=qsize,
            task_factory=self._create_submit_callback,
            staleness_manager=self._staleness_manager,
            enable_tracing=self.config.enable_rollout_tracing,
        )
        # Initialize the dispatcher's async task runner
        self._dispatcher.initialize(logger=logger)

        # Start callback server for weight sync coordination
        self._start_callback_server()

    async def _async_initialize(
        self,
        job: Job,
        server_args: dict[str, Any],
        server_infos: list[LocalInfServerInfo] | None = None,
        *args,
        **kwargs,
    ):
        # Create workers via scheduler
        logger.info("Creating workers via scheduler...")
        worker_ids = self.scheduler.create_workers(job=job)
        logger.info(f"Workers created: {worker_ids}")

        # Wait for workers to be ready
        logger.info("Waiting for workers to be ready...")
        self.workers = self.scheduler.get_workers(role=job.role)
        logger.info(f"Workers ready: {[w.id for w in self.workers]}")

        # Get engine class path for dynamic import on workers
        engine_class = self.inf_engine

        # Create and initialize engines on workers
        logger.info("Creating engines...")
        tasks = [
            self.scheduler.create_engine(
                worker_id=worker.id,
                engine=f"{engine_class.__module__}.{engine_class.__name__}",
                engine_name=self._engine_name(rank),
                config=self.config,
            )
            for rank, worker in enumerate(self.workers)
        ]
        await asyncio.gather(*tasks)
        logger.info("Engine created on all workers!")

        logger.info("Calling engine initialization...")
        # Workers are controller-managed: the controller handles staleness
        # globally, so workers must NOT apply their own dp-scaled staleness
        # constraints. Force train_data_parallel_size=1 unless explicitly
        # configured; an explicit None must not survive, or workers fall back
        # to dividing capacity by dist.get_world_size().
        if kwargs.get("train_data_parallel_size") is None:
            kwargs["train_data_parallel_size"] = 1
        if server_infos is not None:
            # Connecting to existing local servers for evaluation
            self.server_infos = server_infos
            assert len(self.server_infos) == len(self.workers), (
                len(self.server_infos),
                len(self.workers),
            )
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="initialize",
                    engine_name=self._engine_name(rank),
                    # args in `engine_api`
                    engine_id=str(rank),
                    addr=f"{info.host}:{info.port}",
                    *args,
                    **kwargs,
                )
                for rank, (worker, info) in enumerate(
                    zip(self.workers, self.server_infos)
                )
            ]
            await asyncio.gather(*tasks)
        elif (
            self.config.scheduling_strategy.type
            == SchedulingStrategyType.colocation.value
        ):
            # Colocation (AWEX) path: multiple servers share a node, so SLURM does
            # NOT isolate GPUs per worker. We must compute base_gpu_id explicitly and
            # inject `_awex_gpus_per_server` so the worker recomputes base_gpu_id from
            # its own SLURM_LOCALID at runtime (the only value guaranteed unique per
            # node-slot). See SGLangBackend.launch_server.
            #
            # NOTE: this assumes a server fits within one node
            # (gpus_per_server <= n_gpus_per_node). Cross-node TP servers would
            # collapse slots_per_node to 1 and collide; only the SLURM_LOCALID
            # path is collision-safe in that case.
            slots_per_node = max(
                1,
                getattr(self.scheduler, "n_gpus_per_node", 8) // self._gpus_per_server,
            )
            launch_tasks = []
            for rank, worker in enumerate(self.workers):
                per_worker_args = {
                    **server_args,
                    "base_gpu_id": (rank % slots_per_node) * self._gpus_per_server,
                    "_awex_gpus_per_server": self._gpus_per_server,
                }
                launch_tasks.append(
                    self.scheduler.async_call_engine(
                        worker_id=worker.id,
                        method="launch_server",
                        engine_name=self._engine_name(rank),
                        server_args=per_worker_args,
                    )
                )
            self.server_infos = await asyncio.gather(*launch_tasks)
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="initialize",
                    engine_name=self._engine_name(rank),
                    engine_id=str(rank),
                    addr=f"{info.host}:{info.port}",
                    *args,
                    **kwargs,
                )
                for rank, (worker, info) in enumerate(
                    zip(self.workers, self.server_infos)
                )
            ]
            await asyncio.gather(*tasks)
        else:
            # Separation path: each rollout server gets its own SLURM-isolated GPUs,
            # so we must NOT override base_gpu_id (SLURM already sets
            # CUDA_VISIBLE_DEVICES per worker). Use the collective launch + addr-less
            # initialize that the verified disaggregated baseline relies on; the
            # worker discovers its server address via name_resolve.
            self.server_infos = await self._collective_rpc_async(
                "launch_server", server_args=server_args
            )
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="initialize",
                    engine_name=self._engine_name(rank),
                    engine_id=str(rank),
                    *args,
                    **kwargs,
                )
                for rank, worker in enumerate(self.workers)
            ]
            await asyncio.gather(*tasks)

        logger.info("All engines are initialized...")

    def destroy(self):
        # Stop background threads and shutdown the async task runner
        if self._dispatcher is not None:
            self._dispatcher.destroy()

        self._stop_callback_server()

        self._collective_rpc("destroy", http_timeout=60.0)

        # Delete workers via scheduler
        if hasattr(self, "_worker_role"):
            try:
                self.scheduler.delete_workers(role=self._worker_role)
                self.workers.clear()
                logger.info("Workers deleted")
            except Exception:
                logger.error(f"Error deleting workers: {traceback.format_exc()}")

        # Delete proxy workers if initialized
        if self._proxy_started:
            try:
                self.scheduler.delete_workers(role=self._proxy_role)
                self.proxy_workers.clear()
                self.proxy_addrs.clear()
                self._proxy_started = False
                logger.info("Proxy workers deleted")
            except Exception:
                logger.error(f"Error deleting proxy workers: {traceback.format_exc()}")

        # Shutdown proxy gateway if initialized
        self._stop_proxy_gateway()
        with self._futures_lock:
            self._pending_futures.clear()

    def start_proxy(self) -> None:
        """Initialize proxy workers for AgentWorkflow support.

        Creates proxy workers colocated with rollout workers. Each proxy worker
        runs a ProxyRolloutServer that connects to the same inference server
        as its corresponding rollout worker.
        """
        if self._proxy_started:
            logger.warning("Proxy workers already initialized")
            return

        if not self.server_infos:
            raise RuntimeError(
                "Cannot initialize proxy workers: rollout not initialized. "
                "Call initialize() first."
            )

        run_async_task(self._async_start_proxy)
        self._proxy_started = True

    async def _async_start_proxy(self) -> None:
        """Async implementation of proxy worker initialization."""
        command = "areal.experimental.openai.proxy.proxy_rollout_server"
        worker_ids = self.scheduler.fork_workers(
            role=self._proxy_role,
            target_role=self._worker_role,
            command=command,
        )
        logger.info(f"Proxy workers forked: {worker_ids}")

        self.proxy_workers = self.scheduler.get_workers(role=self._proxy_role)
        logger.info(f"Proxy workers: {[w.id for w in self.proxy_workers]}")

        engine_class = f"{self.inf_engine.__module__}.{self.inf_engine.__name__}"

        create_tasks = []
        for rank, worker in enumerate(self.proxy_workers):
            create_tasks.append(
                self.scheduler.create_engine(
                    worker_id=worker.id,
                    engine=engine_class,
                    engine_name=self._proxy_engine_name(rank),
                    config=self.config,
                )
            )
        await asyncio.gather(*create_tasks)
        logger.info("Proxy engines created")

        init_tasks = []
        for rank, (worker, server_info) in enumerate(
            zip(self.proxy_workers, self.server_infos, strict=True)
        ):
            init_tasks.append(
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="initialize",
                    engine_name=self._proxy_engine_name(rank),
                    addr=f"{server_info.host}:{server_info.port}",
                )
            )
            self.proxy_addrs.append(
                f"http://{format_hostport(worker.ip, int(worker.worker_ports[0]))}"
            )
        await asyncio.gather(*init_tasks)

        logger.info(f"Proxy servers initialized. Addresses: {self.proxy_addrs}")

    def get_proxy_addr(self, rank: int) -> str:
        """Get the proxy server address for a given rollout worker rank.

        Parameters
        ----------
        rank : int
            The rank of the rollout worker

        Returns
        -------
        str
            The HTTP address of the corresponding proxy server
        """
        if not self._proxy_started:
            raise RuntimeError(
                "Proxy workers not initialized. Call start_proxy() first."
            )
        if rank >= len(self.proxy_addrs):
            raise IndexError(
                f"Invalid rank {rank}, only {len(self.proxy_addrs)} proxy workers"
            )
        return self.proxy_addrs[rank]

    def start_proxy_gateway(self) -> None:
        """Start the proxy gateway for external access.

        Creates a FastAPI server that routes requests to backend proxy
        workers. Requires ``start_proxy()`` to have been called first.
        """
        if not self._proxy_started:
            raise RuntimeError(
                "Proxy workers not initialized. Call start_proxy() first."
            )
        if self._proxy_gateway_host is not None:
            logger.warning("Proxy gateway already running")
            return

        from areal.experimental.openai.proxy.proxy_gateway import (
            create_proxy_gateway_app,
        )

        agent_cfg = self.config.agent

        app = create_proxy_gateway_app(
            proxy_addrs=self.proxy_addrs,
            admin_api_key=agent_cfg.admin_api_key
            if agent_cfg is not None
            else "areal-admin-key",
        )

        self._proxy_gateway_port = find_free_ports(1)[0]
        self._proxy_gateway_host = gethostip()
        self._proxy_gateway_app = app

        def serve():
            import uvicorn

            try:
                config = uvicorn.Config(
                    app,
                    host="0.0.0.0",
                    port=self._proxy_gateway_port,
                    log_level="warning",
                    access_log=False,
                )
                server = uvicorn.Server(config)
                self._proxy_gateway_server = server
                server.run()
            except Exception:
                logger.error("Proxy gateway thread crashed", exc_info=True)

        self._proxy_gateway_thread = threading.Thread(target=serve, daemon=True)
        self._proxy_gateway_thread.start()

        # Wait for uvicorn to bind the port before propagating the address
        # to worker engines via collective RPC.
        import time

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if (
                self._proxy_gateway_server is not None
                and self._proxy_gateway_server.started
            ):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(
                "Proxy gateway failed to start within 10s. "
                f"Cannot propagate address "
                f"{self._proxy_gateway_host}:{self._proxy_gateway_port}"
            )

        logger.info(
            "Proxy gateway started on "
            f"{self._proxy_gateway_host}:{self._proxy_gateway_port}"
        )

        # Propagate proxy_gateway_addr to all rollout worker engines
        # so that _resolve_workflow can pick it up for online mode.
        self._collective_rpc(
            "set_proxy_gateway_addr",
            addr=self.proxy_gateway_addr,
        )

    @property
    def proxy_gateway_addr(self) -> str:
        """Single URL for external users."""
        if self._proxy_gateway_host is None:
            raise RuntimeError("Proxy gateway not started")
        return f"http://{format_hostport(self._proxy_gateway_host, self._proxy_gateway_port)}"

    def _stop_proxy_gateway(self) -> None:
        """Stop the proxy gateway server if running."""
        if self._proxy_gateway_host is None:
            return
        logger.info("Stopping proxy gateway...")
        if self._proxy_gateway_server is not None:
            self._proxy_gateway_server.should_exit = True
        if self._proxy_gateway_thread is not None:
            self._proxy_gateway_thread.join(timeout=30.0)
            if self._proxy_gateway_thread.is_alive():
                logger.warning(
                    "Proxy gateway thread did not exit within 30s; "
                    "daemon thread will be killed on process exit"
                )
        self._proxy_gateway_app = None
        self._proxy_gateway_server = None
        self._proxy_gateway_thread = None
        self._proxy_gateway_port = None
        self._proxy_gateway_host = None

    def _start_callback_server(self):
        """Start Flask HTTP server to receive callbacks from RolloutCallback."""
        if self._callback_server is not None:
            logger.warning("Callback server already running")
            return

        app = Flask(__name__)
        app.logger.disabled = True

        @app.route("/callback/init_weights_group", methods=["POST"])
        def init_weights_group():
            payload = request.get_json() or {}
            meta = deserialize_value(payload.get("meta"))
            self._callback_loop.run_until_complete(self.init_weights_update_group(meta))
            return jsonify({"status": "ok"})

        @app.route("/callback/update_weights_xccl", methods=["POST"])
        def update_weights():
            payload = request.get_json() or {}
            meta = deserialize_value(payload.get("meta"))
            param_specs = deserialize_value(payload.get("param_specs"))
            self._callback_loop.run_until_complete(
                self.update_weights_from_distributed(meta, param_specs)
            )
            return jsonify({"status": "ok"})

        @app.route("/callback/update_weights_disk", methods=["POST"])
        def update_weights_disk():
            payload = request.get_json() or {}
            meta = deserialize_value(payload.get("meta"))
            self._callback_loop.run_until_complete(self.update_weights_from_disk(meta))
            return jsonify({"status": "ok"})

        @app.route("/callback/pause_generation", methods=["POST"])
        def pause_generation():
            self._callback_loop.run_until_complete(self.pause_generation())
            return jsonify({"status": "ok"})

        @app.route("/callback/continue_generation", methods=["POST"])
        def continue_generation():
            self._callback_loop.run_until_complete(self.continue_generation())
            return jsonify({"status": "ok"})

        @app.route("/callback/onload", methods=["POST"])
        def onload():
            payload = request.get_json() or {}
            tags = payload.get("tags")
            self.onload(tags=tags)
            return jsonify({"status": "ok"})

        @app.route("/callback/offload", methods=["POST"])
        def offload():
            payload = request.get_json() or {}
            tags = payload.get("tags")
            self.offload(tags=tags)
            return jsonify({"status": "ok"})

        @app.route("/callback/rollout_complete", methods=["POST"])
        def rollout_complete():
            payload = request.get_json() or {}
            task_id = payload.get("task_id")
            try:
                self._resolve_task_future(task_id)
                return jsonify({"status": "ok"})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @app.errorhandler(Exception)
        def handle_error(e):
            logger.error(f"Callback handler error: {e}")
            return jsonify({"error": str(e)}), 500

        self._callback_port = find_free_ports(1)[0]
        self._callback_host = gethostip()
        self._callback_app = app
        self._callback_server = make_server(
            self._callback_host, self._callback_port, app, threaded=False
        )

        # Suppress Werkzeug access logs (e.g., "POST /callback/rollout_complete 200 -")
        # Override log_request directly on the request handler class
        self._callback_server.RequestHandlerClass.log_request = (
            lambda self, *args, **kwargs: None
        )

        # Also configure Werkzeug logger level for any other log messages
        import logging as stdlib_logging

        werkzeug_logger = stdlib_logging.getLogger("werkzeug")
        werkzeug_logger.setLevel(stdlib_logging.WARNING)

        def serve_forever():
            # Create and set event loop for this thread
            self._callback_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._callback_loop)
            # Signal that the loop is ready
            self._callback_loop_ready.set()
            logger.info(
                f"Callback server started on {format_hostport(self._callback_host, self._callback_port)}"
            )
            self._callback_server.serve_forever()

        self._callback_server_thread = threading.Thread(
            target=serve_forever, daemon=True
        )
        self._callback_server_thread.start()
        # Wait for loop to be created
        self._callback_loop_ready.wait()

    def _stop_callback_server(self):
        """Stop the callback server if running."""
        if self._callback_server is not None:
            logger.info("Stopping callback server...")
            self._callback_server.shutdown()
            if self._callback_loop is not None:
                self._callback_loop.close()
            self._callback_server = None
            self._callback_app = None
            self._callback_server_thread = None
            self._callback_port = None
            self._callback_host = None
            self._callback_loop = None
            self._callback_loop_ready.clear()

    @property
    def callback_addr(self) -> str:
        """Return callback server address as 'host:port'."""
        if self._callback_host is None or self._callback_port is None:
            raise RuntimeError("Callback server not started")
        return format_hostport(self._callback_host, self._callback_port)

    def _resolve_task_future(self, task_id: int):
        """Resolve a pending future with the task result."""
        with self._futures_lock:
            future = self._pending_futures.pop(task_id, None)
        if future:
            future.get_loop().call_soon_threadsafe(future.set_result, None)

    def _collective_rpc(self, method: str, *args, **kwargs) -> list[Any]:
        return run_async_task(self._collective_rpc_async, method, *args, **kwargs)

    async def _collective_rpc_async(self, method: str, *args, **kwargs) -> list[Any]:
        return await self._generic_collective_rpc_async(
            method, self.workers, self._engine_name, *args, **kwargs
        )

    def _proxy_collective_rpc(self, method: str, *args, **kwargs) -> list[Any]:
        return run_async_task(self._proxy_collective_rpc_async, method, *args, **kwargs)

    async def _proxy_collective_rpc_async(
        self, method: str, *args, **kwargs
    ) -> list[Any]:
        return await self._generic_collective_rpc_async(
            method, self.proxy_workers, self._proxy_engine_name, *args, **kwargs
        )

    async def _generic_collective_rpc_async(
        self,
        method: str,
        workers: list[Worker],
        engine_name_fn: Callable[[int], str],
        *args,
        **kwargs,
    ) -> list[Any]:
        tasks = [
            self.scheduler.async_call_engine(
                worker_id=worker.id,
                method=method,
                engine_name=engine_name_fn(rank),
                *args,
                **kwargs,
            )
            for rank, worker in enumerate(workers)
        ]
        return await asyncio.gather(*tasks)

    def _choose_worker(self) -> tuple[Worker, int]:
        """Choose a worker for the next request using round-robin scheduling.

        Returns
        -------
        tuple[Worker, int]
            The chosen worker object and its rank
        """
        if not self.workers:
            raise RuntimeError("No workers available to choose from.")
        worker = self.workers[self._current_worker_idx]
        rank = self._current_worker_idx
        self._current_worker_idx = (self._current_worker_idx + 1) % len(self.workers)
        return worker, rank

    def _resolve_workflow_str(self, workflow: WorkflowLike | None) -> str | None:
        """Resolve workflow to a string import path.

        Handles RolloutWorkflow, agent workflow instances/classes, string paths,
        and ``None`` (online mode).
        """
        # None workflow = online mode (config-driven)
        if workflow is None:
            return None

        # String paths - return as-is
        if isinstance(workflow, str):
            return workflow

        # RolloutWorkflow classes
        elif isinstance(workflow, type) and issubclass(workflow, RolloutWorkflow):
            return f"{workflow.__module__}.{workflow.__name__}"

        # RolloutWorkflow instances
        elif isinstance(workflow, RolloutWorkflow):
            return f"{workflow.__module__}.{workflow.__class__.__name__}"

        # Agent-like workflow classes
        elif isinstance(workflow, type):
            return f"{workflow.__module__}.{workflow.__name__}"

        # Agent-like workflow instances
        else:
            return f"{workflow.__module__}.{workflow.__class__.__name__}"

    def _resolve_should_accept_fn(
        self, should_accept_fn: Callable[[dict[str, Any]], bool] | str | None
    ):
        if callable(should_accept_fn):
            raise RuntimeError(
                "If given, `should_accept_fn` must be an importable string path, e.g., 'my_module.filter_func'."
            )
        if should_accept_fn is not None:
            try:
                import_from_string(should_accept_fn)
            except Exception:
                raise RuntimeError(
                    f"Failed to import `should_accept_fn` from string path: {should_accept_fn}"
                )
        return should_accept_fn

    def _rollout_stats(self) -> str:
        stats = self._staleness_manager.get_stats()
        return (
            f"enqueued: {stats.enqueued}, "
            f"running: {stats.running}, "
            f"accepted: {stats.accepted}, "
            f"rejected: {stats.rejected}."
        )

    def _create_submit_callback(self, pending_task: _RemoteRolloutTaskInput):
        async def _submit_then_wait() -> _RemoteRolloutResult | None:
            # Choose worker via round-robin
            worker, rank = self._choose_worker()
            engine_name = self._engine_name(rank)

            # NOTE: No need to call `on_rollout_submitted` here.
            # This function will be passed to `BatchTaskDispather` where
            # `on_rollout_submitted` will be called upon dispatching
            task_id = pending_task.task_id

            manager = self.staleness_manager

            try:
                # Set future for this task
                future = asyncio.get_event_loop().create_future()
                with self._futures_lock:
                    self._pending_futures[task_id] = future

                proxy_addr = pending_task.proxy_addr
                if self._proxy_started and proxy_addr is None:
                    proxy_addr = self.get_proxy_addr(rank)
                engine_task_id = await self.scheduler.async_call_engine(
                    worker.id,
                    "submit",
                    engine_name=engine_name,
                    data=pending_task.data,
                    workflow=pending_task.workflow,
                    workflow_kwargs=pending_task.workflow_kwargs,
                    should_accept_fn=pending_task.should_accept_fn,
                    http_timeout=self.config.request_timeout,
                    is_eval=pending_task.is_eval,
                    group_size=pending_task.group_size,
                    task_id=task_id,
                    callback_addr=f"http://{self.callback_addr}/callback/rollout_complete",
                    proxy_addr=proxy_addr,
                )

                assert task_id == engine_task_id, (task_id, engine_task_id)

                # Wait for callback to resolve the future
                await asyncio.wait_for(future, timeout=self.config.request_timeout)

                # Fetch the result
                result = await self.scheduler.async_call_engine(
                    worker.id,
                    "wait_for_task",
                    engine_name=engine_name,
                    task_id=engine_task_id,
                    timeout=0.1,  # A short time to prevent blocking other requests
                    raise_timeout=False,
                    http_timeout=self.config.request_timeout,
                )

                traj = result
                if traj is not None:
                    manager.on_rollout_accepted()
                    if self.config.enable_rollout_tracing:
                        logger.info(
                            f"Finish and accept rollout. {self._rollout_stats()}"
                        )
                    return _RemoteRolloutResult(task_id=task_id, trajectory=traj)

                manager.on_rollout_rejected()
                if self.config.enable_rollout_tracing:
                    logger.info(f"Finish but reject rollout. {self._rollout_stats()}")
                return None

            except TimeoutError:
                if task_id is not None:
                    with self._futures_lock:
                        self._pending_futures.pop(task_id, None)
                manager.on_rollout_rejected()
                logger.error(f"Rollout timed out after {self.config.request_timeout}s")
                return None
            except Exception as exc:
                if task_id is not None:
                    with self._futures_lock:
                        self._pending_futures.pop(task_id, None)
                manager.on_rollout_rejected()
                logger.error("Workflow execution failed: %s", exc, exc_info=True)
                return None

        return _submit_then_wait

    def get_capacity(self):
        return self.staleness_manager.get_capacity()

    def submit(
        self,
        data: dict[str, Any],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: str | None = None,
        task_id: int | None = None,
        is_eval: bool = False,
        group_size: int = 1,
        proxy_addr: str | None = None,
    ) -> int:
        workflow_str = self._resolve_workflow_str(workflow)
        should_accept_fn = self._resolve_should_accept_fn(should_accept_fn)
        if workflow_kwargs is None:
            workflow_kwargs = {}

        # NOTE: RolloutController does not support `should_accept_fn`
        # If the workflow's result should be aborted,
        # `arun_episode` should return None instead.
        if task_id is None:
            task_id = self._task_id_generator.next()
        task_input = _RemoteRolloutTaskInput(
            data=data,
            workflow=workflow_str,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            task_id=task_id,
            is_eval=is_eval,
            group_size=group_size,
            proxy_addr=proxy_addr,
        )

        # Delegate to dispatcher
        self.dispatcher.submit_task_input(task_input)
        return task_id

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[dict[str, Any] | None]:
        # Delegate to dispatcher and extract trajectories
        results = self.dispatcher.wait_results(count, timeout, raise_timeout)
        # Log and trace
        if self.config.enable_rollout_tracing:
            logger.info("Rollout results are ready!")

        return [r.trajectory if r is not None else None for r in results]

    @trace_perf("rollout_controller.rollout_batch", category="scheduler")
    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: str | None = None,
        group_size: int = 1,
    ) -> list[dict[str, Any]]:
        perf_tracer.instant(
            "rollout_controller.rollout_batch",
            category="scheduler",
            args={"data": len(data)},
        )
        for item in data:
            self.submit(
                data=item,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                should_accept_fn=should_accept_fn,
                group_size=group_size,
            )
        results = self.wait(count=len(data))
        # Return list of trajectories
        return [r for r in results if r is not None]

    @trace_perf("rollout_controller.prepare_batch", category="scheduler")
    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ) -> list[dict[str, Any]]:
        """Prepare a batch with controlled staleness.

        Continuously submits from dataloader and waits for results, ensuring at least
        two batches are pending to maximize overlap.

        See :meth:`~areal.api.engine_api.InferenceEngine.prepare_batch` for parameters.
        """

        workflow_str = self._resolve_workflow_str(workflow)
        if workflow_kwargs is None:
            workflow_kwargs = {}

        def task_input_generator():
            for data in cycle_dataloader(dataloader):
                for item in data:
                    yield _RemoteRolloutTaskInput(
                        data=item,
                        workflow=workflow_str,
                        workflow_kwargs=workflow_kwargs,
                        should_accept_fn=should_accept_fn,
                        task_id=self._task_id_generator.next(),
                        group_size=group_size,
                    )

        if not hasattr(self, "data_generator"):
            self.data_generator = task_input_generator()

        # Delegate to dispatcher
        assert dataloader.batch_size is not None
        results = self.dispatcher.active_submit_and_wait(
            self.data_generator, batch_size=dataloader.batch_size, dynamic_bs=dynamic_bs
        )

        # Return list of trajectories
        trajectories = [r.trajectory if r is not None else None for r in results]
        return [t for t in trajectories if t is not None]

    def compute_logp(self, data: list[dict[str, Any]]) -> list[Any]:
        """Compute token log-probabilities for trajectories via remote workers."""
        if len(data) == 0:
            return []

        async def _compute():
            indexed_chunks: list[list[int]] = []
            tasks = []
            n_workers = len(self.workers)
            if n_workers == 0:
                raise RuntimeError("No workers available for compute_logp.")

            for rank, worker in enumerate(self.workers):
                idxs = list(range(rank, len(data), n_workers))
                if not idxs:
                    continue
                chunk = [data[i] for i in idxs]
                indexed_chunks.append(idxs)
                tasks.append(
                    self.scheduler.async_call_engine(
                        worker_id=worker.id,
                        method="compute_logp",
                        engine_name=self._engine_name(rank),
                        data=chunk,
                        http_timeout=self.config.request_timeout,
                    )
                )
            rpc_results = await asyncio.gather(*tasks)
            merged: list[Any] = [None] * len(data)
            for idxs, chunk_result in zip(indexed_chunks, rpc_results):
                if len(chunk_result) != len(idxs):
                    raise RuntimeError(
                        f"compute_logp result length mismatch: got {len(chunk_result)}, expected {len(idxs)}"
                    )
                for out_idx, value in zip(idxs, chunk_result):
                    merged[out_idx] = value
            return merged

        return run_async_task(_compute)

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request.

        This method provides direct access to the inference engine's generation capabilities
        for single requests, bypassing the workflow system.

        Parameters
        ----------
        req : ModelRequest
            The model request containing input data and generation parameters

        Returns
        -------
        ModelResponse
            The generated response from the model
        """
        # Choose worker and delegate
        worker, rank = self._choose_worker()

        # Call agenerate on engine via scheduler
        return await self.scheduler.async_call_engine(
            worker_id=worker.id,
            method="agenerate",
            engine_name=self._engine_name(rank),
            req=req,
        )

    async def init_weights_update_group(self, meta: WeightUpdateMeta) -> None:
        tasks = [
            self.scheduler.async_call_engine(
                worker_id=worker.id,
                method="init_weights_update_group",
                engine_name=self._engine_name(rank),
                meta=meta,
                xccl_group_ranks=[rank],
            )
            for rank, worker in enumerate(self.workers)
        ]
        await asyncio.gather(*tasks)

    async def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ):
        await self._collective_rpc_async(
            "update_weights_from_distributed", meta=meta, param_specs=param_specs
        )

    async def update_weights_from_disk(self, meta: WeightUpdateMeta):
        meta.clear_checkpoint_after_load = False
        await self._collective_rpc_async("update_weights_from_disk", meta=meta)
        shutil.rmtree(meta.path, ignore_errors=True)

    async def pause_generation(self):
        await self._collective_rpc_async("pause_generation")

    def pause_generation_sync(self):
        self._collective_rpc("pause_generation", http_timeout=120.0)

    async def continue_generation(self):
        await self._collective_rpc_async("continue_generation")

    def set_version(self, version: int) -> None:
        with self._version_lock:
            self._version = version
            self._collective_rpc("set_version", version=version, http_timeout=60.0)
            if self._proxy_started:
                self._proxy_collective_rpc(
                    "set_version", version=version, http_timeout=60.0
                )

    def get_version(self) -> int:
        with self._version_lock:
            return self._version

    def pause(self):
        self.dispatcher.pause()
        self._collective_rpc("pause", http_timeout=60.0)

    def resume(self):
        self._collective_rpc("resume", http_timeout=60.0)
        self.dispatcher.resume()

    def offload(self, tags: list[str] | None = None):
        logger.info(
            f"[AWEX] RolloutController.offload(tags={tags}) dispatching to workers"
        )
        self._collective_rpc("offload", tags=tags, http_timeout=120.0)
        logger.info(f"[AWEX] RolloutController.offload(tags={tags}) completed")

    def abort_all_requests(self):
        self._collective_rpc("abort_all_requests", http_timeout=60.0)

    def onload(self, tags: list[str] | None = None):
        self._collective_rpc("onload", tags=tags, http_timeout=120.0)

    def export_stats(self) -> dict[str, float]:
        all_raw_stats = self._collective_rpc(method="export_stats", http_timeout=60.0)
        stats = defaultdict(float)
        counts = defaultdict(int)

        for raw_stats in all_raw_stats:
            for k, v in raw_stats.items():
                if k.endswith("__count"):
                    counts[k] += v
                else:
                    stats[k] += v * raw_stats.get(k + "__count", 0)

        # Average non-count stats
        final_stats = {}
        for k, v in stats.items():
            count_key = k + "__count"
            if count_key in counts and counts[count_key] > 0:
                final_stats[k] = v / counts[count_key]
        return final_stats

    def config_perf_tracer(self, config: PerfTracerConfig, role: str) -> None:
        async def _call():
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="config_perf_tracer",
                    engine_name=self._engine_name(rank),
                    rank=rank,
                    role=role,
                    config=config,
                )
                for rank, worker in enumerate(self.workers)
            ]
            return await asyncio.gather(*tasks)

        run_async_task(_call)

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        self._collective_rpc("save_perf_tracer", step=step, force=force)

    @property
    def staleness_manager(self):
        return self._staleness_manager

    @property
    def dispatcher(
        self,
    ) -> BatchTaskDispatcher[_RemoteRolloutTaskInput, _RemoteRolloutResult]:
        """Get the task dispatcher, ensuring initialization has been called."""
        if self._dispatcher is None:
            raise RuntimeError(
                "RolloutController.initialize() must be called before scheduling rollouts."
            )
        return self._dispatcher

    @property
    def runner(self):
        """For backward compatibility. The runner is now owned by the dispatcher."""
        return self.dispatcher.runner
