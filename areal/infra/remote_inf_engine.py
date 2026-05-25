# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import random
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from datetime import datetime
from logging import Logger
from threading import Lock
from typing import TYPE_CHECKING, Any, Protocol

import aiohttp
import numpy as np
import ray
import requests
import torch.distributed as dist
import uvloop
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api import (
    InferenceEngine,
    LocalInfServerInfo,
    ModelRequest,
    ModelResponse,
    ParamSpec,
    RolloutWorkflow,
    WeightUpdateMeta,
    WorkflowLike,
)
from areal.api.cli_args import InferenceEngineConfig
from areal.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    WeightUpdateRequests,
)
from areal.infra import workflow_context
from areal.infra.platforms import current_platform
from areal.infra.utils.concurrent import get_executor
from areal.infra.utils.http import arequest_with_retry, get_default_connector
from areal.infra.utils.launcher import wait_llm_server_addrs
from areal.infra.utils.proc import kill_process_tree
from areal.utils import logging, name_resolve, names
from areal.utils.data import concat_padded_tensors
from areal.utils.dynamic_import import import_from_string
from areal.utils.network import (
    find_free_ports,
    format_hostport,
    gethostip,
    split_hostport,
)
from areal.utils.perf_tracer import trace_perf

from .workflow_executor import WorkflowExecutor

if TYPE_CHECKING:
    from areal.experimental.openai import InteractionWithTokenLogpReward

RID_CACHE_SIZE = 128

logger = logging.getLogger("RemoteInfEngine")


class GroupedRolloutWorkflow(RolloutWorkflow):
    def __init__(
        self,
        workflow: RolloutWorkflow,
        group_size: int,
        logger: Logger,
    ):
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        self.workflow = workflow
        self.group_size = group_size
        self.logger = logger

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        from areal.experimental.openai import InteractionWithTokenLogpReward

        results = await asyncio.gather(
            *[self.workflow.arun_episode(engine, data) for _ in range(self.group_size)]
        )

        valid_results = [r for r in results if r is not None]

        # All results None -> return None
        if not valid_results:
            return None

        # Some results None -> warn and continue with valid ones
        if len(valid_results) < len(results):
            self.logger.warning(
                f"GroupedRolloutWorkflow: {len(results) - len(valid_results)}/{len(results)} "
                "trajectories returned None, using remaining results"
            )

        # Check if results are InteractionWithTokenLogpReward dicts
        first = valid_results[0]
        if (
            isinstance(first, dict)
            and first
            and all(
                isinstance(v, InteractionWithTokenLogpReward) for v in first.values()
            )
        ):
            # Merge dicts - each result is {completion_id: InteractionWithTokenLogpReward}
            merged: dict[str, InteractionWithTokenLogpReward] = {}
            for result in valid_results:
                merged.update(result)
            return merged if merged else None

        # Otherwise, tensor dicts - concatenate
        concatenated = concat_padded_tensors(valid_results)
        return concatenated if concatenated else None


class RemoteInfBackendProtocol(Protocol):
    """Protocol defining backend-specific operations for remote inference engines.

    This protocol abstracts the differences between various remote inference servers
    (SGLang, vLLM, etc.) by defining a common interface for:
    - Building HTTP requests with backend-specific formats
    - Parsing backend-specific responses
    - Handling weight updates
    - Managing control flow (pause/resume)
    - Supporting optional features (LoRA)

    Implementations can raise NotImplementedError for unsupported features.
    """

    def build_generation_request(
        self, req: ModelRequest, with_lora: bool, version: int
    ) -> HttpRequest:
        """Build HTTP request for text generation.

        Parameters
        ----------
        req : ModelRequest
            The generation request containing input and parameters
        with_lora : bool
            Whether to specify a LoRA to use
        version : int
            The current weight version for versioned LoRA names

        Returns
        -------
        HttpRequest
            The HTTP request with endpoint and payload
        """
        ...

    def parse_generation_response(
        self, response: dict[str, Any]
    ) -> HttpGenerationResult:
        """Parse generation response into standard format.

        Parameters
        ----------
        response : Dict[str, Any]
            The raw JSON response from the server

        Returns
        -------
        HttpGenerationResult
            Parsed result with tokens, logprobs, and stop reason
        """
        ...

    def build_disk_weight_update_requests(
        self, meta: WeightUpdateMeta
    ) -> WeightUpdateRequests:
        """Build requests for loading weights from disk.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing path and configuration

        Returns
        -------
        WeightUpdateRequests
            Collection of HTTP requests (may be multiple for LoRA workflows)
        """
        ...

    def build_distributed_weight_update_requests(
        self,
        meta: WeightUpdateMeta,
        param_specs: list[ParamSpec],
    ) -> WeightUpdateRequests:
        """Build requests for distributed weight update via NCCL/XCCL.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing communication group info
        param_specs : List[ParamSpec]
            Specifications for parameters to be updated

        Returns
        -------
        WeightUpdateRequests
            Collection of HTTP requests for distributed update
        """
        ...

    def build_init_weights_group_request(
        self, addr: str, server_idx: int, meta: WeightUpdateMeta
    ) -> HttpRequest:
        """Build request to initialize weight update xccl group.

        Parameters
        ----------
        addr : str
            Server address
        server_idx : int
            Index of this server in the server list
        meta : WeightUpdateMeta
            Metadata containing communication backend configuration

        Returns
        -------
        HttpRequest
            The HTTP request to initialize the group
        """
        ...

    def get_pause_request(self) -> HttpRequest:
        """Get request to pause generation.

        Returns
        -------
        HttpRequest
            The HTTP request to pause generation

        Raises
        ------
        NotImplementedError
            If pause is not supported by this backend
        """
        ...

    def get_resume_request(self) -> HttpRequest:
        """Get request to resume generation.

        Returns
        -------
        HttpRequest
            The HTTP request to resume generation

        Raises
        ------
        NotImplementedError
            If resume is not supported by this backend
        """
        ...

    def get_health_check_request(self) -> HttpRequest:
        """Get the health check request.

        Returns
        -------
        HttpRequest
            The HTTP request for health checks
        """
        ...

    def get_offload_request(self) -> HttpRequest:
        """Get request to offload model memory.

        Returns
        -------
        HttpRequest
            The HTTP request to offload model memory

        Raises
        ------
        NotImplementedError
            If offload is not supported by this backend
        """
        ...

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        """Get request to onload model memory.

        Parameters
        ----------
        tags : list[str], optional
            Tags to onload specific components. If None, onloads all components.

        Returns
        -------
        HttpRequest
            The HTTP request to onload model memory

        Raises
        ------
        NotImplementedError
            If onload is not supported by this backend
        """
        ...

    def launch_server(self, server_args: dict[str, Any]) -> subprocess.Popen:
        """Launch inference server subprocess.

        Parameters
        ----------
        server_args : dict[str, Any]
            Server configuration arguments for build_cmd_from_args

        Returns
        -------
        subprocess.Popen
            The launched server process
        """
        ...


class RemoteInfEngine(InferenceEngine):
    """
    Base implementation for HTTP-based remote inference engines.

    This class provides common functionality for communicating with remote
    inference servers via HTTP REST APIs. Backend-specific behaviors are
    delegated to an injected RemoteInfBackendProtocol implementation.

    Uses composition pattern - instantiate directly with a backend rather
    than inheriting from this class.

    Parameters
    ----------
    config : InferenceEngineConfig
        Configuration for the inference engine
    backend : RemoteInfBackendProtocol
        Backend implementation providing server-specific behavior
    """

    def __init__(
        self, config: InferenceEngineConfig, backend: RemoteInfBackendProtocol
    ):
        self.config = config
        self.backend = backend

        self.rid_to_address = {}
        # Maintain the addresses for the recent 128 requests
        self.rid_queue = []
        self.addresses = []
        self.server_idx = 0

        self._version = 0

        self.lock = Lock()

        self._workflow_executor: WorkflowExecutor | None = None
        self._initialized = False
        self._proxy_gateway_addr: str | None = None
        self.local_server_processes: list[LocalInfServerInfo] = []

    def _wait_for_server(self, address: str, process: subprocess.Popen | None = None):
        """Wait for a server to become healthy."""
        try:
            host, port = split_hostport(address)
            base_url = f"http://{format_hostport(host, port)}"
        except ValueError:
            base_url = f"http://{address}"
        tik = time.time()
        while time.time() - tik < self.config.setup_timeout:
            if self.check_health(base_url):
                return
            time.sleep(1)
        raise TimeoutError("server launch failed")

    def check_health(self, base_url):
        """Check if server is healthy."""
        try:
            health_req = self.backend.get_health_check_request()
            url = f"{base_url}{health_req.endpoint}"
            response = requests.request(
                health_req.method, url, json=health_req.payload, timeout=30
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def initialize(
        self,
        engine_id: str | None = None,
        addr: str | list[str] | None = None,
        train_data_parallel_size: int | None = None,
    ):
        """Initialize the engine by discovering and connecting to servers.

        Parameters
        ----------
        engine_id : Optional[str]
            Unique identifier for this engine instance
        addr : str | List[str] | None
            Server address(es) to connect to. If None, will auto-discover.
        train_data_parallel_size : int | None
            Data parallel size of the training engine
        """
        if engine_id is None:
            if dist.is_initialized():
                engine_id = str(dist.get_rank())
            else:
                engine_id = uuid.uuid4().hex
        self.engine_id = engine_id
        self.logger = logging.getLogger(f"[RemoteInfEngine Rank {engine_id}]")

        if addr:
            self.addresses = addr if isinstance(addr, list) else [addr]
            self.logger.info("Get server addresses from the `addr` argument.")
        elif len(self.local_server_processes) > 0:
            self.addresses = [
                format_hostport(s.host, s.port) for s in self.local_server_processes
            ]
            self.logger.info("Get server addresses from the local subprocess.")
        elif (
            self.config.experiment_name is not None
            and self.config.trial_name is not None
        ):
            try:
                self.addresses = wait_llm_server_addrs(
                    experiment_name=self.config.experiment_name,
                    trial_name=self.config.trial_name,
                    timeout=1,
                )
                self.logger.info("Get server addresses from name_resolve.")
            except (TimeoutError, RuntimeError):
                self.logger.info(
                    "Failed to get server addresses from name_resolve, "
                    "falling back to environment variable."
                )
                addrs_str = os.getenv("AREAL_LLM_SERVER_ADDRS")
                if addrs_str:
                    # When addr is not provided, fallback to reading addrs from env var
                    self.addresses = addrs_str.split(",")
                    self.logger.info("Get server addresses from environment variable.")

        if not self.addresses:
            raise RuntimeError(
                "No configured inference servers. "
                "Please pass in server addresses by arguments "
                "for `initialize` or environment "
                "variable `AREAL_LLM_SERVER_ADDRS`."
            )

        self.logger.info("Waiting for server ready...")
        for addr_ in self.addresses:
            self._wait_for_server(addr_)
        self.server_idx = random.randint(0, len(self.addresses) - 1)
        self.logger.info("Servers are all ready!")

        self.workflow_executor = WorkflowExecutor(
            config=self.config,
            inference_engine=self,
        )
        self.workflow_executor.initialize(
            logger=self.logger, train_data_parallel_size=train_data_parallel_size
        )
        self._initialized = True

    def destroy(self):
        """Destroy the engine and clean up resources."""
        self._initialized = False
        if self._workflow_executor is not None:
            self._workflow_executor.destroy()
        if len(self.local_server_processes) > 0:
            self.teardown_server()

    @property
    def workflow_executor(self) -> WorkflowExecutor:
        """Get the workflow executor of the inference engine."""
        if self._workflow_executor is None:
            raise RuntimeError("WorkflowExecutor is not initialized")
        return self._workflow_executor

    @workflow_executor.setter
    def workflow_executor(self, workflow_executor: WorkflowExecutor):
        """Set the workflow executor of the inference engine."""
        self._workflow_executor = workflow_executor

    @property
    def initialized(self) -> bool:
        return self._initialized

    def set_version(self, version):
        """Set the current weight version."""
        with self.lock:
            self._version = version

    def get_version(self):
        """Get the current weight version."""
        with self.lock:
            return self._version

    def set_proxy_gateway_addr(self, addr: str) -> None:
        """Set the proxy gateway address.

        Called by ``RolloutController.start_proxy_gateway()`` via
        collective RPC after the proxy gateway is started.
        """
        # HACK: We'd better not have this kind of setter beyond well-defined APIs,
        # but for now it's a workaround for the next release.
        self._proxy_gateway_addr = addr

    def _wrap_openai_agent(self, agent: Any, proxy_addr: str) -> RolloutWorkflow:
        """Wrap an agent workflow in OpenAIProxyWorkflow (HTTP mode only).

        Parameters
        ----------
        agent : Any | None
            The agent workflow to wrap (any class with async run() method).
            ``None`` is valid when ``mode='online'``.
        proxy_addr : str
            HTTP address of the proxy server (required)
        """
        from areal.experimental.openai import OpenAIProxyWorkflow

        agent_cfg = self.config.agent

        return OpenAIProxyWorkflow(
            mode=agent_cfg.mode,
            agent=agent,
            proxy_addr=proxy_addr,
            admin_api_key=agent_cfg.admin_api_key,
            discount=agent_cfg.turn_discount,
            export_style=agent_cfg.export_style,
            subproc_max_workers=agent_cfg.subproc_max_workers,
            proxy_gateway_addr=self._proxy_gateway_addr,
        )

    def _resolve_workflow(
        self,
        workflow: WorkflowLike | None,
        workflow_kwargs: dict[str, Any] | None,
        group_size: int = 1,
        proxy_addr: str | None = None,
    ) -> RolloutWorkflow:
        resolved: RolloutWorkflow

        # 0. None workflow = online mode (config-driven)
        if workflow is None:
            agent_cfg = self.config.agent
            if agent_cfg is None or agent_cfg.mode != "online":
                raise ValueError(
                    "workflow is None but AgentConfig.mode is not 'online'. "
                    "Provide a workflow or set mode='online' in the config."
                )
            if proxy_addr is None:
                raise ValueError("proxy_addr is required for online mode")
            resolved = self._wrap_openai_agent(None, proxy_addr=proxy_addr)
            if group_size > 1:
                resolved = GroupedRolloutWorkflow(resolved, group_size, self.logger)
            return resolved

        # 1. Already a RolloutWorkflow instance
        if isinstance(workflow, RolloutWorkflow):
            if workflow_kwargs is not None:
                self.logger.warning(
                    "workflow_kwargs is ignored when workflow is already an instance"
                )
            resolved = workflow

        # 2. RolloutWorkflow class
        elif isinstance(workflow, type) and issubclass(workflow, RolloutWorkflow):
            if workflow_kwargs is None:
                raise ValueError(
                    f"workflow_kwargs is required when workflow is a class. "
                    f"Got workflow={workflow}, but workflow_kwargs=None."
                )
            resolved = workflow(**workflow_kwargs)

        # 3. String import path
        elif isinstance(workflow, str):
            try:
                imported_obj = import_from_string(workflow)
            except (ValueError, ImportError, AttributeError) as e:
                raise ValueError(
                    f"Failed to import workflow from string {workflow!r}: {e}"
                ) from e

            # Check if it's a RolloutWorkflow class
            if isinstance(imported_obj, type) and issubclass(
                imported_obj, RolloutWorkflow
            ):
                if workflow_kwargs is None:
                    raise ValueError(
                        f"workflow_kwargs is required when workflow is a class or string. "
                        f"Got workflow={workflow}, but workflow_kwargs=None."
                    )
                resolved = imported_obj(**workflow_kwargs)

            # Check if it's a RolloutWorkflow instance
            elif isinstance(imported_obj, RolloutWorkflow):
                if workflow_kwargs is not None:
                    self.logger.warning(
                        "workflow_kwargs is ignored when workflow resolves to an instance"
                    )
                resolved = imported_obj

            # Otherwise, treat it as an agent-like workflow (needs proxy)
            else:
                if proxy_addr is None:
                    raise ValueError(
                        f"proxy_addr is required for agent workflows (non-RolloutWorkflow). "
                        f"Ensure proxy workers are initialized via RolloutController.start_proxy(). "
                        f"Got workflow={workflow!r}"
                    )

                # Instantiate if it's a class
                if isinstance(imported_obj, type):
                    agent = imported_obj(**(workflow_kwargs or {}))
                else:
                    # Already an instance
                    agent = imported_obj

                resolved = self._wrap_openai_agent(agent, proxy_addr=proxy_addr)

        # 4. Callable class (agent-like workflow)
        elif isinstance(workflow, type):
            if proxy_addr is None:
                raise ValueError(
                    "proxy_addr is required for agent workflows (non-RolloutWorkflow). "
                    "Ensure proxy workers are initialized via RolloutController.start_proxy()."
                )
            agent = workflow(**(workflow_kwargs or {}))
            resolved = self._wrap_openai_agent(agent, proxy_addr=proxy_addr)

        # 5. Instance of agent-like workflow
        else:
            if proxy_addr is None:
                raise ValueError(
                    "proxy_addr is required for agent workflows (non-RolloutWorkflow). "
                    "Ensure proxy workers are initialized via RolloutController.start_proxy()."
                )
            if workflow_kwargs is not None:
                self.logger.warning(
                    "workflow_kwargs is ignored when workflow is already an instance"
                )
            resolved = self._wrap_openai_agent(workflow, proxy_addr=proxy_addr)

        # Wrap with GroupedRolloutWorkflow if group_size > 1
        if group_size > 1:
            resolved = GroupedRolloutWorkflow(resolved, group_size, self.logger)

        return resolved

    def _resolve_should_accept_fn(
        self, should_accept_fn: Callable[[dict[str, Any]], bool] | str | None
    ) -> Callable[[dict[str, Any]], bool] | None:
        """Resolve should_accept_fn parameter to a callable or None.

        Parameters
        ----------
        should_accept_fn : Callable[[Dict[str, Any]], bool] | str | None
            The should_accept_fn specification

        Returns
        -------
        Callable[[Dict[str, Any]], bool] | None
            A callable for trajectory filtering, or None

        Raises
        ------
        ValueError
            If string import fails
        TypeError
            If imported object is not callable
        """
        if should_accept_fn is None or callable(should_accept_fn):
            return should_accept_fn

        if isinstance(should_accept_fn, str):
            try:
                func = import_from_string(should_accept_fn)
            except (ValueError, ImportError, AttributeError) as e:
                raise ValueError(
                    f"Failed to import should_accept_fn from string {should_accept_fn!r}: {e}"
                ) from e
            if not callable(func):
                raise TypeError(
                    f"Imported object {func} from {should_accept_fn!r} is not callable"
                )
            return func

        raise TypeError(
            f"Invalid should_accept_fn type: {type(should_accept_fn)}. "
            f"Expected callable or string module path."
        )

    def choose_server(self) -> str:
        """Choose a server based on the routing strategy.

        Returns
        -------
        str
            Selected server address

        Raises
        ------
        NotImplementedError
            If routing strategy other than round-robin is used
        """
        if self.config.routing_strategy == "round_robin":
            server = self.addresses[self.server_idx]
            self.server_idx = (self.server_idx + 1) % len(self.addresses)
            return server
        raise NotImplementedError("Only round-robin scheduling is implemented.")

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request.

        Parameters
        ----------
        req : ModelRequest
            The model request containing input data and generation parameters

        Returns
        -------
        ModelResponse
            The generated response from the model
        """
        # Create a shallow copy of the input request
        # we are going to modify it in-place
        req = req.copy()

        # Populate return_routed_experts from config to metadata
        if self.config.return_routed_experts:
            req.metadata["return_routed_experts"] = True

        # Validate n_samples
        gconfig = req.gconfig
        if gconfig.n_samples != 1:
            raise ValueError(
                "Inference engines do not support n_samples > 1. "
                "Please call generate multiple times with n_samples = 1."
            )

        # Validate max_new_tokens
        max_new_tokens = min(
            gconfig.max_tokens - len(req.input_ids), gconfig.max_new_tokens
        )
        if max_new_tokens <= 0:
            raise RuntimeError(
                f"max_new_tokens ({max_new_tokens}) is non-positive! "
                f"max_tokens={gconfig.max_tokens}, prompt_len={len(req.input_ids)}, "
                f"max_new_tokens={gconfig.max_new_tokens}."
            )

        # Update max_new_tokens in request
        req.gconfig.max_new_tokens = max_new_tokens

        # Make request
        start_time = time.perf_counter()
        accumulated_output_tokens = []
        accumulated_output_logprobs = []
        accumulated_versions = []
        accumulated_routed_experts: list[np.ndarray] = []

        # A single "rid" shares the same server to allow KV cache reuse
        if req.rid in self.rid_to_address:
            server_addr = self.rid_to_address[req.rid]
        else:
            server_addr = self.choose_server()
            if len(self.rid_queue) >= RID_CACHE_SIZE:
                # Remove the oldest entry if cache is full
                oldest_rid = self.rid_queue.pop(0)
                self.rid_to_address.pop(oldest_rid, None)
            self.rid_to_address[req.rid] = server_addr
            self.rid_queue.append(req.rid)

        # Get the shared session from workflow context
        session = await workflow_context.get_aiohttp_session()

        # Deal with rollout interruption
        stop_reason = None
        ori_max_new_tokens = gconfig.max_new_tokens
        while (
            stop_reason not in ["stop", "tool_calls", "length"]
            and len(accumulated_output_tokens) < ori_max_new_tokens
        ):
            # Request is interrupted, wait for some time to avoid interfering
            # with update weights requests
            while self.workflow_executor.is_paused():
                await asyncio.sleep(0.5)

            # Build request using backend
            http_req = self.backend.build_generation_request(
                req,
                with_lora=self.config.use_lora,
                version=self.get_version(),
            )

            # Loop until the generation is complete
            result = await arequest_with_retry(
                session=session,
                addr=server_addr,
                endpoint=http_req.endpoint,
                payload=http_req.payload,
                method=http_req.method,
                max_retries=self.config.request_retries,
                timeout=self.config.request_timeout,
            )

            # Assert response is JSON dict (not text/binary from error pages)
            if not isinstance(result, dict):
                raise ValueError(
                    f"Expected JSON dict response, got {type(result).__name__}"
                )

            # Parse response using backend
            gen_result = self.backend.parse_generation_response(result)
            stop_reason = gen_result.stop_reason

            if (
                req.metadata.get("return_routed_experts", False)
                and gen_result.routed_experts is None
            ):
                if stop_reason != "abort":  # Only validate for successful generations
                    raise RuntimeError(
                        "Requested return_routed_experts=True but received None from SGLang. "
                        "This usually means the model is not a MoE (Mixture of Experts) model. "
                        "Please use a MoE model to get routed_experts information."
                    )

            # Update accumulated outputs
            accumulated_output_tokens.extend(gen_result.output_tokens)
            accumulated_output_logprobs.extend(gen_result.output_logprobs)
            accumulated_versions.extend(
                [self.get_version()] * len(gen_result.output_tokens)
            )
            # Accumulate routed_experts for MoE models
            if gen_result.routed_experts is not None:
                accumulated_routed_experts.append(gen_result.routed_experts)

            # Update request for next iteration
            req.input_ids += gen_result.output_tokens
            req.gconfig.max_new_tokens -= len(gen_result.output_tokens)
            assert req.gconfig.max_new_tokens >= 0, (
                req.gconfig.max_new_tokens,
                len(gen_result.output_tokens),
                len(req.input_ids),
            )

        # Final abort handling
        if stop_reason == "abort":
            # If stop_reason is "abort", the only reason we exit the loop is
            # len(accumulated_output_tokens) >= gconfig.max_new_tokens
            # so the actual reason is length
            stop_reason = "length"

        latency = time.perf_counter() - start_time

        accumulated_routed_experts = (
            np.concatenate(accumulated_routed_experts)
            if accumulated_routed_experts
            else None
        )

        response = ModelResponse(
            input_tokens=req.input_ids[
                : len(req.input_ids) - len(accumulated_output_tokens)
            ],
            input_images=req.image_data,
            output_tokens=accumulated_output_tokens,
            output_logprobs=accumulated_output_logprobs,
            output_versions=accumulated_versions,
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,  # Simplified for non-streaming
            tokenizer=req.tokenizer,
            processor=req.processor,
            routed_experts=accumulated_routed_experts,
        )
        return response

    def init_weights_update_group(
        self, meta: WeightUpdateMeta, xccl_group_ranks: list[int] | None = None
    ) -> Future[None]:
        """Initialize the weight update process group for distributed weight updates.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update
        xccl_group_ranks : list[int] | None, optional
            Explicit rank assignment for each remote inference worker, aligned with
            ``self.addresses`` (same length, same order).

            - If provided, worker at ``self.addresses[i]`` will initialize the
            communication group using rank ``xccl_group_ranks[i]``.
            - If None, ranks are assigned by address order: rank ``i`` for
            ``self.addresses[i]``.

        Returns
        -------
        Future[None]
            A future object representing the asynchronous initialization operation
        """
        assert meta.type == "xccl"

        self.logger.info(
            "Initializing weight update group: group=%s, addresses=%s",
            meta.nccl_group_name,
            self.addresses,
        )

        fut = get_executor().submit(
            _init_weights_update_group_remote,
            self.backend,
            meta,
            self.addresses,
            self.config.request_timeout,
            xccl_group_ranks,
        )

        def callback(fut):
            if fut.cancelled():
                return
            if fut.exception() is not None:
                self.logger.error(
                    "Failed to initialize %s group for distributed weight update for %s: %s",
                    current_platform.communication_backend.upper(),
                    meta.nccl_group_name,
                    repr(fut.exception()),
                )
                return
            self.logger.info(
                f"Initialized {current_platform.communication_backend.upper()} group "
                f"for distributed weight update for {meta.nccl_group_name}."
            )

        fut.add_done_callback(callback)
        return fut

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> Future[None]:
        """Update weights in the inference engine from distributed memory.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update
        param_specs : List[ParamSpec]
            A list of parameter specifications for the weights to be updated

        Returns
        -------
        Future[None]
            A future object representing the asynchronous weight update operation
        """
        assert meta.type == "xccl"

        fut = get_executor().submit(
            _update_weights_from_distributed,
            self.backend,
            meta,
            param_specs,
            self.addresses,
            self.config.request_timeout,
        )

        return fut

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        """Update weights in the inference engine from disk.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update

        Returns
        -------
        Future[None]
            A future object representing the asynchronous weight update operation
        """
        assert meta.type == "disk"

        tik = time.perf_counter()

        # Use ProcessPool to bypass python GIL for running async coroutines
        if self.config.experiment_name is None or self.config.trial_name is None:
            raise RuntimeError(
                "Experiment and trial names must be set for disk-based weight updates."
            )

        fut = get_executor().submit(
            _update_weights_from_disk,
            self.backend,
            self.config.experiment_name,
            self.config.trial_name,
            self.get_version(),
            self.addresses,
            meta,
            self.config.request_retries,
            self.config.request_timeout,
        )

        def callback(fut):
            respond_time = fut.result()
            self.logger.info(
                f"Loading weights from disk done "
                f"in {(time.perf_counter() - tik):.2f}s. "
                f"Respond time: {respond_time:.2f}s."
            )
            if meta.clear_checkpoint_after_load:
                shutil.rmtree(meta.path, ignore_errors=True)

        fut.add_done_callback(callback)
        return fut

    def submit(
        self,
        data: dict[str, Any],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        task_id: int | None = None,
        callback_addr: str | None = None,
        is_eval: bool = False,
        proxy_addr: str | None = None,
    ) -> int:
        """Submit a request to the inference engine and return immediately.

        Parameters
        ----------
        data : Dict[str, Any]
            The input data for rollout
        workflow : WorkflowLike
            The workflow to use for rollout generation
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        should_accept_fn : Callable[[Dict[str, Any]], bool] | str, optional
            A function or module path for trajectory filtering
        group_size : int
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).
        task_id : int, optional
            The task ID to use. If None, a new task ID will be generated internally.
        is_eval : bool, optional
            Whether this is an evaluation workflow. Affects variables like trajectory dump path
            and statistics keys. By default False.
        proxy_addr : str, optional
            HTTP address of the proxy server for AgentWorkflow. If provided,
            AgentWorkflow will use this proxy instead of a local one.
        """
        if workflow is None and (
            self.config.agent is None or self.config.agent.mode != "online"
        ):
            raise ValueError(
                "workflow must be specified for submit (unless mode='online')"
            )
        if callback_addr:
            self.workflow_executor.dispatcher.register_callback(task_id, callback_addr)

        # Resolve workflow to a RolloutWorkflow instance
        resolved_workflow = self._resolve_workflow(
            workflow, workflow_kwargs, group_size, proxy_addr=proxy_addr
        )
        resolved_should_accept_fn = self._resolve_should_accept_fn(should_accept_fn)

        return self.workflow_executor.submit(
            data,
            workflow=resolved_workflow,
            should_accept_fn=resolved_should_accept_fn,
            task_id=task_id,
            is_eval=is_eval,
        )

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[dict[str, Any] | None]:
        """Wait for a specified number of requests to complete.

        Parameters
        ----------
        count : int
            The number of accepted trajectories to wait for
        timeout : float, optional
            Timeout in seconds
        raise_timeout : bool, optional
            Whether to raise a TimeoutError when the timeout is exceeded, by default True

        Returns
        -------
        list[dict[str, Any] | None]
            A list of trajectory dictionaries. Each element may be None for rejected trajectories.
            Returns an empty list if timeout exceeded and raise_timeout is False.
        """
        return self.workflow_executor.wait(
            count, timeout=timeout, raise_timeout=raise_timeout
        )

    def wait_for_task(
        self, task_id: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> dict[str, Any] | None:
        """Wait for a specific submitted task to complete."""
        return self.workflow_executor.wait_for_task(task_id, timeout, raise_timeout)

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
    ) -> list[dict[str, Any]]:
        """Submit a batch of requests and wait for results.

        This method does not support asynchronous rollout and should be used for offline
        data collection or debugging, not in production experiments.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            A list of input data dictionaries for rollout
        workflow : WorkflowLike
            The workflow to use for rollout generation
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        group_size : int
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).

        Returns
        -------
        list[dict[str, Any]]
            A list of trajectory dictionaries, one per accepted rollout result.
            Each trajectory is a dict of tensors with shape [batch_size, seqlen, ...],
            where batch_size can vary per trajectory depending on the workflow output.
        """
        assert workflow is not None, "Workflow must be specified for rollout_batch."

        # Resolve workflow to a RolloutWorkflow instance
        resolved_workflow = self._resolve_workflow(
            workflow, workflow_kwargs, group_size
        )

        return self.workflow_executor.rollout_batch(
            data=data,
            workflow=resolved_workflow,
        )

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ) -> list[dict[str, Any]]:
        """Asynchronously submit and wait until a full batch is ready.

        Parameters
        ----------
        dataloader : StatefulDataLoader
            The data loader to pull data from
        workflow : WorkflowLike
            The workflow to use for rollout generation
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        should_accept_fn : Callable[[Dict[str, Any]], bool] | str, optional
            A function or module path for trajectory filtering
        group_size : int
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).
        dynamic_bs : bool, optional
            If True, enables dynamic batch sizing. Default is False.

        Returns
        -------
        list[dict[str, Any]]
            A list of trajectory dictionaries, one per accepted rollout result.
            Each trajectory is a dict of tensors with shape [batch_size, seqlen, ...],
            where batch_size can vary per trajectory depending on the workflow output.
        """
        assert workflow is not None, "Workflow must be specified for prepare_batch."

        # Resolve workflow to a RolloutWorkflow instance
        resolved_workflow = self._resolve_workflow(
            workflow, workflow_kwargs, group_size
        )
        resolved_should_accept_fn = self._resolve_should_accept_fn(should_accept_fn)

        return self.workflow_executor.prepare_batch(
            dataloader=dataloader,
            workflow=resolved_workflow,
            should_accept_fn=resolved_should_accept_fn,
            dynamic_bs=dynamic_bs,
        )

    @trace_perf("remote_inf_engine.pause_generation", category="misc")
    def pause_generation(self):
        """Pause request submission for async rollout."""
        pause_req = self.backend.get_pause_request()
        self._run_request_on_all_servers(pause_req)

        # The above http request may require some time to be scheduled and executed.
        # The following line waits until all requests are indeed dropped.
        time.sleep(self.config.pause_grace_period)

    @trace_perf("remote_inf_engine.continue_generation", category="misc")
    def continue_generation(self):
        """Resume request submission for async rollout."""
        resume_req = self.backend.get_resume_request()
        self._run_request_on_all_servers(resume_req)

    def pause(self):
        """Pause request submission for async rollout.
        Used during evaluation to prevent data over generation.
        """
        return self.workflow_executor.pause()

    def resume(self):
        """Resume request submission for async rollout."""
        return self.workflow_executor.resume()

    def offload(self) -> None:
        """Offload model memory on all servers."""
        offload_req = self.backend.get_offload_request()
        self._run_request_on_all_servers(offload_req)

    def onload(self, tags: list[str] | None = None) -> None:
        """Onload model memory on all servers."""
        onload_req = self.backend.get_onload_request(tags=tags)
        self._run_request_on_all_servers(onload_req)

    def _run_request_on_all_servers(self, req: HttpRequest):
        async def _fn():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
                read_bufsize=1024 * 1024 * 10,
                connector=get_default_connector(),
            ) as session:
                jobs = []
                for addr in self.addresses:
                    jobs.append(
                        arequest_with_retry(
                            session=session,
                            addr=addr,
                            endpoint=req.endpoint,
                            payload=req.payload,
                            method=req.method,
                            max_retries=self.config.request_retries,
                            timeout=self.config.request_timeout,
                        )
                    )
                await asyncio.gather(*jobs)

        uvloop.run(_fn())

    def launch_server(self, server_args: dict[str, Any]) -> LocalInfServerInfo:
        """Launch a local inference server."""
        server_args["host"] = gethostip()
        server_args["port"] = find_free_ports(1)[0]
        process = self.backend.launch_server(server_args)
        address = format_hostport(server_args["host"], server_args["port"])
        server_info = LocalInfServerInfo(
            host=server_args["host"],
            port=server_args["port"],
            process=process,
        )
        try:
            self._wait_for_server(address, process=process)
            self.local_server_processes.append(server_info)
            if ray.is_initialized():
                # do not return with process for ray as it is not picklable
                return LocalInfServerInfo(
                    host=server_args["host"],
                    port=server_args["port"],
                    process=None,
                )
            return server_info
        except TimeoutError:
            logger.warning(
                f"Launch local server timeouted at {address} after {self.config.setup_timeout}s."
            )
            self._shutdown_one_server(server_info)
            raise

    def _shutdown_one_server(self, server_info: LocalInfServerInfo):
        addr = format_hostport(server_info.host, server_info.port)
        if addr in self.addresses:
            self.addresses.remove(addr)
        if server_info.process.poll() is not None:
            return
        kill_process_tree(server_info.process.pid, graceful=True)

    def teardown_server(self):
        """Teardown all locally launched servers."""
        for server_info in self.local_server_processes:
            self._shutdown_one_server(server_info)
        self.local_server_processes.clear()


# Helper functions that run in ProcessPoolExecutor


def _update_weights_from_disk(
    backend: RemoteInfBackendProtocol,
    experiment_name: str,
    trial_name: str,
    model_version: int,
    addresses: list[str],
    meta: WeightUpdateMeta,
    request_retries: int,
    request_timeout: float,
):
    """Helper to update weights from disk in a separate process."""

    async def _fn():
        update_name = names.update_weights_from_disk(
            experiment_name, trial_name, model_version
        )
        save_timestamp = float(name_resolve.wait(update_name, timeout=120))
        load_timestamp = datetime.now().timestamp()

        # Get requests from backend with version for LoRA name
        weight_reqs = backend.build_disk_weight_update_requests(meta)

        # Execute all requests
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=request_timeout),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        ) as session:
            for http_req in weight_reqs.requests:
                jobs = [
                    arequest_with_retry(
                        session=session,
                        addr=addr,
                        endpoint=http_req.endpoint,
                        payload=http_req.payload,
                        method=http_req.method,
                        max_retries=request_retries,
                        timeout=request_timeout,
                    )
                    for addr in addresses
                ]
                await asyncio.gather(*jobs)

        return load_timestamp - save_timestamp

    return uvloop.run(_fn())


def _init_weights_update_group_remote(
    backend: RemoteInfBackendProtocol,
    meta: WeightUpdateMeta,
    addresses: list[str],
    request_timeout: float,
    xccl_group_ranks: list[int] | None = None,
):
    """Helper to initialize weight update group in a separate process.

    If xccl_group_ranks is provided, it must have the same length as addresses and will be
    used as the per-address rank passed to the backend request builder.
    Otherwise, ranks default to enumerate(addresses).
    """

    if xccl_group_ranks is not None and len(xccl_group_ranks) != len(addresses):
        raise ValueError(
            f"xccl_group_ranks must have the same length as addresses "
            f"(got {len(xccl_group_ranks)} vs {len(addresses)})"
        )

    async def _fn():
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=request_timeout),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        ) as session:
            jobs = []
            for i, addr in enumerate(addresses):
                xccl_group_rank = (
                    xccl_group_ranks[i] if xccl_group_ranks is not None else i
                )
                http_req = backend.build_init_weights_group_request(
                    addr, xccl_group_rank, meta
                )
                jobs.append(
                    arequest_with_retry(
                        session=session,
                        addr=addr,
                        endpoint=http_req.endpoint,
                        payload=http_req.payload,
                        method=http_req.method,
                        max_retries=1,
                        timeout=request_timeout,
                    )
                )
            results = await asyncio.gather(*jobs, return_exceptions=True)
            for _idx, _r in enumerate(results):
                if isinstance(_r, Exception):
                    logger.error(
                        "init_weights_update_group request %d to %s failed: %s",
                        _idx,
                        addresses[_idx],
                        _r,
                    )
            # Re-raise first exception if any failed
            for _r in results:
                if isinstance(_r, Exception):
                    raise _r

    return uvloop.run(_fn())


def _update_weights_from_distributed(
    backend: RemoteInfBackendProtocol,
    meta: WeightUpdateMeta,
    param_specs: list[ParamSpec],
    addresses: list[str],
    request_timeout: float,
):
    """Helper to update weights from distributed memory in a separate process."""

    async def _fn():
        # Get requests from backend
        weight_reqs = backend.build_distributed_weight_update_requests(
            meta, param_specs
        )

        # Execute all requests sequentially (they may have dependencies)
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=request_timeout),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        ) as session:
            for http_req in weight_reqs.requests:
                jobs = [
                    arequest_with_retry(
                        session=session,
                        addr=addr,
                        endpoint=http_req.endpoint,
                        payload=http_req.payload,
                        method=http_req.method,
                        max_retries=1,
                        timeout=request_timeout,
                    )
                    for addr in addresses
                ]
                await asyncio.gather(*jobs)

    return uvloop.run(_fn())
