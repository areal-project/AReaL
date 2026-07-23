# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from typing import Any

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api import (
    InferenceEngine,
    LocalInfServerInfo,
    ModelAllocation,
    ModelRequest,
    ModelResponse,
    ParamSpec,
    Scheduler,
    WeightUpdateMeta,
    WorkflowLike,
)
from areal.api.cli_args import InferenceEngineConfig, PerfTracerConfig, vLLMConfig
from areal.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    WeightUpdateRequests,
    detect_image_mime,
    get_versioned_lora_name,
)
from areal.infra import RemoteInfEngine, RolloutController, WorkflowExecutor
from areal.infra.platforms import current_platform
from areal.infra.utils.launcher import TRITON_CACHE_PATH
from areal.utils import logging, perf_tracer, stats_tracker
from areal.utils.network import format_host_for_url
from areal.utils.vllm_response import parse_vllm_generation_response

logger = logging.getLogger("vLLMEngine")


class VLLMBackend:
    """vLLM-specific backend implementation for remote inference."""

    @staticmethod
    def build_server_env(env: Mapping[str, str]) -> dict[str, str]:
        _env = dict(env)
        triton_cache_path = _env.get("TRITON_CACHE_PATH", TRITON_CACHE_PATH)
        _env["TRITON_CACHE_PATH"] = os.path.join(triton_cache_path, str(uuid.uuid4()))

        vllm_cache_path = _env.get("VLLM_CACHE_ROOT")
        if vllm_cache_path:
            _env["VLLM_CACHE_ROOT"] = os.path.join(vllm_cache_path, str(uuid.uuid4()))
        _env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        return _env

    def build_generation_request(
        self, req: ModelRequest, with_lora: bool, version: int
    ) -> HttpRequest:
        """Build vLLM generation request."""
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids

        # NOTE: vLLM uses flat payload structure, not nested sampling_params
        payload = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": stop_token_ids,
            "ignore_eos": gconfig.ignore_eos,
            "skip_special_tokens": gconfig.skip_special_tokens,
            "frequency_penalty": gconfig.frequency_penalty,
            "return_tokens_as_token_ids": True,
            "logprobs": 0,
            "use_beam_search": gconfig.use_beam_search,
            "stream": False,
        }
        if gconfig.stop:
            payload["stop"] = gconfig.stop

        if with_lora:
            lora_name = gconfig.lora_name
            if not lora_name:
                raise ValueError(
                    "LoRA name (gconfig.lora_name) is required when use_lora is enabled."
                )
            payload["model"] = get_versioned_lora_name(lora_name, version)

        if req.vision_msg_vllm:
            images = iter(req.image_data)
            parsed_input = req.vision_msg_vllm[0]
            for msg in parsed_input:
                if isinstance(msg["content"], list):
                    for content in msg["content"]:
                        if content.get("type") == "image_url":
                            try:
                                base64_img = next(images)
                            except StopIteration:
                                raise ValueError(
                                    "Not enough images in req.image_data to match image_url entries."
                                )
                            mime = detect_image_mime(base64_img)
                            content["image_url"] = {
                                "url": f"data:{mime};base64,{base64_img}"
                            }
            payload["messages"] = parsed_input.copy()
            payload["logprobs"] = True
            return HttpRequest(endpoint="/v1/chat/completions", payload=payload)
        else:
            payload["prompt"] = req.input_ids.copy()
            return HttpRequest(endpoint="/v1/completions", payload=payload)

    def parse_generation_response(
        self, response: dict[str, Any]
    ) -> HttpGenerationResult:
        """Parse vLLM generation response."""
        return parse_vllm_generation_response(response)

    def build_score_request(
        self, input_ids: list[int], target_len: int, with_lora: bool, version: int
    ) -> HttpRequest:
        payload: dict[str, Any] = {
            "prompt": input_ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 1,
            "prompt_logprobs": 1,
            "echo": True,
        }
        if with_lora:
            raise NotImplementedError(
                "LoRA scoring request is not supported in vLLM teacher compute_logp yet."
            )
        return HttpRequest(endpoint="/v1/completions", payload=payload)

    def parse_score_response(
        self, response: dict[str, Any], target_len: int
    ) -> list[float]:
        choices = response.get("choices")
        if not choices:
            raise ValueError("vLLM response missing choices for score request")
        prompt_logprobs = choices[0].get("prompt_logprobs")
        if prompt_logprobs is None:
            raise ValueError("vLLM response missing prompt_logprobs for score request")
        if len(prompt_logprobs) < target_len + 1:
            raise ValueError(
                f"prompt_logprobs too short: got {len(prompt_logprobs)}, need {target_len + 1}"
            )
        sliced = prompt_logprobs[-target_len:]
        token_logps: list[float] = []
        for item in sliced:
            if not item:
                token_logps.append(0.0)
                continue
            top = next(iter(item.values()))
            token_logps.append(float(top["logprob"] if isinstance(top, dict) else top))
        return token_logps

    def build_disk_weight_update_requests(
        self, meta: WeightUpdateMeta
    ) -> WeightUpdateRequests:
        """Build vLLM disk weight update requests."""
        if meta.use_lora:
            if meta.version is None:
                raise ValueError("Version is required for LoRA update.")
            lora_name = get_versioned_lora_name(meta.lora_name, meta.version)
            endpoint = "/v1/load_lora_adapter"
            payload = {
                "lora_path": str(meta.path),
                "lora_name": lora_name,
            }
        else:
            endpoint = "/areal_update_weights"
            payload = {"model_path": str(meta.path)}

        return WeightUpdateRequests(
            requests=[HttpRequest(endpoint=endpoint, payload=payload)]
        )

    def build_distributed_weight_update_requests(
        self,
        meta: WeightUpdateMeta,
        param_specs: list[ParamSpec],
    ) -> WeightUpdateRequests:
        """Build vLLM distributed weight update requests."""
        # vLLM uses two-step process: set metadata, then update
        # vLLM uses two-step process: set metadata, then update
        base_payload = {
            "names": [pspec.name for pspec in param_specs],
            "dtypes": [pspec.dtype for pspec in param_specs],
            "shapes": [pspec.shape for pspec in param_specs],
            "group_name": meta.nccl_group_name,
        }

        if meta.use_lora:
            if meta.version is None:
                raise ValueError("Version is required for LoRA update.")
            lora_name = get_versioned_lora_name(meta.lora_name, meta.version)
            lora_payload = {
                "lora_name": lora_name,
                "lora_int_id": meta.lora_int_id,
                "lora_target_modules": meta.peft_config["target_modules"],
                "lora_rank": meta.peft_config["r"],
                "lora_alpha": meta.peft_config["lora_alpha"],
                "lora_bias": meta.peft_config["bias"],
                "base_model_name": meta.base_model_name,
            }
            payload = {**base_payload, **lora_payload}
            meta_endpoint = "/areal_set_update_weight_meta_lora"
            update_endpoint = "/areal_update_weights_lora_xccl"
        else:
            payload = base_payload
            meta_endpoint = "/areal_set_update_weight_meta"
            update_endpoint = "/areal_update_weights_xccl"

        return WeightUpdateRequests(
            requests=[
                HttpRequest(
                    endpoint=meta_endpoint,
                    payload=payload,
                ),
                HttpRequest(
                    endpoint=update_endpoint,
                    payload={} if not meta.use_lora else payload,
                ),
            ]
        )

    def build_init_weights_group_request(
        self, addr: str, server_idx: int, meta: WeightUpdateMeta
    ) -> HttpRequest:
        """Build vLLM init weights group request."""
        assert meta.gen_allocation is not None
        gen_parallel = meta.gen_allocation.parallel
        rank_offset = 1 + server_idx * gen_parallel.tp_size * gen_parallel.pp_size
        payload = {
            "master_address": format_host_for_url(meta.nccl_master_address),
            "master_port": str(meta.nccl_master_port),
            "rank_offset": rank_offset,
            "world_size": gen_parallel.world_size + 1,
            "backend": current_platform.communication_backend,
            "group_name": meta.nccl_group_name,
        }
        return HttpRequest(endpoint="/areal_init_weights_update_group", payload=payload)

    def get_pause_request(self) -> HttpRequest:
        """Get vLLM pause request."""
        return HttpRequest(endpoint="/areal_pause_generation", payload={})

    def get_resume_request(self) -> HttpRequest:
        """Get vLLM resume request."""
        return HttpRequest(endpoint="/areal_continue_generation", payload={})

    def get_health_check_request(self) -> HttpRequest:
        """Get vLLM health check request."""
        return HttpRequest(endpoint="/health", payload={}, method="GET")

    def get_abort_all_request(self) -> HttpRequest:
        raise NotImplementedError("vLLM does not support abort_all_requests")

    def get_offload_request(self, tags: list[str] | None = None) -> HttpRequest:
        """Get vLLM offload request.

        Uses vLLM's /sleep endpoint to offload model memory to CPU.
        Default level is 1.

        Parameters
        ----------
        tags : list[str], optional
            Accepted for RemoteInfEngine API compatibility. vLLM sleep does not
            support component-specific tags, so this value is ignored.
        """
        return HttpRequest(endpoint="/sleep", payload={}, method="POST")

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        """Get vLLM onload request.

        Uses vLLM's /wake_up endpoint to reload model memory from CPU.
        vLLM reads parameters from query string.

        Parameters
        ----------
        tags : list[str], optional
            Tags to wake up specific components. If None, wakes up all components.
        """
        if tags is not None:
            # Build query string with multiple tags parameters
            tags_query = "&".join([f"tags={tag}" for tag in tags])
            endpoint = f"/wake_up?{tags_query}"
        else:
            endpoint = "/wake_up"
        return HttpRequest(endpoint=endpoint, payload={}, method="POST")

    def launch_server(self, server_args: dict[str, Any]) -> subprocess.Popen:
        """Launch vLLM server subprocess."""
        cmd = vLLMConfig.build_cmd_from_args(server_args)
        _env = self.build_server_env(os.environ)

        logger.info(f"Launching vLLM server with command: {' '.join(cmd)}")
        return subprocess.Popen(
            cmd,
            env=_env,
            stdout=sys.stdout,
            stderr=sys.stdout,
        )


class RemotevLLMEngine(InferenceEngine):
    """vLLM remote inference engine.

    This class delegates all functionality to RemoteInfEngine with
    a VLLMBackend implementation. It maintains the same public API for
    backward compatibility.

    Parameters
    ----------
    config : InferenceEngineConfig
        Configuration for the inference engine
    """

    def __init__(self, config: InferenceEngineConfig):
        self.config = config
        # Pure composition - create internal engine with vLLM backend
        self._engine = RemoteInfEngine(config, VLLMBackend())

    @classmethod
    def from_pretrained(
        cls,
        tokenizer_path: str | None = None,
        dp_size: int = 1,
        max_concurrent_rollouts: int | None = None,
        **kwargs,
    ) -> "RemoteInfEngine":
        """Create a RemoteInfEngine without kwargs instead of InferenceEngineConfig.

        Parameters
        ----------
        tokenizer_path: str | None = None
            Path to the tokenizer
        dp_size : int
            Data parallelism size
        max_concurrent_rollouts : int | None
            Maximum concurrent rollouts
        **kwargs : dict
            Additional config parameters passed to InferenceEngineConfig

        Returns
        -------
        RemoteInfEngine
        """

        backend_str = f"vllm:d{dp_size}"

        config = InferenceEngineConfig(
            backend=backend_str,
            max_concurrent_rollouts=max_concurrent_rollouts,
            tokenizer_path=tokenizer_path,
            **kwargs,
        )

        engine = cls(config)

        return engine

    def initialize(
        self,
        engine_id: str | None = None,
        addr: str | list[str] | None = None,
        train_data_parallel_size: int | None = None,
    ):
        """Initialize the engine by discovering and connecting to servers."""
        if train_data_parallel_size is None:
            train_data_parallel_size = ModelAllocation.from_str(
                self.config.backend, name="rollout"
            ).parallel.data_parallel_size
        return self._engine.initialize(engine_id, addr, train_data_parallel_size)

    def destroy(self):
        """Destroy the engine and clean up resources."""
        return self._engine.destroy()

    @property
    def initialized(self) -> bool:
        return self._engine.initialized

    @property
    def workflow_executor(self) -> WorkflowExecutor:
        """Get the workflow executor of the inference engine."""
        return self._engine.workflow_executor

    def set_version(self, version: int):
        """Set the current weight version."""
        return self._engine.set_version(version)

    def get_version(self) -> int:
        """Get the current weight version."""
        return self._engine.get_version()

    def set_proxy_gateway_addr(self, addr: str) -> None:
        self._engine.set_proxy_gateway_addr(addr)

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request."""
        return await self._engine.agenerate(req)

    def init_weights_update_group(
        self, meta: WeightUpdateMeta, xccl_group_ranks: list[int] | None = None
    ) -> Future[None]:
        """Initialize the weight update process group."""
        return self._engine.init_weights_update_group(
            meta, xccl_group_ranks=xccl_group_ranks
        )

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> Future[None]:
        """Update weights from distributed memory."""
        return self._engine.update_weights_from_distributed(meta, param_specs)

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        """Update weights from disk."""
        return self._engine.update_weights_from_disk(meta)

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
        """Submit a request to the inference engine."""
        return self._engine.submit(
            data=data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
            task_id=task_id,
            callback_addr=callback_addr,
            is_eval=is_eval,
            proxy_addr=proxy_addr,
        )

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[dict[str, Any] | None]:
        """Wait for a specified number of requests to complete."""
        return self._engine.wait(count, timeout, raise_timeout)

    def wait_for_task(
        self, task_id: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> dict[str, Any] | None:
        """Wait for a specific task to complete by task_id."""
        return self._engine.wait_for_task(task_id, timeout, raise_timeout)

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
    ) -> dict[str, Any]:
        """Submit a batch of requests and wait for results.

        This method does not support asynchronous rollout and should be used for offline
        data collection or debugging, not in production experiments.
        """
        return self._engine.rollout_batch(
            data=data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            group_size=group_size,
        )

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ):
        """Asynchronously submit and wait until a full batch is ready."""
        return self._engine.prepare_batch(
            dataloader=dataloader,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
            dynamic_bs=dynamic_bs,
        )

    def compute_logp(self, data: list[dict[str, Any]]) -> list[torch.Tensor]:
        return self._engine.compute_logp(data)

    def pause(self):
        return self._engine.pause()

    def resume(self):
        return self._engine.resume()

    def pause_generation(self):
        return self._engine.pause_generation()

    def continue_generation(self):
        return self._engine.continue_generation()

    def launch_server(self, server_args: dict[str, Any]) -> LocalInfServerInfo:
        return self._engine.launch_server(server_args)

    def teardown_server(self):
        return self._engine.teardown_server()

    def offload(self, tags: list[str] | None = None):
        return self._engine.offload(tags=tags)

    def onload(self, tags: list[str] | None = None):
        return self._engine.onload(tags=tags)

    def export_stats(self) -> dict[str, float]:
        return stats_tracker.export_all(reduce_group=None)

    @classmethod
    def as_controller(cls, config: InferenceEngineConfig, scheduler: Scheduler):
        if config._version == "v2":
            from areal.v2.inference_service.controller.controller import (
                RolloutControllerV2,
            )

            return RolloutControllerV2(config=config, scheduler=scheduler)
        return RolloutController(cls, config=config, scheduler=scheduler)

    def clear_batches(self, shard_ids: list[str] | None = None) -> None:
        """Drain this worker's client-side RTensor fetch buffer.

        Called via RPC by ``TrainController.clear_batches`` at step end so
        cross-node consumer DP heads release cached tensors. See #1209.
        Non-DP-head ranks receive no positional args via
        ``_call_workers`` (see train_controller.py:575-577) — accept the
        no-args call and noop, since their ``_fetch_buffer`` is empty.
        """
        from areal.infra.rpc.rtensor import clear_fetch_buffer

        if shard_ids:
            clear_fetch_buffer(shard_ids)

    def fetch_buffer_stats(self) -> dict[str, int]:
        """Expose local fetch-buffer stats for post-step drain verification."""
        from areal.infra.rpc.rtensor import fetch_buffer_stats

        return fetch_buffer_stats()

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        perf_tracer.save(step=step, force=force)

    def config_perf_tracer(
        self, config: PerfTracerConfig, rank: int, role: str
    ) -> None:
        if perf_tracer.is_configured():
            return
        perf_tracer.configure(config, rank=rank, role=role)
