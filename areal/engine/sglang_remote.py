# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import numpy as np
import pybase64
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
from areal.api.cli_args import InferenceEngineConfig, PerfTracerConfig, SGLangConfig
from areal.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    WeightUpdateRequests,
    get_versioned_lora_name,
)
from areal.infra import RemoteInfEngine, RolloutController, WorkflowExecutor
from areal.infra.platforms import current_platform
from areal.infra.utils.launcher import TRITON_CACHE_PATH
from areal.utils import perf_tracer, stats_tracker
from areal.utils.network import format_host_for_url


class SGLangBackend:
    """SGLang-specific backend implementation for remote inference."""

    def build_generation_request(
        self, req: ModelRequest, with_lora: bool, version: int
    ) -> HttpRequest:
        """Build SGLang generation request."""
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids
        stop = gconfig.stop

        if gconfig.use_beam_search:
            raise NotImplementedError(
                "Currently Beam search is not supported in SGLang backend."
            )

        sample_params = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_new_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": stop_token_ids,
            "ignore_eos": gconfig.ignore_eos,
            "skip_special_tokens": gconfig.skip_special_tokens,
            "frequency_penalty": gconfig.frequency_penalty,
        }
        if stop:
            sample_params["stop"] = stop

        payload = {
            "input_ids": req.input_ids.copy(),
            "image_data": req.image_data,  # ImageObject or str
            "sampling_params": sample_params,
            "return_logprob": True,
            "stream": False,
        }

        # Add return_routed_experts to payload if set
        if req.metadata.get("return_routed_experts", False):
            payload["return_routed_experts"] = True
        # Add LoRA if initialized
        if with_lora:
            lora_name = gconfig.lora_name
            if not lora_name:
                raise ValueError(
                    "LoRA name (gconfig.lora_name) is required when use_lora is enabled."
                )
            payload["lora_path"] = get_versioned_lora_name(lora_name, version)

        return HttpRequest(endpoint="/generate", payload=payload)

    def parse_generation_response(
        self, response: dict[str, Any]
    ) -> HttpGenerationResult:
        """Parse SGLang generation response."""
        meta_info = response["meta_info"]
        finish_reason = meta_info["finish_reason"]
        stop_reason = finish_reason["type"]
        stop_message = finish_reason.get("message", "")

        # Extract routed_experts information if available
        routed_experts = meta_info.get("routed_experts", None)
        if routed_experts is not None:
            num_sgl_token = (
                meta_info["prompt_tokens"] + meta_info["completion_tokens"] - 1
            )
            # Extract expert_id and reshape to (num_sgl_token, num_layers*expert_top_k)
            routed_experts = np.frombuffer(
                pybase64.b64decode(routed_experts.encode("utf-8")), dtype=np.int32
            ).reshape(num_sgl_token, -1)

        if stop_reason == "abort" and stop_message.startswith("Abort before prefill"):
            return HttpGenerationResult(
                output_tokens=[],
                output_logprobs=[],
                stop_reason=stop_reason,
                routed_experts=routed_experts,
            )

        output_tokens = [x[1] for x in meta_info["output_token_logprobs"]]
        output_logprobs = [x[0] for x in meta_info["output_token_logprobs"]]

        return HttpGenerationResult(
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            stop_reason=stop_reason,
            routed_experts=routed_experts,
        )

    def build_score_request(
        self, input_ids: list[int], target_len: int, with_lora: bool, version: int
    ) -> HttpRequest:
        payload: dict[str, Any] = {
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
            "return_logprob": True,
            "logprob_start_len": max(0, len(input_ids) - target_len - 1),
            "top_logprobs_num": 0,
            "stream": False,
        }
        if with_lora:
            raise NotImplementedError(
                "LoRA scoring request is not supported in SGLang teacher compute_logp yet."
            )
        return HttpRequest(endpoint="/generate", payload=payload)

    def parse_score_response(
        self, response: dict[str, Any], target_len: int
    ) -> list[float]:
        meta_info = response.get("meta_info")
        if meta_info is None:
            raise ValueError("SGLang response missing meta_info for score request")
        # SGLang returns [logprob, token_id, ...]
        all_logprobs = [float(x[0]) for x in meta_info.get("input_token_logprobs", [])]
        if len(all_logprobs) < target_len:
            raise ValueError(
                f"SGLang returned insufficient input_token_logprobs: {len(all_logprobs)} < {target_len}"
            )
        return all_logprobs[-target_len:]

    def build_disk_weight_update_requests(
        self, meta: WeightUpdateMeta
    ) -> WeightUpdateRequests:
        """Build SGLang disk weight update requests."""
        if meta.use_lora:
            if not meta.lora_name:
                raise ValueError("LoRA name is required for LoRA update.")
            if meta.version is None:
                raise ValueError("Version is required for LoRA update.")
            lora_name = get_versioned_lora_name(meta.lora_name, meta.version)
            # Load new LoRA
            requests = [
                HttpRequest(
                    endpoint="/load_lora_adapter",
                    payload={"lora_name": lora_name, "lora_path": str(meta.path)},
                )
            ]
            # Unload the version that has fallen outside the retention window so
            # sglang does not accumulate one adapter per train step (which leaks
            # VRAM and eventually hangs). Kept versions cover off-policy rollouts
            # (max_head_offpolicyness). Best-effort: the stale adapter may have
            # already been evicted or never loaded.
            keep = meta.lora_keep_versions
            if keep > 0 and meta.version - keep >= 0:
                stale_name = get_versioned_lora_name(
                    meta.lora_name, meta.version - keep
                )
                requests.append(
                    HttpRequest(
                        endpoint="/unload_lora_adapter",
                        payload={"lora_name": stale_name},
                        best_effort=True,
                    )
                )
            return WeightUpdateRequests(requests=requests)
        else:
            # Full model update
            return WeightUpdateRequests(
                requests=[
                    HttpRequest(
                        endpoint="/update_weights_from_disk",
                        payload={
                            "model_path": str(meta.path),
                            "abort_all_requests": True,
                        },
                    )
                ]
            )

    def build_distributed_weight_update_requests(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> WeightUpdateRequests:
        """Build SGLang distributed weight update requests.

        Note: SGLang distributed weight update (NCCL-based) does not support LoRA.
        For LoRA weight updates with SGLang, use disk-based update mode instead.
        """
        if meta.use_lora:
            raise ValueError(
                "SGLang distributed (XCCL/NCCL) weight update does not support LoRA. "
                "Use weight_update_mode='disk' for LoRA weight updates with SGLang."
            )
        return WeightUpdateRequests(
            requests=[
                HttpRequest(
                    endpoint="/update_weights_from_distributed",
                    payload={
                        "names": [pspec.name for pspec in param_specs],
                        "dtypes": [pspec.dtype for pspec in param_specs],
                        "shapes": [pspec.shape for pspec in param_specs],
                        "group_name": meta.nccl_group_name,
                        "abort_all_requests": True,
                    },
                )
            ]
        )

    def build_init_weights_group_request(
        self, addr: str, server_idx: int, meta: WeightUpdateMeta
    ) -> HttpRequest:
        """Build SGLang init weights group request.

        Supports two scenarios:

        1. **PP=1** (original): Single NCCL group spanning all TP workers across
           all DP instances. ``rank_offset`` is based on ``tp_size``.

        2. **PP>1, per-PP-rank groups**: The training engine creates a separate
           NCCL group per PP stage. The group name encodes the PP rank
           (e.g., ``update_weight_group_0``). Only sglang workers at that PP
           rank participate, so ``rank_offset`` is based on ``tp_size`` only,
           ``world_size = n_servers * tp_size + 1``, and ``pp_rank`` is
           included in the payload.

        All three training engines (Megatron, FSDP, Archon) use per-PP-rank
        group naming (``update_weight_group_{pp_rank}``) when PP>1, so the
        per-PP-rank path is always taken for PP>1.
        """
        assert meta.gen_allocation is not None
        gen_parallel = meta.gen_allocation.parallel
        group_name = meta.nccl_group_name

        # Determine if training side uses per-PP-rank groups.
        # Per-PP-rank groups are identified by group names ending with _{digit}
        # and pp_size > 1. All engines use this pattern when PP>1.
        per_pp_groups = False
        if gen_parallel.pp_size > 1:
            try:
                _suffix = group_name.rsplit("_", 1)[-1]
                int(_suffix)
                per_pp_groups = True
            except (ValueError, IndexError):
                per_pp_groups = False

        if per_pp_groups:
            # Scenario 2: PP>1 with per-PP-rank groups.
            # Extract pp_rank from the group name suffix.
            pp_rank = int(group_name.rsplit("_", 1)[-1])

            tp_size = gen_parallel.tp_size
            pp_size = gen_parallel.pp_size
            # gen_parallel.world_size = dp_size * tp_size * pp_size on the
            # inference side. n_servers (number of sglang server replicas)
            # therefore equals dp_size whether or not DP-attention is
            # configured: the AReaL allocation always materialises one server
            # replica per DP shard, each running ``tp_size * pp_size`` workers.
            n_servers = gen_parallel.world_size // (tp_size * pp_size)

            # Each server contributes exactly ``tp_size`` workers per PP stage.
            # Across all replicas this PP stage has ``n_servers * tp_size``
            # inference workers (= dp_size * tp_size), all of which join the
            # per-PP NCCL group together with the trainer.
            rank_offset = 1 + server_idx * tp_size

            # world_size for this group: TP workers across all DP replicas at
            # this PP rank + 1 (trainer).
            world_size = n_servers * tp_size + 1

            payload = {
                "master_address": format_host_for_url(meta.nccl_master_address),
                "master_port": str(meta.nccl_master_port),
                "rank_offset": rank_offset,
                "world_size": world_size,
                "backend": current_platform.communication_backend,
                "group_name": group_name,
                "pp_rank": pp_rank,
            }
        else:
            instance_size = gen_parallel.tp_size * gen_parallel.pp_size
            rank_offset = 1 + server_idx * instance_size
            payload = {
                "master_address": format_host_for_url(meta.nccl_master_address),
                "master_port": str(meta.nccl_master_port),
                "rank_offset": rank_offset,
                "world_size": gen_parallel.world_size + 1,
                "backend": current_platform.communication_backend,
                "group_name": group_name,
            }

        return HttpRequest(endpoint="/init_weights_update_group", payload=payload)

    def get_pause_request(self) -> HttpRequest:
        """Get SGLang pause request."""
        return HttpRequest(endpoint="/pause_generation", payload={})

    def get_resume_request(self) -> HttpRequest:
        """Get SGLang resume request."""
        return HttpRequest(endpoint="/continue_generation", payload={})

    def get_health_check_request(self) -> HttpRequest:
        """Get SGLang health check request."""
        return HttpRequest(endpoint="/health", payload={}, method="GET")

    def get_offload_request(self) -> HttpRequest:
        """Get SGLang offload request."""
        return HttpRequest(endpoint="/release_memory_occupation", payload={})

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        """Get SGLang onload request.

        Parameters:
        ----------
        tags: list[str], optional
            Available tags for multi-stage resume: weights, kv_cache
        """
        payload = {"tags": tags} if tags is not None else {}
        return HttpRequest(endpoint="/resume_memory_occupation", payload=payload)

    def launch_server(self, server_args: dict[str, Any]) -> subprocess.Popen:
        """Launch SGLang server subprocess."""
        cmd = SGLangConfig.build_cmd_from_args(server_args)
        _env = os.environ.copy()
        triton_cache_path = _env.get("TRITON_CACHE_PATH", TRITON_CACHE_PATH)
        _env["TRITON_CACHE_PATH"] = os.path.join(triton_cache_path, str(uuid.uuid4()))

        return subprocess.Popen(
            cmd,
            env=_env,
            stdout=sys.stdout,
            stderr=sys.stdout,
        )


class RemoteSGLangEngine(InferenceEngine):
    """SGLang remote inference engine.

    This class delegates all functionality to RemoteInfEngine with
    an SGLangBackend implementation. It maintains the same public API.

    Parameters
    ----------
    config : InferenceEngineConfig
        Configuration for the inference engine
    """

    def __init__(self, config: InferenceEngineConfig):
        self.config = config
        # Pure composition - create internal engine with SGLang backend
        self._engine = RemoteInfEngine(config, SGLangBackend())

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

        backend_str = f"sglang:d{dp_size}"

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
        return self._engine.set_proxy_gateway_addr(addr)

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

    def offload(self):
        return self._engine.offload()

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
