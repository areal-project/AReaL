# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from typing import Any

import requests
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
from areal.utils.hf_utils import media_marker_token_ids, vision_pad_tokens
from areal.utils.network import format_host_for_url
from areal.utils.offload import sanitize_tms_env_vars

logger = logging.getLogger("vLLMEngine")


def _check_placeholder_count(req: ModelRequest) -> None:
    """Fail here rather than let the server fail opaquely.

    vLLM replaces one placeholder per media item. If the collapsed prompt has a
    different number, its renderer raises an assertion and the caller sees an
    HTTP 500 with no indication of which prompt was wrong -- the exact-token
    check never runs, because rendering fails first. This turns that into a
    named error naming both counts, before the request leaves the process.
    """
    processor = getattr(req, "processor", None)
    tokenizer = getattr(req, "tokenizer", None) or getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError(
            "A multimodal vLLM request needs a tokenizer (directly or via its "
            "processor) so the collapsed prompt can be checked against the "
            "media it carries. Skipping the check here only moves the failure "
            "to the server, where it surfaces as an opaque render error."
        )
    marker_ids = media_marker_token_ids(tokenizer, processor=processor)
    if not marker_ids:
        raise ValueError(
            "Could not resolve this model's collapsed media marker, so the "
            "prompt cannot be checked against the media it carries. The "
            "processor should declare boi_token or image_token."
        )
    found = sum(1 for token in req.collapsed_input_ids if token in marker_ids)
    expected = len(req.image_data or ())
    if found != expected:
        raise ValueError(
            f"Collapsed prompt carries {found} media placeholder(s) for "
            f"{expected} media item(s). vLLM replaces exactly one placeholder "
            f"per item, so it would fail to render this prompt. The collapsed "
            f"prompt must be built from the same rendered text as input_ids."
        )


class VLLMBackend:
    """vLLM-specific backend implementation for remote inference."""

    @staticmethod
    def build_server_env(env: Mapping[str, str]) -> dict[str, str]:
        _env = sanitize_tms_env_vars(env)
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
        """Build a vLLM generation request.

        Text and multimodal rollouts share one endpoint. The engine consumes the
        token ids we send rather than re-rendering messages, so the prompt that
        produces the behaviour logprobs is the prompt we record and train on.

        For media, ``token_ids`` carries the collapsed prompt (one placeholder
        per item) because vLLM expands placeholders itself, and
        ``expected_token_ids`` carries the expanded prompt the server must
        reproduce before it will generate.
        """
        gconfig = req.gconfig
        if gconfig.use_beam_search:
            raise NotImplementedError(
                "use_beam_search is not supported by the vLLM generate API; "
                "vLLM exposes beam search through a separate entrypoint."
            )

        sampling_params: dict[str, Any] = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": gconfig.stop_token_ids,
            "ignore_eos": gconfig.ignore_eos,
            "skip_special_tokens": gconfig.skip_special_tokens,
            "frequency_penalty": gconfig.frequency_penalty,
            # 0 = the sampled token's own logprob, no top-k alternatives.
            "logprobs": 0,
        }
        if gconfig.stop:
            # Stop strings require server-side detokenization, so `detokenize`
            # must stay enabled whenever they are present.
            sampling_params["stop"] = gconfig.stop

        payload: dict[str, Any] = {
            "request_id": req.rid,
            "sampling_params": sampling_params,
            "stream": False,
        }

        has_media = bool(req.image_data)
        if has_media:
            # Prohibit this model's media placeholders at the sampler; vLLM
            # resolves these strings to token ids server side. Filtering them
            # out of the response is too late for an interrupted generation:
            # the abort loop appends each partial segment to both prompts
            # before the client-side filter runs, so a sampled placeholder
            # would reach the next request and fail media matching or the
            # exact-token check. The response filter stays as a backstop.
            bad_words = vision_pad_tokens(req.tokenizer, processor=req.processor)
            if bad_words:
                sampling_params["bad_words"] = list(bad_words)
            if not req.collapsed_input_ids:
                raise ValueError(
                    "A multimodal vLLM request needs collapsed_input_ids: the "
                    "prompt with one unexpanded placeholder per media item. "
                    "vLLM expands placeholders itself, so sending the expanded "
                    "input_ids would expand them twice. Build both forms where "
                    "the prompt is constructed."
                )
            _check_placeholder_count(req)
            payload["token_ids"] = req.collapsed_input_ids.copy()
            payload["expected_token_ids"] = req.input_ids.copy()
            payload["content_parts"] = [
                {
                    "type": "image_url",
                    "url": f"data:{detect_image_mime(img)};base64,{img}",
                }
                for img in req.image_data
            ]
        else:
            payload["token_ids"] = req.input_ids.copy()

        if with_lora:
            lora_name = gconfig.lora_name
            if not lora_name:
                raise ValueError(
                    "LoRA name (gconfig.lora_name) is required when use_lora is enabled."
                )
            payload["model"] = get_versioned_lora_name(lora_name, version)

        return HttpRequest(endpoint="/inference/v1/generate", payload=payload)

    def parse_generation_response(
        self, response: dict[str, Any]
    ) -> HttpGenerationResult:
        """Parse vLLM generation response."""
        choice = response["choices"][0]
        stop_reason = choice.get("finish_reason") or "stop"
        output_tokens = list(choice.get("token_ids") or [])

        logprobs_payload = choice.get("logprobs") or {}
        content = logprobs_payload.get("content") or []
        output_logprobs = [entry["logprob"] for entry in content]
        # Behavior logprobs must remain parallel with generated tokens.
        if len(output_logprobs) != len(output_tokens):
            raise ValueError(
                f"vLLM returned {len(output_tokens)} token ids but "
                f"{len(output_logprobs)} logprobs; they must stay parallel."
            )

        return HttpGenerationResult(
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            stop_reason=stop_reason,
        )

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
            "backend": meta.backend
            if meta.backend is not None
            else current_platform.communication_backend,
            "group_name": meta.nccl_group_name,
        }
        return HttpRequest(endpoint="/areal_init_weights_update_group", payload=payload)

    def build_awex_init_request(
        self, meta: WeightUpdateMeta, engine_rank: int, num_engines: int
    ) -> HttpRequest:
        """Build vLLM Awex init request."""
        payload = {
            "meta_server_addr": meta.meta_server_addr,
            "engine_rank": engine_rank,
            "num_engines": num_engines,
            "comm_backend": meta.comm_backend or "file",
            "enable_debug_mode": meta.enable_debug_mode,
            "debug_mode_config": meta.debug_mode_config or {},
            "disable_weights_exchange_pipeline": meta.disable_weights_exchange_pipeline,
            "enable_colocate_mode": meta.enable_colocate_mode,
            "weights_exchange_ipc_backend": meta.weights_exchange_ipc_backend or "cuda",
            "weights_comm_nccl_group_size": meta.weights_comm_nccl_group_size,
            "weights_validation_steps": meta.weights_validation_steps,
            "validate_weights_every_n_steps": meta.validate_weights_every_n_steps,
        }
        if meta.dump_weights_list_for_validation:
            payload["dump_weights_list_for_validation"] = (
                meta.dump_weights_list_for_validation
            )
        if meta.dump_weights_dir_for_validation:
            payload["dump_weights_dir_for_validation"] = (
                meta.dump_weights_dir_for_validation
            )
        if meta.nnodes is not None:
            payload["nnodes"] = meta.nnodes
        if meta.node_rank is not None:
            payload["node_rank"] = meta.node_rank
        return HttpRequest(endpoint="/areal_awex_init", payload=payload)

    def build_awex_update_request(
        self, meta: WeightUpdateMeta, step_id: int, kwargs: dict | None
    ) -> HttpRequest:
        """Build vLLM Awex update request."""
        payload = {"step_id": step_id, "kwargs": kwargs or {}}
        return HttpRequest(endpoint="/areal_awex_update", payload=payload)

    def get_pause_request(self) -> HttpRequest:
        """Get vLLM pause request."""
        return HttpRequest(endpoint="/areal_pause_generation", payload={})

    def get_resume_request(self) -> HttpRequest:
        """Get vLLM resume request."""
        return HttpRequest(endpoint="/areal_continue_generation", payload={})

    def get_health_check_request(self) -> HttpRequest:
        """Get vLLM health check request."""
        return HttpRequest(endpoint="/health", payload={}, method="GET")

    def get_metrics_request(self) -> HttpRequest:
        """Get vLLM Prometheus metrics scrape request."""
        return HttpRequest(endpoint="/metrics", payload={}, method="GET")

    def get_offload_request(self) -> HttpRequest:
        """Get vLLM offload request.

        Uses vLLM's /sleep endpoint to offload model memory to CPU.
        Default level is 1.
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


def _sum_prometheus_counter(text: str, metric: str) -> float | None:
    """Sum every labeled series of a Prometheus counter across engines.

    Matches both ``metric`` and the ``metric + "_total"`` name that
    prometheus_client emits for counters. Returns None when the metric is
    absent from the scrape.
    """
    total: float | None = None
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        head = line.split("{", 1)[0].split(" ", 1)[0]
        if head == metric or head == metric + "_total":
            try:
                total = (total or 0.0) + float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                continue
    return total


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
        # Cumulative spec-decode counters at the previous export, for windowed
        # acceptance-rate deltas.
        self._spec_decode_prev: dict[str, float] = {}

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
        engine_rank: int | None = None,
        num_engines: int | None = None,
    ):
        """Initialize the engine by discovering and connecting to servers."""
        if train_data_parallel_size is None:
            train_data_parallel_size = ModelAllocation.from_str(
                self.config.backend, name="rollout"
            ).parallel.data_parallel_size
        return self._engine.initialize(
            engine_id,
            addr,
            train_data_parallel_size,
            engine_rank=engine_rank,
            num_engines=num_engines,
        )

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

    def update_weights_from_awex(
        self,
        meta: WeightUpdateMeta,
        step_id: int | None = None,
        kwargs: dict | None = None,
    ) -> Future[None]:
        """Update weights via Awex."""
        return self._engine.update_weights_from_awex(
            meta, step_id=step_id, kwargs=kwargs
        )

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
        stats = stats_tracker.export_all(reduce_group=None)
        try:
            stats.update(self._collect_spec_decode_stats())
        except Exception as e:  # metrics must never interrupt training
            logger.warning(f"Failed to collect spec-decode stats: {e}")
        return stats

    def _collect_spec_decode_stats(self) -> dict[str, float]:
        """Scrape vLLM's ``/metrics`` for speculative-decoding counters and
        return the acceptance stats over the window since the last export.

        Returns an empty dict when speculative decoding is disabled or the
        metrics endpoint is unavailable, so the caller stays a no-op unless a
        draft head is actually running.
        """
        get_metrics = getattr(self._engine.backend, "get_metrics_request", None)
        if get_metrics is None:
            return {}
        req = get_metrics()
        totals = {"drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0}
        found = False
        for addr in self._engine.addresses:
            try:
                resp = requests.request(
                    req.method, f"http://{addr}{req.endpoint}", timeout=3
                )
                if resp.status_code != 200:
                    continue
            except requests.exceptions.RequestException:
                continue
            for key, name in (
                ("drafts", "vllm:spec_decode_num_drafts"),
                ("draft_tokens", "vllm:spec_decode_num_draft_tokens"),
                ("accepted", "vllm:spec_decode_num_accepted_tokens"),
            ):
                value = _sum_prometheus_counter(resp.text, name)
                if value is not None:
                    totals[key] += value
                    found = True
        if not found:
            return {}

        prev = self._spec_decode_prev
        self._spec_decode_prev = totals
        d_draft_tokens = totals["draft_tokens"] - prev.get("draft_tokens", 0.0)
        d_accepted = totals["accepted"] - prev.get("accepted", 0.0)
        d_drafts = totals["drafts"] - prev.get("drafts", 0.0)
        if d_draft_tokens <= 0 or d_drafts <= 0:
            return {}
        # Emit companion ``__count`` keys so RolloutController.export_stats keeps
        # these (it drops keys without a count) and, weighting by the window's
        # draft/draft-token totals, aggregates the per-worker rates into the
        # correct global ratios rather than a plain mean.
        return {
            "rollout/spec_decode/acceptance_rate": d_accepted / d_draft_tokens,
            "rollout/spec_decode/acceptance_rate__count": d_draft_tokens,
            "rollout/spec_decode/mean_accepted_len": d_accepted / d_drafts,
            "rollout/spec_decode/mean_accepted_len__count": d_drafts,
        }

    @classmethod
    def as_controller(cls, config: InferenceEngineConfig, scheduler: Scheduler):
        if config._version == "v2":
            from areal.experimental.inference_service.controller.controller import (
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

    def clear_all_local_rtensors(self) -> dict[str, int]:
        """Forcibly drain actor-local RTensor storage at step boundary.

        Run after ``clear_batches`` as a defensive sweep for any RTensors
        created by auxiliary RPC calls (stats returns, etc.) that aren't
        in the standard rollout_batch/adv_batch lifecycle. See #1209.
        """
        from areal.infra.rpc.rtensor import clear_all_local

        n_storage, n_buffer = clear_all_local()
        return {"storage_cleared": n_storage, "fetch_buffer_cleared": n_buffer}

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        perf_tracer.save(step=step, force=force)

    def config_perf_tracer(
        self, config: PerfTracerConfig, rank: int, role: str
    ) -> None:
        if perf_tracer.is_configured():
            return
        perf_tracer.configure(config, rank=rank, role=role)
