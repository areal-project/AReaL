# SPDX-License-Identifier: Apache-2.0

"""SGLang-specific inference bridge backend."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

from areal.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    get_versioned_lora_name,
)
from areal.utils import logging

if TYPE_CHECKING:
    from areal.api.io_struct import ModelRequest

logger = logging.getLogger("SGLangBridge")

# Probe switch for the qwen3 behav-logp investigation: log what the server
# actually returned at the parse point, before any downstream aggregation.
_DEBUG_LOGPROBS = os.environ.get("AREAL_DEBUG_BRIDGE_LOGPROBS", "0") == "1"
_SGLANG_TOP_K_ALL_THRESHOLD = 1_000_000


def _normalize_sglang_top_k(top_k: int) -> int:
    """Translate AReaL's large "all vocab" top-k sentinel to SGLang's -1."""
    if top_k >= _SGLANG_TOP_K_ALL_THRESHOLD:
        return -1
    return top_k


class SGLangBridgeBackend:
    """SGLang-specific backend for :class:`InfBridge`.

    Mirrors the relevant subset of
    :class:`areal.engine.sglang_remote.SGLangBackend`.
    """

    def __init__(self, use_awex_memory_endpoints: bool | None = None) -> None:
        if use_awex_memory_endpoints is None:
            raw = os.environ.get("DTE_USE_AWEX_MEMORY_ENDPOINTS")
            if raw is None or raw.strip() == "":
                raw = os.environ.get("AWEX_USE_MEMORY_ENDPOINTS", "0")
            use_awex_memory_endpoints = raw.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self.use_awex_memory_endpoints = use_awex_memory_endpoints

    # -- generation ---------------------------------------------------------

    def build_generation_request(
        self,
        req: ModelRequest,
        with_lora: bool,
        version: int = -1,
    ) -> HttpRequest:
        """Build a ``/generate`` request for SGLang."""
        gconfig = req.gconfig

        if gconfig.use_beam_search:
            raise NotImplementedError(
                "Beam search is not supported in the SGLang bridge backend."
            )

        # Compute effective max_new_tokens
        max_new_tokens = min(
            gconfig.max_tokens - len(req.input_ids),
            gconfig.max_new_tokens,
        )

        sampling_params: dict[str, Any] = {
            "top_p": gconfig.top_p,
            "top_k": _normalize_sglang_top_k(gconfig.top_k),
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": gconfig.stop_token_ids,
            "ignore_eos": gconfig.ignore_eos,
            "skip_special_tokens": gconfig.skip_special_tokens,
            "frequency_penalty": gconfig.frequency_penalty,
        }
        if gconfig.stop:
            sampling_params["stop"] = gconfig.stop

        payload: dict[str, Any] = {
            "input_ids": list(req.input_ids),
            "image_data": req.image_data,
            "sampling_params": sampling_params,
            "return_logprob": True,
            "stream": False,
        }

        if _DEBUG_LOGPROBS:
            logger.info(
                "bridge request: input_len=%d sampling_params=%s",
                len(req.input_ids),
                sampling_params,
            )

        if req.metadata.get("return_routed_experts", False):
            payload["return_routed_experts"] = True

        if with_lora:
            lora_name = gconfig.lora_name
            if not lora_name:
                raise ValueError(
                    "LoRA name (gconfig.lora_name) is required when use_lora "
                    "is enabled."
                )
            payload["lora_path"] = get_versioned_lora_name(lora_name, version)

        return HttpRequest(endpoint="/generate", payload=payload)

    # -- response parsing ---------------------------------------------------

    def parse_generation_response(
        self,
        response: dict[str, Any],
    ) -> HttpGenerationResult:
        """Parse SGLang ``/generate`` JSON into :class:`HttpGenerationResult`."""
        import pybase64

        meta_info = response["meta_info"]
        finish_reason = meta_info["finish_reason"]
        stop_reason: str = finish_reason["type"]
        stop_message: str = finish_reason.get("message", "")

        # Routed experts (MoE)
        routed_experts: np.ndarray | None = None
        raw_experts = meta_info.get("routed_experts", None)
        if raw_experts is not None:
            num_sgl_token = (
                meta_info["prompt_tokens"] + meta_info["completion_tokens"] - 1
            )
            routed_experts = np.frombuffer(
                pybase64.b64decode(raw_experts.encode("utf-8")),
                dtype=np.int32,
            ).reshape(num_sgl_token, -1)

        if stop_reason == "abort" and stop_message.startswith("Abort before prefill"):
            return HttpGenerationResult(
                output_tokens=[],
                output_logprobs=[],
                stop_reason=stop_reason,
                routed_experts=routed_experts,
            )

        if "output_token_logprobs" not in meta_info:
            raise ValueError(
                "Malformed SGLang response: output_token_logprobs is missing "
                "from meta_info."
            )
        output_token_logprobs = meta_info["output_token_logprobs"]
        output_tokens = [x[1] for x in output_token_logprobs]
        output_logprobs = [x[0] for x in output_token_logprobs]

        if _DEBUG_LOGPROBS and output_logprobs:
            logger.info(
                "bridge parse: n=%d mean_logp=%.4f first5=%s stop=%s",
                len(output_logprobs),
                sum(output_logprobs) / len(output_logprobs),
                [round(x, 4) for x in output_logprobs[:5]],
                stop_reason,
            )

        return HttpGenerationResult(
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            stop_reason=stop_reason,
            routed_experts=routed_experts,
        )

    # -- pause / resume -----------------------------------------------------

    def get_pause_request(self) -> HttpRequest:
        return HttpRequest(endpoint="/pause_generation", payload={})

    def get_resume_request(self) -> HttpRequest:
        return HttpRequest(endpoint="/continue_generation", payload={})

    def get_offload_request(self, tags: list[str] | None = None) -> HttpRequest:
        payload = {"tags": tags} if tags is not None else {}
        if self.use_awex_memory_endpoints:
            return HttpRequest(endpoint="/awex/release_memory", payload=payload)
        return HttpRequest(endpoint="/release_memory_occupation", payload=payload)

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        payload = {"tags": tags} if tags is not None else {}
        if self.use_awex_memory_endpoints:
            return HttpRequest(endpoint="/awex/resume_memory", payload=payload)
        return HttpRequest(endpoint="/resume_memory_occupation", payload=payload)

    def get_generation_max_new_tokens(self, http_req: HttpRequest) -> int:
        return int(http_req.payload["sampling_params"]["max_new_tokens"])

    def patch_generation_request(
        self,
        http_req: HttpRequest,
        req: ModelRequest,
        accumulated_tokens: list[int],
        remaining_tokens: int,
    ) -> None:
        http_req.payload["input_ids"] = list(req.input_ids) + accumulated_tokens
        http_req.payload["sampling_params"]["max_new_tokens"] = remaining_tokens
