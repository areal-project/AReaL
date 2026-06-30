# SPDX-License-Identifier: Apache-2.0

"""SGLang-specific inference bridge backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from areal.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    get_versioned_lora_name,
)

if TYPE_CHECKING:
    from areal.api.io_struct import ModelRequest


class SGLangBridgeBackend:
    """SGLang-specific backend for :class:`InfBridge`.

    Mirrors the relevant subset of
    :class:`areal.engine.sglang_remote.SGLangBackend`.
    """

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
            "top_k": gconfig.top_k,
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

        # Handle abort-before-prefill: no output tokens
        output_token_logprobs = meta_info.get("output_token_logprobs", [])
        output_tokens = [x[1] for x in output_token_logprobs]
        output_logprobs = [x[0] for x in output_token_logprobs]

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

    def get_offload_request(self) -> HttpRequest:
        return HttpRequest(endpoint="/release_memory_occupation", payload={})

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        payload = {"tags": tags} if tags is not None else {}
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
