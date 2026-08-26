# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific inference bridge backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from areal.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    detect_image_mime,
    get_versioned_lora_name,
)
from areal.engine.vllm_remote import _check_placeholder_count
from areal.utils.hf_utils import vision_pad_tokens

if TYPE_CHECKING:
    from areal.api.io_struct import ModelRequest


class VLLMBridgeBackend:
    """vLLM-specific backend for :class:`InfBridge`.

    Mirrors the relevant subset of
    :class:`areal.engine.vllm_remote.VLLMBackend`.
    """

    def build_generation_request(
        self,
        req: ModelRequest,
        with_lora: bool,
        version: int = -1,
    ) -> HttpRequest:
        """Build an ``/inference/v1/generate`` request."""
        gconfig = req.gconfig
        if gconfig.use_beam_search:
            raise NotImplementedError(
                "use_beam_search is not supported by the vLLM generate API; "
                "vLLM exposes beam search through a separate entrypoint."
            )

        # Compute effective max_new_tokens (cap by remaining context window)
        max_new_tokens = min(
            gconfig.max_tokens - len(req.input_ids),
            gconfig.max_new_tokens,
        )

        sampling_params: dict[str, Any] = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_tokens": max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": gconfig.stop_token_ids,
            "ignore_eos": gconfig.ignore_eos,
            "skip_special_tokens": gconfig.skip_special_tokens,
            "frequency_penalty": gconfig.frequency_penalty,
            # 0 = the sampled token's own logprob, no top-k alternatives.
            "logprobs": 0,
        }
        if gconfig.stop:
            sampling_params["stop"] = gconfig.stop

        payload: dict[str, Any] = {
            "request_id": req.rid,
            "sampling_params": sampling_params,
            "stream": False,
        }

        if with_lora:
            lora_name = gconfig.lora_name
            if not lora_name:
                raise ValueError(
                    "LoRA name (gconfig.lora_name) is required when use_lora is enabled."
                )
            payload["model"] = get_versioned_lora_name(lora_name, version)

        if req.image_data:
            # Prohibited at the sampler, not just filtered from the response; see
            # areal.engine.vllm_remote.VLLMBackend for why.
            bad_words = vision_pad_tokens(req.tokenizer, processor=req.processor)
            if bad_words:
                sampling_params["bad_words"] = list(bad_words)
            if not req.collapsed_input_ids:
                raise ValueError(
                    "A multimodal vLLM request needs collapsed_input_ids: the "
                    "prompt with one unexpanded placeholder per media item."
                )
            _check_placeholder_count(req)
            payload["token_ids"] = list(req.collapsed_input_ids)
            payload["expected_token_ids"] = list(req.input_ids)
            payload["content_parts"] = [
                {
                    "type": "image_url",
                    "url": f"data:{detect_image_mime(img)};base64,{img}",
                }
                for img in req.image_data
            ]
        else:
            payload["token_ids"] = list(req.input_ids)

        return HttpRequest(endpoint="/inference/v1/generate", payload=payload)

    def parse_generation_response(
        self,
        response: dict[str, Any],
    ) -> HttpGenerationResult:
        """Parse vLLM JSON into :class:`HttpGenerationResult`."""
        choice = response["choices"][0]
        stop_reason = choice.get("finish_reason") or "stop"
        output_tokens = list(choice.get("token_ids") or [])

        content = (choice.get("logprobs") or {}).get("content") or []
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

    def get_pause_request(self) -> HttpRequest:
        return HttpRequest(endpoint="/areal_pause_generation", payload={})

    def get_resume_request(self) -> HttpRequest:
        return HttpRequest(endpoint="/areal_continue_generation", payload={})

    def get_offload_request(self) -> HttpRequest:
        return HttpRequest(endpoint="/sleep", payload={}, method="POST")

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        if tags is not None:
            from urllib.parse import urlencode

            tags_query = urlencode({"tags": tags}, doseq=True)
            endpoint = f"/wake_up?{tags_query}"
        else:
            endpoint = "/wake_up"
        return HttpRequest(endpoint=endpoint, payload={}, method="POST")

    def get_generation_max_new_tokens(self, http_req: HttpRequest) -> int:
        return int(http_req.payload["sampling_params"]["max_tokens"])

    def patch_generation_request(
        self,
        http_req: HttpRequest,
        req: ModelRequest,
        accumulated_tokens: list[int],
        remaining_tokens: int,
    ) -> None:
        """Resume an interrupted generation from the tokens produced so far.

        Both prompt representations have to grow by the same suffix. Advancing
        only ``token_ids`` would leave ``expected_token_ids`` describing a
        different sequence, and the server's exact-token check would reject
        every retry -- which happens after each weight update, so uninterrupted
        tests would still pass.
        """
        http_req.payload["sampling_params"]["max_tokens"] = remaining_tokens
        if "expected_token_ids" in http_req.payload:
            http_req.payload["token_ids"] = (
                list(req.collapsed_input_ids) + accumulated_tokens
            )
            http_req.payload["expected_token_ids"] = (
                list(req.input_ids) + accumulated_tokens
            )
        else:
            http_req.payload["token_ids"] = list(req.input_ids) + accumulated_tokens
