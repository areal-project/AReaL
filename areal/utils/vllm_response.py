# SPDX-License-Identifier: Apache-2.0

from typing import Any

from areal.api.io_struct import HttpGenerationResult

# vLLM uses -9999.0 both when a chat logprob is absent and when a lower value
# is clamped, so the original sampling evidence cannot be recovered:
# https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/entrypoints/openai/chat_completion/protocol.py#L67-L70
# https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/entrypoints/openai/chat_completion/serving.py#L1721-L1727
_AMBIGUOUS_VLLM_LOGPROB = -9999.0


def parse_vllm_generation_response(
    response: dict[str, Any],
) -> HttpGenerationResult:
    """Normalize a vLLM generation response into AReaL's shared result type."""
    meta_info = response["choices"][0]
    stop_reason = meta_info["finish_reason"]

    if "tokens" in meta_info["logprobs"]:
        output_tokens = [
            int(token.split(":")[1]) for token in meta_info["logprobs"]["tokens"]
        ]
        output_logprobs = meta_info["logprobs"]["token_logprobs"]
    elif "content" in meta_info["logprobs"]:
        outputs = meta_info["logprobs"]["content"]
        output_tokens = [int(token["token"].split(":")[1]) for token in outputs]
        output_logprobs = [token["logprob"] for token in outputs]
    else:
        raise ValueError("Unexpected vLLM response format.")

    for index, logprob in enumerate(output_logprobs):
        if logprob == _AMBIGUOUS_VLLM_LOGPROB:
            raise ValueError(
                f"vLLM output_logprobs[{index}] is {_AMBIGUOUS_VLLM_LOGPROB}, which "
                "may represent a missing or clamped logprob; exact sampling "
                "evidence is required."
            )

    if stop_reason == "abort" and len(output_tokens) == 0:
        return HttpGenerationResult(
            output_tokens=[],
            output_logprobs=[],
            stop_reason=stop_reason,
        )

    return HttpGenerationResult(
        output_tokens=output_tokens,
        output_logprobs=output_logprobs,
        stop_reason=stop_reason,
    )
