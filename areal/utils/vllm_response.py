# SPDX-License-Identifier: Apache-2.0

import math
import re
from typing import Any

from areal.api.io_struct import HttpGenerationResult

# vLLM uses -9999.0 both when a chat logprob is absent and when a lower value
# is clamped, so the original sampling evidence cannot be recovered:
# https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/entrypoints/openai/chat_completion/protocol.py#L67-L70
# https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/entrypoints/openai/chat_completion/serving.py#L1721-L1727
_AMBIGUOUS_VLLM_LOGPROB = -9999.0


def parse_vllm_generation_response(
    response: dict[str, Any],
    *,
    score_centering_top_k: int = 0,
) -> HttpGenerationResult:
    """Normalize a vLLM generation response into AReaL's shared result type."""
    if (
        isinstance(score_centering_top_k, bool)
        or not isinstance(score_centering_top_k, int)
        or not 0 <= score_centering_top_k <= 1024
    ):
        raise ValueError("Score-centering head width must be an integer in [0, 1024].")
    meta_info = response["choices"][0]
    stop_reason = meta_info["finish_reason"]

    if score_centering_top_k:
        tokens = meta_info["logprobs"].get("tokens")
        if not isinstance(tokens, list):
            raise ValueError("vLLM score centering requires native text completions.")
        if any(
            not isinstance(token, str)
            or re.fullmatch(r"token_id:[0-9]+", token) is None
            for token in tokens
        ):
            raise ValueError("vLLM sampled tokens require token_id:<integer> encoding.")

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

    head_ids, head_logprobs = None, None
    if score_centering_top_k and output_tokens:
        if "tokens" not in meta_info["logprobs"]:
            raise ValueError("vLLM score centering requires native text completions.")
        head_ids, head_logprobs = _parse_sampler_heads(
            meta_info["logprobs"], output_tokens, output_logprobs, score_centering_top_k
        )

    return HttpGenerationResult(
        output_tokens=output_tokens,
        output_logprobs=output_logprobs,
        stop_reason=stop_reason,
        sampler_head_ids=head_ids,
        sampler_head_logprobs=head_logprobs,
    )


def _parse_sampler_heads(
    logprobs: dict[str, Any],
    output_tokens: list[int],
    output_logprobs: list[float],
    width: int,
) -> tuple[list[list[int]], list[list[float]]]:
    """Recover vLLM's top-K head without including an extra sampled tail token."""
    from areal.utils.score_centering import validate_heads

    rows = logprobs.get("top_logprobs")
    if not isinstance(rows, list) or len(rows) != len(output_tokens):
        raise ValueError("Missing or misaligned vLLM sampler heads.")
    if len(output_logprobs) != len(output_tokens):
        raise ValueError("Misaligned vLLM sampled-token logprobs.")

    head_ids, head_logprobs = [], []
    for sampled_id, sampled_logprob, row in zip(
        output_tokens, output_logprobs, rows, strict=True
    ):
        if not isinstance(row, dict) or len(row) not in (width, width + 1):
            raise ValueError("vLLM sampler head must contain K or K+1 entries.")
        ids, values = [], []
        for token, value in row.items():
            if (
                not isinstance(token, str)
                or re.fullmatch(r"token_id:[0-9]+", token) is None
            ):
                raise ValueError("vLLM sampler heads require token_id:<integer> keys.")
            ids.append(int(token.removeprefix("token_id:")))
            values.append(value)
        validate_heads([ids], [values], 1, len(ids))
        if _AMBIGUOUS_VLLM_LOGPROB in values:
            raise ValueError(
                "vLLM sampler head contains a missing or clamped -9999 logprob."
            )
        if sampled_id not in ids:
            raise ValueError("vLLM completion head is missing the sampled token.")
        sample_index = ids.index(sampled_id)
        if (
            isinstance(sampled_logprob, bool)
            or not isinstance(sampled_logprob, (float, int))
            or not math.isfinite(sampled_logprob)
            or sampled_logprob > 0
            or not math.isclose(
                values[sample_index], sampled_logprob, rel_tol=1e-5, abs_tol=1e-5
            )
        ):
            raise ValueError(
                "Sampled-token and head logprobs describe different distributions."
            )
        # vLLM emits the sampled token plus the true top-K, deduplicating when
        # the sample is in the head. For K+1 entries, remove the sample rather
        # than sorting/truncating the union: ties must not make head membership
        # depend on which token was sampled.
        if len(ids) == width + 1:
            ids.pop(sample_index)
            tail_logprob = values.pop(sample_index)
            boundary_logprob = min(values)
            if tail_logprob > boundary_logprob and not math.isclose(
                tail_logprob, boundary_logprob, rel_tol=1e-5, abs_tol=1e-5
            ):
                raise ValueError(
                    "vLLM sampled tail token outranks the returned top-K head."
                )
        head_ids.append(ids)
        head_logprobs.append(values)
    validate_heads(head_ids, head_logprobs, len(output_tokens), width)
    return head_ids, head_logprobs
