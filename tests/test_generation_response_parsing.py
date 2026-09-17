from typing import Any

import pytest

from areal.engine.vllm_remote import VLLMBackend
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend
from areal.v2.inference_service.vllm.bridge import VLLMBridgeBackend


@pytest.fixture(params=[VLLMBackend, VLLMBridgeBackend])
def vllm_backend(request: pytest.FixtureRequest) -> Any:
    return request.param()


@pytest.mark.parametrize("response_format", ["completion", "chat"])
def test_vllm_rejects_ambiguous_sampling_logprob(
    vllm_backend: Any, response_format: str
) -> None:
    if response_format == "completion":
        logprobs = {
            "tokens": ["token:42"],
            "token_logprobs": [-9999.0],
        }
    else:
        logprobs = {
            "content": [{"token": "token:42", "logprob": -9999.0}],
        }
    response = {"choices": [{"finish_reason": "stop", "logprobs": logprobs}]}

    with pytest.raises(ValueError, match=r"output_logprobs\[0\].*-9999\.0"):
        vllm_backend.parse_generation_response(response)


def test_vllm_accepts_zero_sampling_logprob(vllm_backend: Any) -> None:
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "logprobs": {
                    "tokens": ["token:42"],
                    "token_logprobs": [0.0],
                },
            }
        ]
    }

    result = vllm_backend.parse_generation_response(response)

    assert result.output_tokens == [42]
    assert result.output_logprobs == [0.0]


def test_sglang_v2_accepts_abort_before_prefill_without_evidence() -> None:
    response = {
        "meta_info": {
            "finish_reason": {
                "type": "abort",
                "message": "Abort before prefill due to engine pause",
            }
        }
    }

    result = SGLangBridgeBackend().parse_generation_response(response)

    assert result.output_tokens == []
    assert result.output_logprobs == []
    assert result.stop_reason == "abort"


@pytest.mark.parametrize(
    "finish_reason",
    [
        pytest.param({"type": "stop"}, id="normal-stop"),
        pytest.param(
            {"type": "abort", "message": "Abort after prefill"},
            id="abort-after-prefill",
        ),
    ],
)
def test_sglang_v2_requires_sampling_evidence_otherwise(
    finish_reason: dict[str, str],
) -> None:
    response = {"meta_info": {"finish_reason": finish_reason}}

    with pytest.raises(ValueError, match="output_token_logprobs"):
        SGLangBridgeBackend().parse_generation_response(response)
