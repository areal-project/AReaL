"""Unit tests for InfBridge -- _AsyncGenerateEngine protocol via pluggable backend."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import ModelRequest, ModelResponse, get_versioned_lora_name
from areal.experimental.inference_service.data_proxy.pause import PauseState
from areal.experimental.inference_service.inf_bridge import InfBridge
from areal.experimental.inference_service.sglang.bridge import SGLangBridgeBackend
from areal.experimental.inference_service.vllm.bridge import VLLMBridgeBackend

# =============================================================================
# Helpers
# =============================================================================


def _make_sglang_response(
    token_logprobs: list[tuple[float, int]],
    finish_reason_type: str = "stop",
) -> dict[str, Any]:
    """Build a minimal SGLang /generate JSON response.

    Args:
        token_logprobs: List of (logprob, token_id) pairs.
        finish_reason_type: One of "stop", "abort", "length", "tool_calls".

    Returns:
        Dict matching SGLang /generate response schema.
    """
    return {
        "meta_info": {
            "finish_reason": {"type": finish_reason_type},
            "output_token_logprobs": token_logprobs,
        },
    }


def _make_vllm_response(
    tokens: list[int],
    token_logprobs: list[float],
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """Build a minimal vLLM completions JSON response."""
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "logprobs": {
                    "tokens": [f"token:{token}" for token in tokens],
                    "token_logprobs": token_logprobs,
                },
            }
        ]
    }


def _make_request(
    input_ids: list[int] | None = None,
    max_new_tokens: int = 20,
    max_tokens: int = 32768,
    n_samples: int = 1,
    greedy: bool = False,
    temperature: float = 1.0,
    metadata: dict[str, Any] | None = None,
    lora_name: str | None = None,
) -> ModelRequest:
    """Create a ModelRequest with sensible defaults for testing."""
    if input_ids is None:
        input_ids = [1, 2, 3]
    gconfig = GenerationHyperparameters(
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        max_tokens=max_tokens,
        greedy=greedy,
        temperature=temperature,
    )
    if lora_name is not None:
        gconfig.lora_name = lora_name
    return ModelRequest(
        input_ids=input_ids,
        gconfig=gconfig,
        metadata=metadata or {},
    )


def _make_bridge(
    pause_state: PauseState | None = None,
    backend: SGLangBridgeBackend | VLLMBridgeBackend | None = None,
    **kwargs: Any,
) -> InfBridge:
    """Create an InfBridge with a pluggable backend and sensible defaults."""
    if pause_state is None:
        pause_state = PauseState()
    if backend is None:
        backend = SGLangBridgeBackend()
    kwargs.setdefault("backend_addr", "http://mock")
    kwargs.setdefault("resubmit_wait", 0.01)
    return InfBridge(
        backend=backend,
        pause_state=pause_state,
        **kwargs,
    )


# =============================================================================
# TestInfBridge
# =============================================================================


class TestInfBridge:
    """Tests for InfBridge abort/resubmit loop and ModelResponse construction."""

    # -- 1. Normal stop ---------------------------------------------------------

    @pytest.mark.asyncio
    async def test_normal_stop_returns_model_response(self):
        """SGLang returns stop -- verify all ModelResponse fields populated."""
        pause_state = PauseState()
        bridge = _make_bridge(pause_state=pause_state)

        sglang_resp = _make_sglang_response([(-0.5, 100), (-0.3, 101)], "stop")
        bridge._send_request = AsyncMock(return_value=sglang_resp)

        req = _make_request(input_ids=[1, 2, 3])
        resp = await bridge.agenerate(req)

        assert isinstance(resp, ModelResponse)
        assert resp.output_tokens == [100, 101]
        assert resp.output_logprobs == [-0.5, -0.3]
        assert resp.stop_reason == "stop"
        assert resp.input_tokens == [1, 2, 3]
        assert resp.latency > 0
        bridge._send_request.assert_called_once()

    # -- 2. Single abort then stop -----------------------------------------------

    @pytest.mark.asyncio
    async def test_single_abort_then_stop_accumulates_tokens(self):
        """One abort, then stop -- verify token accumulation across resubmits."""
        call_count = 0

        async def mock_send(http_req, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_sglang_response([(-0.5, 100), (-0.3, 101)], "abort")
            return _make_sglang_response([(-0.4, 200), (-0.2, 201)], "stop")

        bridge = _make_bridge()
        bridge._send_request = mock_send

        req = _make_request(input_ids=[1, 2, 3], max_new_tokens=20)
        resp = await bridge.agenerate(req)

        assert call_count == 2
        assert resp.output_tokens == [100, 101, 200, 201]
        assert resp.output_logprobs == [-0.5, -0.3, -0.4, -0.2]
        assert resp.stop_reason == "stop"

    # -- 3. Resubmit passes input_ids + accumulated ------------------------------

    @pytest.mark.asyncio
    async def test_resubmit_input_ids_extended(self):
        """Verify resubmit passes input_ids + accumulated output as new input."""
        calls: list[dict[str, Any]] = []

        async def mock_send(http_req, **kwargs):
            # Capture the payload that InfBridge sends
            calls.append(
                {
                    "input_ids": list(http_req.payload["input_ids"]),
                    "params": dict(http_req.payload["sampling_params"]),
                }
            )
            if len(calls) == 1:
                return _make_sglang_response([(-0.5, 100), (-0.3, 101)], "abort")
            return _make_sglang_response([(-0.4, 200)], "stop")

        bridge = _make_bridge()
        bridge._send_request = mock_send

        req = _make_request(input_ids=[1, 2, 3], max_new_tokens=20)
        await bridge.agenerate(req)

        # First call: original input_ids
        assert calls[0]["input_ids"] == [1, 2, 3]

        # Second call: original + accumulated output tokens
        assert calls[1]["input_ids"] == [1, 2, 3, 100, 101]

    # -- 4. max_new_tokens decremented -------------------------------------------

    @pytest.mark.asyncio
    async def test_max_new_tokens_decremented_on_resubmit(self):
        """Verify max_new_tokens decreases by accumulated token count."""
        calls: list[dict[str, Any]] = []

        async def mock_send(http_req, **kwargs):
            calls.append(
                {
                    "input_ids": list(http_req.payload["input_ids"]),
                    "params": dict(http_req.payload["sampling_params"]),
                }
            )
            if len(calls) == 1:
                return _make_sglang_response([(-0.5, 100), (-0.3, 101)], "abort")
            return _make_sglang_response([(-0.4, 200)], "stop")

        bridge = _make_bridge()
        bridge._send_request = mock_send

        req = _make_request(input_ids=[1, 2, 3], max_new_tokens=20)
        await bridge.agenerate(req)

        # First call: full max_new_tokens
        assert calls[0]["params"]["max_new_tokens"] == 20

        # Second call: decremented by 2 (the 2 tokens from first call)
        assert calls[1]["params"]["max_new_tokens"] == 18

    # -- 5. max_new_tokens exhausted becomes length ------------------------------

    @pytest.mark.asyncio
    async def test_max_new_tokens_exhausted_becomes_length(self):
        """When accumulated tokens reach max_new_tokens, stop_reason='length'."""
        call_count = 0

        async def mock_send(http_req, **kwargs):
            nonlocal call_count
            call_count += 1
            # Always return 5 tokens with abort
            return _make_sglang_response(
                [(-0.1, 10), (-0.1, 11), (-0.1, 12), (-0.1, 13), (-0.1, 14)],
                "abort",
            )

        bridge = _make_bridge(max_resubmit_retries=10)
        bridge._send_request = mock_send

        req = _make_request(input_ids=[1, 2], max_new_tokens=10)
        resp = await bridge.agenerate(req)

        # First call returns 5 tokens (abort), second call max_new_tokens=5,
        # returns 5 more tokens (abort), third call max_new_tokens=0 -> break as length
        assert call_count == 2
        assert len(resp.output_tokens) == 10
        assert resp.stop_reason == "length"

    # -- 6. Max retries final abort becomes length --------------------------------

    @pytest.mark.asyncio
    async def test_max_retries_final_abort_becomes_length(self):
        """After max retries, final abort is converted to 'length'."""
        sglang_resp = _make_sglang_response([(-0.1, 10)], "abort")

        bridge = _make_bridge(max_resubmit_retries=3)
        bridge._send_request = AsyncMock(return_value=sglang_resp)

        req = _make_request(input_ids=[1, 2], max_new_tokens=100)
        resp = await bridge.agenerate(req)

        assert bridge._send_request.call_count == 3
        assert resp.stop_reason == "length"
        assert len(resp.output_tokens) == 3  # 1 token per retry

    # -- 7. Paused blocks until resumed ------------------------------------------

    @pytest.mark.asyncio
    async def test_paused_blocks_until_resumed(self):
        """While paused, agenerate waits; after resume, it completes."""
        sglang_resp = _make_sglang_response([(-0.5, 100)], "stop")

        pause_state = PauseState()
        bridge = _make_bridge(pause_state=pause_state)
        bridge._send_request = AsyncMock(return_value=sglang_resp)

        await pause_state.set_paused(True)
        req = _make_request(input_ids=[1, 2, 3], max_new_tokens=5)
        task = asyncio.create_task(bridge.agenerate(req))
        await asyncio.sleep(0.05)  # let it start
        assert not task.done()  # blocked by pause

        await pause_state.set_paused(False)  # unblock
        resp = await asyncio.wait_for(task, timeout=2.0)
        assert resp.stop_reason == "stop"
        assert resp.output_tokens == [100]

    # -- 8. tool_calls stop reason passthrough -----------------------------------

    @pytest.mark.asyncio
    async def test_tool_calls_stop_reason_passthrough(self):
        """stop_reason='tool_calls' exits the loop normally."""
        sglang_resp = _make_sglang_response([(-0.5, 100), (-0.3, 101)], "tool_calls")

        bridge = _make_bridge()
        bridge._send_request = AsyncMock(return_value=sglang_resp)

        req = _make_request(input_ids=[1, 2], max_new_tokens=20)
        resp = await bridge.agenerate(req)

        assert resp.stop_reason == "tool_calls"
        assert resp.output_tokens == [100, 101]
        bridge._send_request.assert_called_once()

    # -- 9. output_versions populated correctly ----------------------------------

    @pytest.mark.asyncio
    async def test_output_versions_populated(self):
        """Verify output_versions = [version] * len(output_tokens)."""
        sglang_resp = _make_sglang_response(
            [(-0.5, 100), (-0.3, 101), (-0.2, 102)], "stop"
        )

        bridge = _make_bridge(version=42)
        bridge._send_request = AsyncMock(return_value=sglang_resp)

        req = _make_request(input_ids=[1, 2, 3])
        resp = await bridge.agenerate(req)

        assert resp.output_versions == [42, 42, 42]
        assert len(resp.output_versions) == len(resp.output_tokens)

    @pytest.mark.asyncio
    async def test_output_versions_tracks_version_changes(self):
        """output_versions uses the version at call time, not construction time."""
        sglang_resp = _make_sglang_response([(-0.5, 100)], "stop")

        bridge = _make_bridge(version=1)
        bridge._send_request = AsyncMock(return_value=sglang_resp)

        bridge.set_version(7)
        req = _make_request(input_ids=[1, 2, 3])
        resp = await bridge.agenerate(req)

        assert resp.output_versions == [7]

    # -- 10. n_samples validation ------------------------------------------------

    @pytest.mark.asyncio
    async def test_n_samples_validation_raises_value_error(self):
        """n_samples != 1 raises ValueError."""
        bridge = _make_bridge()

        req = _make_request(input_ids=[1, 2, 3], n_samples=4)
        with pytest.raises(ValueError, match="n_samples=1"):
            await bridge.agenerate(req)

    # -- 11. max_new_tokens capped by max_tokens ---------------------------------

    @pytest.mark.asyncio
    async def test_max_new_tokens_capped_by_max_tokens(self):
        """max_new_tokens = min(max_tokens - input_len, max_new_tokens)."""
        calls: list[dict[str, Any]] = []

        async def mock_send(http_req, **kwargs):
            calls.append(
                {
                    "input_ids": list(http_req.payload["input_ids"]),
                    "params": dict(http_req.payload["sampling_params"]),
                }
            )
            return _make_sglang_response([(-0.5, 100)], "stop")

        bridge = _make_bridge()
        bridge._send_request = mock_send

        # input_len=3, max_tokens=10 -> effective max_new_tokens = min(7, 20) = 7
        req = _make_request(input_ids=[1, 2, 3], max_new_tokens=20, max_tokens=10)
        await bridge.agenerate(req)

        assert calls[0]["params"]["max_new_tokens"] == 7

    # -- 12. set_version / get_version -------------------------------------------

    @pytest.mark.asyncio
    async def test_set_get_version(self):
        """Version tracking works via set_version / get_version."""
        bridge = _make_bridge(version=0)

        assert bridge.get_version() == 0
        bridge.set_version(5)
        assert bridge.get_version() == 5
        bridge.set_version(99)
        assert bridge.get_version() == 99

    # -- 13. greedy sets temperature zero ----------------------------------------

    @pytest.mark.asyncio
    async def test_greedy_sets_temperature_zero(self):
        """gconfig.greedy=True -> sampling_params temperature=0.0 in HTTP payload."""
        calls: list[dict[str, Any]] = []

        async def mock_send(http_req, **kwargs):
            calls.append(
                {
                    "input_ids": list(http_req.payload["input_ids"]),
                    "params": dict(http_req.payload["sampling_params"]),
                }
            )
            return _make_sglang_response([(-0.5, 100)], "stop")

        bridge = _make_bridge()
        bridge._send_request = mock_send

        req = _make_request(input_ids=[1, 2, 3], greedy=True, temperature=0.8)
        await bridge.agenerate(req)

        assert calls[0]["params"]["temperature"] == 0.0

    # -- 14. length stop reason passthrough --------------------------------------

    @pytest.mark.asyncio
    async def test_length_stop_reason_passthrough(self):
        """stop_reason='length' exits the loop normally without resubmit."""
        sglang_resp = _make_sglang_response([(-0.5, 100)], "length")

        bridge = _make_bridge()
        bridge._send_request = AsyncMock(return_value=sglang_resp)

        req = _make_request(input_ids=[1, 2], max_new_tokens=20)
        resp = await bridge.agenerate(req)

        assert resp.stop_reason == "length"
        bridge._send_request.assert_called_once()


class TestVLLMBridgeBackend:
    @pytest.mark.asyncio
    async def test_vllm_text_abort_then_stop_accumulates_tokens(self):
        """vLLM text completions resubmit by extending prompt + shrinking max_tokens."""
        calls: list[dict[str, Any]] = []

        async def mock_send(http_req, **kwargs):
            calls.append(
                {
                    "prompt": list(http_req.payload["prompt"]),
                    "max_tokens": http_req.payload["max_tokens"],
                }
            )
            if len(calls) == 1:
                return _make_vllm_response([100, 101], [-0.5, -0.3], "abort")
            return _make_vllm_response([200], [-0.2], "stop")

        bridge = _make_bridge(backend=VLLMBridgeBackend())
        bridge._send_request = mock_send

        req = _make_request(input_ids=[1, 2, 3], max_new_tokens=5)
        resp = await bridge.agenerate(req)

        assert calls == [
            {"prompt": [1, 2, 3], "max_tokens": 5},
            {"prompt": [1, 2, 3, 100, 101], "max_tokens": 3},
        ]
        assert resp.output_tokens == [100, 101, 200]
        assert resp.output_logprobs == [-0.5, -0.3, -0.2]
        assert resp.stop_reason == "stop"

    def test_vllm_build_generation_request_for_text(self):
        """vLLM bridge uses flat completions payload for text requests."""
        backend = VLLMBridgeBackend()
        req = _make_request(input_ids=[11, 12], max_new_tokens=7)

        http_req = backend.build_generation_request(req, with_lora=False, version=0)

        assert http_req.endpoint == "/v1/completions"
        assert http_req.payload["prompt"] == [11, 12]
        assert http_req.payload["max_tokens"] == 7
        assert http_req.payload["stream"] is False

    def test_vllm_parse_generation_response_for_chat_format(self):
        """vLLM bridge parses chat logprobs content format."""
        backend = VLLMBridgeBackend()
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {"token": "token:42", "logprob": -0.1},
                            {"token": "token:43", "logprob": -0.2},
                        ]
                    },
                }
            ]
        }

        result = backend.parse_generation_response(response)

        assert result.output_tokens == [42, 43]
        assert result.output_logprobs == [-0.1, -0.2]
        assert result.stop_reason == "stop"

    # -- 4. max_new_tokens capped by max_tokens ----------------------------------

    @pytest.mark.asyncio
    async def test_vllm_max_new_tokens_capped_by_max_tokens(self):
        """vLLM: max_tokens = min(max_tokens - input_len, max_new_tokens)."""
        calls: list[dict[str, Any]] = []

        async def mock_send(http_req, **kwargs):
            calls.append(
                {
                    "prompt": list(http_req.payload["prompt"]),
                    "max_tokens": http_req.payload["max_tokens"],
                }
            )
            return _make_vllm_response([100], [-0.5], "stop")

        bridge = _make_bridge(backend=VLLMBridgeBackend())
        bridge._send_request = mock_send

        # input_len=3, max_tokens=10 -> effective max_new_tokens = min(7, 20) = 7
        req = _make_request(input_ids=[1, 2, 3], max_new_tokens=20, max_tokens=10)
        await bridge.agenerate(req)

        assert calls[0]["max_tokens"] == 7

    # -- 5. greedy sets temperature zero -----------------------------------------

    @pytest.mark.asyncio
    async def test_vllm_greedy_sets_temperature_zero(self):
        """vLLM: gconfig.greedy=True -> temperature=0.0 in HTTP payload."""
        calls: list[dict[str, Any]] = []

        async def mock_send(http_req, **kwargs):
            calls.append({"temperature": http_req.payload["temperature"]})
            return _make_vllm_response([100], [-0.5], "stop")

        bridge = _make_bridge(backend=VLLMBridgeBackend())
        bridge._send_request = mock_send

        req = _make_request(input_ids=[1, 2, 3], greedy=True, temperature=0.8)
        await bridge.agenerate(req)

        assert calls[0]["temperature"] == 0.0

    # -- 6. LoRA name in payload --------------------------------------------------

    def test_vllm_lora_name_in_payload(self):
        """vLLM: with_lora=True sets payload['model'] to versioned lora name."""
        backend = VLLMBridgeBackend()
        req = _make_request(input_ids=[1, 2], lora_name="my_lora")

        http_req = backend.build_generation_request(req, with_lora=True, version=3)

        expected_model = get_versioned_lora_name("my_lora", 3)
        assert http_req.payload["model"] == expected_model

    # -- 7. Vision request uses chat endpoint ------------------------------------

    def test_vllm_vision_request_uses_chat_endpoint(self):
        """vLLM: vision_msg_vllm field routes request to /v1/chat/completions."""
        backend = VLLMBridgeBackend()
        gconfig = GenerationHyperparameters(
            n_samples=1,
            max_new_tokens=20,
            max_tokens=32768,
        )
        vision_msgs = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "placeholder"},
                        },
                    ],
                }
            ]
        ]
        req = ModelRequest(
            input_ids=[1, 2, 3],
            gconfig=gconfig,
            metadata={},
            vision_msg_vllm=vision_msgs,
            image_data=["base64data"],
        )

        http_req = backend.build_generation_request(req, with_lora=False, version=0)

        assert http_req.endpoint == "/v1/chat/completions"
        assert "messages" in http_req.payload

    # -- 8. Pause blocks vLLM backend -------------------------------------------

    @pytest.mark.asyncio
    async def test_vllm_pause_blocks_until_resumed(self):
        """vLLM: while paused, agenerate waits; after resume, it completes."""
        vllm_resp = _make_vllm_response([100], [-0.5], "stop")

        pause_state = PauseState()
        bridge = _make_bridge(pause_state=pause_state, backend=VLLMBridgeBackend())
        bridge._send_request = AsyncMock(return_value=vllm_resp)

        await pause_state.set_paused(True)
        req = _make_request(input_ids=[1, 2, 3], max_new_tokens=5)
        task = asyncio.create_task(bridge.agenerate(req))
        await asyncio.sleep(0.05)  # let it start
        assert not task.done()  # blocked by pause

        await pause_state.set_paused(False)  # unblock
        resp = await asyncio.wait_for(task, timeout=2.0)
        assert resp.stop_reason == "stop"
        assert resp.output_tokens == [100]

    # -- 9. length stop reason passthrough ---------------------------------------

    @pytest.mark.asyncio
    async def test_vllm_length_stop_reason_passthrough(self):
        """vLLM: stop_reason='length' exits the loop normally without resubmit."""
        vllm_resp = _make_vllm_response([100], [-0.5], "length")

        bridge = _make_bridge(backend=VLLMBridgeBackend())
        bridge._send_request = AsyncMock(return_value=vllm_resp)

        req = _make_request(input_ids=[1, 2], max_new_tokens=20)
        resp = await bridge.agenerate(req)

        assert resp.stop_reason == "length"
        bridge._send_request.assert_called_once()

    # -- 10. output_versions populated correctly ---------------------------------

    @pytest.mark.asyncio
    async def test_vllm_output_versions_populated(self):
        """vLLM: output_versions = [version] * len(output_tokens)."""
        vllm_resp = _make_vllm_response([100, 101, 102], [-0.5, -0.3, -0.2], "stop")

        bridge = _make_bridge(version=42, backend=VLLMBridgeBackend())
        bridge._send_request = AsyncMock(return_value=vllm_resp)

        req = _make_request(input_ids=[1, 2, 3])
        resp = await bridge.agenerate(req)

        assert resp.output_versions == [42, 42, 42]
        assert len(resp.output_versions) == len(resp.output_tokens)

    # -- 11. max retries abort becomes length ------------------------------------

    @pytest.mark.asyncio
    async def test_vllm_max_retries_abort_becomes_length(self):
        """vLLM: after max retries, final abort is converted to 'length'."""
        vllm_resp = _make_vllm_response([10], [-0.1], "abort")

        bridge = _make_bridge(max_resubmit_retries=3, backend=VLLMBridgeBackend())
        bridge._send_request = AsyncMock(return_value=vllm_resp)

        req = _make_request(input_ids=[1, 2], max_new_tokens=100)
        resp = await bridge.agenerate(req)

        assert bridge._send_request.call_count == 3
        assert resp.stop_reason == "length"
        assert len(resp.output_tokens) == 3  # 1 token per retry
