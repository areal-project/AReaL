# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from areal.api.cli_args import GenerationHyperparameters, InferenceEngineConfig
from areal.api.io_struct import ModelRequest
from areal.engine.sglang_remote import SGLangBackend
from areal.infra import remote_inf_engine
from areal.infra.remote_inf_engine import RemoteInfEngine


def _make_sglang_response(
    token_logprobs: list[tuple[float, int]],
    finish_reason_type: str = "stop",
    *,
    routed_experts: np.ndarray | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> dict[str, Any]:
    meta_info: dict[str, Any] = {
        "finish_reason": {
            "type": finish_reason_type,
            "message": "",
        },
        "output_token_logprobs": token_logprobs,
    }
    if routed_experts is not None:
        if prompt_tokens is None or completion_tokens is None:
            raise ValueError("prompt_tokens and completion_tokens are required")
        meta_info["prompt_tokens"] = prompt_tokens
        meta_info["completion_tokens"] = completion_tokens
        meta_info["routed_experts"] = base64.b64encode(
            np.asarray(routed_experts, dtype=np.int32).tobytes()
        ).decode("utf-8")
    return {"meta_info": meta_info}


def _make_engine(*, return_routed_experts: bool = True) -> RemoteInfEngine:
    engine = RemoteInfEngine(
        InferenceEngineConfig(
            backend="sglang:d1",
            return_routed_experts=return_routed_experts,
        ),
        SGLangBackend(),
    )
    engine.addresses = ["mock-server"]
    engine._workflow_executor = SimpleNamespace(is_paused=lambda: False)
    return engine


def _make_request() -> ModelRequest:
    return ModelRequest(
        input_ids=[1, 2, 3],
        gconfig=GenerationHyperparameters(max_new_tokens=5, max_tokens=16),
    )


@pytest.mark.asyncio
async def test_agenerate_r3_resubmit_drops_duplicate_prefix(monkeypatch):
    """Abort/resubmit keeps one routing row sequence instead of duplicating prefix."""
    calls: list[dict[str, Any]] = []
    first_routing = np.arange(1, 1 + 4 * 4, dtype=np.int32).reshape(4, 4)
    second_routing = np.concatenate(
        [
            first_routing,
            np.arange(101, 101 + 4, dtype=np.int32).reshape(1, 4),
        ],
        axis=0,
    )

    async def mock_get_session():
        return object()

    async def mock_request(**kwargs):
        payload = kwargs["payload"]
        calls.append(dict(payload))
        if len(calls) == 1:
            return _make_sglang_response(
                [(-0.5, 100), (-0.3, 101)],
                "abort",
                routed_experts=first_routing,
                prompt_tokens=3,
                completion_tokens=2,
            )
        return _make_sglang_response(
            [(-0.2, 200)],
            "stop",
            routed_experts=second_routing,
            prompt_tokens=5,
            completion_tokens=1,
        )

    monkeypatch.setattr(
        remote_inf_engine.workflow_context,
        "get_aiohttp_session",
        mock_get_session,
    )
    monkeypatch.setattr(remote_inf_engine, "arequest_with_retry", mock_request)

    resp = await _make_engine().agenerate(_make_request())

    assert calls[0]["return_routed_experts"] is True
    assert calls[1]["return_routed_experts"] is True
    assert calls[1]["input_ids"] == [1, 2, 3, 100, 101]
    assert resp.input_tokens == [1, 2, 3]
    assert resp.output_tokens == [100, 101, 200]
    np.testing.assert_array_equal(resp.routed_experts, second_routing)


@pytest.mark.asyncio
async def test_agenerate_r3_resubmit_shorter_routing_raises(monkeypatch):
    """A shorter resubmit routing payload cannot cover already accepted rows."""
    first_routing = np.arange(1, 1 + 4 * 4, dtype=np.int32).reshape(4, 4)
    shorter_routing = np.arange(1, 1 + 3 * 4, dtype=np.int32).reshape(3, 4)
    calls = 0

    async def mock_get_session():
        return object()

    async def mock_request(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _make_sglang_response(
                [(-0.5, 100), (-0.3, 101)],
                "abort",
                routed_experts=first_routing,
                prompt_tokens=3,
                completion_tokens=2,
            )
        return _make_sglang_response(
            [(-0.2, 200)],
            "stop",
            routed_experts=shorter_routing,
            prompt_tokens=2,
            completion_tokens=2,
        )

    monkeypatch.setattr(
        remote_inf_engine.workflow_context,
        "get_aiohttp_session",
        mock_get_session,
    )
    monkeypatch.setattr(remote_inf_engine, "arequest_with_retry", mock_request)

    with pytest.raises(RuntimeError, match="shorter than the previously accumulated"):
        await _make_engine().agenerate(_make_request())
