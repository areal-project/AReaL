# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

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


def _make_score_response(token_logps: list[float]) -> dict[str, Any]:
    """SGLang score response carrying one log-prob per response token."""
    return {"meta_info": {"input_token_logprobs": [[lp, 0] for lp in token_logps]}}


class _MockScoreBackend(SGLangBackend):
    """Backend whose score parse returns exactly the values it was given."""

    def __init__(self, token_logps: list[float]) -> None:
        super().__init__()
        self._token_logps = token_logps

    def parse_score_response(
        self, response: dict[str, Any], target_len: int
    ) -> list[float]:
        assert len(self._token_logps) == target_len
        return list(self._token_logps)


class _ScoreResp:
    def __init__(self, token_logps: list[float]) -> None:
        self._token_logps = token_logps

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return _make_score_response(self._token_logps)


def _run_compute_logp(input_ids, loss_mask, token_logps, monkeypatch):
    engine = RemoteInfEngine(
        InferenceEngineConfig(backend="sglang:d1"),
        _MockScoreBackend(token_logps),
    )
    engine.addresses = ["mock-server"]
    monkeypatch.setattr(
        remote_inf_engine.requests,
        "request",
        lambda *a, **k: _ScoreResp(token_logps),
    )
    return engine.compute_logp([{"input_ids": input_ids, "loss_mask": loss_mask}])[0]


def test_compute_logp_writes_teacher_scores_at_prediction_positions(monkeypatch):
    """Teacher scores land one slot left of the response token.

    ``loss_mask`` marks response tokens, but the actor predicts token t at
    position t - 1 and downstream code rolls the mask and student log-probs left
    by one while passing ``teacher_logp`` unchanged. Writing at the response
    index would offset every teacher score by one position.
    """
    input_ids = torch.tensor([[10, 11, 12, 20, 21, 22]])
    loss_mask = torch.tensor([[0, 0, 0, 1, 1, 1]])
    token_logps = [-0.25, -0.5, -0.75]

    out = _run_compute_logp(input_ids, loss_mask, token_logps, monkeypatch)

    assert out[0, 2].item() == pytest.approx(-0.25)
    assert out[0, 3].item() == pytest.approx(-0.5)
    assert out[0, 4].item() == pytest.approx(-0.75)
    # Response-token positions stay zero-filled.
    assert out[0, 5].item() == 0.0
    # No teacher score may leak into the prompt region.
    assert out[0, 0].item() == 0.0
    assert out[0, 1].item() == 0.0


def test_compute_logp_single_token_response_keeps_teacher_signal(monkeypatch):
    """A one-token response previously lost its only teacher score."""
    input_ids = torch.tensor([[10, 11, 12, 20]])
    loss_mask = torch.tensor([[0, 0, 0, 1]])
    token_logps = [-0.4]

    out = _run_compute_logp(input_ids, loss_mask, token_logps, monkeypatch)

    assert out[0, 2].item() == pytest.approx(-0.4)
    assert out[0, 3].item() == 0.0
