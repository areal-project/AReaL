# SPDX-License-Identifier: Apache-2.0

"""Request identity and server timing correlation without a running SGLang server."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from areal.api.io_struct import ModelRequest, ModelResponse
from areal.infra.controller.rollout_controller import RolloutController
from areal.utils.session_request_ids import SessionRequestIds


def test_session_request_ids_increment_and_restore_nested_context():
    """Each physical send gets a new ID under the active session UUID."""
    outer = SessionRequestIds("outer")
    inner = SessionRequestIds("inner")

    assert SessionRequestIds.current() is None
    with outer.activate():
        assert outer.request_rid(outer.next_call_index(), 1) == "outer:1.1"
        with inner.activate():
            assert SessionRequestIds.current() is inner
            assert inner.request_rid(inner.next_call_index(), 1) == "inner:1.1"
        assert SessionRequestIds.current() is outer
        assert outer.request_rid(outer.next_call_index(), 1) == "outer:2.1"
    assert SessionRequestIds.current() is None


def test_session_request_ids_passes_sglang_latency_to_callback():
    """The callback gets the SGLang duration and physical RID once."""
    observed: list[tuple[str, float, float]] = []
    ids = SessionRequestIds("sample")
    ids.set_server_timing_callback(
        lambda rid, duration, ended_at: observed.append((rid, duration, ended_at))
    )

    rid = ids.request_rid(ids.next_call_index(), 1)
    ids.record_server_timing(rid, 7.5, 12.0)

    assert observed == [("sample:1.1", 7.5, 12.0)]


@pytest.mark.asyncio
async def test_proxy_controller_transfers_session_uuid_across_worker_rpc():
    """The selected worker receives the UUID without changing the caller request."""
    controller = RolloutController.__new__(RolloutController)
    controller._choose_worker = lambda: (SimpleNamespace(id="worker"), 0)
    controller._engine_name = lambda rank: f"rollout-{rank}"
    controller.scheduler = SimpleNamespace(async_call_engine=AsyncMock())
    request = ModelRequest()

    with SessionRequestIds("sample").activate():
        await controller.agenerate(request)

    sent = controller.scheduler.async_call_engine.await_args.kwargs["req"]
    assert sent.metadata["_areal_sglang_session_uuid"] == "sample"
    assert sent.metadata["_areal_sglang_call_index"] == 1
    assert "_areal_sglang_session_uuid" not in request.metadata


@pytest.mark.asyncio
async def test_proxy_controller_records_worker_server_latency_once():
    """The RPC response returns SGLang latency to the proxy session scope."""
    controller = RolloutController.__new__(RolloutController)
    controller._choose_worker = lambda: (SimpleNamespace(id="worker"), 0)
    controller._engine_name = lambda rank: f"rollout-{rank}"
    controller.scheduler = SimpleNamespace(
        async_call_engine=AsyncMock(
            return_value=ModelResponse(
                sglang_server_timings=[("sample:1.1", 7.0, 100.0)],
                sglang_worker_finished_at=101.0,
            )
        )
    )
    observed: list[tuple[str, float, float]] = []
    request_ids = SessionRequestIds("sample")
    request_ids.set_server_timing_callback(
        lambda rid, duration, ended_at: observed.append((rid, duration, ended_at))
    )

    with request_ids.activate():
        await controller.agenerate(ModelRequest())

    assert len(observed) == 1
    assert observed[0][0] == "sample:1.1"
    assert observed[0][1] == 7.0
