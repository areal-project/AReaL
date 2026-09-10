# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

io_struct = pytest.importorskip("sglang.srt.managers.io_struct")
from areal.v2.inference_service.sglang.rpc_proxy import RpcProxy  # noqa: E402
from areal.v2.inference_service.sglang.scheduler import (  # noqa: E402
    AwexSchedulerBridge,
)

RpcReqInput = io_struct.RpcReqInput
RpcReqOutput = io_struct.RpcReqOutput


class _FakeSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []

    async def send_pyobj(self, value):
        self.sent.append(value)

    async def recv_pyobj(self):
        await asyncio.sleep(0)
        return self.messages.pop(0)


@pytest.mark.asyncio
async def test_rpc_discards_late_ack_and_result(monkeypatch):
    proxy = RpcProxy.__new__(RpcProxy)
    proxy._lock = asyncio.Lock()
    proxy._default_timeout_s = 1.0
    proxy._rpc_socket = _FakeSocket(
        [
            RpcReqOutput(True, "", rid="old-rid"),
            RpcReqOutput(True, "", rid="new-rid"),
        ]
    )
    proxy._result_pull = _FakeSocket(
        [
            {"rid": "old-rid", "value": "old"},
            {"rid": "new-rid", "value": "new"},
        ]
    )
    monkeypatch.setattr(
        "areal.v2.inference_service.sglang.rpc_proxy.uuid4",
        lambda: SimpleNamespace(hex="new-rid"),
    )

    result = await proxy.collective_rpc_with_result("method", timeout_s=1.0)

    assert result == "new"
    assert proxy._rpc_socket.sent[0].rid == "new-rid"


def test_scheduler_dispatcher_copies_request_rid(monkeypatch):
    monkeypatch.delenv("AREAL_AWEX_RESULT_IPC", raising=False)
    scheduler = MagicMock(tp_rank=0, dp_rank=0)
    scheduler.handle_rpc_request.return_value = RpcReqOutput(True, "")
    bridge = AwexSchedulerBridge(scheduler)

    bridge.bind()
    response = scheduler.handle_rpc_request(
        RpcReqInput(method="awex_randomize_parameters", rid="request-rid")
    )

    assert response.rid == "request-rid"
    scheduler.init_request_dispatcher.assert_called_once_with()


def test_scheduler_deducts_queue_time_from_pg_timeout(monkeypatch):
    monkeypatch.delenv("AREAL_AWEX_RESULT_IPC", raising=False)
    monkeypatch.setattr(
        "areal.v2.inference_service.sglang.scheduler.time.monotonic", lambda: 14.0
    )
    bridge = AwexSchedulerBridge(MagicMock(tp_rank=0, dp_rank=0))
    bridge._adapter = MagicMock()

    bridge.awex_init_weights_update_group(
        pair_name="pair",
        operation_id="operation",
        operation_expiry_monotonic=20.0,
        process_group_timeout_s=9.0,
    )

    assert (
        bridge._adapter.init_weight_update_group.call_args.kwargs[
            "process_group_timeout_s"
        ]
        == 6.0
    )


def test_scheduler_tombstone_blocks_late_init(monkeypatch):
    monkeypatch.delenv("AREAL_AWEX_RESULT_IPC", raising=False)
    bridge = AwexSchedulerBridge(MagicMock(tp_rank=0, dp_rank=0))
    bridge.awex_teardown_weight_update_group("pair", "operation")

    with pytest.raises(RuntimeError, match="was cancelled"):
        bridge.awex_init_weights_update_group(
            pair_name="pair", operation_id="operation"
        )
    assert bridge._adapter is None
