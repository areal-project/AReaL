"""Unit tests for training-service data proxy."""

from __future__ import annotations

from typing import Any

import httpx
import orjson
import pytest
import pytest_asyncio
import torch

from areal.experimental.training_service.data_proxy.app import create_app
from areal.experimental.training_service.data_proxy.config import TrainDataProxyConfig
from areal.experimental.training_service.data_proxy.dispatcher import Dispatcher
from areal.experimental.training_service.data_proxy.topology import (
    WorkerInfo,
    WorkerTopology,
)
from areal.infra.controller.train_controller import _dispatch_tensors
from areal.infra.rpc.serialization import deserialize_value, serialize_value

ADMIN_KEY = "dp-admin-key"


# ------------------------------------------------------------------
# aiohttp mock helpers
# ------------------------------------------------------------------


class _FakeAiohttpResponse:
    def __init__(self, content: bytes, status: int = 200):
        self._content = content
        self.status = status

    async def read(self) -> bytes:
        return self._content


class _AsyncCM:
    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *args):
        pass


class _NoOpSession:
    async def close(self):
        pass


class _CapturingSession:
    def __init__(self, *, post_handler=None):
        self.captured_payloads: list[dict[str, Any]] = []
        self._post_handler = post_handler

    def post(self, url, *, data=b"", headers=None):
        _ = headers
        payload = orjson.loads(data)
        self.captured_payloads.append(
            {
                "url": url,
                "payload": payload,
                "args": deserialize_value(payload["args"]),
                "kwargs": deserialize_value(payload["kwargs"]),
            }
        )
        if self._post_handler:
            content = self._post_handler(url, data, headers)
        else:
            content = orjson.dumps({"status": "success", "result": payload["args"]})
        return _AsyncCM(_FakeAiohttpResponse(content))

    async def close(self):
        pass


# ------------------------------------------------------------------
# Fake Dispatcher for app-level route tests
# ------------------------------------------------------------------


class _FakeDispatchRequest:
    def __init__(
        self,
        parent: _FakeDispatcher,
        path: str,
        *,
        pad_eval_batch: bool = False,
    ):
        self._parent = parent
        self._path = path
        self._pad_eval_batch = pad_eval_batch

    async def get(self) -> bytes:
        return b'{"status":"success","result":{"path":"get"}}'

    async def post(self, body: bytes) -> bytes:
        _ = body
        self._parent.dispatch_calls.append(
            {"path": self._path, "pad_eval_batch": self._pad_eval_batch}
        )
        return b'{"status":"success","result":{"path":"compute"}}'


class _FakeBroadcastRequest:
    def __init__(self, parent: _FakeDispatcher, path: str):
        self._parent = parent
        self._path = path

    async def get(self) -> list[bytes]:
        return self._parent.broadcast_get_return

    async def post(self, body: bytes) -> list[bytes]:
        _ = body
        return self._parent.broadcast_return


class _FakeDispatcher:
    def __init__(self):
        self.broadcast_return: list[bytes] = [
            b'{"status":"success","result":{"ok":true}}'
        ]
        self.broadcast_get_return: list[bytes] = [
            b'{"status":"success","result":{"path":"broadcast_get"}}'
        ]
        self.dispatch_calls: list[dict[str, Any]] = []

    def dispatch(
        self, path: str, *, pad_eval_batch: bool = False
    ) -> _FakeDispatchRequest:
        return _FakeDispatchRequest(self, path, pad_eval_batch=pad_eval_batch)

    def broadcast(self, path: str) -> _FakeBroadcastRequest:
        return _FakeBroadcastRequest(self, path)


@pytest.fixture
def config() -> TrainDataProxyConfig:
    return TrainDataProxyConfig(
        host="127.0.0.1",
        port=18082,
        worker_addrs=["http://worker-0:19001", "http://worker-1:19001"],
        admin_api_key=ADMIN_KEY,
        request_timeout=10.0,
    )


@pytest_asyncio.fixture
async def app_client(config):
    app = create_app(config)
    app.state.topology = WorkerTopology(
        workers=[
            WorkerInfo(addr="http://worker-0:19001", rank=0, dp_rank=0, dp_size=2),
            WorkerInfo(addr="http://worker-1:19001", rank=1, dp_rank=1, dp_size=2),
        ],
        dp_heads=[0, 1],
        dp_size=2,
        dp_groups=[[0], [1]],
    )
    app.state.dispatcher = _FakeDispatcher()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield app, c


def _admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


class TestDataProxyBasics:
    @pytest.mark.asyncio
    async def test_health_and_topology(self, app_client):
        _app, client = app_client
        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json()["dp_size"] == 2

        topo = await client.get("/topology")
        assert topo.status_code == 200
        assert len(topo.json()["workers"]) == 2

    @pytest.mark.asyncio
    async def test_train_batch_uses_dispatcher(self, app_client):
        _app, client = app_client
        resp = await client.post("/train_batch", content=b"{}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @pytest.mark.asyncio
    async def test_ppo_actor_compute_logp_uses_dispatcher(self, app_client):
        _app, client = app_client
        resp = await client.post("/ppo/actor/compute_logp", content=b"{}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @pytest.mark.asyncio
    async def test_sft_train_uses_dispatcher(self, app_client):
        _app, client = app_client
        resp = await client.post("/sft/train", content=b"{}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @pytest.mark.asyncio
    async def test_rw_train_uses_dispatcher(self, app_client):
        _app, client = app_client
        resp = await client.post("/rw/train", content=b"{}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @pytest.mark.asyncio
    async def test_eval_routes_opt_in_to_padding(self, app_client):
        app, client = app_client

        await client.post("/eval_batch", content=b"{}")
        await client.post("/sft/evaluate", content=b"{}")
        await client.post("/rw/evaluate", content=b"{}")

        assert app.state.dispatcher.dispatch_calls == [
            {"path": "/eval_batch", "pad_eval_batch": True},
            {"path": "/sft/evaluate", "pad_eval_batch": True},
            {"path": "/rw/evaluate", "pad_eval_batch": True},
        ]

    @pytest.mark.asyncio
    async def test_training_routes_do_not_enable_padding(self, app_client):
        app, client = app_client

        await client.post("/train_batch", content=b"{}")
        await client.post("/forward_batch", content=b"{}")
        await client.post("/sft/train", content=b"{}")
        await client.post("/ppo/actor/update", content=b"{}")
        await client.post("/ppo/critic/update", content=b"{}")
        await client.post("/rw/train", content=b"{}")

        assert app.state.dispatcher.dispatch_calls == [
            {"path": "/train_batch", "pad_eval_batch": False},
            {"path": "/forward_batch", "pad_eval_batch": False},
            {"path": "/sft/train", "pad_eval_batch": False},
            {"path": "/ppo/actor/update", "pad_eval_batch": False},
            {"path": "/ppo/critic/update", "pad_eval_batch": False},
            {"path": "/rw/train", "pad_eval_batch": False},
        ]

    @pytest.mark.asyncio
    async def test_optimizer_step_empty_broadcast_returns_502(self, app_client):
        app, client = app_client
        app.state.dispatcher.broadcast_return = []
        resp = await client.post("/optimizer_step", content=b"{}")
        assert resp.status_code == 502
        assert "No worker responses" in resp.json()["error"]

    @pytest.mark.asyncio
    async def test_export_stats_uses_broadcast_get(self, app_client):
        app, client = app_client
        app.state.dispatcher.broadcast_get_return = [
            b'{"status":"success","result":{"stats":1}}'
        ]
        resp = await client.get("/export_stats")
        assert resp.status_code == 200
        assert resp.json()["result"]["stats"] == 1


def _make_dispatcher(
    *,
    dp_size: int,
    dp_heads: list[int],
    dp_ranks: list[int],
    session=None,
) -> Dispatcher:
    workers = [
        WorkerInfo(
            addr=f"http://worker-{i}:19001",
            rank=i,
            dp_rank=dp_ranks[i],
            dp_size=dp_size,
            is_dp_head=(i in dp_heads),
        )
        for i in range(len(dp_ranks))
    ]
    max_dp_rank = max(dp_ranks) if dp_ranks else 0
    dp_groups = [[] for _ in range(max_dp_rank + 1)]
    for i, dp_rank in enumerate(dp_ranks):
        dp_groups[dp_rank].append(i)
    topology = WorkerTopology(
        workers=workers,
        dp_heads=dp_heads,
        dp_size=dp_size,
        dp_groups=dp_groups,
    )
    return Dispatcher(
        topology=topology,
        request_timeout=10.0,
        _session=session or _NoOpSession(),
    )


def _make_tensor_item(seq_len: int) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.randint(0, 100, (1, seq_len), dtype=torch.long),
        "attention_mask": torch.ones((1, seq_len), dtype=torch.bool),
    }


class TestDispatcherParityWithTrainController:
    def test_partition_inputs_matches_train_controller_dispatch(self):
        dispatcher = _make_dispatcher(dp_size=2, dp_heads=[0, 1], dp_ranks=[0, 1])
        req = dispatcher.dispatch("/any")

        batch = [
            _make_tensor_item(16),
            _make_tensor_item(8),
            _make_tensor_item(12),
            _make_tensor_item(10),
        ]
        args = [batch]
        kwargs: dict[str, object] = {"tag": "x"}

        dp_args, dp_kwargs, group_indices = req._partition_inputs(
            args=args,
            kwargs=kwargs,
            group_size=1,
        )

        expected_splits, expected_indices = _dispatch_tensors(batch, 2, group_size=1)
        assert group_indices == expected_indices
        assert dp_args[0] == expected_splits
        assert dp_kwargs["tag"] == ["x", "x"]

    def test_partition_inputs_respects_group_size_atomicity(self):
        dispatcher = _make_dispatcher(dp_size=2, dp_heads=[0, 1], dp_ranks=[0, 1])
        req = dispatcher.dispatch("/any")

        batch = [
            _make_tensor_item(16),
            _make_tensor_item(8),
            _make_tensor_item(12),
            _make_tensor_item(10),
        ]
        dp_args, _dp_kwargs, group_indices = req._partition_inputs(
            args=[batch],
            kwargs={},
            group_size=2,
        )

        assert len(dp_args[0]) == 2
        for idxs in group_indices:
            assert len(idxs) % 2 == 0
            for i in range(0, len(idxs), 2):
                left, right = idxs[i], idxs[i + 1]
                assert right == left + 1
                assert left % 2 == 0

    @pytest.mark.asyncio
    async def test_fan_out_dispatches_only_to_dp_heads_for_model_parallel(
        self,
    ):
        session = _CapturingSession()
        dispatcher = _make_dispatcher(
            dp_size=1, dp_heads=[0], dp_ranks=[0, 0], session=session
        )
        req = dispatcher.dispatch("/forward_batch")

        dp_args = [[[_make_tensor_item(16), _make_tensor_item(8)]]]
        dp_kwargs = {"output_seqlens": [[2, 3]]}

        await req._fan_out(dp_args=dp_args, dp_kwargs=dp_kwargs)

        captured = session.captured_payloads
        assert len(captured) == 2
        assert captured[0]["args"] != []
        assert captured[0]["kwargs"] == {"output_seqlens": [2, 3]}
        assert "rpc_meta" not in captured[0]["payload"]
        assert captured[1]["args"] == []
        assert captured[1]["kwargs"] == {}
        assert "rpc_meta" not in captured[1]["payload"]

    @pytest.mark.asyncio
    async def test_fan_out_omits_rpc_meta_for_algorithm_paths(
        self,
    ):
        session = _CapturingSession()
        dispatcher = _make_dispatcher(
            dp_size=1, dp_heads=[0], dp_ranks=[0, 0], session=session
        )
        req = dispatcher.dispatch("/sft/train")

        dp_args = [[[_make_tensor_item(16), _make_tensor_item(8)]]]
        dp_kwargs = {"output_seqlens": [[2, 3]]}

        await req._fan_out(dp_args=dp_args, dp_kwargs=dp_kwargs)

        captured = session.captured_payloads
        assert len(captured) == 2
        assert "rpc_meta" not in captured[0]["payload"]
        assert "rpc_meta" not in captured[1]["payload"]

    @pytest.mark.asyncio
    async def test_fan_out_dispatches_per_dp_shard_when_dp_size_two(
        self,
    ):
        session = _CapturingSession()
        dispatcher = _make_dispatcher(
            dp_size=2, dp_heads=[0, 1], dp_ranks=[0, 1], session=session
        )
        req = dispatcher.dispatch("/forward_batch")

        shard0 = [_make_tensor_item(4)]
        shard1 = [_make_tensor_item(12), _make_tensor_item(6)]
        dp_args = [[shard0, shard1]]
        dp_kwargs = {"output_seqlens": [[1], [2, 1]]}

        await req._fan_out(dp_args=dp_args, dp_kwargs=dp_kwargs)

        captured = session.captured_payloads
        assert len(captured) == 2
        assert len(captured[0]["args"][0]) == len(shard0)
        assert len(captured[1]["args"][0]) == len(shard1)
        assert (
            captured[0]["args"][0][0]["input_ids"].shape == shard0[0]["input_ids"].shape
        )
        assert (
            captured[1]["args"][0][0]["input_ids"].shape == shard1[0]["input_ids"].shape
        )
        assert captured[0]["kwargs"] == {"output_seqlens": [1]}
        assert captured[1]["kwargs"] == {"output_seqlens": [2, 1]}

    @pytest.mark.asyncio
    async def test_dispatch_post_merges_results_in_original_order(
        self,
    ):
        def _shard_handler(url, data, headers):
            _ = url, headers
            payload = orjson.loads(data)
            args = deserialize_value(payload["args"])
            shard = args[0] if args else []
            shard_result = [int(item["attention_mask"].sum().item()) for item in shard]
            return orjson.dumps(
                {
                    "status": "success",
                    "result": serialize_value(shard_result),
                }
            )

        session = _CapturingSession(post_handler=_shard_handler)
        dispatcher = _make_dispatcher(
            dp_size=2, dp_heads=[0, 1], dp_ranks=[0, 1], session=session
        )

        batch = [
            _make_tensor_item(5),
            _make_tensor_item(11),
            _make_tensor_item(7),
            _make_tensor_item(13),
        ]
        body = orjson.dumps(
            {
                "args": serialize_value([batch]),
                "kwargs": serialize_value({}),
            }
        )

        result_bytes = await dispatcher.dispatch("/forward_batch").post(body)
        result_payload = orjson.loads(result_bytes)
        merged = deserialize_value(result_payload["result"])

        assert merged == [5, 11, 7, 13]

    @pytest.mark.asyncio
    async def test_dispatch_post_does_not_pad_training_routes(self):
        def _shard_handler(url, data, headers):
            _ = url, headers
            payload = orjson.loads(data)
            args = deserialize_value(payload["args"])
            shard = args[0] if args else []
            return orjson.dumps(
                {
                    "status": "success",
                    "result": serialize_value(
                        [int(item["attention_mask"].sum().item()) for item in shard]
                    ),
                }
            )

        session = _CapturingSession(post_handler=_shard_handler)
        dispatcher = _make_dispatcher(
            dp_size=2, dp_heads=[0, 1], dp_ranks=[0, 1], session=session
        )

        batch = [_make_tensor_item(5), _make_tensor_item(11), _make_tensor_item(7)]
        body = orjson.dumps(
            {
                "args": serialize_value([batch]),
                "kwargs": serialize_value({}),
            }
        )

        with pytest.raises(ValueError, match="divisible by K"):
            await dispatcher.dispatch("/train_batch").post(body)

    @pytest.mark.asyncio
    async def test_dispatch_post_pads_eval_routes_only(self):
        def _shard_handler(url, data, headers):
            _ = url, headers
            payload = orjson.loads(data)
            args = deserialize_value(payload["args"])
            shard = args[0] if args else []
            return orjson.dumps(
                {
                    "status": "success",
                    "result": serialize_value(
                        [int(item["attention_mask"].sum().item()) for item in shard]
                    ),
                }
            )

        session = _CapturingSession(post_handler=_shard_handler)
        dispatcher = _make_dispatcher(
            dp_size=2, dp_heads=[0, 1], dp_ranks=[0, 1], session=session
        )

        batch = [_make_tensor_item(5), _make_tensor_item(11), _make_tensor_item(7)]
        body = orjson.dumps(
            {
                "args": serialize_value([batch]),
                "kwargs": serialize_value({}),
            }
        )

        result_bytes = await dispatcher.dispatch(
            "/eval_batch", pad_eval_batch=True
        ).post(body)
        result_payload = orjson.loads(result_bytes)
        merged = deserialize_value(result_payload["result"])

        assert merged == [5, 11, 7, 0]


class TestScalarFanOut:
    """Tests for _scalar_fan_out: non-partitionable payloads must reach all workers."""

    @pytest.mark.asyncio
    async def test_non_partitionable_tensor_fans_out_to_all_workers(self):
        session = _CapturingSession()
        dispatcher = _make_dispatcher(
            dp_size=1, dp_heads=[0], dp_ranks=[0, 0], session=session
        )

        packed = {"tensor_a": [1, 2, 3]}
        body = orjson.dumps(
            {
                "args": serialize_value([packed]),
                "kwargs": serialize_value({}),
            }
        )

        await dispatcher.dispatch("/forward_batch").post(body)

        captured = session.captured_payloads
        assert len(captured) == 2

        assert captured[0]["args"] == [packed]
        assert captured[0]["kwargs"] == {}

        assert captured[1]["args"] == []
        assert captured[1]["kwargs"] == {}

    @pytest.mark.asyncio
    async def test_returns_first_dp_head_with_non_contiguous_heads(self):
        def _echo_handler(url, data, headers):
            _ = headers
            addr = url.rsplit("/", 1)[0]
            return orjson.dumps({"status": "success", "result": serialize_value(addr)})

        session = _CapturingSession(post_handler=_echo_handler)
        dispatcher = _make_dispatcher(
            dp_size=2,
            dp_heads=[1, 3],
            dp_ranks=[0, 0, 1, 1],
            session=session,
        )

        body = orjson.dumps(
            {
                "args": serialize_value(["scalar_value"]),
                "kwargs": serialize_value({}),
            }
        )

        result_bytes = await dispatcher.dispatch("/train_batch").post(body)
        result_payload = orjson.loads(result_bytes)

        assert result_payload == {
            "status": "success",
            "result": serialize_value("http://worker-1:19001"),
        }

    @pytest.mark.asyncio
    async def test_partitionable_list_still_uses_tensor_dispatch(self):
        def _shard_handler(url, data, headers):
            _ = url, headers
            payload = orjson.loads(data)
            args = deserialize_value(payload["args"])
            shard = args[0] if args else []
            return orjson.dumps(
                {"status": "success", "result": serialize_value(len(shard))}
            )

        session = _CapturingSession(post_handler=_shard_handler)
        dispatcher = _make_dispatcher(
            dp_size=2, dp_heads=[0, 1], dp_ranks=[0, 1], session=session
        )

        batch = [_make_tensor_item(4), _make_tensor_item(8)]
        body = orjson.dumps(
            {
                "args": serialize_value([batch]),
                "kwargs": serialize_value({}),
            }
        )

        result_bytes = await dispatcher.dispatch("/forward_batch").post(body)
        result_payload = orjson.loads(result_bytes)
        merged = deserialize_value(result_payload["result"])

        assert merged == [1, 1]

    @pytest.mark.asyncio
    async def test_single_worker_single_dp_head(self):
        session = _CapturingSession()
        dispatcher = _make_dispatcher(
            dp_size=1, dp_heads=[0], dp_ranks=[0], session=session
        )

        body = orjson.dumps(
            {
                "args": serialize_value(["value"]),
                "kwargs": serialize_value({}),
            }
        )

        await dispatcher.dispatch("/eval_batch").post(body)

        captured = session.captured_payloads
        assert len(captured) == 1
        assert captured[0]["args"] == ["value"]
