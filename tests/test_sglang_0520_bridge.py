# SPDX-License-Identifier: Apache-2.0

"""Focused contract checks for AReaL's SGLang 0.5.20 scheduler bridge."""

import json
from inspect import signature
from types import SimpleNamespace

import zmq
from sglang.srt.managers.io_struct import (
    RpcReqInput,
    RpcReqOutput,
    msgpack_decode,
    msgpack_encode,
    sock_recv,
    sock_send,
)

from areal.v2.inference_service.sglang.pp_bridge import PPSchedulerBridge
from areal.v2.inference_service.sglang.rpc_proxy import RpcProxy
from areal.v2.inference_service.sglang.scheduler import (
    AwexSchedulerBridge,
    areal_run_scheduler_process,
)


def test_awex_rpc_payload_survives_sglang_msgpack_and_keeps_lists(monkeypatch):
    monkeypatch.delenv("AREAL_AWEX_RESULT_IPC", raising=False)
    scheduler = SimpleNamespace(ps=SimpleNamespace(tp_rank=0, dp_rank=0))
    bridge = AwexSchedulerBridge(scheduler)
    received = []
    bridge.awex_release_memory = lambda tags=None: received.append(tags)
    bridge.bind()

    request = RpcReqInput(
        method="awex_release_memory",
        parameters={"payload": json.dumps({"tags": ["kv_cache"]})},
    )
    decoded = msgpack_decode(msgpack_encode(request))
    getattr(scheduler, decoded.method)(**decoded.parameters)

    assert received == [["kv_cache"]]


def test_rpc_proxy_uses_blocking_msgpack_socket_with_list_payload():
    ctx = zmq.Context()
    server = ctx.socket(zmq.PAIR)
    client = ctx.socket(zmq.PAIR)
    server.bind("inproc://awex-rpc-test")
    client.connect("inproc://awex-rpc-test")
    client.setsockopt(zmq.RCVTIMEO, 5000)
    proxy = object.__new__(RpcProxy)
    proxy._rpc_socket = client

    try:
        sock_send(server, RpcReqOutput(success=True, message=""))
        proxy.collective_rpc("awex_release_memory", tags=["kv_cache"])
        request = sock_recv(server, flags=zmq.DONTWAIT)
        assert isinstance(request, RpcReqInput)
        assert request.method == "awex_release_memory"
        assert json.loads(request.parameters["payload"]) == {"tags": ["kv_cache"]}
    finally:
        client.close(linger=0)
        server.close(linger=0)
        ctx.term()


def test_scheduler_process_accepts_sglang_dp_controller_arguments():
    from sglang.srt.managers.scheduler import run_scheduler_process

    assert list(signature(areal_run_scheduler_process).parameters) == list(
        signature(run_scheduler_process).parameters
    )


def test_pp_bridge_skips_only_other_pipeline_stage():
    groups = {}
    calls = []

    def init_group(req):
        calls.append(("init", req.group_name))
        groups[req.group_name] = object()
        return True, "initialized"

    updater = SimpleNamespace(
        _model_update_group=groups,
        update_weights_from_distributed=lambda *args: (True, "updated"),
        destroy_weights_update_group=lambda name: (True, "destroyed"),
    )
    runner = SimpleNamespace(
        ps=SimpleNamespace(pp_rank=1, tp_rank=0), weight_updater=updater
    )
    worker = SimpleNamespace(model_runner=runner, init_weights_update_group=init_group)
    scheduler = SimpleNamespace(tp_worker=worker)
    PPSchedulerBridge(scheduler, SimpleNamespace(pp_size=2)).bind()

    assert worker.init_weights_update_group(
        SimpleNamespace(group_name="update_weight_group_0")
    )[0]
    assert groups["update_weight_group_0"] is None
    assert updater.update_weights_from_distributed([], [], [], "update_weight_group_0")[
        0
    ]
    assert updater.destroy_weights_update_group("update_weight_group_0")[0]
    assert "update_weight_group_0" not in groups
    assert calls == []

    assert worker.init_weights_update_group(
        SimpleNamespace(group_name="update_weight_group_1")
    )[0]
    assert calls == [("init", "update_weight_group_1")]


def test_launch_server_consumes_six_values_and_closes_proxy(monkeypatch):
    from sglang.srt.entrypoints import engine, http_server

    from areal.v2.inference_service.sglang import launch_server, rpc_proxy, scheduler

    calls = []
    monkeypatch.setattr(scheduler, "create_result_ipc", lambda: "ipc://result")
    monkeypatch.setattr(
        engine.Engine,
        "_launch_subprocesses",
        lambda **kwargs: (
            object(),
            object(),
            object(),
            SimpleNamespace(scheduler_infos=[]),
            object(),
            ["weight-cache-daemon"],
        ),
    )
    monkeypatch.setattr(
        engine.Engine,
        "_terminate_weight_cache_daemons",
        lambda procs: calls.append(("terminate", procs)),
    )
    monkeypatch.setattr(
        http_server,
        "_setup_and_run_http_server",
        lambda *args, **kwargs: calls.append(("serve", args[0])),
    )
    monkeypatch.setattr(
        "areal.v2.inference_service.sglang.awex.register_awex_endpoints",
        lambda app, proxy: calls.append(("register", proxy)),
    )

    class FakeProxy:
        def __init__(self, port_args, result_ipc):
            assert result_ipc == "ipc://result"

        def close(self):
            calls.append(("close", None))

    monkeypatch.setattr(rpc_proxy, "RpcProxy", FakeProxy)
    args = object()
    launch_server.areal_launch_server(args)

    assert [name for name, _ in calls] == ["register", "serve", "close", "terminate"]
    assert calls[-1][1] == ["weight-cache-daemon"]


def test_awex_adapter_reads_parallel_ranks_from_sglang_state(monkeypatch):
    import torch.distributed as dist

    from areal.v2.weight_update.awex.sglang_adapter import AwexSGLangAdapter

    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setenv("LOCAL_RANK", "7")
    scheduler = SimpleNamespace(
        ps=SimpleNamespace(
            tp_rank=2,
            pp_rank=1,
            gpu_id=3,
            attn_tp_rank=1,
            attn_tp_size=2,
            attn_dp_rank=1,
        ),
        server_args=SimpleNamespace(tp_size=4, pp_size=2, dp_size=1),
    )
    adapter = object.__new__(AwexSGLangAdapter)
    adapter._scheduler = scheduler

    context = adapter._get_model_context()

    assert context["tp_rank"] == 2
    assert context["pp_rank"] == 1
    assert context["attn_tp_rank"] == 1
    assert context["attn_dp_rank"] == 1
    assert context["local_rank"] == 3
