# SPDX-License-Identifier: Apache-2.0

"""CPU coverage for the opt-in Theta native SGLang/NCCL protocol."""

from types import SimpleNamespace

import pytest

from areal.api import ModelAllocation, ParamSpec, WeightUpdateMeta
from areal.engine import sglang_remote
from areal.engine.sglang_remote import SGLangBackend
from areal.infra import remote_inf_engine
from areal.infra.remote_inf_engine import RemoteInfEngine


@pytest.fixture
def launched_servers(monkeypatch):
    """Capture the real command builder's output without starting GPU workers."""
    monkeypatch.delenv("AREAL_SGLANG_FORK", raising=False)
    monkeypatch.delenv("AWEX_META_SERVER_ADDR", raising=False)
    calls = []

    def launch(cmd, **kwargs):
        calls.append((cmd, kwargs["env"]))
        return SimpleNamespace(pid=1)

    monkeypatch.setattr(sglang_remote.subprocess, "Popen", launch)
    return calls


def _meta(tp=8, pp=1, dp=2):
    return WeightUpdateMeta(
        type="xccl",
        gen_allocation=ModelAllocation.from_str(f"sglang:d{dp}p{pp}t{tp}"),
        nccl_master_address="127.0.0.1",
        nccl_master_port=29501,
        nccl_group_name="update_weight_group_0",
    )


@pytest.mark.parametrize(
    "fork,awex,entrypoint",
    [
        (None, False, "areal.v2.inference_service.sglang.launch_server"),
        ("theta", False, "sglang.launch_server"),
        ("theta", True, "areal.engine.awex.sglang_plugin"),
    ],
)
def test_launch_selects_native_theta_only_without_awex(
    monkeypatch, launched_servers, fork, awex, entrypoint
):
    """Opting into Theta bypasses the incompatible v2 wrapper, preserving AWEX."""
    if fork:
        monkeypatch.setenv("AREAL_SGLANG_FORK", fork)
    backend = SGLangBackend()
    args = {"model_path": "model", "tp_size": 8, "port": 30000}
    if awex:
        args["awex_meta_server_addr"] = "127.0.0.1:1234"

    backend.launch_server(args)

    cmd, env = launched_servers[0]
    assert cmd == [
        "python3",
        "-m",
        entrypoint,
        "--model-path",
        "model",
        "--tp-size",
        "8",
        "--port",
        "30000",
    ]
    assert backend.get_health_check_request().endpoint == (
        "/model_info" if awex else "/health"
    )
    if fork:
        assert env["AREAL_SGLANG_FORK"] == fork


def test_awex_environment_keeps_plugin_entrypoint(monkeypatch, launched_servers):
    """Existing AWEX environment selection takes precedence over native Theta."""
    monkeypatch.setenv("AREAL_SGLANG_FORK", "theta")
    monkeypatch.setenv("AWEX_META_SERVER_ADDR", "127.0.0.1:1234")

    SGLangBackend().launch_server({"model_path": "model"})

    assert launched_servers[0][0][2] == "areal.engine.awex.sglang_plugin"


@pytest.mark.parametrize(
    "unsupported,match",
    [
        ({"pp_size": 2}, "rollout pp_size=1"),
        ({"dp_size": 2}, "server dp_size=1"),
        ({"speculative_algorithm": "EAGLE"}, "draft model"),
    ],
)
def test_theta_launch_rejects_unsupported_native_update_topology(
    monkeypatch, launched_servers, unsupported, match
):
    """Reject configurations that native update endpoints cannot service safely."""
    monkeypatch.setenv("AREAL_SGLANG_FORK", "theta")

    with pytest.raises(ValueError, match=match):
        SGLangBackend().launch_server({"model_path": "model", **unsupported})

    assert launched_servers == []


def test_theta_init_groups_use_native_schema_and_disjoint_tp_ranks(monkeypatch):
    """Two TP8 replicas join trainer rank 0 without AReaL-only PP fields."""
    monkeypatch.setenv("AREAL_SGLANG_FORK", "theta")
    monkeypatch.setattr(
        sglang_remote, "current_platform", SimpleNamespace(communication_backend="nccl")
    )
    backend = SGLangBackend()
    requests = [
        backend.build_init_weights_group_request("addr", index, _meta())
        for index in range(2)
    ]

    assert requests[0].endpoint == "/init_weights_update_group"
    assert requests[0].payload == {
        "master_address": "127.0.0.1",
        "master_port": 29501,
        "rank_offset": 1,
        "world_size": 17,
        "group_name": "update_weight_group_0",
        "backend": "nccl",
    }
    assert requests[1].payload["rank_offset"] == 9
    assert requests[1].payload["world_size"] == 17


def test_theta_init_rejects_inference_pipeline_parallelism(monkeypatch):
    """Native Theta computes update ranks from TP only and has no pp_rank field."""
    monkeypatch.setenv("AREAL_SGLANG_FORK", "theta")

    with pytest.raises(ValueError, match="rollout pp_size=1"):
        SGLangBackend().build_init_weights_group_request("addr", 0, _meta(pp=2))


def test_theta_pause_update_resume_sends_native_protocol(monkeypatch):
    """Drain requests before each bucketed HF update and resume only afterwards."""
    monkeypatch.setenv("AREAL_SGLANG_FORK", "theta")
    calls = []
    backend = SGLangBackend()
    engine = RemoteInfEngine.__new__(RemoteInfEngine)
    engine.backend = backend
    engine.config = SimpleNamespace(pause_grace_period=0)
    engine._run_request_on_all_servers = lambda req: calls.append(
        (req.endpoint, req.payload)
    )

    async def record_http(**kwargs):
        calls.append((kwargs["endpoint"], kwargs["payload"]))
        return {"success": True, "message": "updated"}

    monkeypatch.setattr(remote_inf_engine, "arequest_with_retry", record_http)
    first_bucket = [
        ParamSpec("model.layers.3.attention.kv_b_proj.weight", (16, 4), "bfloat16")
    ]
    second_bucket = [ParamSpec("lm_head.weight", (32, 8), "bfloat16")]

    pause = getattr(RemoteInfEngine.pause_generation, "__wrapped__")
    resume = getattr(RemoteInfEngine.continue_generation, "__wrapped__")
    pause(engine)
    for bucket in (first_bucket, second_bucket):
        remote_inf_engine._update_weights_from_distributed(
            backend, _meta(), bucket, ["127.0.0.1:30000"], request_timeout=10
        )
    resume(engine)

    assert calls == [
        ("/pause_generation", {}),
        ("/pause_generation", {"mode": "in_place"}),
        (
            "/update_weights_from_distributed",
            {
                "names": [first_bucket[0].name],
                "dtypes": ["bfloat16"],
                "shapes": [(16, 4)],
                "group_name": "update_weight_group_0",
                "abort_all_requests": True,
            },
        ),
        (
            "/update_weights_from_distributed",
            {
                "names": [second_bucket[0].name],
                "dtypes": ["bfloat16"],
                "shapes": [(32, 8)],
                "group_name": "update_weight_group_0",
                "abort_all_requests": True,
            },
        ),
        ("/continue_generation", {}),
    ]
