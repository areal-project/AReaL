# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from areal.engine import sglang_remote


@pytest.fixture(autouse=True)
def _cuda_device_namespace(monkeypatch):
    monkeypatch.setattr(
        sglang_remote.current_platform, "device_control_env_var", "CUDA_VISIBLE_DEVICES"
    )


@pytest.mark.parametrize("nnodes", [1, 8])
def test_colocated_launch_registers_unique_ranks_with_reordered_workers(
    monkeypatch, nnodes
):
    """Slurm placement overrides stale controller offsets for every TP rank."""
    from areal.engine.awex.sglang_plugin import _resolve_transfer_rank

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    for name in ("AREAL_SGLANG_BASE_GPU_ID", "AWEX_TRANSFER_RANK", "RANK"):
        monkeypatch.delenv(name, raising=False)
    commands = []
    children = []
    monkeypatch.setattr(
        sglang_remote.SGLangConfig,
        "build_cmd_from_args",
        lambda args: commands.append(args.copy()) or ["sglang-server"],
    )
    monkeypatch.setattr(
        sglang_remote.subprocess,
        "Popen",
        lambda *args, **kwargs: children.append(kwargs["env"]) or MagicMock(),
    )
    ranks = []
    for node in range(nnodes):
        for local_id in range(2):
            monkeypatch.setenv("SLURM_LOCALID", str(local_id))
            sglang_remote.SGLangBackend().launch_server(
                {
                    "port": 19000,
                    "nccl_port": 19001,
                    "base_gpu_id": (1 - local_id) * 4,
                    "_awex_gpus_per_server": 4,
                }
            )
            base = commands[-1]["base_gpu_id"]
            monkeypatch.setenv(
                "AWEX_TRANSFER_RANK_BASE", children[-1]["AWEX_TRANSFER_RANK_BASE"]
            )
            ranks.extend(
                _resolve_transfer_rank(
                    infer_world_size=8 * nnodes,
                    gpu_id=base + tp_rank,
                    node_id=node,
                    nnodes=nnodes,
                    instance_world_size=4,
                    server_base_gpu_id=base,
                )
                for tp_rank in range(4)
            )

    assert ranks == list(range(8 * nnodes))


@pytest.mark.parametrize("nnodes", [1, 8])
@pytest.mark.parametrize("explicit_base", [None, "0"])
def test_colocated_isolated_replicas_keep_unique_transfer_ranks(
    monkeypatch, nnodes, explicit_base
):
    """Logical GPU zero must preserve each isolated replica's dense offset."""
    from areal.engine.awex.sglang_plugin import _resolve_transfer_rank

    for name in ("AWEX_TRANSFER_RANK", "RANK", "AREAL_SGLANG_BASE_GPU_ID"):
        monkeypatch.delenv(name, raising=False)
    if explicit_base is not None:
        monkeypatch.setenv("AREAL_SGLANG_BASE_GPU_ID", explicit_base)
    build_cmd = MagicMock(return_value=["sglang-server"])
    launch = MagicMock()
    monkeypatch.setattr(sglang_remote.SGLangConfig, "build_cmd_from_args", build_cmd)
    monkeypatch.setattr(sglang_remote.subprocess, "Popen", launch)
    ranks = []

    for node in range(nnodes):
        for replica in range(2):
            transfer_base = replica * 4
            monkeypatch.setenv(
                "CUDA_VISIBLE_DEVICES",
                ",".join(str(transfer_base + rank) for rank in range(4)),
            )
            monkeypatch.setenv("SLURM_LOCALID", str(1 - replica))
            sglang_remote.SGLangBackend().launch_server(
                {
                    "port": 19000,
                    "nccl_port": 19001,
                    "base_gpu_id": transfer_base,
                    "_awex_gpus_per_server": 4,
                }
            )
            assert build_cmd.call_args.args[0]["base_gpu_id"] == 0
            child_env = launch.call_args.kwargs["env"]
            assert child_env["AWEX_TRANSFER_RANK_BASE"] == str(transfer_base)
            monkeypatch.setenv(
                "AWEX_TRANSFER_RANK_BASE", child_env["AWEX_TRANSFER_RANK_BASE"]
            )
            ranks.extend(
                _resolve_transfer_rank(
                    infer_world_size=8 * nnodes,
                    gpu_id=tp_rank,
                    node_id=node,
                    nnodes=nnodes,
                    instance_world_size=4,
                    server_base_gpu_id=0,
                )
                for tp_rank in range(4)
            )

    assert ranks == list(range(8 * nnodes))


@pytest.mark.parametrize("visible_devices", ["2,5", "0,1,2,3,4,5,6,7"])
def test_colocated_launch_preserves_explicit_dense_transfer_base(
    monkeypatch, visible_devices
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
    monkeypatch.setenv("SLURM_LOCALID", "1")
    monkeypatch.delenv("AREAL_SGLANG_BASE_GPU_ID", raising=False)
    build_cmd = MagicMock(return_value=["sglang-server"])
    launch = MagicMock()
    monkeypatch.setattr(sglang_remote.SGLangConfig, "build_cmd_from_args", build_cmd)
    monkeypatch.setattr(sglang_remote.subprocess, "Popen", launch)

    sglang_remote.SGLangBackend().launch_server(
        {
            "port": 19000,
            "nccl_port": 19001,
            "base_gpu_id": 0,
            "_awex_gpus_per_server": 2,
            "_awex_transfer_rank_base": 4,
        }
    )

    assert launch.call_args.kwargs["env"]["AWEX_TRANSFER_RANK_BASE"] == "4"
    assert "_awex_transfer_rank_base" not in build_cmd.call_args.args[0]
