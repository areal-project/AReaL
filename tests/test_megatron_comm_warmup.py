"""Unit tests for the communicator warmup's platform portability.

The collectives themselves need a distributed environment, but the device
resolution and the platform guards can be exercised on CPU by driving the
unbound method with a stand-in engine.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

from areal.engine.megatron_engine import MegatronEngine


@pytest.fixture
def engine():
    return SimpleNamespace(
        device=torch.device("cuda", 3),
        cpu_group=object(),
        logger=MagicMock(),
    )


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("AREAL_COMM_WARMUP", "1")


def _single_rank_topology():
    mpu = MagicMock()
    mpu.get_pipeline_model_parallel_world_size.return_value = 1
    return mpu


def test_warmup_is_skipped_on_cpu_platforms(engine):
    with (
        patch("areal.engine.megatron_engine.current_platform") as platform,
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "all_reduce") as all_reduce,
        patch.object(dist, "barrier") as barrier,
    ):
        platform.device_type = "cpu"
        MegatronEngine.warmup_communicators(engine)

    all_reduce.assert_not_called()
    barrier.assert_not_called()


def test_probes_are_allocated_on_the_engine_device_without_touching_torch_cuda(
    engine,
):
    with (
        patch("areal.engine.megatron_engine.current_platform") as platform,
        patch("areal.engine.megatron_engine.mpu", _single_rank_topology()),
        patch("torch.distributed.distributed_c10d._world", SimpleNamespace(pg_map={})),
        patch("torch.cuda") as torch_cuda,
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_world_size", return_value=1),
        patch.object(dist, "get_rank", return_value=0),
        patch.object(dist, "barrier"),
        patch("torch.ones") as ones,
    ):
        platform.device_type = "cuda"
        ones.return_value = MagicMock()
        MegatronEngine.warmup_communicators(engine)

    assert ones.call_count >= 2
    for call in ones.call_args_list:
        assert call.kwargs["device"] is engine.device

    torch_cuda.current_device.assert_not_called()
    torch_cuda.synchronize.assert_not_called()
    platform.synchronize.assert_called_once_with()
