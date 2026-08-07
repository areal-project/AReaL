"""Unit tests for the communicator warmup's device and group selection.

The collectives themselves need a distributed environment, but the platform
guards and the choice of target group can be exercised on CPU by driving the
unbound method with a stand-in engine.
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

from areal.engine.megatron_engine import MegatronEngine

_WORLD = object()


@pytest.fixture
def engine():
    return SimpleNamespace(
        device=torch.device("cuda", 3),
        cpu_group=object(),
        logger=MagicMock(),
        config=SimpleNamespace(warmup_communicators=True),
    )


@pytest.fixture(autouse=True)
def no_local_rank(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)


def _topology(pp_size, pp_group, ep_size=1):
    mpu = MagicMock()
    mpu.get_expert_model_parallel_world_size.return_value = ep_size
    mpu.get_pipeline_model_parallel_world_size.return_value = pp_size
    mpu.get_pipeline_model_parallel_group.return_value = pp_group
    mpu.get_pipeline_model_parallel_rank.return_value = 1
    mpu.get_pipeline_model_parallel_prev_rank.return_value = 0
    mpu.get_pipeline_model_parallel_next_rank.return_value = 2
    mpu.is_pipeline_first_stage.return_value = False
    mpu.is_pipeline_last_stage.return_value = False
    return mpu


class _P2PRecorder:
    def __init__(self):
        self.groups = []

    def send_like(self, tensor, peer, group=None, *args, **kwargs):
        self.groups.append(group)
        return MagicMock()

    def p2p_op(self, op, tensor, peer, group=None, *args, **kwargs):
        self.groups.append(group)
        return MagicMock()


def _platform_patches(stack, device_type):
    for target in (
        "areal.engine.megatron_engine.current_platform",
        "areal.infra.platforms.current_platform",
    ):
        platform = stack.enter_context(patch(target))
        platform.device_type = device_type
        platform.current_device.return_value = 3


def _run_warmup(engine, pp_size, pp_group, backend="nccl", ep_size=1):
    recorder = _P2PRecorder()
    patches = [
        patch(
            "areal.engine.megatron_engine.mpu", _topology(pp_size, pp_group, ep_size)
        ),
        patch(
            "torch.distributed.distributed_c10d._world",
            SimpleNamespace(pg_map={}, default_pg=_WORLD),
        ),
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_backend", return_value=backend),
        patch.object(dist, "get_world_size", return_value=pp_size),
        patch.object(dist, "get_rank", return_value=1),
        patch.object(dist, "barrier"),
        patch.object(dist, "all_reduce"),
        patch.object(dist, "all_to_all_single"),
        patch.object(dist, "reduce_scatter_tensor"),
        patch.object(dist, "all_gather_into_tensor"),
        patch.object(dist, "batch_isend_irecv", return_value=[]),
        patch.object(dist, "send", side_effect=recorder.send_like),
        patch.object(dist, "recv", side_effect=recorder.send_like),
        patch.object(dist, "isend", side_effect=recorder.send_like),
        patch.object(dist, "irecv", side_effect=recorder.send_like),
        patch.object(dist, "P2POp", side_effect=recorder.p2p_op),
        patch("torch.zeros", return_value=MagicMock()),
        patch("torch.empty", return_value=MagicMock()),
        patch("torch.empty_like", return_value=MagicMock()),
    ]
    with ExitStack() as stack:
        _platform_patches(stack, "cuda")
        for p in patches:
            stack.enter_context(p)
        MegatronEngine.warmup_communicators(engine)
    return recorder


def test_warmup_is_skipped_when_not_enabled(engine):
    engine.config.warmup_communicators = False
    with ExitStack() as stack:
        _platform_patches(stack, "cuda")
        stack.enter_context(patch.object(dist, "is_initialized", return_value=True))
        all_reduce = stack.enter_context(patch.object(dist, "all_reduce"))
        MegatronEngine.warmup_communicators(engine)

    all_reduce.assert_not_called()


def test_warmup_is_skipped_on_cpu_platforms(engine):
    with ExitStack() as stack:
        _platform_patches(stack, "cpu")
        stack.enter_context(patch.object(dist, "is_initialized", return_value=True))
        all_reduce = stack.enter_context(patch.object(dist, "all_reduce"))
        isend = stack.enter_context(patch.object(dist, "isend"))
        MegatronEngine.warmup_communicators(engine)

    all_reduce.assert_not_called()
    isend.assert_not_called()


def test_pipeline_warmup_targets_the_pipeline_group(engine):
    """Megatron routes unbatched pipeline p2p through the pipeline group.

    Warming the default group instead leaves the communicator that the train
    step actually uses cold, which is the allocation this warmup exists to
    front-load.
    """
    pp_group = object()

    recorder = _run_warmup(engine, pp_size=4, pp_group=pp_group)

    assert recorder.groups
    assert set(recorder.groups) == {pp_group}


def test_pipeline_warmup_covers_the_world_group_when_pp_size_is_two(engine):
    """At pp_size == 2 Megatron sends one direction over the default group."""
    pp_group = object()

    recorder = _run_warmup(engine, pp_size=2, pp_group=pp_group)

    assert set(recorder.groups) == {pp_group, _WORLD}


def test_pipeline_warmup_stays_on_the_pipeline_group_for_ucc(engine):
    pp_group = object()

    recorder = _run_warmup(engine, pp_size=2, pp_group=pp_group, backend="ucc")

    assert set(recorder.groups) == {pp_group}


def test_no_pipeline_warmup_without_pipeline_parallelism(engine):
    recorder = _run_warmup(engine, pp_size=1, pp_group=object())

    assert recorder.groups == []


def test_all_to_all_is_warmed_only_when_experts_are_sharded(engine):
    """all_to_all buffers persist, so a group that never dispatches pays for nothing."""
    with ExitStack() as stack:
        _platform_patches(stack, "cuda")
        stack.enter_context(
            patch("areal.engine.megatron_engine.mpu", _topology(1, object(), ep_size=1))
        )
        stack.enter_context(
            patch(
                "torch.distributed.distributed_c10d._world",
                SimpleNamespace(pg_map={}, default_pg=_WORLD),
            )
        )
        stack.enter_context(patch.object(dist, "is_initialized", return_value=True))
        stack.enter_context(patch.object(dist, "get_world_size", return_value=4))
        stack.enter_context(patch.object(dist, "barrier"))
        stack.enter_context(patch.object(dist, "reduce_scatter_tensor"))
        stack.enter_context(patch.object(dist, "all_gather_into_tensor"))
        stack.enter_context(patch("torch.zeros", return_value=MagicMock()))
        stack.enter_context(patch("torch.empty", return_value=MagicMock()))
        stack.enter_context(patch("torch.empty_like", return_value=MagicMock()))
        all_to_all = stack.enter_context(patch.object(dist, "all_to_all_single"))
        MegatronEngine.warmup_communicators(engine)

    all_to_all.assert_not_called()
