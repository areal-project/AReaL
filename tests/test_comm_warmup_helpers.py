"""Unit tests for the transport warmup helpers.

Only metadata and tensor shapes are asserted here; the collectives are
stubbed, so these run without a distributed environment or a GPU.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

from areal.engine.core.distributed import (
    nccl_process_groups,
    warmup_all_to_all_transports,
    warmup_collective_transports,
    warmup_p2p_transports,
    warmup_sharded_transports,
)


class _FakeGroup:
    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return f"<FakeGroup {self._name}>"


@pytest.fixture(autouse=True)
def no_local_rank(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)


@contextmanager
def _registry(*entries: tuple[_FakeGroup, str, int]):
    backends = {group: backend for group, backend, _ in entries}
    sizes = {group: size for group, _, size in entries}
    world = SimpleNamespace(pg_map={group: (backends[group], None) for group in sizes})
    with ExitStack() as stack:
        stack.enter_context(patch("torch.distributed.distributed_c10d._world", world))
        stack.enter_context(patch.object(dist, "is_initialized", return_value=True))
        stack.enter_context(
            patch.object(dist, "get_backend", side_effect=lambda group: backends[group])
        )
        stack.enter_context(
            patch.object(dist, "get_world_size", side_effect=lambda group: sizes[group])
        )
        yield


@contextmanager
def _cuda_platform(device_type="cuda", world_size=4):
    with ExitStack() as stack:
        platform = stack.enter_context(patch("areal.infra.platforms.current_platform"))
        platform.device_type = device_type
        platform.current_device.return_value = 0
        stack.enter_context(patch.object(dist, "is_initialized", return_value=True))
        stack.enter_context(
            patch.object(dist, "get_world_size", return_value=world_size)
        )
        yield stack


def test_enumeration_is_empty_before_init():
    with patch.object(dist, "is_initialized", return_value=False):
        assert nccl_process_groups() == []


def test_enumeration_skips_gloo_and_solo_groups():
    nccl = _FakeGroup("nccl")
    gloo = _FakeGroup("gloo")
    solo = _FakeGroup("solo")
    with _registry((nccl, "nccl", 8), (gloo, "gloo", 8), (solo, "nccl", 1)):
        assert nccl_process_groups() == [nccl]


def test_enumeration_preserves_creation_order():
    first = _FakeGroup("first")
    second = _FakeGroup("second")
    with _registry((first, "nccl", 4), (second, "nccl", 4)):
        assert nccl_process_groups() == [first, second]


def test_collective_warmup_is_a_noop_on_cpu():
    with _cuda_platform(device_type="cpu") as stack:
        all_reduce = stack.enter_context(patch.object(dist, "all_reduce"))
        warmup_collective_transports(_FakeGroup("dp"))
    all_reduce.assert_not_called()


def test_collective_warmup_dedupes_and_covers_every_probe_size():
    group = _FakeGroup("dp")
    with _cuda_platform() as stack:
        stack.enter_context(patch("torch.zeros", return_value=MagicMock()))
        all_reduce = stack.enter_context(patch.object(dist, "all_reduce"))
        warmup_collective_transports(group, None, group)

    assert all_reduce.call_count == 2
    assert {c.kwargs["group"] for c in all_reduce.call_args_list} == {group}


def test_all_to_all_warmup_truncates_to_a_multiple_of_the_world_size():
    group = _FakeGroup("ep")
    sizes = []
    with _cuda_platform(world_size=3) as stack:
        stack.enter_context(
            patch(
                "torch.zeros",
                side_effect=lambda n, **kw: sizes.append(n) or MagicMock(),
            )
        )
        stack.enter_context(patch("torch.empty_like", return_value=MagicMock()))
        stack.enter_context(patch.object(dist, "all_to_all_single"))
        warmup_all_to_all_transports(group)

    assert sizes
    assert all(size % 3 == 0 for size in sizes)


def test_all_to_all_warmup_skips_probes_smaller_than_the_world_size():
    group = _FakeGroup("ep")
    with _cuda_platform(world_size=4096) as stack:
        stack.enter_context(patch("torch.zeros", return_value=MagicMock()))
        stack.enter_context(patch("torch.empty_like", return_value=MagicMock()))
        all_to_all = stack.enter_context(patch.object(dist, "all_to_all_single"))
        warmup_all_to_all_transports(group, numels=(1024,))

    all_to_all.assert_not_called()


def test_sharded_warmup_is_a_noop_without_a_group():
    with patch.object(dist, "reduce_scatter_tensor") as reduce_scatter:
        warmup_sharded_transports(None)
    reduce_scatter.assert_not_called()


def test_sharded_warmup_is_a_noop_for_a_single_rank_group():
    with _cuda_platform(world_size=1) as stack:
        reduce_scatter = stack.enter_context(
            patch.object(dist, "reduce_scatter_tensor")
        )
        warmup_sharded_transports(_FakeGroup("dp"))
    reduce_scatter.assert_not_called()


def test_sharded_warmup_shapes_the_shard_to_the_world_size():
    shapes = []
    with _cuda_platform(world_size=4) as stack:
        stack.enter_context(
            patch(
                "torch.zeros",
                side_effect=lambda n, **kw: shapes.append(n) or MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "torch.empty",
                side_effect=lambda n, **kw: shapes.append(n) or MagicMock(),
            )
        )
        stack.enter_context(patch.object(dist, "reduce_scatter_tensor"))
        stack.enter_context(patch.object(dist, "all_gather_into_tensor"))
        warmup_sharded_transports(_FakeGroup("dp"), numel=1024)

    assert shapes == [1024, 256]


def test_p2p_warmup_is_a_noop_for_a_single_stage_pipeline():
    with _cuda_platform() as stack:
        isend = stack.enter_context(patch.object(dist, "isend"))
        warmup_p2p_transports(
            _FakeGroup("pp"),
            prev_rank=0,
            next_rank=0,
            has_prev=False,
            has_next=False,
        )
    isend.assert_not_called()


def test_p2p_warmup_passes_the_group_explicitly():
    group = _FakeGroup("pp")
    with _cuda_platform() as stack:
        stack.enter_context(patch("torch.zeros", return_value=MagicMock()))
        stack.enter_context(patch("torch.empty", return_value=MagicMock()))
        stack.enter_context(patch.object(dist, "batch_isend_irecv", return_value=[]))
        stack.enter_context(patch.object(dist, "irecv", return_value=MagicMock()))
        stack.enter_context(patch.object(dist, "P2POp", return_value=MagicMock()))
        isend = stack.enter_context(
            patch.object(dist, "isend", return_value=MagicMock())
        )
        warmup_p2p_transports(
            group,
            prev_rank=0,
            next_rank=2,
            has_prev=True,
            has_next=True,
        )

    assert isend.call_count
    assert {c.kwargs["group"] for c in isend.call_args_list} == {group}


def test_torch_device_is_built_from_the_platform_device_type():
    group = _FakeGroup("dp")
    with _cuda_platform() as stack:
        zeros = stack.enter_context(patch("torch.zeros", return_value=MagicMock()))
        stack.enter_context(patch.object(dist, "all_reduce"))
        warmup_collective_transports(group, numels=(8,))

    assert zeros.call_args.kwargs["device"] == torch.device("cuda", 0)
