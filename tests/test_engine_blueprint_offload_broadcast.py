"""Offloaded engines broadcast over CPU only when they expose a CPU mirror group."""

from types import SimpleNamespace

import pytest

from areal.infra.rpc.guard.engine_blueprint import resolve_broadcast_target


def _engine(is_offload: bool, with_cpu_group: bool):
    fields = {
        "is_offload": is_offload,
        "context_and_model_parallel_group": "device-group",
    }
    if with_cpu_group:
        fields["cpu_model_parallel_group"] = "cpu-group"
    return SimpleNamespace(**fields)


def test_offloaded_engine_uses_cpu_mirror_group():
    group, device = resolve_broadcast_target(
        _engine(is_offload=True, with_cpu_group=True), device="cuda:0"
    )
    assert group == "cpu-group"
    assert device == "cpu"


def test_engine_without_cpu_group_keeps_device_broadcast():
    """FSDP tracks is_offload but has no CPU mirror group."""
    group, device = resolve_broadcast_target(
        _engine(is_offload=True, with_cpu_group=False), device="cuda:0"
    )
    assert group == "device-group"
    assert device == "cuda:0"


def test_resident_engine_keeps_device_broadcast():
    group, device = resolve_broadcast_target(
        _engine(is_offload=False, with_cpu_group=True), device="cuda:0"
    )
    assert group == "device-group"
    assert device == "cuda:0"


def test_cpu_staged_method_uses_cpu_mirror_group():
    engine = _engine(is_offload=False, with_cpu_group=True)
    engine.cpu_staged_rpc_methods = frozenset({"train_batch"})

    group, device = resolve_broadcast_target(
        engine, device="cuda:0", method_name="train_batch"
    )

    assert group == "cpu-group"
    assert device == "cpu"


def test_cpu_staged_method_without_cpu_group_fails_fast():
    engine = _engine(is_offload=False, with_cpu_group=False)
    engine.cpu_staged_rpc_methods = frozenset({"train_batch"})

    with pytest.raises(RuntimeError, match="cpu_model_parallel_group is None"):
        resolve_broadcast_target(engine, device="cuda:0", method_name="train_batch")
