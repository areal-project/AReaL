# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from areal.engine.awex.colocate_writer import (
    AwexMegatronAdapter as V1AwexMegatronAdapter,
)
from areal.engine.megatron_utils.gpu_staged_optimizer import (
    GPUStagedAdamW,
    GPUStagedAdamWConfig,
)
from areal.engine.megatron_utils.gpu_staged_optimizer_checkpoint import (
    prepare_managed_checkpoint_save,
)
from areal.engine.megatron_utils.optimizer_chain import (
    iter_megatron_optimizer_leaves,
)
from areal.v2.weight_update.awex.megatron_adapter import (
    AwexMegatronAdapter as V2AwexMegatronAdapter,
)

CUDA_AVAILABLE = torch.cuda.is_available()
AWEX_ADAPTERS = [V1AwexMegatronAdapter, V2AwexMegatronAdapter]


def _make_adapter(adapter_cls, optimizer):
    wrapper = SimpleNamespace(optimizer=optimizer)
    chained = SimpleNamespace(chained_optimizers=[wrapper])
    engine = SimpleNamespace(optimizer=chained, device=torch.device("cuda"), model=None)
    return adapter_cls(engine)


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_managed_release_resume_preserves_cpu_slab_views(
    adapter_cls, monkeypatch
) -> None:
    """Both AWEX adapters drain managed state and never replace its CPU views."""
    param = torch.nn.Parameter(
        torch.linspace(-1, 1, 23, device="cuda", dtype=torch.bfloat16)
    )
    optimizer = GPUStagedAdamW(
        [param],
        lr=2e-3,
        staged_config=GPUStagedAdamWConfig(
            buffer_count=2, bucket_size_mb=7 * 4 / (1024 * 1024)
        ),
    )
    optimizer.bind_owned_params(optimizer.param_groups)
    param.decoupled_grad = torch.linspace(
        1, -1, param.numel(), device="cuda", dtype=torch.bfloat16
    )
    optimizer.step()
    adapter = _make_adapter(adapter_cls, optimizer)
    slabs = optimizer.cpu_slabs
    assert slabs is not None
    state = optimizer.state[param]
    original_objects = dict(state)
    original_storage = {
        "master_param": slabs.master.untyped_storage().data_ptr(),
        "exp_avg": slabs.exp_avg.untyped_storage().data_ptr(),
        "exp_avg_sq": slabs.exp_avg_sq.untyped_storage().data_ptr(),
    }
    drain_calls = 0
    original_drain = optimizer.drain

    def tracked_drain() -> None:
        nonlocal drain_calls
        drain_calls += 1
        original_drain()

    def forbidden_lifecycle() -> None:
        raise AssertionError("AWEX must not migrate managed optimizer state")

    monkeypatch.setattr(optimizer, "drain", tracked_drain)
    monkeypatch.setattr(optimizer, "offload_to_cpu", forbidden_lifecycle)
    monkeypatch.setattr(optimizer, "restore_from_cpu", forbidden_lifecycle)
    monkeypatch.setenv("AWEX_OPT_OFFLOAD_VIA_HDO", "1")

    adapter.release_memory(tags=["optimizer"])
    adapter.release_memory(tags=["optimizer"])
    assert drain_calls == 1
    original_values = {key: tensor.clone() for key, tensor in state.items()}
    prepare_managed_checkpoint_save(
        SimpleNamespace(optimizer=optimizer), async_save=False
    )
    assert drain_calls == 2
    adapter.resume_memory(tags=["optimizer"])
    adapter.resume_memory(tags=["optimizer"])
    assert drain_calls == 2

    assert optimizer.residency == "CPU_RESIDENT"
    for key, tensor in state.items():
        assert tensor is original_objects[key]
        assert tensor.device.type == "cpu"
        assert tensor.is_pinned()
        assert tensor.untyped_storage().data_ptr() == original_storage[key]
        torch.testing.assert_close(tensor, original_values[key], rtol=0.0, atol=0.0)
    assert (
        sum(
            tensor.numel()
            for param_state in optimizer.state.values()
            for tensor in param_state.values()
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda
        )
        == 0
    )


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_mixed_chain_dispatches_managed_and_ordinary_optimizers(
    adapter_cls,
) -> None:
    """Managed and ordinary wrappers retain their distinct lifecycle behavior."""
    managed_param = torch.nn.Parameter(
        torch.tensor([1.0, -1.0], device="cuda", dtype=torch.bfloat16)
    )
    managed = GPUStagedAdamW([managed_param], lr=1e-2)
    managed.bind_owned_params(managed.param_groups)
    managed_state = managed.state[managed_param]
    managed_objects = dict(managed_state)
    managed_storage = {
        key: tensor.untyped_storage().data_ptr()
        for key, tensor in managed_state.items()
    }

    ordinary_param = torch.nn.Parameter(torch.tensor([2.0, -3.0], device="cuda"))
    ordinary = torch.optim.AdamW([ordinary_param], lr=1e-2)
    ordinary_param.grad = torch.tensor([0.5, -0.25], device="cuda")
    ordinary.step()
    expected_ordinary = {
        key: value.detach().clone()
        for key, value in ordinary.state[ordinary_param].items()
        if key in ("exp_avg", "exp_avg_sq")
    }

    chained = SimpleNamespace(
        chained_optimizers=[
            SimpleNamespace(optimizer=managed),
            SimpleNamespace(optimizer=ordinary),
        ]
    )
    engine = SimpleNamespace(optimizer=chained, device=torch.device("cuda"), model=None)
    adapter = adapter_cls(engine)

    adapter.release_memory(tags=["optimizer"])
    assert managed.residency == "CPU_RESIDENT"
    assert all(
        ordinary.state[ordinary_param][key].device.type == "cpu"
        for key in expected_ordinary
    )
    adapter.resume_memory(tags=["optimizer"])

    for key, tensor in managed_state.items():
        assert tensor is managed_objects[key]
        assert tensor.is_pinned()
        assert tensor.untyped_storage().data_ptr() == managed_storage[key]
    for key, expected_value in expected_ordinary.items():
        actual = ordinary.state[ordinary_param][key]
        assert actual.is_cuda
        torch.testing.assert_close(actual, expected_value, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_non_managed_release_resume_behavior_is_unchanged(adapter_cls) -> None:
    """Ordinary AdamW moments still move CPU on release and CUDA on resume."""
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=1e-2)
    param.grad = torch.tensor([0.25, -0.5], device="cuda")
    optimizer.step()
    expected = {
        key: value.detach().clone()
        for key, value in optimizer.state[param].items()
        if key in ("exp_avg", "exp_avg_sq")
    }
    adapter = _make_adapter(adapter_cls, optimizer)

    adapter.release_memory(tags=["optimizer"])
    for key in expected:
        assert optimizer.state[param][key].device.type == "cpu"

    adapter.resume_memory(tags=["optimizer"])
    for key, expected_value in expected.items():
        actual = optimizer.state[param][key]
        assert actual.is_cuda
        torch.testing.assert_close(actual, expected_value, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_nested_chain_dispatches_managed_optimizer(
    adapter_cls, monkeypatch
) -> None:
    """Nested MCore chains must still find and drain each managed leaf once."""
    param = torch.nn.Parameter(
        torch.tensor([1.0, -1.0], device="cuda", dtype=torch.bfloat16)
    )
    optimizer = GPUStagedAdamW([param], lr=1e-2)
    optimizer.bind_owned_params(optimizer.param_groups)
    original_drain = optimizer.drain
    drain_calls = 0

    def tracked_drain() -> None:
        nonlocal drain_calls
        drain_calls += 1
        original_drain()

    monkeypatch.setattr(optimizer, "drain", tracked_drain)
    leaf = SimpleNamespace(optimizer=optimizer)
    nested = SimpleNamespace(chained_optimizers=[leaf])
    top = SimpleNamespace(chained_optimizers=[nested])
    engine = SimpleNamespace(optimizer=top, device=torch.device("cuda"), model=None)
    adapter = adapter_cls(engine)

    adapter.release_memory(tags=["optimizer"])
    adapter.resume_memory(tags=["optimizer"])

    assert drain_calls == 1


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_hdo_opt_in_preserves_wrapper_lifecycle(adapter_cls, monkeypatch) -> None:
    """HDO opt-in must keep delegating release/resume to the wrapper methods."""
    param = torch.nn.Parameter(torch.tensor([1.0, -1.0], device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=1e-2)
    param.grad = torch.tensor([0.5, -0.25], device="cuda")
    optimizer.step()
    lifecycle_calls: list[str] = []
    wrapper = SimpleNamespace(
        optimizer=optimizer,
        offload_to_cpu=lambda: lifecycle_calls.append("offload"),
        restore_from_cpu=lambda: lifecycle_calls.append("restore"),
    )
    chained = SimpleNamespace(chained_optimizers=[wrapper])
    engine = SimpleNamespace(optimizer=chained, device=torch.device("cuda"), model=None)
    adapter = adapter_cls(engine)
    monkeypatch.setenv("AWEX_OPT_OFFLOAD_VIA_HDO", "1")

    adapter.release_memory(tags=["optimizer"])
    adapter.resume_memory(tags=["optimizer"])

    assert lifecycle_calls == ["offload", "restore"]


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_empty_optimizer_release_resume_is_idempotent(adapter_cls) -> None:
    """Missing optimizers must remain a repeatable lifecycle no-op."""
    engine = SimpleNamespace(optimizer=None, device=torch.device("cuda"), model=None)
    adapter = adapter_cls(engine)

    adapter.release_memory(tags=["optimizer"])
    adapter.release_memory(tags=["optimizer"])
    adapter.resume_memory(tags=["optimizer"])
    adapter.resume_memory(tags=["optimizer"])


def test_optimizer_leaf_iterator_is_stable_and_deduplicates_by_identity() -> None:
    """Nested leaves are yielded DFS-left-to-right and at most once."""
    leaf_a = object()
    leaf_b = object()
    leaf_c = object()
    nested = SimpleNamespace(chained_optimizers=[leaf_b, leaf_a])
    root = SimpleNamespace(chained_optimizers=[leaf_a, nested, leaf_c, nested, leaf_b])

    assert list(iter_megatron_optimizer_leaves(root)) == [leaf_a, leaf_b, leaf_c]


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_nested_mixed_chain_dispatches_each_leaf_once(
    adapter_cls, monkeypatch
) -> None:
    """Nested managed, ordinary and HDO leaves dispatch once in DFS order."""
    calls: list[str] = []
    managed_param = torch.nn.Parameter(
        torch.tensor([1.0, -1.0], device="cuda", dtype=torch.bfloat16)
    )
    managed = GPUStagedAdamW([managed_param], lr=1e-2)
    managed.bind_owned_params(managed.param_groups)
    managed_state = managed.state[managed_param]
    managed_storage = {
        key: tensor.untyped_storage().data_ptr()
        for key, tensor in managed_state.items()
    }
    original_drain = managed.drain

    def tracked_drain() -> None:
        calls.append("managed.drain")
        original_drain()

    monkeypatch.setattr(managed, "drain", tracked_drain)
    managed_leaf = SimpleNamespace(
        optimizer=managed,
        offload_to_cpu=lambda: calls.append("managed.offload"),
        restore_from_cpu=lambda: calls.append("managed.restore"),
    )

    ordinary_param = torch.nn.Parameter(torch.tensor([2.0, -3.0], device="cuda"))
    ordinary = torch.optim.AdamW([ordinary_param], lr=1e-2)
    ordinary_param.grad = torch.tensor([0.5, -0.25], device="cuda")
    ordinary.step()
    ordinary_expected = {
        key: value.detach().clone()
        for key, value in ordinary.state[ordinary_param].items()
        if key in ("exp_avg", "exp_avg_sq")
    }
    ordinary_leaf = SimpleNamespace(optimizer=ordinary)

    hdo_leaf = SimpleNamespace(
        state={},
        offload_to_cpu=lambda: calls.append("hdo.offload"),
        restore_from_cpu=lambda: calls.append("hdo.restore"),
    )
    middle = SimpleNamespace(chained_optimizers=[ordinary_leaf, managed_leaf, hdo_leaf])
    root = SimpleNamespace(
        chained_optimizers=[managed_leaf, middle, hdo_leaf, ordinary_leaf]
    )
    engine = SimpleNamespace(optimizer=root, device=torch.device("cuda"), model=None)
    adapter = adapter_cls(engine)
    monkeypatch.setenv("AWEX_OPT_OFFLOAD_VIA_HDO", "1")

    adapter.release_memory(tags=["optimizer"])
    assert calls == ["managed.drain", "hdo.offload"]
    assert all(
        ordinary.state[ordinary_param][key].device.type == "cpu"
        for key in ordinary_expected
    )
    adapter.release_memory(tags=["optimizer"])
    assert calls == ["managed.drain", "hdo.offload"]

    adapter.resume_memory(tags=["optimizer"])
    assert calls == ["managed.drain", "hdo.offload", "hdo.restore"]
    adapter.resume_memory(tags=["optimizer"])
    assert calls == ["managed.drain", "hdo.offload", "hdo.restore"]

    for key, tensor in managed_state.items():
        assert tensor.device.type == "cpu"
        assert tensor.is_pinned()
        assert tensor.untyped_storage().data_ptr() == managed_storage[key]
    assert not any(
        tensor.is_cuda
        for state in managed.state.values()
        for tensor in state.values()
        if isinstance(tensor, torch.Tensor)
    )
    for key, expected in ordinary_expected.items():
        actual = ordinary.state[ordinary_param][key]
        assert actual.is_cuda
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_hdo_methods_are_ignored_without_opt_in(adapter_cls, monkeypatch) -> None:
    """Without opt-in, an HDO-capable wrapper follows ordinary state migration."""
    param = torch.nn.Parameter(torch.tensor([1.0, -1.0], device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=1e-2)
    param.grad = torch.tensor([0.5, -0.25], device="cuda")
    optimizer.step()
    expected = {
        key: value.detach().clone()
        for key, value in optimizer.state[param].items()
        if key in ("exp_avg", "exp_avg_sq")
    }
    lifecycle_calls: list[str] = []
    leaf = SimpleNamespace(
        optimizer=optimizer,
        offload_to_cpu=lambda: lifecycle_calls.append("offload"),
        restore_from_cpu=lambda: lifecycle_calls.append("restore"),
    )
    engine = SimpleNamespace(optimizer=leaf, device=torch.device("cuda"), model=None)
    adapter = adapter_cls(engine)
    monkeypatch.delenv("AWEX_OPT_OFFLOAD_VIA_HDO", raising=False)

    adapter.release_memory(tags=["optimizer"])
    assert lifecycle_calls == []
    assert all(optimizer.state[param][key].device.type == "cpu" for key in expected)
    adapter.resume_memory(tags=["optimizer"])
    assert lifecycle_calls == []
    for key, expected_value in expected.items():
        actual = optimizer.state[param][key]
        assert actual.is_cuda
        torch.testing.assert_close(actual, expected_value, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.parametrize("cycle_kind", ["self", "multi"])
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_optimizer_chain_cycle_logs_and_processes_reachable_leaf_once(
    adapter_cls, cycle_kind, monkeypatch
) -> None:
    """Self and multi-node cycles terminate while reachable leaves run once."""
    calls: list[str] = []
    leaf = SimpleNamespace(
        state={},
        offload_to_cpu=lambda: calls.append("offload"),
        restore_from_cpu=lambda: calls.append("restore"),
    )
    first = SimpleNamespace(chained_optimizers=[])
    if cycle_kind == "self":
        first.chained_optimizers.extend([first, leaf, leaf])
    else:
        second = SimpleNamespace(chained_optimizers=[first, leaf])
        first.chained_optimizers.extend([second, leaf])
    engine = SimpleNamespace(optimizer=first, device=torch.device("cuda"), model=None)
    adapter = adapter_cls(engine)
    monkeypatch.setenv("AWEX_OPT_OFFLOAD_VIA_HDO", "1")
    warnings: list[str] = []

    class _RecordingLogger:
        @staticmethod
        def info(message, *args) -> None:
            del message, args

        @staticmethod
        def warning(message, *args) -> None:
            warnings.append(message % args)

    monkeypatch.setattr(sys.modules[adapter_cls.__module__], "logger", _RecordingLogger)

    adapter.release_memory(tags=["optimizer"])
    adapter.resume_memory(tags=["optimizer"])

    assert calls == ["offload", "restore"]
    assert sum("optimizer chain cycle" in message for message in warnings) == 2


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_empty_chain_release_resume_is_idempotent(adapter_cls) -> None:
    """An empty chain contains no accidental ordinary lifecycle leaf."""
    chain = SimpleNamespace(chained_optimizers=[])
    engine = SimpleNamespace(optimizer=chain, device=torch.device("cuda"), model=None)
    adapter = adapter_cls(engine)

    adapter.release_memory(tags=["optimizer"])
    adapter.release_memory(tags=["optimizer"])
    adapter.resume_memory(tags=["optimizer"])
    adapter.resume_memory(tags=["optimizer"])


def test_optimizer_chain_deep_dfs_fallback_and_shared_cycle_are_stable() -> None:
    """DFS keeps valid DAG leaves while deduplicating shared/cyclic paths."""
    leaf_a = object()
    leaf_b = object()
    leaf_c = object()
    shared = SimpleNamespace(optimizers=[leaf_b, leaf_c])
    branch = SimpleNamespace(chained_optimizers=[shared])
    root = SimpleNamespace(chained_optimizers=[leaf_a, branch, shared, leaf_a])
    branch.chained_optimizers.append(root)
    warnings: list[str] = []

    class _RecordingLogger:
        @staticmethod
        def warning(message, *args) -> None:
            warnings.append(message % args)

    leaves = list(iter_megatron_optimizer_leaves(root, logger=_RecordingLogger))

    assert leaves == [leaf_a, leaf_b, leaf_c]
    assert len(warnings) == 1
    assert "optimizer chain cycle" in warnings[0]

    # The public spelling has precedence; ``optimizers`` is only a fallback.
    both_spellings = SimpleNamespace(chained_optimizers=[leaf_a], optimizers=[leaf_b])
    assert list(iter_megatron_optimizer_leaves(both_spellings)) == [leaf_a]

    deep = SimpleNamespace(optimizers=[leaf_c])
    for _ in range(128):
        deep = SimpleNamespace(chained_optimizers=[deep])
    assert list(iter_megatron_optimizer_leaves(deep)) == [leaf_c]


def test_optimizer_chain_reads_child_descriptor_once() -> None:
    """Dynamic chain attributes must not be observed in two different states."""
    leaf = object()

    class _DynamicChain:
        reads = 0

        @property
        def chained_optimizers(self):
            self.reads += 1
            if self.reads > 1:
                raise AssertionError("chained_optimizers was read more than once")
            return [leaf]

    chain = _DynamicChain()

    assert list(iter_megatron_optimizer_leaves(chain)) == [leaf]
    assert chain.reads == 1


def test_optimizer_chain_primary_descriptor_does_not_read_fallback() -> None:
    """A present canonical descriptor suppresses all fallback side effects."""
    leaf = object()

    class _BothDescriptors:
        primary_reads = 0
        fallback_reads = 0

        @property
        def chained_optimizers(self):
            self.primary_reads += 1
            return [leaf]

        @property
        def optimizers(self):
            self.fallback_reads += 1
            raise AssertionError("fallback descriptor must not be read")

    chain = _BothDescriptors()

    assert list(iter_megatron_optimizer_leaves(chain)) == [leaf]
    assert chain.primary_reads == 1
    assert chain.fallback_reads == 0


def test_optimizer_chain_descriptor_attribute_error_is_not_treated_as_missing() -> None:
    """A broken canonical descriptor must not silently select the fallback."""

    class _BrokenDescriptor:
        fallback_reads = 0

        @property
        def chained_optimizers(self):
            raise AttributeError("descriptor failed")

        @property
        def optimizers(self):
            self.fallback_reads += 1
            return [object()]

    chain = _BrokenDescriptor()

    with pytest.raises(AttributeError, match="descriptor failed"):
        list(iter_megatron_optimizer_leaves(chain))
    assert chain.fallback_reads == 0


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_invalid_nested_chain_is_atomic(adapter_cls) -> None:
    """A malformed nested chain fails before any leaf state or tag is changed."""
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=1e-2)
    param.grad = torch.tensor([0.25, -0.5], device="cuda")
    optimizer.step()
    state = optimizer.state[param]
    original_objects = dict(state)
    original_values = {
        key: value.detach().clone()
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    ordinary_leaf = SimpleNamespace(optimizer=optimizer)
    malformed = SimpleNamespace(chained_optimizers=1)
    root = SimpleNamespace(chained_optimizers=[ordinary_leaf, malformed])
    engine = SimpleNamespace(optimizer=root, device=torch.device("cuda"), model=None)
    adapter = adapter_cls(engine)

    with pytest.raises(
        TypeError,
        match="optimizer chain attribute 'chained_optimizers'.*must be iterable",
    ):
        adapter.release_memory(tags=["optimizer"])

    assert "optimizer" not in adapter._released_tags
    for key, original in original_objects.items():
        assert state[key] is original
        assert state[key].device == original.device
        torch.testing.assert_close(state[key], original_values[key], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.parametrize("available_method", ["offload_to_cpu", "restore_from_cpu"])
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_partial_hdo_interface_falls_back_to_ordinary_lifecycle(
    adapter_cls, available_method, monkeypatch
) -> None:
    """A partial HDO interface must not switch lifecycle strategy mid-cycle."""
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=1e-2)
    param.grad = torch.tensor([0.25, -0.5], device="cuda")
    optimizer.step()
    expected = {
        key: optimizer.state[param][key].detach().clone()
        for key in ("exp_avg", "exp_avg_sq")
    }
    calls: list[str] = []
    wrapper = SimpleNamespace(optimizer=optimizer)
    setattr(wrapper, available_method, lambda: calls.append(available_method))
    root = SimpleNamespace(chained_optimizers=[wrapper])
    engine = SimpleNamespace(optimizer=root, device=torch.device("cuda"), model=None)
    adapter = adapter_cls(engine)
    monkeypatch.setenv("AWEX_OPT_OFFLOAD_VIA_HDO", "1")

    adapter.release_memory(tags=["optimizer"])
    released_to_cpu = all(
        optimizer.state[param][key].device.type == "cpu" for key in expected
    )
    wrapper.offload_to_cpu = lambda: calls.append("dynamic.offload")
    wrapper.restore_from_cpu = lambda: calls.append("dynamic.restore")
    adapter.resume_memory(tags=["optimizer"])

    assert released_to_cpu
    for key, value in expected.items():
        assert optimizer.state[param][key].device.type == "cuda"
        torch.testing.assert_close(
            optimizer.state[param][key], value, rtol=0.0, atol=0.0
        )
    assert calls == []


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.parametrize(
    ("offload_method", "restore_method"),
    [(None, lambda: None), (lambda: None, None), (object(), lambda: None)],
)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_partial_hdo_interface_with_noncallable_method_stays_ordinary(
    adapter_cls, offload_method, restore_method, monkeypatch
) -> None:
    """Both HDO endpoints must be callable or neither endpoint is used."""
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=1e-2)
    param.grad = torch.tensor([0.25, -0.5], device="cuda")
    optimizer.step()
    expected = optimizer.state[param]["exp_avg"].detach().clone()
    wrapper = SimpleNamespace(
        optimizer=optimizer,
        offload_to_cpu=offload_method,
        restore_from_cpu=restore_method,
    )
    engine = SimpleNamespace(optimizer=wrapper, device=torch.device("cuda"), model=None)
    adapter = adapter_cls(engine)
    monkeypatch.setenv("AWEX_OPT_OFFLOAD_VIA_HDO", "1")

    adapter.release_memory(tags=["optimizer"])
    assert optimizer.state[param]["exp_avg"].device.type == "cpu"
    adapter.resume_memory(tags=["optimizer"])

    actual = optimizer.state[param]["exp_avg"]
    assert actual.is_cuda
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.parametrize("mutation", ["delete", "replace"])
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_hdo_cycle_uses_release_time_classification_and_methods(
    adapter_cls, mutation, monkeypatch
) -> None:
    """Resume consumes the release snapshot after leaf capabilities mutate."""
    calls: list[str] = []
    leaf = SimpleNamespace(
        state={},
        offload_to_cpu=lambda: calls.append("release.offload"),
        restore_from_cpu=lambda: calls.append("release.restore"),
    )
    engine = SimpleNamespace(optimizer=leaf, device=torch.device("cuda"), model=None)
    adapter = adapter_cls(engine)
    monkeypatch.setenv("AWEX_OPT_OFFLOAD_VIA_HDO", "1")

    adapter.release_memory(tags=["optimizer"])
    if mutation == "delete":
        del leaf.offload_to_cpu
        del leaf.restore_from_cpu
    else:
        leaf.offload_to_cpu = lambda: calls.append("mutated.offload")
        leaf.restore_from_cpu = lambda: calls.append("mutated.restore")
    monkeypatch.delenv("AWEX_OPT_OFFLOAD_VIA_HDO")
    adapter.resume_memory(tags=["optimizer"])

    assert calls == ["release.offload", "release.restore"]
    assert adapter._optimizer_lifecycle_cycle is None


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_release_failure_rolls_back_cycle_tag_and_ordinary_state(
    adapter_cls, monkeypatch
) -> None:
    """A later leaf failure restores earlier ordinary state and clears metadata."""
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=1e-2)
    param.grad = torch.tensor([0.25, -0.5], device="cuda")
    optimizer.step()
    expected = {
        key: optimizer.state[param][key].detach().clone()
        for key in ("exp_avg", "exp_avg_sq")
    }
    calls: list[str] = []

    def fail_offload() -> None:
        calls.append("offload")
        raise RuntimeError("injected HDO release failure")

    ordinary = SimpleNamespace(optimizer=optimizer)
    failing_hdo = SimpleNamespace(
        state={},
        offload_to_cpu=fail_offload,
        restore_from_cpu=lambda: calls.append("restore"),
    )
    root = SimpleNamespace(chained_optimizers=[ordinary, failing_hdo])
    engine = SimpleNamespace(optimizer=root, device=torch.device("cuda"), model=None)
    adapter = adapter_cls(engine)
    monkeypatch.setenv("AWEX_OPT_OFFLOAD_VIA_HDO", "1")

    with pytest.raises(RuntimeError, match="injected HDO release failure"):
        adapter.release_memory(tags=["optimizer"])

    assert calls == ["offload", "restore"]
    assert "optimizer" not in adapter._released_tags
    assert adapter._optimizer_lifecycle_cycle is None
    if hasattr(adapter, "_offloaded_optimizer_states"):
        assert adapter._offloaded_optimizer_states == {}
    for key, expected_value in expected.items():
        actual = optimizer.state[param][key]
        assert actual.is_cuda
        torch.testing.assert_close(actual, expected_value, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_failed_leaf_restore_does_not_block_prior_leaf_rollback(
    adapter_cls, monkeypatch
) -> None:
    """A leaf whose release failed must not block rollback of earlier leaves."""
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=1e-2)
    param.grad = torch.tensor([0.25, -0.5], device="cuda")
    optimizer.step()
    expected = optimizer.state[param]["exp_avg"].detach().clone()

    def fail_offload() -> None:
        raise RuntimeError("original release failure")

    def fail_restore() -> None:
        raise RuntimeError("restore called without successful offload")

    failing_hdo = SimpleNamespace(
        state={}, offload_to_cpu=fail_offload, restore_from_cpu=fail_restore
    )
    root = SimpleNamespace(
        chained_optimizers=[SimpleNamespace(optimizer=optimizer), failing_hdo]
    )
    engine = SimpleNamespace(optimizer=root, device=torch.device("cuda"), model=None)
    adapter = adapter_cls(engine)
    monkeypatch.setenv("AWEX_OPT_OFFLOAD_VIA_HDO", "1")

    with pytest.raises(RuntimeError, match="original release failure"):
        adapter.release_memory(tags=["optimizer"])

    actual = optimizer.state[param]["exp_avg"]
    assert actual.is_cuda
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert "optimizer" not in adapter._released_tags
    assert adapter._optimizer_lifecycle_cycle is None
    if hasattr(adapter, "_offloaded_optimizer_states"):
        assert adapter._offloaded_optimizer_states == {}


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_resume_failure_retries_without_replaying_restored_hdo(
    adapter_cls, monkeypatch
) -> None:
    """A retained release plan remains retryable after a later restore fails."""
    calls: list[str] = []
    first_restored = False
    second_should_fail = True

    def first_restore() -> None:
        nonlocal first_restored
        if first_restored:
            raise RuntimeError("first HDO restore replayed")
        first_restored = True
        calls.append("first.restore")

    def second_restore() -> None:
        if second_should_fail:
            raise RuntimeError("second restore failure")
        calls.append("second.restore")

    first = SimpleNamespace(
        state={},
        offload_to_cpu=lambda: calls.append("first.offload"),
        restore_from_cpu=first_restore,
    )
    second = SimpleNamespace(
        state={},
        offload_to_cpu=lambda: calls.append("second.offload"),
        restore_from_cpu=second_restore,
    )
    root = SimpleNamespace(chained_optimizers=[first, second])
    engine = SimpleNamespace(optimizer=root, device=torch.device("cuda"), model=None)
    adapter = adapter_cls(engine)
    monkeypatch.setenv("AWEX_OPT_OFFLOAD_VIA_HDO", "1")

    adapter.release_memory(tags=["optimizer"])
    with pytest.raises(RuntimeError, match="second restore failure"):
        adapter.resume_memory(tags=["optimizer"])
    assert "optimizer" in adapter._released_tags
    assert adapter._optimizer_lifecycle_cycle is not None

    second_should_fail = False
    adapter.resume_memory(tags=["optimizer"])

    assert calls == [
        "first.offload",
        "second.offload",
        "first.restore",
        "second.restore",
    ]
    assert "optimizer" not in adapter._released_tags
    assert adapter._optimizer_lifecycle_cycle is None


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_release_terminal_sync_failure_rolls_back_optimizer(
    adapter_cls, monkeypatch
) -> None:
    """A release-level synchronization failure performs an equivalent rollback."""
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=1e-2)
    param.grad = torch.tensor([0.25, -0.5], device="cuda")
    optimizer.step()
    expected = optimizer.state[param]["exp_avg"].detach().clone()
    engine = SimpleNamespace(
        optimizer=SimpleNamespace(optimizer=optimizer),
        device=torch.device("cuda"),
        model=None,
    )
    adapter = adapter_cls(engine)
    original_synchronize = torch.cuda.synchronize
    synchronize_calls = 0
    fail_at = 2 if adapter_cls is V1AwexMegatronAdapter else 1

    def injected_synchronize(*args, **kwargs) -> None:
        nonlocal synchronize_calls
        synchronize_calls += 1
        if synchronize_calls == fail_at:
            raise RuntimeError("injected terminal synchronize failure")
        original_synchronize(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "synchronize", injected_synchronize)

    with pytest.raises(RuntimeError, match="terminal synchronize failure"):
        adapter.release_memory(tags=["optimizer"])

    actual = optimizer.state[param]["exp_avg"]
    assert actual.is_cuda
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert "optimizer" not in adapter._released_tags
    assert adapter._optimizer_lifecycle_cycle is None
    if hasattr(adapter, "_offloaded_optimizer_states"):
        assert adapter._offloaded_optimizer_states == {}


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_ordinary_cycle_uses_release_time_base_optimizer(adapter_cls) -> None:
    """Ordinary restore targets the base optimizer captured by the release plan."""
    first_param = torch.nn.Parameter(torch.tensor([1.0, -2.0], device="cuda"))
    first_optimizer = torch.optim.AdamW([first_param], lr=1e-2)
    first_param.grad = torch.tensor([0.25, -0.5], device="cuda")
    first_optimizer.step()
    expected = first_optimizer.state[first_param]["exp_avg"].detach().clone()

    second_param = torch.nn.Parameter(torch.tensor([3.0, -4.0], device="cuda"))
    second_optimizer = torch.optim.AdamW([second_param], lr=1e-2)
    second_param.grad = torch.tensor([0.75, -1.0], device="cuda")
    second_optimizer.step()

    leaf = SimpleNamespace(optimizer=first_optimizer)
    engine = SimpleNamespace(optimizer=leaf, device=torch.device("cuda"), model=None)
    adapter = adapter_cls(engine)

    adapter.release_memory(tags=["optimizer"])
    leaf.optimizer = second_optimizer
    adapter.resume_memory(tags=["optimizer"])

    actual = first_optimizer.state[first_param]["exp_avg"]
    assert actual.is_cuda
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_v2_distinct_optimizers_sharing_param_keep_distinct_state() -> None:
    """The v2 ordinary side-table must not collide on a shared Parameter key."""
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], device="cuda"))
    first_optimizer = torch.optim.AdamW([param], lr=0.0)
    param.grad = torch.tensor([0.5, -0.5], device="cuda")
    first_optimizer.step()
    second_optimizer = torch.optim.AdamW([param], lr=0.0)
    param.grad = torch.tensor([1.5, -1.5], device="cuda")
    second_optimizer.step()
    expected_first = first_optimizer.state[param]["exp_avg"].detach().clone()
    expected_second = second_optimizer.state[param]["exp_avg"].detach().clone()
    root = SimpleNamespace(
        chained_optimizers=[
            SimpleNamespace(optimizer=first_optimizer),
            SimpleNamespace(optimizer=second_optimizer),
        ]
    )
    engine = SimpleNamespace(optimizer=root, device=torch.device("cuda"), model=None)
    adapter = V2AwexMegatronAdapter(engine)

    adapter.release_memory(tags=["optimizer"])
    adapter.resume_memory(tags=["optimizer"])

    torch.testing.assert_close(
        first_optimizer.state[param]["exp_avg"], expected_first, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        second_optimizer.state[param]["exp_avg"], expected_second, rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_resume_terminal_sync_retry_does_not_replay_hdo(
    adapter_cls, monkeypatch
) -> None:
    """A terminal sync retry starts after the fully restored entry suffix."""
    calls: list[str] = []
    leaf = SimpleNamespace(
        state={},
        offload_to_cpu=lambda: calls.append("offload"),
        restore_from_cpu=lambda: calls.append("restore"),
    )
    engine = SimpleNamespace(optimizer=leaf, device=torch.device("cuda"), model=None)
    adapter = adapter_cls(engine)
    monkeypatch.setenv("AWEX_OPT_OFFLOAD_VIA_HDO", "1")
    adapter.release_memory(tags=["optimizer"])

    original_synchronize = torch.cuda.synchronize
    fail_once = True

    def injected_synchronize(*args, **kwargs) -> None:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise RuntimeError("resume terminal sync failure")
        original_synchronize(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "synchronize", injected_synchronize)
    with pytest.raises(RuntimeError, match="resume terminal sync failure"):
        adapter.resume_memory(tags=["optimizer"])

    cycle = adapter._optimizer_lifecycle_cycle
    assert cycle is not None
    assert cycle.resume_index == len(cycle.entries)
    assert "optimizer" in adapter._released_tags
    adapter.resume_memory(tags=["optimizer"])

    assert calls == ["offload", "restore"]
    assert adapter._optimizer_lifecycle_cycle is None
    assert "optimizer" not in adapter._released_tags


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_multiple_rollback_failures_continue_and_preserve_original(
    adapter_cls, monkeypatch
) -> None:
    """Every rollback is attempted and errors become notes on the release error."""
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=1e-2)
    param.grad = torch.tensor([0.25, -0.5], device="cuda")
    optimizer.step()
    expected = optimizer.state[param]["exp_avg"].detach().clone()
    calls: list[str] = []

    def failed_restore(name: str):
        def restore() -> None:
            calls.append(f"{name}.restore")
            raise RuntimeError(f"{name} rollback failure")

        return restore

    first_hdo = SimpleNamespace(
        state={},
        offload_to_cpu=lambda: calls.append("first.offload"),
        restore_from_cpu=failed_restore("first"),
    )

    def final_offload() -> None:
        calls.append("final.offload")
        raise RuntimeError("primary release failure")

    final_hdo = SimpleNamespace(
        state={},
        offload_to_cpu=final_offload,
        restore_from_cpu=failed_restore("final"),
    )
    root = SimpleNamespace(
        chained_optimizers=[
            SimpleNamespace(optimizer=optimizer),
            first_hdo,
            final_hdo,
        ]
    )
    engine = SimpleNamespace(optimizer=root, device=torch.device("cuda"), model=None)
    adapter = adapter_cls(engine)
    monkeypatch.setenv("AWEX_OPT_OFFLOAD_VIA_HDO", "1")

    with pytest.raises(RuntimeError, match="primary release failure") as exc_info:
        adapter.release_memory(tags=["optimizer"])

    assert calls == [
        "first.offload",
        "final.offload",
        "final.restore",
        "first.restore",
    ]
    assert len(exc_info.value.__notes__) >= 2
    actual = optimizer.state[param]["exp_avg"]
    assert actual.is_cuda
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_failed_hdo_rollback_is_retained_and_blocks_next_release(
    adapter_cls, monkeypatch
) -> None:
    """An HDO rollback failure must remain recoverable, not be forgotten."""
    calls: list[str] = []

    def first_restore() -> None:
        calls.append("first.restore")
        raise RuntimeError("first rollback failure")

    first = SimpleNamespace(
        state={},
        offload_to_cpu=lambda: calls.append("first.offload"),
        restore_from_cpu=first_restore,
    )

    def final_offload() -> None:
        calls.append("final.offload")
        raise RuntimeError("primary release failure")

    final = SimpleNamespace(
        state={},
        offload_to_cpu=final_offload,
        restore_from_cpu=lambda: calls.append("final.restore"),
    )
    root = SimpleNamespace(chained_optimizers=[first, final])
    engine = SimpleNamespace(optimizer=root, device=torch.device("cuda"), model=None)
    adapter = adapter_cls(engine)
    monkeypatch.setenv("AWEX_OPT_OFFLOAD_VIA_HDO", "1")

    with pytest.raises(RuntimeError, match="primary release failure"):
        adapter.release_memory(tags=["optimizer"])

    assert calls == [
        "first.offload",
        "final.offload",
        "final.restore",
        "first.restore",
    ]
    assert adapter._optimizer_rollback_recovery is not None
    assert "optimizer" not in adapter._released_tags

    with pytest.raises(RuntimeError, match="unresolved rollback state"):
        adapter.release_memory(tags=["optimizer"])
    assert calls == [
        "first.offload",
        "final.offload",
        "final.restore",
        "first.restore",
    ]


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_hdo_rollback_recovery_retries_only_pending_entries(
    adapter_cls, monkeypatch
) -> None:
    """Recovery retries retain failures and never replay a completed restore."""
    calls: list[str] = []
    first_restore_fails = True

    def first_restore() -> None:
        calls.append("first.restore")
        if first_restore_fails:
            raise RuntimeError("first rollback failure")

    first = SimpleNamespace(
        state={},
        offload_to_cpu=lambda: calls.append("first.offload"),
        restore_from_cpu=first_restore,
    )

    def final_offload() -> None:
        calls.append("final.offload")
        raise RuntimeError("primary release failure")

    final = SimpleNamespace(
        state={},
        offload_to_cpu=final_offload,
        restore_from_cpu=lambda: calls.append("final.restore"),
    )
    engine = SimpleNamespace(
        optimizer=SimpleNamespace(chained_optimizers=[first, final]),
        device=torch.device("cuda"),
        model=None,
    )
    adapter = adapter_cls(engine)
    monkeypatch.setenv("AWEX_OPT_OFFLOAD_VIA_HDO", "1")

    with pytest.raises(RuntimeError, match="primary release failure"):
        adapter.release_memory(tags=["optimizer"])
    with pytest.raises(RuntimeError, match="first rollback failure"):
        adapter._retry_optimizer_rollback_recovery()

    assert calls.count("final.restore") == 1
    assert calls.count("first.restore") == 2
    assert adapter._optimizer_rollback_recovery is not None

    first_restore_fails = False
    adapter._retry_optimizer_rollback_recovery()

    assert calls.count("final.restore") == 1
    assert calls.count("first.restore") == 3
    assert adapter._optimizer_rollback_recovery is None
    assert adapter._optimizer_lifecycle_cycle is None
    assert "optimizer" not in adapter._released_tags
    if hasattr(adapter, "_offloaded_optimizer_states"):
        assert adapter._offloaded_optimizer_states == {}


@pytest.mark.parametrize("adapter_cls", AWEX_ADAPTERS)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_mixed_hdo_ordinary_recovery_retries_every_pending_action(
    adapter_cls, monkeypatch
) -> None:
    """Mixed rollback failures retain both journals and retry only pending work."""
    calls: list[str] = []

    class RestoreFailState(dict):
        restore_fails = True

        def __setitem__(self, key, value):
            if key == "exp_avg" and isinstance(value, torch.Tensor) and value.is_cuda:
                calls.append("ordinary.restore")
                if self.restore_fails:
                    raise RuntimeError("ordinary rollback failure")
            return super().__setitem__(key, value)

    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=0.0)
    param.grad = torch.tensor([0.25, -0.5], device="cuda")
    optimizer.step()
    expected = optimizer.state[param]["exp_avg"].detach().clone()
    guarded_state = RestoreFailState(optimizer.state[param])
    optimizer.state[param] = guarded_state
    hdo_restore_fails = True

    def hdo_restore() -> None:
        calls.append("hdo.restore")
        if hdo_restore_fails:
            raise RuntimeError("HDO rollback failure")

    hdo = SimpleNamespace(
        state={},
        offload_to_cpu=lambda: calls.append("hdo.offload"),
        restore_from_cpu=hdo_restore,
    )

    def final_offload() -> None:
        calls.append("final.offload")
        raise RuntimeError("primary release failure")

    final = SimpleNamespace(
        state={},
        offload_to_cpu=final_offload,
        restore_from_cpu=lambda: calls.append("final.restore"),
    )
    root = SimpleNamespace(
        chained_optimizers=[SimpleNamespace(optimizer=optimizer), hdo, final]
    )
    adapter = adapter_cls(
        SimpleNamespace(optimizer=root, device=torch.device("cuda"), model=None)
    )
    monkeypatch.setenv("AWEX_OPT_OFFLOAD_VIA_HDO", "1")

    with pytest.raises(RuntimeError, match="primary release failure") as exc_info:
        adapter.release_memory(tags=["optimizer"])
    assert any("ordinary rollback failure" in note for note in exc_info.value.__notes__)
    assert any("HDO rollback failure" in note for note in exc_info.value.__notes__)
    assert adapter._optimizer_rollback_recovery is not None
    with pytest.raises(RuntimeError, match="unresolved rollback state"):
        adapter.release_memory(tags=["optimizer"])

    with pytest.raises(RuntimeError, match="HDO rollback failure") as recovery_info:
        adapter._retry_optimizer_rollback_recovery()
    assert any(
        "ordinary rollback failure" in note for note in recovery_info.value.__notes__
    )

    guarded_state.restore_fails = False
    hdo_restore_fails = False
    completed_before = calls.count("final.restore")
    adapter._retry_optimizer_rollback_recovery()

    assert calls.count("final.restore") == completed_before
    assert adapter._optimizer_rollback_recovery is None
    assert adapter._optimizer_lifecycle_cycle is None
    assert "optimizer" not in adapter._released_tags
    if hasattr(adapter, "_offloaded_optimizer_states"):
        assert adapter._offloaded_optimizer_states == {}
    torch.testing.assert_close(
        optimizer.state[param]["exp_avg"], expected, rtol=0.0, atol=0.0
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_legacy_te_cache_purge_failure_rolls_back_optimizer(
    monkeypatch,
) -> None:
    """A post-release TE cache failure remains inside the release transaction."""
    import builtins

    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=0.0)
    param.grad = torch.tensor([0.25, -0.5], device="cuda")
    optimizer.step()
    expected = optimizer.state[param]["exp_avg"].detach().clone()
    leaf = SimpleNamespace(optimizer=optimizer)
    engine = SimpleNamespace(optimizer=leaf, device=torch.device("cuda"), model=None)
    adapter = V1AwexMegatronAdapter(engine)
    original_import = builtins.__import__

    def injected_import(name, *args, **kwargs):
        if name == "transformer_engine.pytorch.module.base":
            raise RuntimeError("TE cache purge failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", injected_import)
    with pytest.raises(RuntimeError, match="TE cache purge failure"):
        adapter.release_memory(tags=["optimizer"])

    actual = optimizer.state[param]["exp_avg"]
    assert actual.is_cuda
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert "optimizer" not in adapter._released_tags
    assert adapter._optimizer_lifecycle_cycle is None
    assert adapter._optimizer_rollback_recovery is None


@pytest.mark.parametrize("failure_index", [0, 1])
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_legacy_te_cache_first_and_middle_delete_are_transactional(
    monkeypatch, failure_index
) -> None:
    """First and middle delete failures restore exact cache key/value objects."""
    from transformer_engine.pytorch.module import base as te_base

    class IndexedDeleteFailure(dict):
        delete_count = 0

        def __delitem__(self, key):
            index = self.delete_count
            self.delete_count += 1
            if index == failure_index:
                raise RuntimeError(f"TE delete {failure_index} failure")
            return super().__delitem__(key)

    expected = {"first": object(), "middle": object(), "last": object()}
    cache = IndexedDeleteFailure(expected)
    monkeypatch.setattr(te_base, "_dummy_wgrads", cache)
    param = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=0.0)
    param.grad = torch.ones_like(param)
    optimizer.step()
    state_before = optimizer.state[param]["exp_avg"].detach().clone()
    adapter = V1AwexMegatronAdapter(
        SimpleNamespace(
            optimizer=SimpleNamespace(optimizer=optimizer),
            device=torch.device("cuda"),
            model=None,
        )
    )

    with pytest.raises(RuntimeError, match=f"TE delete {failure_index} failure"):
        adapter.release_memory(tags=["optimizer"])

    assert set(cache) == set(expected)
    for key, value in expected.items():
        assert cache[key] is value
    torch.testing.assert_close(
        optimizer.state[param]["exp_avg"], state_before, rtol=0.0, atol=0.0
    )
    assert optimizer.state[param]["exp_avg"].is_cuda
    assert adapter._released_tags == set()
    assert adapter._optimizer_rollback_recovery is None
    assert adapter._te_cache_purge_undo is None


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_legacy_te_cache_delete_mutates_then_raises_is_recoverable(
    monkeypatch,
) -> None:
    """A mapping that raises after deletion must not escape the TE undo journal."""
    from transformer_engine.pytorch.module import base as te_base

    class MutateThenRaise(dict):
        def __delitem__(self, key):
            super().__delitem__(key)
            if key == "bad":
                raise RuntimeError("TE delete mutated then failed")

    expected = {"ok": object(), "bad": object()}
    cache = MutateThenRaise(expected)
    monkeypatch.setattr(te_base, "_dummy_wgrads", cache)
    param = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=0.0)
    param.grad = torch.ones_like(param)
    optimizer.step()
    state_before = optimizer.state[param]["exp_avg"].detach().clone()
    adapter = V1AwexMegatronAdapter(
        SimpleNamespace(
            optimizer=SimpleNamespace(optimizer=optimizer),
            device=torch.device("cuda"),
            model=None,
        )
    )

    with pytest.raises(RuntimeError, match="TE delete mutated then failed"):
        adapter.release_memory(tags=["optimizer"])

    assert set(cache) == set(expected)
    for key, value in expected.items():
        assert cache[key] is value
    torch.testing.assert_close(
        optimizer.state[param]["exp_avg"], state_before, rtol=0.0, atol=0.0
    )
    assert optimizer.state[param]["exp_avg"].is_cuda
    assert adapter._released_tags == set()
    assert adapter._optimizer_rollback_recovery is None
    assert adapter._te_cache_purge_undo is None


@pytest.mark.parametrize("failure_index", [0, 1])
@pytest.mark.parametrize("mutate_then_raise", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, ImportError])
def test_awex_legacy_te_cache_delete_failure_matrix_restores_in_reverse(
    failure_index, mutate_then_raise, error_type
) -> None:
    """Pre-registered undo covers pre/post-mutation errors for every key."""
    operations: list[str] = []

    class TransactionalFailureCache(dict):
        delete_index = 0

        def __delitem__(self, key):
            index = self.delete_index
            self.delete_index += 1
            operations.append(f"delete:{key}")
            if index == failure_index:
                if mutate_then_raise:
                    super().__delitem__(key)
                raise error_type("injected cache delete failure")
            return super().__delitem__(key)

        def __setitem__(self, key, value):
            operations.append(f"restore:{key}")
            return super().__setitem__(key, value)

    expected = {"first": object(), "middle": object(), "last": object()}
    cache = TransactionalFailureCache(expected)
    adapter = V1AwexMegatronAdapter(SimpleNamespace())

    with pytest.raises(error_type, match="injected cache delete failure") as exc_info:
        adapter._purge_te_cache_transactionally(adapter._snapshot_te_cache(cache))
    adapter._rollback_te_cache_purge(exc_info.value)

    assert operations[-len(expected) :] == [
        f"restore:{key}" for key in reversed(tuple(expected))
    ]
    assert set(cache) == set(expected)
    for key, value in expected.items():
        assert cache[key] is value
    assert adapter._te_cache_purge_undo is None


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_legacy_te_cache_operation_import_error_is_not_swallowed(
    monkeypatch,
) -> None:
    """Only a missing TE import may be ignored, not cache-operation ImportError."""
    from transformer_engine.pytorch.module import base as te_base

    class ImportErrorDuringDelete(dict):
        def __delitem__(self, key):
            if key == "bad":
                raise ImportError("cache delete import error")
            return super().__delitem__(key)

        def __setitem__(self, key, value):
            if key == "ok":
                raise RuntimeError("cache undo failure")
            return super().__setitem__(key, value)

    expected = {"ok": object(), "bad": object()}
    cache = ImportErrorDuringDelete(expected)
    monkeypatch.setattr(te_base, "_dummy_wgrads", cache)
    param = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=0.0)
    param.grad = torch.ones_like(param)
    optimizer.step()
    state_before = optimizer.state[param]["exp_avg"].detach().clone()
    adapter = V1AwexMegatronAdapter(
        SimpleNamespace(
            optimizer=SimpleNamespace(optimizer=optimizer),
            device=torch.device("cuda"),
            model=None,
        )
    )

    with pytest.raises(ImportError, match="cache delete import error"):
        adapter.release_memory(tags=["optimizer"])

    assert set(cache) == set(expected)
    for key, value in expected.items():
        assert cache[key] is value
    torch.testing.assert_close(
        optimizer.state[param]["exp_avg"], state_before, rtol=0.0, atol=0.0
    )
    assert optimizer.state[param]["exp_avg"].is_cuda
    assert adapter._released_tags == set()
    assert adapter._optimizer_rollback_recovery is not None
    assert adapter._te_cache_purge_undo is not None


def test_awex_legacy_te_cache_snapshot_bypasses_items_override(monkeypatch) -> None:
    """The trusted dict snapshot never executes a subclass items override."""
    from transformer_engine.pytorch.module import base as te_base

    class ItemsImportError(dict):
        items_calls = 0

        def items(self):
            self.items_calls += 1
            raise ImportError("cache items import error")

        def __delitem__(self, key):
            if key == "bad":
                super().__delitem__(key)
                raise ImportError("cache delete import error")
            return super().__delitem__(key)

    expected = {"ok": object(), "bad": object()}
    cache = ItemsImportError(expected)
    monkeypatch.setattr(te_base, "_dummy_wgrads", cache)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    adapter = V1AwexMegatronAdapter(SimpleNamespace(optimizer=None, model=None))

    with pytest.raises(ImportError, match="cache delete import error"):
        adapter.release_memory(tags=["optimizer"])

    assert cache.items_calls == 0
    assert set(cache) == set(expected)
    for key, value in expected.items():
        assert cache[key] is value
    assert adapter._released_tags == set()
    assert adapter._optimizer_lifecycle_cycle is None
    assert adapter._optimizer_rollback_recovery is None
    assert adapter._te_cache_purge_undo is None


@pytest.mark.parametrize("error_type", [RuntimeError, ImportError])
def test_awex_legacy_te_cache_items_mutates_then_raises_is_transactional(
    monkeypatch, error_type
) -> None:
    """A trusted snapshot bypasses items() and enables complete delete rollback."""
    from transformer_engine.pytorch.module import base as te_base

    class ItemsMutateThenRaise(dict):
        items_calls = 0

        def items(self):
            self.items_calls += 1
            dict.__delitem__(self, "bad")
            raise error_type("cache items mutated then failed")

        def __delitem__(self, key):
            if key == "bad":
                super().__delitem__(key)
                raise error_type("cache delete mutated then failed")
            return super().__delitem__(key)

    expected = {"ok": object(), "bad": object()}
    cache = ItemsMutateThenRaise(expected)
    monkeypatch.setattr(te_base, "_dummy_wgrads", cache)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    adapter = V1AwexMegatronAdapter(SimpleNamespace(optimizer=None, model=None))

    with pytest.raises(error_type, match="cache delete mutated then failed"):
        adapter.release_memory(tags=["optimizer"])

    assert cache.items_calls == 0
    assert set(cache) == set(expected)
    for key, value in expected.items():
        assert cache[key] is value
    assert adapter._released_tags == set()
    assert adapter._optimizer_lifecycle_cycle is None
    assert adapter._optimizer_rollback_recovery is None
    assert adapter._te_cache_purge_undo is None


def test_awex_legacy_te_cache_delete_side_effect_new_key_is_removed() -> None:
    """Rollback reconstructs the exact trusted snapshot after delete side effects."""

    class AddsKeyThenFails(dict):
        def __delitem__(self, key):
            if key == "bad":
                dict.__setitem__(self, "injected", object())
                super().__delitem__(key)
                raise RuntimeError("delete added key then failed")
            return super().__delitem__(key)

    expected = {"ok": object(), "bad": object()}
    cache = AddsKeyThenFails(expected)
    adapter = V1AwexMegatronAdapter(SimpleNamespace())
    journal = adapter._snapshot_te_cache(cache)

    with pytest.raises(RuntimeError, match="delete added key then failed") as exc_info:
        adapter._purge_te_cache_transactionally(journal)
    adapter._rollback_te_cache_purge(exc_info.value)

    assert set(cache) == set(expected)
    for key, value in expected.items():
        assert cache[key] is value
    assert adapter._te_cache_purge_undo is None


def test_awex_legacy_te_cache_unsupported_mapping_fails_before_dynamic_access(
    monkeypatch,
) -> None:
    """Non-dict mappings fail closed before optimizer or mapping side effects."""
    from transformer_engine.pytorch.module import base as te_base

    class UnsupportedMapping:
        dynamic_calls = 0

        def items(self):
            self.dynamic_calls += 1
            raise AssertionError("unsupported mapping was dynamically read")

        def __delitem__(self, key):
            self.dynamic_calls += 1
            raise AssertionError("unsupported mapping was modified")

    cache = UnsupportedMapping()
    monkeypatch.setattr(te_base, "_dummy_wgrads", cache)
    lifecycle_calls: list[str] = []
    leaf = SimpleNamespace(
        state={},
        offload_to_cpu=lambda: lifecycle_calls.append("offload"),
        restore_from_cpu=lambda: lifecycle_calls.append("restore"),
    )
    adapter = V1AwexMegatronAdapter(SimpleNamespace(optimizer=leaf, model=None))
    monkeypatch.setenv("AWEX_OPT_OFFLOAD_VIA_HDO", "1")

    with pytest.raises(
        TypeError, match=r"must be a dict or dict subclass.*UnsupportedMapping"
    ):
        adapter.release_memory(tags=["optimizer"])

    assert cache.dynamic_calls == 0
    assert lifecycle_calls == []
    assert adapter._released_tags == set()
    assert adapter._optimizer_lifecycle_cycle is None
    assert adapter._optimizer_rollback_recovery is None
    assert adapter._te_cache_purge_undo is None


def test_awex_legacy_te_cache_recovery_targets_release_time_cache(
    monkeypatch,
) -> None:
    """Replacing the TE global cache does not redirect a pending rollback."""
    from transformer_engine.pytorch.module import base as te_base

    expected = {"first": object(), "second": object()}
    original_cache = dict(expected)
    replacement_cache = {"replacement": object()}
    adapter = V1AwexMegatronAdapter(SimpleNamespace())

    adapter._purge_te_cache_transactionally(adapter._snapshot_te_cache(original_cache))
    monkeypatch.setattr(te_base, "_dummy_wgrads", replacement_cache)
    adapter._restore_te_cache_items()

    assert set(original_cache) == set(expected)
    for key, value in expected.items():
        assert original_cache[key] is value
    assert set(replacement_cache) == {"replacement"}
    assert adapter._te_cache_purge_undo is None


def test_awex_legacy_te_cache_descriptor_import_error_is_not_swallowed(
    monkeypatch,
) -> None:
    """A cache descriptor ImportError occurs after, and outside, module import."""
    from transformer_engine.pytorch.module import base as te_base

    monkeypatch.delattr(te_base, "_dummy_wgrads")

    def descriptor_failure(name):
        if name == "_dummy_wgrads":
            raise ImportError("cache descriptor import error")
        raise AttributeError(name)

    monkeypatch.setattr(te_base, "__getattr__", descriptor_failure, raising=False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    adapter = V1AwexMegatronAdapter(SimpleNamespace(optimizer=None, model=None))

    with pytest.raises(ImportError, match="cache descriptor import error"):
        adapter.release_memory(tags=["optimizer"])

    assert adapter._released_tags == set()
    assert adapter._optimizer_lifecycle_cycle is None
    assert adapter._optimizer_rollback_recovery is None
    assert adapter._te_cache_purge_undo is None


def test_awex_legacy_te_module_import_error_keeps_compatibility(
    monkeypatch,
) -> None:
    """An ImportError raised by the TE module import skips only cache purge."""
    import builtins

    original_import = builtins.__import__

    def missing_te_import(name, *args, **kwargs):
        if name == "transformer_engine.pytorch.module.base":
            raise ImportError("TE module import failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_te_import)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    adapter = V1AwexMegatronAdapter(SimpleNamespace(optimizer=None, model=None))

    adapter.release_memory(tags=["optimizer"])
    assert adapter._released_tags == {"optimizer"}
    assert adapter._optimizer_lifecycle_cycle is not None
    assert adapter._te_cache_purge_undo is None

    adapter.resume_memory(tags=["optimizer"])
    assert adapter._released_tags == set()
    assert adapter._optimizer_lifecycle_cycle is None


@pytest.mark.parametrize("failure_stage", ["read", "delete"])
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_legacy_te_cache_operation_failure_rolls_back_optimizer(
    monkeypatch, failure_stage
) -> None:
    """TE cache discovery and partial purge are part of optimizer release."""
    from transformer_engine.pytorch.module import base as te_base

    class FailingCache(dict):
        items_calls = 0

        def items(self):
            self.items_calls += 1
            if failure_stage == "read":
                raise RuntimeError("TE cache read failure")
            return super().items()

        def __delitem__(self, key):
            if failure_stage == "delete" and key == "bad":
                raise RuntimeError("TE cache delete failure")
            return super().__delitem__(key)

    cache = FailingCache(ok=object(), bad=object())
    monkeypatch.setattr(te_base, "_dummy_wgrads", cache)
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=0.0)
    param.grad = torch.tensor([0.25, -0.5], device="cuda")
    optimizer.step()
    expected = optimizer.state[param]["exp_avg"].detach().clone()
    engine = SimpleNamespace(
        optimizer=SimpleNamespace(optimizer=optimizer),
        device=torch.device("cuda"),
        model=None,
    )
    adapter = V1AwexMegatronAdapter(engine)

    if failure_stage == "read":
        adapter.release_memory(tags=["optimizer"])
        assert cache.items_calls == 0
        assert cache == {}
        adapter.resume_memory(tags=["optimizer"])
    else:
        with pytest.raises(RuntimeError, match="TE cache delete failure"):
            adapter.release_memory(tags=["optimizer"])

    actual = optimizer.state[param]["exp_avg"]
    assert actual.is_cuda
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert set(cache) == (set() if failure_stage == "read" else {"ok", "bad"})
    assert "optimizer" not in adapter._released_tags
    assert adapter._optimizer_lifecycle_cycle is None
    assert adapter._optimizer_rollback_recovery is None
    assert adapter._te_cache_purge_undo is None


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_legacy_te_cache_rollback_failure_is_retryable(monkeypatch) -> None:
    """A failed TE undo remains fail-closed until its explicit retry succeeds."""
    from transformer_engine.pytorch.module import base as te_base

    class RecoverableCache(dict):
        restore_fails = True

        def __delitem__(self, key):
            if key == "bad":
                raise RuntimeError("TE cache delete failure")
            return super().__delitem__(key)

        def __setitem__(self, key, value):
            if key == "ok" and self.restore_fails:
                raise RuntimeError("TE cache rollback failure")
            return super().__setitem__(key, value)

    cache = RecoverableCache(ok=object(), bad=object())
    monkeypatch.setattr(te_base, "_dummy_wgrads", cache)
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=0.0)
    param.grad = torch.tensor([0.25, -0.5], device="cuda")
    optimizer.step()
    expected = optimizer.state[param]["exp_avg"].detach().clone()
    engine = SimpleNamespace(
        optimizer=SimpleNamespace(optimizer=optimizer),
        device=torch.device("cuda"),
        model=None,
    )
    adapter = V1AwexMegatronAdapter(engine)

    with pytest.raises(RuntimeError, match="TE cache delete failure") as exc_info:
        adapter.release_memory(tags=["optimizer"])

    assert any("TE cache rollback failed" in note for note in exc_info.value.__notes__)
    assert optimizer.state[param]["exp_avg"].is_cuda
    torch.testing.assert_close(
        optimizer.state[param]["exp_avg"], expected, rtol=0.0, atol=0.0
    )
    assert adapter._optimizer_rollback_recovery is not None
    assert adapter._te_cache_purge_undo is not None
    with pytest.raises(RuntimeError, match="unresolved rollback state"):
        adapter.release_memory(tags=["optimizer"])
    with pytest.raises(RuntimeError, match="TE cache rollback failure"):
        adapter._retry_optimizer_rollback_recovery()

    cache.restore_fails = False
    adapter._retry_optimizer_rollback_recovery()

    assert set(cache) == {"ok", "bad"}
    assert adapter._optimizer_rollback_recovery is None
    assert adapter._te_cache_purge_undo is None
    assert adapter._optimizer_lifecycle_cycle is None
    assert "optimizer" not in adapter._released_tags


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_legacy_weights_failure_restores_optimizer_and_te_cache(
    monkeypatch,
) -> None:
    """A failure after optimizer release rolls back both optimizer and TE cache."""
    from transformer_engine.pytorch.module import base as te_base

    cache_value = object()
    cache = {"cached": cache_value}
    monkeypatch.setattr(te_base, "_dummy_wgrads", cache)
    param = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=0.0)
    param.grad = torch.ones_like(param)
    optimizer.step()
    expected = optimizer.state[param]["exp_avg"].detach().clone()
    adapter = V1AwexMegatronAdapter(
        SimpleNamespace(
            optimizer=SimpleNamespace(optimizer=optimizer),
            device=torch.device("cuda"),
            model=None,
        )
    )

    def fail_weights_release() -> None:
        raise RuntimeError("weights release failure")

    monkeypatch.setattr(adapter, "_offload_model_weights", fail_weights_release)
    with pytest.raises(RuntimeError, match="weights release failure"):
        adapter.release_memory(tags=["optimizer", "weights"])

    assert cache == {"cached": cache_value}
    assert cache["cached"] is cache_value
    assert optimizer.state[param]["exp_avg"].is_cuda
    torch.testing.assert_close(
        optimizer.state[param]["exp_avg"], expected, rtol=0.0, atol=0.0
    )
    assert adapter._released_tags == set()
    assert adapter._optimizer_lifecycle_cycle is None
    assert adapter._optimizer_rollback_recovery is None
    assert adapter._te_cache_purge_undo is None


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_legacy_terminal_sync_failure_restores_te_cache(monkeypatch) -> None:
    """Terminal synchronization remains inside the TE/optimizer transaction."""
    from transformer_engine.pytorch.module import base as te_base

    cache_value = object()
    cache = {"cached": cache_value}
    monkeypatch.setattr(te_base, "_dummy_wgrads", cache)
    param = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=0.0)
    param.grad = torch.ones_like(param)
    optimizer.step()
    expected = optimizer.state[param]["exp_avg"].detach().clone()
    adapter = V1AwexMegatronAdapter(
        SimpleNamespace(
            optimizer=SimpleNamespace(optimizer=optimizer),
            device=torch.device("cuda"),
            model=None,
        )
    )
    original_synchronize = torch.cuda.synchronize
    synchronize_calls = 0

    def fail_terminal_sync(*args, **kwargs) -> None:
        nonlocal synchronize_calls
        synchronize_calls += 1
        if synchronize_calls == 2:
            raise RuntimeError("terminal sync failure")
        original_synchronize(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "synchronize", fail_terminal_sync)
    with pytest.raises(RuntimeError, match="terminal sync failure"):
        adapter.release_memory(tags=["optimizer"])

    assert cache == {"cached": cache_value}
    assert cache["cached"] is cache_value
    assert optimizer.state[param]["exp_avg"].is_cuda
    torch.testing.assert_close(
        optimizer.state[param]["exp_avg"], expected, rtol=0.0, atol=0.0
    )
    assert adapter._released_tags == set()
    assert adapter._optimizer_rollback_recovery is None
    assert adapter._te_cache_purge_undo is None


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AWEX residency")
def test_awex_v2_entry_side_table_prunes_only_restored_prefix(monkeypatch) -> None:
    """Entry journals are independent and consumed as each entry restores."""
    first_param = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
    first = torch.optim.AdamW([first_param], lr=0.0)
    first_param.grad = torch.ones_like(first_param)
    first.step()
    second_param = torch.nn.Parameter(torch.tensor([2.0], device="cuda"))
    second = torch.optim.AdamW([second_param], lr=0.0)
    second_param.grad = torch.ones_like(second_param)
    second.step()
    should_fail = True

    def restore_gate() -> None:
        if should_fail:
            raise RuntimeError("restore gate failure")

    gate = SimpleNamespace(
        state={}, offload_to_cpu=lambda: None, restore_from_cpu=restore_gate
    )
    root = SimpleNamespace(
        chained_optimizers=[
            SimpleNamespace(optimizer=first),
            gate,
            SimpleNamespace(optimizer=second),
        ]
    )
    engine = SimpleNamespace(optimizer=root, device=torch.device("cuda"), model=None)
    adapter = V2AwexMegatronAdapter(engine)
    monkeypatch.setenv("AWEX_OPT_OFFLOAD_VIA_HDO", "1")
    adapter.release_memory(tags=["optimizer"])

    with pytest.raises(RuntimeError, match="restore gate failure"):
        adapter.resume_memory(tags=["optimizer"])

    assert first.state[first_param]["exp_avg"].is_cuda
    assert second.state[second_param]["exp_avg"].device.type == "cpu"
    assert set(adapter._offloaded_optimizer_states) == {2}
    should_fail = False
    adapter.resume_memory(tags=["optimizer"])

    assert second.state[second_param]["exp_avg"].is_cuda
    assert adapter._offloaded_optimizer_states == {}
