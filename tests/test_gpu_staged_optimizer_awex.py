# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
