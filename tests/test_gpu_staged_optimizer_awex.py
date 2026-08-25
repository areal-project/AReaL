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
    """Both AWEX adapters recycle staging slots without replacing CPU views."""
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
    original_slots = tuple(optimizer._slots)
    original_slot_ids = {id(slot) for slot in original_slots}
    assert len(original_slot_ids) == optimizer.staged_config.buffer_count
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

    monkeypatch.setattr(optimizer, "drain", tracked_drain)
    adapter.release_memory(tags=["optimizer"])
    adapter.release_memory(tags=["optimizer"])
    assert drain_calls == 1
    assert optimizer._slots == []
    assert optimizer._slot_machine is None
    original_values = {key: tensor.clone() for key, tensor in state.items()}
    optimizer.prepare_checkpoint_save()
    assert drain_calls == 2
    adapter.resume_memory(tags=["optimizer"])
    adapter.resume_memory(tags=["optimizer"])
    assert drain_calls == 2
    assert len(optimizer._slots) == optimizer.staged_config.buffer_count
    assert all(id(slot) not in original_slot_ids for slot in optimizer._slots)
    assert optimizer._slot_machine is not None

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
def test_awex_releases_mixed_managed_and_ordinary_optimizer_chain(
    adapter_cls,
) -> None:
    """The original ordinary migration remains the non-staged fallback."""
    managed_param = torch.nn.Parameter(
        torch.tensor([1.0, -1.0], device="cuda", dtype=torch.bfloat16)
    )
    managed = GPUStagedAdamW([managed_param], lr=1e-2)
    managed.bind_owned_params(managed.param_groups)
    ordinary_param = torch.nn.Parameter(torch.tensor([2.0, -3.0], device="cuda"))
    ordinary = torch.optim.AdamW([ordinary_param], lr=1e-2)
    ordinary_param.grad = torch.tensor([0.5, -0.25], device="cuda")
    ordinary.step()
    ordinary_state = ordinary.state[ordinary_param]
    # Native Muon uses this state name; v1 AWEX must migrate it alongside
    # AdamW's moments when the optimizer is not CPU-staged.
    ordinary_state["momentum_buffer"] = torch.tensor([0.25, -0.5], device="cuda")
    expected_state = {
        key: value.detach().cpu().clone()
        for key, value in ordinary_state.items()
        if isinstance(value, torch.Tensor)
    }
    original_devices = {
        key: value.device
        for key, value in ordinary_state.items()
        if isinstance(value, torch.Tensor)
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

    assert managed._slots == []
    assert not any(
        value.is_cuda
        for value in ordinary_state.values()
        if isinstance(value, torch.Tensor)
    )

    adapter.resume_memory(tags=["optimizer"])

    assert len(managed._slots) == managed.staged_config.buffer_count
    for key, expected in expected_state.items():
        value = ordinary_state[key]
        assert value.device == original_devices[key]
        torch.testing.assert_close(value.cpu(), expected, rtol=0.0, atol=0.0)
