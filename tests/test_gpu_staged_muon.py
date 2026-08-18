# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from areal.engine.megatron_utils.gpu_staged_muon import (
    GPUStagedMuon,
    GPUStagedMuonConfig,
    _validate_official_ownership,
)
from areal.engine.megatron_utils.gpu_staged_optimizer import (
    GPUStagedAdamW,
    GPUStagedAdamWConfig,
)


def _config(*, slot_numel: int = 64, buffer_count: int = 2) -> GPUStagedMuonConfig:
    return GPUStagedMuonConfig(
        buffer_count=buffer_count,
        slot_size_mb=slot_numel * 4 / (1024 * 1024),
        split_qkv=False,
        fp32_matmul_prec="highest",
    )


def _identity_orthogonalize(
    param: torch.Tensor, update: torch.Tensor, **kwargs
) -> torch.Tensor:
    del param, kwargs
    return update


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_muon_owned_matrices_are_indivisible_pinned_slab_units() -> None:
    """Each official-owned 2D matrix must map to one complete staged unit."""
    params = [
        torch.nn.Parameter(torch.randn(4, 5, device="cuda", dtype=torch.bfloat16)),
        torch.nn.Parameter(torch.randn(3, 7, device="cuda", dtype=torch.bfloat16)),
    ]
    groups = [
        {
            "params": params,
            "lr": 0.02,
            "momentum": 0.9,
            "weight_decay": 0.01,
        }
    ]
    optimizer = GPUStagedMuon(
        groups,
        staged_config=_config(slot_numel=32),
        orthogonalize=_identity_orthogonalize,
        matmul_precision=nullcontext,
        nesterov=False,
        weight_decay_method="decoupled",
    )
    optimizer.bind_owned_params(optimizer.param_groups)

    assert [unit.param for unit in optimizer.units] == params
    assert [unit.numel for unit in optimizer.units] == [20, 21]
    assert optimizer.residency == "CPU_RESIDENT"
    assert optimizer.cuda_state_numel == 0
    assert optimizer.cpu_slabs is not None
    assert optimizer.cpu_slabs.master.dtype is torch.float32
    assert optimizer.cpu_slabs.momentum.dtype is torch.float32
    assert optimizer.cpu_slabs.master.is_pinned()
    assert optimizer.cpu_slabs.momentum.is_pinned()
    for param in params:
        assert set(optimizer.state[param]) == {"master_param", "momentum_buffer"}
        for value in optimizer.state[param].values():
            assert value.device.type == "cpu"
            assert value.dtype is torch.float32
    first_state = optimizer.state[params[0]]["master_param"]
    assert (
        first_state.untyped_storage().data_ptr()
        == optimizer.cpu_slabs.master.untyped_storage().data_ptr()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_muon_and_scalar_adamw_state_schemas_are_isolated() -> None:
    """Official classification leaves receive disjoint Muon and AdamW schemas."""
    matrix = torch.nn.Parameter(torch.ones(4, 4, device="cuda", dtype=torch.bfloat16))
    scalar = torch.nn.Parameter(torch.ones(4, device="cuda", dtype=torch.bfloat16))
    muon = GPUStagedMuon(
        [{"params": [matrix], "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}],
        staged_config=_config(),
        orthogonalize=_identity_orthogonalize,
        matmul_precision=nullcontext,
        nesterov=False,
        weight_decay_method="decoupled",
    )
    muon.bind_owned_params(muon.param_groups)
    adam = GPUStagedAdamW(
        [scalar],
        lr=0.1,
        staged_config=GPUStagedAdamWConfig(buffer_count=1, bucket_size_mb=1),
    )
    adam.bind_owned_params(adam.param_groups)

    assert set(muon.state[matrix]) == {"master_param", "momentum_buffer"}
    assert set(adam.state[scalar]) == {"master_param", "exp_avg", "exp_avg_sq"}
    assert "exp_avg" not in muon.state[matrix]
    assert "momentum_buffer" not in adam.state[scalar]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_muon_gpu_residency_is_bounded_by_max_unit_not_total_state() -> None:
    """More owner matrices grow only CPU slabs, not resident CUDA slots."""
    observed_staging = []
    observed_cpu_state = []
    observed_optimizer_peak_bytes = []
    for matrix_count in (1, 8):
        gc.collect()
        torch.cuda.empty_cache()
        params = [
            torch.nn.Parameter(torch.ones(8, 8, device="cuda", dtype=torch.bfloat16))
            for _ in range(matrix_count)
        ]
        before_optimizer = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        optimizer = GPUStagedMuon(
            [
                {
                    "params": params,
                    "lr": 0.1,
                    "momentum": 0.9,
                    "weight_decay": 0.0,
                }
            ],
            staged_config=_config(slot_numel=64, buffer_count=2),
            orthogonalize=_identity_orthogonalize,
            matmul_precision=nullcontext,
            nesterov=False,
            weight_decay_method="decoupled",
        )
        optimizer.bind_owned_params(optimizer.param_groups)
        observed_optimizer_peak_bytes.append(
            torch.cuda.max_memory_allocated() - before_optimizer
        )
        observed_staging.append(optimizer.gpu_staging_numel)
        assert optimizer.cpu_slabs is not None
        observed_cpu_state.append(
            optimizer.cpu_slabs.master.numel() + optimizer.cpu_slabs.momentum.numel()
        )
        assert optimizer.cuda_state_numel == 0
        del optimizer, params

    assert observed_staging == [2 * 64 * 4, 2 * 64 * 4]
    assert observed_cpu_state == [2 * 64, 2 * 8 * 64]
    assert observed_optimizer_peak_bytes[1] == observed_optimizer_peak_bytes[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_muon_step_matches_reference_across_accumulated_gradients() -> None:
    """Three staged steps with accumulation=2 match the exact owner update."""
    torch.manual_seed(41)
    initial = torch.randn(5, 6, device="cuda", dtype=torch.float32)
    param = torch.nn.Parameter(initial.to(torch.bfloat16))
    optimizer = GPUStagedMuon(
        [{"params": [param], "lr": 0.03, "momentum": 0.8, "weight_decay": 0.02}],
        staged_config=_config(slot_numel=32, buffer_count=1),
        orthogonalize=_identity_orthogonalize,
        matmul_precision=nullcontext,
        nesterov=True,
        weight_decay_method="decoupled",
    )
    optimizer.bind_owned_params(optimizer.param_groups)
    reference_master = param.detach().float().clone()
    reference_momentum = torch.zeros_like(reference_master)

    for step in range(3):
        accumulated = torch.zeros_like(reference_master)
        for accumulation in range(2):
            accumulated.add_(0.01 * (step + 1) + 0.02 * accumulation)
        param.decoupled_grad = accumulated
        reference_master.mul_(1.0 - 0.03 * 0.02)
        reference_momentum.lerp_(accumulated, 0.2)
        reference_update = accumulated.lerp(reference_momentum, 0.8)
        reference_master.add_(reference_update, alpha=-0.03)
        optimizer.step()
        optimizer.drain()

    assert optimizer.cpu_slabs is not None
    torch.testing.assert_close(
        optimizer.cpu_slabs.master.view_as(reference_master),
        reference_master.cpu(),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        optimizer.cpu_slabs.momentum.view_as(reference_momentum),
        reference_momentum.cpu(),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        param.float(), reference_master.to(torch.bfloat16).float(), rtol=0.0, atol=0.0
    )
    assert optimizer.residency == "CPU_RESIDENT"
    assert optimizer.cuda_state_numel == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_muon_slot_reuse_waits_for_d2h_and_drain_commits_cpu_authority() -> None:
    """A single slot is fenced before reuse and remains pending until drain."""
    params = [
        torch.nn.Parameter(torch.ones(8, 8, device="cuda", dtype=torch.bfloat16))
        for _ in range(2)
    ]

    def delayed_identity(
        param: torch.Tensor, update: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        del param, kwargs
        torch.cuda._sleep(50_000_000)
        return update

    optimizer = GPUStagedMuon(
        [{"params": params, "lr": 0.03, "momentum": 0.8, "weight_decay": 0.0}],
        staged_config=_config(slot_numel=64, buffer_count=1),
        orthogonalize=delayed_identity,
        matmul_precision=nullcontext,
        nesterov=False,
        weight_decay_method="decoupled",
    )
    optimizer.bind_owned_params(optimizer.param_groups)
    assert optimizer._slot_machine is not None
    original_wait = optimizer._slot_machine._wait_for_slot
    waited: list[int] = []

    def recorded_wait(slot_index: int) -> None:
        waited.append(slot_index)
        original_wait(slot_index)

    optimizer._slot_machine._wait_for_slot = recorded_wait
    for index, param in enumerate(params):
        param.decoupled_grad = torch.full_like(
            param, 0.01 * (index + 1), dtype=torch.float32
        )

    optimizer.step()

    assert waited == [0]
    assert optimizer._slot_machine.phases == ("D2H_PENDING",)
    assert not optimizer._slots[0].d2h_done.query()
    assert optimizer.residency == "STEP_ACTIVE"
    optimizer.drain()
    assert optimizer._slot_machine.phases == ("FREE",)
    assert optimizer.residency == "CPU_RESIDENT"
    assert optimizer.cpu_slabs is not None
    for unit in optimizer.units:
        state = optimizer.state[unit.param]
        torch.testing.assert_close(
            state["master_param"].to(torch.bfloat16),
            unit.param.cpu(),
            rtol=0.0,
            atol=0.0,
        )
        assert (
            state["master_param"].untyped_storage().data_ptr()
            == optimizer.cpu_slabs.master.untyped_storage().data_ptr()
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_muon_rejects_duplicate_nonmatrix_and_undersized_slot() -> None:
    """Malformed ownership and split-prone capacity fail before state is usable."""
    matrix = torch.nn.Parameter(torch.ones(8, 8, device="cuda", dtype=torch.bfloat16))

    def make(groups, slot_numel=128):
        return GPUStagedMuon(
            groups,
            staged_config=_config(slot_numel=slot_numel),
            orthogonalize=_identity_orthogonalize,
            matmul_precision=nullcontext,
            nesterov=False,
            weight_decay_method="decoupled",
        )

    duplicate = make(
        [{"params": [matrix, matrix], "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}]
    )
    with pytest.raises(ValueError, match="duplicate"):
        duplicate.bind_owned_params(duplicate.param_groups)

    vector = torch.nn.Parameter(torch.ones(8, device="cuda", dtype=torch.bfloat16))
    nonmatrix = make(
        [{"params": [vector], "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}]
    )
    with pytest.raises(ValueError, match="2D matrix"):
        nonmatrix.bind_owned_params(nonmatrix.param_groups)

    undersized = make(
        [{"params": [matrix], "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}],
        slot_numel=63,
    )
    with pytest.raises(ValueError, match="slot is too small"):
        undersized.bind_owned_params(undersized.param_groups)


def test_official_ownership_validation_rejects_duplicate_and_missing_owner() -> None:
    """Official owner lists must map each local leaf parameter exactly once."""
    param = torch.nn.Parameter(torch.ones(2, 2))
    leaf = SimpleNamespace(
        param_groups=[{"params": [param], "is_expert_parallel": False}]
    )
    process_group = SimpleNamespace(rank=lambda: 0, size=lambda: 2)
    pg_collection = SimpleNamespace(dp_cp=process_group, expt_dp=process_group)

    duplicate = SimpleNamespace(
        chained_optimizers=[leaf],
        pg_collection=pg_collection,
        dp_cp_params_list=[[param], [param]],
        expt_dp_params_list=None,
    )
    with pytest.raises(RuntimeError, match="owner lists contain duplicate"):
        _validate_official_ownership(duplicate)

    missing = SimpleNamespace(
        chained_optimizers=[leaf],
        pg_collection=pg_collection,
        dp_cp_params_list=[[], []],
        expt_dp_params_list=None,
    )
    with pytest.raises(RuntimeError, match="does not match owner lists"):
        _validate_official_ownership(missing)

    distributed_group = SimpleNamespace(rank=lambda: 0, size=lambda: 2)
    distributed_without_owner_lists = SimpleNamespace(
        chained_optimizers=[leaf],
        pg_collection=SimpleNamespace(
            dp_cp=distributed_group,
            expt_dp=distributed_group,
        ),
        dp_cp_params_list=None,
        expt_dp_params_list=None,
    )
    param_before = param.detach().clone()
    group_params_before = tuple(leaf.param_groups[0]["params"])
    with pytest.raises(RuntimeError, match="owner list"):
        _validate_official_ownership(distributed_without_owner_lists)
    torch.testing.assert_close(param, param_before, rtol=0.0, atol=0.0)
    assert all(
        actual is expected
        for actual, expected in zip(
            leaf.param_groups[0]["params"], group_params_before, strict=True
        )
    )


def test_official_ownership_validation_accepts_missing_lists_only_for_dp1() -> None:
    """MCore's list elision is legal only for an actual size-one DP group."""
    param = torch.nn.Parameter(torch.ones(2, 2))
    leaf = SimpleNamespace(
        param_groups=[{"params": [param], "is_expert_parallel": False}]
    )
    process_group = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    official = SimpleNamespace(
        chained_optimizers=[leaf],
        pg_collection=SimpleNamespace(
            dp_cp=process_group,
            expt_dp=process_group,
        ),
        dp_cp_params_list=None,
        expt_dp_params_list=None,
    )

    _validate_official_ownership(official, [param])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_muon_checkpoint_roundtrip_preserves_cpu_slab_residency(
    tmp_path: Path,
) -> None:
    """A synchronous Muon load restores both CPU slab state kinds transactionally."""
    param = torch.nn.Parameter(torch.ones(4, 4, device="cuda", dtype=torch.bfloat16))
    optimizer = GPUStagedMuon(
        [{"params": [param], "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}],
        staged_config=GPUStagedMuonConfig(
            buffer_count=1,
            slot_size_mb=1,
            checkpoint_snapshot_root=str(tmp_path),
            checkpoint_snapshot_chunk_mb=1,
        ),
        orthogonalize=_identity_orthogonalize,
        matmul_precision=nullcontext,
        nesterov=False,
        weight_decay_method="decoupled",
    )
    optimizer.bind_owned_params(optimizer.param_groups)
    optimizer.offload_to_cpu()
    optimizer.restore_from_cpu()
    assert optimizer.residency == "CPU_RESIDENT"
    saved = optimizer.state_dict()
    saved = {
        "state": {
            state_id: {key: value.clone() for key, value in state.items()}
            for state_id, state in saved["state"].items()
        },
        "param_groups": [dict(group) for group in saved["param_groups"]],
    }
    optimizer.cpu_slabs.master.fill_(11.0)
    optimizer.cpu_slabs.momentum.fill_(13.0)
    optimizer.begin_checkpoint_load()
    optimizer.load_state_dict(saved)
    optimizer.prepare_checkpoint_load()
    optimizer.commit_checkpoint_load()

    assert optimizer.checkpoint_lifecycle == "CLEAN"
    assert optimizer.residency == "CPU_RESIDENT"
    assert optimizer.cuda_state_numel == 0
    assert optimizer.cpu_slabs.master.is_pinned()
    assert optimizer.cpu_slabs.momentum.is_pinned()
    for key, expected in saved["state"][0].items():
        torch.testing.assert_close(
            optimizer.state[param][key], expected, rtol=0.0, atol=0.0
        )
