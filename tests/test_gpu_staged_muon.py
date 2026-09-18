# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
from types import SimpleNamespace

import pytest
import torch
from megatron.core.optimizer.muon import TensorParallelMuon

from areal.engine.megatron_utils.gpu_staged_muon import (
    GPUStagedMuon,
    GPUStagedMuonConfig,
    _validate_muon_parallel_topology,
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
    )


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
        native_optimizer=TensorParallelMuon(groups, use_decoupled_weight_decay=True),
        staged_config=_config(slot_numel=32),
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
        native_optimizer=TensorParallelMuon(
            [{"params": [matrix], "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}],
            use_decoupled_weight_decay=True,
        ),
        staged_config=_config(),
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
            native_optimizer=TensorParallelMuon(
                [
                    {
                        "params": params,
                        "lr": 0.1,
                        "momentum": 0.9,
                        "weight_decay": 0.0,
                    }
                ],
                use_decoupled_weight_decay=True,
            ),
            staged_config=_config(slot_numel=64, buffer_count=2),
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

    assert observed_staging == [2 * 64 * 3, 2 * 64 * 3]
    assert observed_cpu_state == [2 * 64, 2 * 8 * 64]
    assert observed_optimizer_peak_bytes[1] == observed_optimizer_peak_bytes[0]


@pytest.mark.parametrize("use_nesterov", [False, True])
@pytest.mark.parametrize("schedule_hyperparameters", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_muon_steps_match_official_tensor_parallel_muon(
    use_nesterov: bool,
    schedule_hyperparameters: bool,
) -> None:
    """Staged updates match native Muon with fixed or scheduled settings."""
    torch.manual_seed(20260829)
    initial = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        for shape in ((5, 7), (8, 3))
    ]
    staged_params = [torch.nn.Parameter(value.clone()) for value in initial]
    baseline_params = [torch.nn.Parameter(value.float()) for value in initial]
    kwargs = {
        "lr": 0.03,
        "momentum_beta": 0.8,
        "use_nesterov": use_nesterov,
        "weight_decay": 0.02,
        "use_decoupled_weight_decay": True,
        "fp32_matmul_prec": "highest",
        "coefficient_type": "quintic",
        "num_ns_steps": 3,
        "scale_mode": "spectral",
        "extra_scale_factor": 1.0,
        "mode": "duplicated",
    }
    baseline = TensorParallelMuon(baseline_params, **kwargs)
    staged = GPUStagedMuon(
        [
            {
                "params": staged_params,
                "lr": kwargs["lr"],
                "momentum": kwargs["momentum_beta"],
                "weight_decay": kwargs["weight_decay"],
            }
        ],
        staged_config=_config(slot_numel=64, buffer_count=1),
        weight_decay_method=baseline.weight_decay_method,
        native_optimizer=baseline,
    )
    staged.bind_owned_params(staged.param_groups)

    for step in range(5):
        if schedule_hyperparameters:
            settings = {
                "lr": (0.0, 0.01, 0.03, 0.015, 0.0)[step],
                "weight_decay": (0.02, 0.04, 0.0, 0.01, 0.03)[step],
                "momentum": (0.8, 0.85, 0.9, 0.85, 0.8)[step],
            }
            for group in (*staged.param_groups, *baseline.param_groups):
                group.update(settings)
        for param_index, (staged_param, baseline_param) in enumerate(
            zip(staged_params, baseline_params, strict=True)
        ):
            grad = torch.randn_like(staged_param).mul_(0.1 + step + param_index)
            staged_param.decoupled_grad = grad
            baseline_param.grad = grad.float()
        staged.step()
        baseline.step()
        staged.drain()

        for staged_param, baseline_param in zip(
            staged_params, baseline_params, strict=True
        ):
            staged_state = staged.state[staged_param]
            baseline_state = baseline.state[baseline_param]
            torch.testing.assert_close(
                staged_state["master_param"],
                baseline_param.detach().cpu(),
                rtol=3e-6,
                atol=3e-6,
            )
            torch.testing.assert_close(
                staged_state["momentum_buffer"],
                baseline_state["momentum_buffer"].cpu(),
                rtol=3e-6,
                atol=3e-6,
            )
            torch.testing.assert_close(
                staged_param,
                baseline_param.detach().bfloat16(),
                rtol=0.0,
                atol=0.0,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_muon_checkpoint_resume_matches_official_tensor_parallel_muon() -> None:
    """Muon CPU state resumes on the official optimization trajectory."""
    torch.manual_seed(20260830)
    initial = torch.randn(6, 5, device="cuda", dtype=torch.bfloat16)
    kwargs = {
        "lr": 0.02,
        "momentum_beta": 0.85,
        "use_nesterov": True,
        "weight_decay": 0.03,
        "use_decoupled_weight_decay": True,
        "fp32_matmul_prec": "highest",
        "num_ns_steps": 3,
        "mode": "duplicated",
    }
    baseline_param = torch.nn.Parameter(initial.float())
    baseline = TensorParallelMuon([baseline_param], **kwargs)
    source_param = torch.nn.Parameter(initial.clone())

    def make_staged(param: torch.nn.Parameter) -> GPUStagedMuon:
        optimizer = GPUStagedMuon(
            [
                {
                    "params": [param],
                    "lr": kwargs["lr"],
                    "momentum": kwargs["momentum_beta"],
                    "weight_decay": kwargs["weight_decay"],
                }
            ],
            staged_config=_config(slot_numel=32, buffer_count=1),
            weight_decay_method=baseline.weight_decay_method,
            native_optimizer=baseline,
        )
        optimizer.bind_owned_params(optimizer.param_groups)
        return optimizer

    source = make_staged(source_param)
    for _ in range(3):
        grad = torch.randn_like(source_param)
        source_param.decoupled_grad = grad
        baseline_param.grad = grad.float()
        source.step()
        baseline.step()
        source.drain()

    live_checkpoint = source.state_dict()
    checkpoint = {
        "state": {
            state_id: {key: value.clone() for key, value in state.items()}
            for state_id, state in live_checkpoint["state"].items()
        },
        "param_groups": [dict(group) for group in live_checkpoint["param_groups"]],
    }
    resumed_param = torch.nn.Parameter(source_param.detach().clone())
    resumed = make_staged(resumed_param)
    resumed.begin_checkpoint_load()
    resumed.load_state_dict(checkpoint)
    resumed.complete_checkpoint_load()
    resumed.offload_to_cpu()
    resumed.restore_from_cpu()
    assert resumed.cuda_state_numel == 0
    assert resumed.cpu_slabs.master.is_pinned()
    assert resumed.cpu_slabs.momentum.is_pinned()

    for _ in range(3):
        grad = torch.randn_like(resumed_param)
        resumed_param.decoupled_grad = grad
        baseline_param.grad = grad.float()
        resumed.step()
        baseline.step()
        resumed.drain()

        resumed_state = resumed.state[resumed_param]
        baseline_state = baseline.state[baseline_param]
        torch.testing.assert_close(
            resumed_state["master_param"],
            baseline_param.detach().cpu(),
            rtol=3e-6,
            atol=3e-6,
        )
        torch.testing.assert_close(
            resumed_state["momentum_buffer"],
            baseline_state["momentum_buffer"].cpu(),
            rtol=3e-6,
            atol=3e-6,
        )
        torch.testing.assert_close(
            resumed_param,
            baseline_param.detach().bfloat16(),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_muon_rejects_duplicate_nonmatrix_and_undersized_slot() -> None:
    """Malformed ownership and split-prone capacity fail before state is usable."""
    matrix = torch.nn.Parameter(torch.ones(8, 8, device="cuda", dtype=torch.bfloat16))

    def make(groups, slot_numel=128):
        return GPUStagedMuon(
            groups,
            native_optimizer=TensorParallelMuon(
                groups, use_decoupled_weight_decay=True
            ),
            staged_config=_config(slot_numel=slot_numel),
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


def test_muon_topology_accepts_explicit_expert_partition_metadata() -> None:
    """MCore expert comm keeps the partition axis without the dense TP flag."""
    param = torch.nn.Parameter(torch.ones(2, 2))
    param.expert_tp = True
    param.tensor_model_parallel = False
    param.partition_dim = 0
    param.partition_stride = 1
    singleton = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    pg_collection = SimpleNamespace(
        tp=singleton,
        expt_tp=singleton,
        dp_cp=singleton,
        expt_dp=singleton,
    )
    optimizer = SimpleNamespace(
        pg_collection=pg_collection,
        mode="duplicated",
        param_groups=[{"params": [param], "is_expert_parallel": True}],
    )
    official = SimpleNamespace(pg_collection=pg_collection)

    _validate_muon_parallel_topology(
        official,
        [optimizer],
        [param],
        tp_mode="duplicated",
    )
