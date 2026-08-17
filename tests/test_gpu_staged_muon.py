# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import importlib.util
import json
import os
import signal
import subprocess
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from areal.engine.megatron_utils.gpu_staged_muon import (
    GPUStagedEmptyOptimizer,
    GPUStagedMuon,
    GPUStagedMuonConfig,
    _freeze_owner_schema,
    _has_materialized_parameter_state,
    _make_staged_layerwise_class,
    _validate_muon_parallel_topology,
    _validate_official_ownership,
    get_megatron_optimizer_with_gpu_staged_muon,
)
from areal.engine.megatron_utils.gpu_staged_optimizer import (
    GPUStagedAdamW,
    GPUStagedAdamWConfig,
)
from areal.utils.network import find_free_ports


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


@pytest.mark.parametrize(
    ("owner_lists", "group", "match"),
    [
        ([[torch.nn.Parameter(torch.ones(2, 2))]], (2, 0), "length"),
        ([[], []], (2, 2), "outside"),
    ],
)
def test_official_ownership_validation_rejects_short_list_and_invalid_rank(
    owner_lists,
    group,
    match,
) -> None:
    """Rank coverage and owner-list cardinality are validated explicitly."""
    param = torch.nn.Parameter(torch.ones(2, 2))
    leaf = SimpleNamespace(
        param_groups=[{"params": [param], "is_expert_parallel": False}]
    )
    size, rank = group
    process_group = SimpleNamespace(rank=lambda: rank, size=lambda: size)
    official = SimpleNamespace(
        chained_optimizers=[leaf],
        pg_collection=SimpleNamespace(
            dp_cp=process_group,
            expt_dp=SimpleNamespace(rank=lambda: 0, size=lambda: 1),
        ),
        dp_cp_params_list=owner_lists,
        expt_dp_params_list=None,
    )

    with pytest.raises(RuntimeError, match=match):
        _validate_official_ownership(official, [param])


def test_official_ownership_validation_rejects_extra_parameter() -> None:
    """An owner list cannot silently introduce a parameter outside the model."""
    param = torch.nn.Parameter(torch.ones(2, 2))
    extra = torch.nn.Parameter(torch.zeros(2, 2))
    leaf = SimpleNamespace(
        param_groups=[{"params": [param, extra], "is_expert_parallel": False}]
    )
    process_group = SimpleNamespace(rank=lambda: 0, size=lambda: 2)
    official = SimpleNamespace(
        chained_optimizers=[leaf],
        pg_collection=SimpleNamespace(
            dp_cp=process_group,
            expt_dp=SimpleNamespace(rank=lambda: 0, size=lambda: 1),
        ),
        dp_cp_params_list=[[param, extra], []],
        expt_dp_params_list=None,
    )

    with pytest.raises(RuntimeError, match="extra"):
        _validate_official_ownership(official, [param])


def test_official_ownership_validation_rejects_swapped_dense_expert_leaf_mapping() -> (
    None
):
    """Correct owner schemas cannot hide swapped local dense/expert leaf flags."""
    dense = torch.nn.Parameter(torch.ones(2, 2))
    expert = torch.nn.Parameter(torch.zeros(2, 2))
    expert.allreduce = False
    leaf = SimpleNamespace(
        param_groups=[
            {"params": [dense], "is_expert_parallel": True},
            {"params": [expert], "is_expert_parallel": False},
        ]
    )
    process_group = SimpleNamespace(rank=lambda: 0, size=lambda: 2)
    official = SimpleNamespace(
        chained_optimizers=[leaf],
        pg_collection=SimpleNamespace(
            dp_cp=process_group,
            expt_dp=process_group,
        ),
        dp_cp_params_list=[[dense], []],
        expt_dp_params_list=[[expert], []],
    )

    dense_before = dense.detach().clone()
    expert_before = expert.detach().clone()
    groups_before = tuple(tuple(group["params"]) for group in leaf.param_groups)
    allocated_before = (
        torch.cuda.memory_allocated() if torch.cuda.is_available() else None
    )
    with pytest.raises(RuntimeError) as error:
        _validate_official_ownership(official, [dense, expert])
    message = str(error.value)
    assert "local ownership" in message
    assert "domain=dp_cp" in message
    assert "owner_rank=0" in message
    assert "expected=" in message and "local=" in message
    torch.testing.assert_close(dense, dense_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(expert, expert_before, rtol=0.0, atol=0.0)
    for group, expected_params in zip(leaf.param_groups, groups_before, strict=True):
        assert all(
            actual is expected
            for actual, expected in zip(group["params"], expected_params, strict=True)
        )
    if allocated_before is not None:
        assert torch.cuda.memory_allocated() == allocated_before


def test_official_ownership_validation_rejects_cross_domain_parameter() -> None:
    """A parameter may never be listed in both dense and expert domains."""
    dense = torch.nn.Parameter(torch.ones(2, 2))
    expert = torch.nn.Parameter(torch.zeros(2, 2))
    expert.allreduce = False
    leaf = SimpleNamespace(
        param_groups=[
            {"params": [dense], "is_expert_parallel": False},
            {"params": [expert], "is_expert_parallel": True},
        ]
    )
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 2)
    official = SimpleNamespace(
        chained_optimizers=[leaf],
        pg_collection=SimpleNamespace(dp_cp=group, expt_dp=group),
        dp_cp_params_list=[[dense, expert], []],
        expt_dp_params_list=[[expert], []],
    )

    with pytest.raises(RuntimeError, match="both ownership domains"):
        _validate_official_ownership(official, [dense, expert])


def test_official_ownership_validation_rejects_local_cross_domain_parameter() -> None:
    """Swapped or duplicated local flags cannot merge away a domain error."""
    param = torch.nn.Parameter(torch.ones(2, 2))
    leaf = SimpleNamespace(
        param_groups=[
            {"params": [param], "is_expert_parallel": False},
            {"params": [param], "is_expert_parallel": True},
        ]
    )
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    official = SimpleNamespace(
        chained_optimizers=[leaf],
        pg_collection=SimpleNamespace(dp_cp=group, expt_dp=group),
        dp_cp_params_list=None,
        expt_dp_params_list=None,
    )

    with pytest.raises(RuntimeError, match="local parameter.*both ownership domains"):
        _validate_official_ownership(official, [param])


@pytest.mark.parametrize("missing_domain", ["dp_cp", "expt_dp"])
def test_official_ownership_validation_rejects_missing_domain_parameter(
    missing_domain,
) -> None:
    """Dense and expert schemas close independently, not only as a union."""
    dense = torch.nn.Parameter(torch.ones(2, 2))
    expert = torch.nn.Parameter(torch.zeros(2, 2))
    expert.allreduce = False
    leaf = SimpleNamespace(
        param_groups=[
            {"params": [dense], "is_expert_parallel": False},
            {"params": [expert], "is_expert_parallel": True},
        ]
    )
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 2)
    dense_lists = [[], []] if missing_domain == "dp_cp" else [[dense], []]
    expert_lists = [[], []] if missing_domain == "expt_dp" else [[expert], []]
    official = SimpleNamespace(
        chained_optimizers=[leaf],
        pg_collection=SimpleNamespace(dp_cp=group, expt_dp=group),
        dp_cp_params_list=dense_lists,
        expt_dp_params_list=expert_lists,
    )

    with pytest.raises(RuntimeError, match=f"domain={missing_domain}"):
        _validate_official_ownership(official, [dense, expert])


def test_official_ownership_domains_support_distinct_group_sizes() -> None:
    """Dense and expert owner schemas use their own group rank and cardinality."""
    dense = torch.nn.Parameter(torch.ones(2, 2))
    expert = torch.nn.Parameter(torch.zeros(2, 2))
    expert.allreduce = False
    leaf = SimpleNamespace(
        param_groups=[
            {"params": [dense], "is_expert_parallel": False},
            {"params": [expert], "is_expert_parallel": True},
        ]
    )
    official = SimpleNamespace(
        chained_optimizers=[leaf],
        pg_collection=SimpleNamespace(
            dp_cp=SimpleNamespace(rank=lambda: 0, size=lambda: 2),
            expt_dp=SimpleNamespace(rank=lambda: 0, size=lambda: 3),
        ),
        dp_cp_params_list=[[dense], []],
        expt_dp_params_list=[[expert], [], []],
    )

    _validate_official_ownership(official, [dense, expert])

    official.expt_dp_params_list = [[expert], []]
    with pytest.raises(RuntimeError, match="expt_dp owner list length"):
        _validate_official_ownership(official, [dense, expert])


def test_empty_defaultdict_entries_are_not_materialized_optimizer_state() -> None:
    """Torch optimizers may touch an empty state entry before allocating tensors."""
    param = torch.nn.Parameter(torch.ones(2, 2))
    state = defaultdict(dict)
    state[param]
    optimizer = SimpleNamespace(state=state)

    assert not _has_materialized_parameter_state(optimizer)
    state[param]["momentum_buffer"] = torch.zeros_like(param)
    assert _has_materialized_parameter_state(optimizer)


@pytest.mark.parametrize(
    "overlap_fields",
    [
        ("overlap_param_gather",),
        ("overlap_param_gather_with_optimizer_step",),
        ("overlap_param_gather", "overlap_param_gather_with_optimizer_step"),
    ],
)
def test_staged_muon_overlap_is_rejected_before_model_access(
    overlap_fields,
) -> None:
    """Both unsupported overlap modes fail before the official builder runs."""
    config = SimpleNamespace(
        use_distributed_optimizer=False,
        optimizer="adam",
        decoupled_weight_decay=True,
        bf16=True,
        fp16=False,
        optimizer_cuda_graph=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        use_precision_aware_optimizer=False,
    )
    for overlap_field in overlap_fields:
        setattr(config, overlap_field, True)

    class UntouchedModel:
        @property
        def parameters(self):
            pytest.fail("overlap rejection must happen before model access")

    config_before = vars(config).copy()
    with pytest.raises(ValueError, match="overlap_param_gather"):
        get_megatron_optimizer_with_gpu_staged_muon(
            config,
            [UntouchedModel()],
            GPUStagedMuonConfig(),
        )
    assert vars(config) == config_before


@pytest.mark.parametrize("tp_mode", ["blockwise", "distributed"])
def test_staged_muon_unverified_tp_modes_are_rejected_before_model_access(
    tp_mode: str,
) -> None:
    """Only the duplicated EO 0.3 contract has real multi-GPU evidence."""
    config = SimpleNamespace(
        use_distributed_optimizer=False,
        optimizer="adam",
        decoupled_weight_decay=True,
        bf16=True,
        fp16=False,
        optimizer_cuda_graph=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        use_precision_aware_optimizer=False,
    )

    class UntouchedModel:
        @property
        def parameters(self):
            pytest.fail("TP mode rejection must happen before model access")

    with pytest.raises(ValueError, match="only the verified duplicated TP mode"):
        get_megatron_optimizer_with_gpu_staged_muon(
            config,
            [UntouchedModel()],
            GPUStagedMuonConfig(tp_mode=tp_mode),
        )


@pytest.mark.parametrize(
    ("tp_size", "expt_tp_size"),
    [(2, 1), (1, 2), (2, 2)],
)
def test_staged_muon_factory_rejects_tp_multibuffer_before_model_or_builder_access(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    expt_tp_size: int,
) -> None:
    """Dense TP and expert TP reject multiple slots before any model side effect."""
    import megatron.core.optimizer.muon as muon_module

    config = SimpleNamespace(
        use_distributed_optimizer=False,
        optimizer="adam",
        decoupled_weight_decay=True,
        bf16=True,
        fp16=False,
        optimizer_cuda_graph=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        use_precision_aware_optimizer=False,
    )
    tp_group = SimpleNamespace(size=lambda: tp_size, rank=lambda: 0)
    expt_tp_group = SimpleNamespace(size=lambda: expt_tp_size, rank=lambda: 0)
    pg_collection = SimpleNamespace(
        tp=tp_group,
        expt_tp=expt_tp_group,
    )
    monkeypatch.setattr(
        muon_module,
        "get_megatron_muon_optimizer",
        lambda *args, **kwargs: pytest.fail(
            "TP buffer rejection must happen before the official builder"
        ),
    )

    class UntouchedModel:
        def parameters(self):
            pytest.fail("TP buffer rejection must happen before model access")

    config_before = vars(config).copy()
    staged_config = GPUStagedMuonConfig(buffer_count=2)
    allocated_before = (
        torch.cuda.memory_allocated() if torch.cuda.is_available() else None
    )
    with pytest.raises(
        ValueError,
        match=(rf"tp_size={tp_size}, expt_tp_size={expt_tp_size}, buffer_count=2"),
    ):
        get_megatron_optimizer_with_gpu_staged_muon(
            config,
            [UntouchedModel()],
            staged_config,
            pg_collection=pg_collection,
        )
    assert vars(config) == config_before
    assert staged_config.buffer_count == 2
    if allocated_before is not None:
        assert torch.cuda.memory_allocated() == allocated_before


def test_staged_muon_factory_allows_dp_only_multibuffer_past_early_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A singleton TP/ETP topology keeps the existing DP-only multi-slot path."""
    import megatron.core.optimizer as optimizer_module

    config = SimpleNamespace(
        use_distributed_optimizer=False,
        optimizer="adam",
        decoupled_weight_decay=True,
        bf16=True,
        fp16=False,
        optimizer_cuda_graph=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        use_precision_aware_optimizer=False,
    )
    singleton_group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    pg_collection = SimpleNamespace(tp=singleton_group, expt_tp=singleton_group)
    monkeypatch.setattr(optimizer_module, "HAVE_EMERGING_OPTIMIZERS", False)

    with pytest.raises(ImportError, match="emerging-optimizers backend"):
        get_megatron_optimizer_with_gpu_staged_muon(
            config,
            [],
            GPUStagedMuonConfig(buffer_count=2),
            pg_collection=pg_collection,
        )


def test_staged_muon_all_empty_owner_domain_skips_collective(monkeypatch) -> None:
    """A rank-indexed all-empty domain performs no communication or allocation."""
    wrapper_cls = _make_staged_layerwise_class()
    wrapper = object.__new__(wrapper_cls)
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    wrapper.pg_collection = SimpleNamespace(dp_cp=group, expt_dp=group)
    wrapper.dp_cp_params_list = [[], []]
    wrapper.expt_dp_params_list = None
    wrapper._staged_owner_schema = {
        "dense": _freeze_owner_schema("dense", wrapper.dp_cp_params_list),
        "expert": None,
    }
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather",
        lambda *args, **kwargs: pytest.fail("all-empty domain must not communicate"),
    )

    wrapper.allgather_params()


def test_staged_muon_zero_numel_owner_domain_skips_collective(monkeypatch) -> None:
    """A schema containing only zero-numel tensors preserves MCore's early exit."""
    wrapper_cls = _make_staged_layerwise_class()
    wrapper = object.__new__(wrapper_cls)
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    empty = torch.nn.Parameter(torch.empty(0))
    wrapper.pg_collection = SimpleNamespace(dp_cp=group, expt_dp=group)
    wrapper.dp_cp_params_list = [[empty], []]
    wrapper.expt_dp_params_list = None
    wrapper._staged_owner_schema = {
        "dense": _freeze_owner_schema("dense", wrapper.dp_cp_params_list),
        "expert": None,
    }
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather",
        lambda *args, **kwargs: pytest.fail("zero-numel domain must not communicate"),
    )

    wrapper.allgather_params()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_staged_muon_owner_domain_rejects_mixed_dtype_before_collective(
    monkeypatch,
) -> None:
    """One flattened domain cannot safely mix CUDA tensor dtypes."""
    wrapper_cls = _make_staged_layerwise_class()
    wrapper = object.__new__(wrapper_cls)
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    fp16 = torch.nn.Parameter(torch.ones(2, device="cuda", dtype=torch.float16))
    bf16 = torch.nn.Parameter(torch.ones(2, device="cuda", dtype=torch.bfloat16))
    wrapper.pg_collection = SimpleNamespace(dp_cp=group, expt_dp=group)
    wrapper.dp_cp_params_list = [[fp16, bf16]]
    wrapper.expt_dp_params_list = None
    wrapper._staged_owner_schema = {
        "dense": _freeze_owner_schema("dense", wrapper.dp_cp_params_list),
        "expert": None,
    }
    monkeypatch.setattr(
        torch.distributed,
        "all_gather",
        lambda *args, **kwargs: pytest.fail("invalid domain must not communicate"),
    )

    with pytest.raises(RuntimeError, match="inconsistent parameter device or dtype"):
        wrapper.allgather_params()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_staged_muon_owner_domain_rejects_bound_metadata_drift_before_collective(
    monkeypatch,
) -> None:
    """A bound tensor cannot change shape/numel without invalidating the schema."""
    wrapper_cls = _make_staged_layerwise_class()
    wrapper = object.__new__(wrapper_cls)
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    param = torch.nn.Parameter(torch.ones(2, device="cuda", dtype=torch.float32))
    wrapper.pg_collection = SimpleNamespace(dp_cp=group, expt_dp=group)
    wrapper.dp_cp_params_list = [[param]]
    wrapper.expt_dp_params_list = None
    wrapper._staged_owner_schema = {
        "dense": _freeze_owner_schema("dense", wrapper.dp_cp_params_list),
        "expert": None,
    }
    param.data = torch.ones(3, device="cuda", dtype=torch.float32)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather",
        lambda *args, **kwargs: pytest.fail(
            "metadata drift must be rejected before collective"
        ),
    )

    with pytest.raises(RuntimeError, match="metadata changed after bind"):
        wrapper.allgather_params()


@pytest.mark.parametrize("mutation", ["shape", "stride", "dtype", "storage", "order"])
def test_staged_muon_public_step_rejects_owner_metadata_drift_before_mutation(
    mutation: str,
) -> None:
    """The public step validates frozen owner authority before Chained step."""
    wrapper_cls = _make_staged_layerwise_class()
    wrapper = object.__new__(wrapper_cls)
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    first = torch.nn.Parameter(
        torch.ones((2, 2) if mutation == "stride" else 2, dtype=torch.float32)
    )
    second = torch.nn.Parameter(torch.full((2,), 2.0, dtype=torch.float32))
    wrapper.pg_collection = SimpleNamespace(dp_cp=group, expt_dp=group)
    wrapper.dp_cp_params_list = [[first, second]]
    wrapper.expt_dp_params_list = None
    wrapper._staged_owner_schema = {
        "dense": _freeze_owner_schema("dense", wrapper.dp_cp_params_list),
        "expert": None,
    }
    untouched = second.detach().clone()
    if mutation == "shape":
        first.data = torch.ones(3, dtype=torch.float32)
    elif mutation == "stride":
        first.data = first.data.t()
    elif mutation == "dtype":
        first.data = torch.ones(2, dtype=torch.float64)
    elif mutation == "storage":
        first.data = first.detach().clone()
    elif mutation == "order":
        wrapper.dp_cp_params_list[0][:] = [second, first]
    else:
        raise AssertionError(f"unknown mutation {mutation}")

    with pytest.raises(RuntimeError, match="after bind"):
        wrapper.step()
    torch.testing.assert_close(second, untouched, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_staged_muon_public_step_rejects_owner_device_drift_before_mutation() -> None:
    """Moving a bound owner tensor cannot redirect the all-gather destination."""
    wrapper_cls = _make_staged_layerwise_class()
    wrapper = object.__new__(wrapper_cls)
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    param = torch.nn.Parameter(torch.ones(2, device="cuda", dtype=torch.float32))
    wrapper.pg_collection = SimpleNamespace(dp_cp=group, expt_dp=group)
    wrapper.dp_cp_params_list = [[param]]
    wrapper.expt_dp_params_list = None
    wrapper._staged_owner_schema = {
        "dense": _freeze_owner_schema("dense", wrapper.dp_cp_params_list),
        "expert": None,
    }
    param.data = torch.ones(2, device="cpu", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="metadata changed after bind"):
        wrapper.step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_staged_muon_owner_domain_rejects_storage_pointer_drift() -> None:
    """A storage reallocation cannot retain stale all-gather authority."""
    wrapper_cls = _make_staged_layerwise_class()
    wrapper = object.__new__(wrapper_cls)
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    param = torch.nn.Parameter(torch.arange(4, device="cuda", dtype=torch.float32))
    owner_lists = [[param]]
    wrapper._staged_owner_schema = {
        "dense": _freeze_owner_schema("dense", owner_lists),
        "expert": None,
    }

    storage = param.untyped_storage()
    original_pointer = storage.data_ptr()
    original_cdata = int(storage._cdata)
    storage.resize_(8 * 1024 * 1024)
    assert storage.data_ptr() != original_pointer
    assert param.untyped_storage() is storage
    assert int(storage._cdata) == original_cdata

    _, _, _, error = wrapper._validate_owner_domain(
        domain="dense", params_list=owner_lists, group=group
    )

    assert error is not None
    assert "metadata changed after bind" in error
    assert "storage_nbytes" in error
    assert "storage_data_ptr" in error
    assert "param_data_ptr" in error


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_staged_muon_owner_domain_rejects_param_data_pointer_drift() -> None:
    """A shifted view is rejected even when its storage base and capacity match."""
    wrapper_cls = _make_staged_layerwise_class()
    wrapper = object.__new__(wrapper_cls)
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    base = torch.arange(8, device="cuda", dtype=torch.float32)
    param = torch.nn.Parameter(base[:4])
    owner_lists = [[param]]
    wrapper._staged_owner_schema = {
        "dense": _freeze_owner_schema("dense", owner_lists),
        "expert": None,
    }
    storage = param.untyped_storage()
    storage_pointer = storage.data_ptr()
    storage_nbytes = storage.nbytes()
    param.data = base[1:5]
    assert param.untyped_storage() is storage
    assert param.untyped_storage().data_ptr() == storage_pointer
    assert param.untyped_storage().nbytes() == storage_nbytes

    _, _, _, error = wrapper._validate_owner_domain(
        domain="dense", params_list=owner_lists, group=group
    )

    assert error is not None
    assert "param_data_ptr" in error
    assert "storage_offset" in error


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_staged_muon_owner_domain_rejects_restored_capacity_with_new_pointer() -> None:
    """Restoring nbytes cannot legitimize a storage whose allocation moved."""
    wrapper_cls = _make_staged_layerwise_class()
    wrapper = object.__new__(wrapper_cls)
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    param = torch.nn.Parameter(torch.arange(4, device="cuda", dtype=torch.float32))
    owner_lists = [[param]]
    wrapper._staged_owner_schema = {
        "dense": _freeze_owner_schema("dense", owner_lists),
        "expert": None,
    }
    storage = param.untyped_storage()
    original_pointer = storage.data_ptr()
    original_nbytes = storage.nbytes()
    storage.resize_(8 * 1024 * 1024)
    blocker = torch.empty(original_nbytes // 4, device="cuda", dtype=torch.float32)
    assert blocker.data_ptr() == original_pointer
    storage.resize_(original_nbytes)
    assert storage.nbytes() == original_nbytes
    assert storage.data_ptr() != original_pointer

    _, _, _, error = wrapper._validate_owner_domain(
        domain="dense", params_list=owner_lists, group=group
    )

    assert error is not None
    assert "storage_data_ptr" in error
    assert "param_data_ptr" in error


def test_staged_muon_owner_domain_length_mismatch_fails_closed(monkeypatch) -> None:
    """Owner schemas cannot be silently zipped against a different group size."""
    wrapper_cls = _make_staged_layerwise_class()
    wrapper = object.__new__(wrapper_cls)
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    wrapper.pg_collection = SimpleNamespace(dp_cp=group, expt_dp=group)
    wrapper.dp_cp_params_list = [[]]
    wrapper.expt_dp_params_list = None
    wrapper._staged_owner_schema = {
        "dense": _freeze_owner_schema("dense", wrapper.dp_cp_params_list),
        "expert": None,
    }
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="length 1 does not match group size 2"):
        wrapper.allgather_params()


@pytest.mark.skipif(
    importlib.util.find_spec("emerging_optimizers") is not None,
    reason="missing-dependency fail-closed path requires an isolated base environment",
)
def test_staged_muon_factory_missing_dependency_is_mutation_free() -> None:
    """A missing official backend fails before model or config mutation."""
    config = SimpleNamespace(
        use_distributed_optimizer=False,
        optimizer="adam",
        decoupled_weight_decay=True,
        bf16=True,
        fp16=False,
        optimizer_cuda_graph=False,
        overlap_param_gather=False,
        use_precision_aware_optimizer=False,
    )
    model = torch.nn.Linear(2, 2, bias=False)
    model_before = model.weight.detach().clone()
    params_before = tuple(model.parameters())
    config_before = vars(config).copy()
    allocated_before = (
        torch.cuda.memory_allocated() if torch.cuda.is_available() else None
    )

    with pytest.raises(ImportError, match="emerging-optimizers backend"):
        get_megatron_optimizer_with_gpu_staged_muon(
            config,
            [model],
            GPUStagedMuonConfig(),
        )

    torch.testing.assert_close(model.weight, model_before, rtol=0.0, atol=0.0)
    assert all(
        actual is expected
        for actual, expected in zip(model.parameters(), params_before, strict=True)
    )
    assert vars(config) == config_before
    assert not hasattr(model.weight, "decoupled_grad")
    if allocated_before is not None:
        assert torch.cuda.memory_allocated() == allocated_before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_staged_muon_factory_rejects_unsupported_official_wrapper(
    monkeypatch,
) -> None:
    """The instance-level adapter fails closed on unknown MCore leaf wrappers."""
    import megatron.core.optimizer as optimizer_module
    import megatron.core.optimizer.muon as muon_module
    from megatron.core.optimizer.layer_wise_optimizer import (
        LayerWiseDistributedOptimizer,
    )

    param = torch.nn.Parameter(torch.ones(2, 2, device="cuda"))
    leaf = SimpleNamespace(
        param_groups=[{"params": [param], "is_expert_parallel": False}]
    )
    process_group = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    official = object.__new__(LayerWiseDistributedOptimizer)
    official.chained_optimizers = [leaf]
    official.pg_collection = SimpleNamespace(
        dp_cp=process_group,
        expt_dp=process_group,
    )
    official.dp_cp_params_list = None
    official.expt_dp_params_list = None
    official.async_allgather = False
    monkeypatch.setattr(optimizer_module, "HAVE_EMERGING_OPTIMIZERS", True)
    monkeypatch.setattr(
        "areal.engine.megatron_utils.gpu_staged_muon.importlib.metadata.version",
        lambda package: "0.3.0" if package == "emerging-optimizers" else "0.17.0",
    )

    def fake_builder(
        config,
        model_chunks,
        config_overrides=None,
        use_gloo_process_groups=True,
        layer_wise_distributed_optimizer=False,
        pg_collection=None,
    ):
        del (
            config,
            model_chunks,
            config_overrides,
            use_gloo_process_groups,
            layer_wise_distributed_optimizer,
            pg_collection,
        )
        return official

    monkeypatch.setattr(muon_module, "get_megatron_muon_optimizer", fake_builder)
    config = SimpleNamespace(
        optimizer="adam",
        decoupled_weight_decay=True,
        use_distributed_optimizer=False,
        bf16=True,
        fp16=False,
        optimizer_cuda_graph=False,
        overlap_param_gather=False,
        use_precision_aware_optimizer=False,
    )
    model = SimpleNamespace(parameters=lambda: iter([param]))

    with pytest.raises(TypeError, match="unsupported official Muon wrapper"):
        get_megatron_optimizer_with_gpu_staged_muon(
            config,
            [model],
            _config(),
        )


def _topology_parameter(*, expert_tp: bool = False) -> torch.nn.Parameter:
    param = torch.nn.Parameter(torch.ones(4, 4))
    param.expert_tp = expert_tp
    param.tensor_model_parallel = True
    param.partition_dim = 0
    param.partition_stride = 1
    param.allreduce = not expert_tp
    return param


def _topology_group() -> SimpleNamespace:
    return SimpleNamespace(size=lambda: 1, rank=lambda: 0)


@pytest.mark.parametrize("missing_group", ["tp", "expt_tp", "dp_cp", "expt_dp"])
def test_muon_topology_preflight_rejects_missing_process_group(
    missing_group: str,
) -> None:
    """Every official TP and ownership group is mandatory before allocation."""
    param = _topology_parameter()
    groups = {
        name: _topology_group()
        for name in ("tp", "expt_tp", "dp_cp", "expt_dp")
        if name != missing_group
    }
    pg_collection = SimpleNamespace(**groups)
    leaf = SimpleNamespace(
        pg_collection=pg_collection,
        mode="duplicated",
        param_groups=[{"params": [param], "is_expert_parallel": False}],
    )
    official = SimpleNamespace(pg_collection=pg_collection)

    with pytest.raises(RuntimeError, match=f"missing process group {missing_group}"):
        _validate_muon_parallel_topology(
            official, [leaf], [param], tp_mode="duplicated"
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expert_tp", "yes", "invalid expert_tp"),
        ("tensor_model_parallel", False, "metadata is inconsistent"),
        ("partition_dim", 2, "invalid TP partition_dim"),
        ("partition_stride", 0, "invalid TP partition_stride"),
    ],
)
def test_muon_topology_preflight_rejects_corrupt_tp_metadata(
    field: str, value: object, message: str
) -> None:
    """Corrupt TP metadata fails before staged state or slots can be created."""
    param = _topology_parameter()
    setattr(param, field, value)
    pg_collection = SimpleNamespace(
        tp=_topology_group(),
        expt_tp=_topology_group(),
        dp_cp=_topology_group(),
        expt_dp=_topology_group(),
    )
    leaf = SimpleNamespace(
        pg_collection=pg_collection,
        mode="duplicated",
        param_groups=[{"params": [param], "is_expert_parallel": False}],
    )

    with pytest.raises(RuntimeError, match=message):
        _validate_muon_parallel_topology(
            SimpleNamespace(pg_collection=pg_collection),
            [leaf],
            [param],
            tp_mode="duplicated",
        )


def test_muon_topology_keeps_tp_and_dp_expert_domains_independent() -> None:
    """TP routing uses expert_tp, while owner routing remains group metadata."""
    param = _topology_parameter(expert_tp=True)
    param.allreduce = True
    pg_collection = SimpleNamespace(
        tp=_topology_group(),
        expt_tp=_topology_group(),
        dp_cp=_topology_group(),
        expt_dp=_topology_group(),
    )
    leaf = SimpleNamespace(
        pg_collection=pg_collection,
        mode="duplicated",
        param_groups=[{"params": [param], "is_expert_parallel": False}],
    )

    _validate_muon_parallel_topology(
        SimpleNamespace(pg_collection=pg_collection),
        [leaf],
        [param],
        tp_mode="duplicated",
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("collection", "different process-group collections"),
        ("mode", "TP mode does not match"),
    ],
)
def test_muon_topology_preflight_rejects_official_wrapper_mismatch(
    change: str, message: str
) -> None:
    """The adapter cannot reinterpret a wrapper built for another topology."""
    param = _topology_parameter()
    pg_collection = SimpleNamespace(
        tp=_topology_group(),
        expt_tp=_topology_group(),
        dp_cp=_topology_group(),
        expt_dp=_topology_group(),
    )
    leaf = SimpleNamespace(
        pg_collection=(SimpleNamespace() if change == "collection" else pg_collection),
        mode=("distributed" if change == "mode" else "duplicated"),
        param_groups=[{"params": [param], "is_expert_parallel": False}],
    )

    with pytest.raises(RuntimeError, match=message):
        _validate_muon_parallel_topology(
            SimpleNamespace(pg_collection=pg_collection),
            [leaf],
            [param],
            tp_mode="duplicated",
        )


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


class _TinyOfficialMuonModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(
            8, 8, bias=True, device="cuda", dtype=torch.bfloat16
        )
        self.norm = torch.nn.LayerNorm(8, device="cuda", dtype=torch.bfloat16)
        self.embedding = torch.nn.Parameter(
            torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
        )
        self.embedding.is_embedding_or_output_parameter = True
        self.config = SimpleNamespace(
            num_attention_heads=1,
            num_query_groups=1,
            kv_channels=8,
        )
        from megatron.core.distributed import DistributedDataParallelConfig

        self.ddp_config = DistributedDataParallelConfig()


class _OnlyMuonModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = torch.nn.Parameter(
            torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
        )
        self.config = SimpleNamespace(
            num_attention_heads=1,
            num_query_groups=1,
            kv_channels=4,
        )
        from megatron.core.distributed import DistributedDataParallelConfig

        self.ddp_config = DistributedDataParallelConfig()


class _OnlyScalarModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Parameter(
            torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
        )
        self.embedding.is_embedding_or_output_parameter = True
        self.bias = torch.nn.Parameter(
            torch.randn(4, device="cuda", dtype=torch.bfloat16)
        )
        self.config = SimpleNamespace(
            num_attention_heads=1,
            num_query_groups=1,
            kv_channels=4,
        )
        from megatron.core.distributed import DistributedDataParallelConfig

        self.ddp_config = DistributedDataParallelConfig()


def _official_muon_config():
    from megatron.core.optimizer import OptimizerConfig

    return OptimizerConfig(
        optimizer="adam",
        lr=2e-3,
        min_lr=0.0,
        weight_decay=0.01,
        bf16=True,
        use_distributed_optimizer=False,
        main_grads_dtype=torch.float32,
        main_params_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        muon_split_qkv=False,
        muon_fp32_matmul_prec="highest",
        muon_num_ns_steps=3,
        muon_use_nesterov=True,
    )


def _official_staged_config() -> GPUStagedMuonConfig:
    return GPUStagedMuonConfig(
        buffer_count=1,
        slot_size_mb=1,
        split_qkv=False,
        fp32_matmul_prec="highest",
        num_ns_steps=3,
        use_nesterov=True,
    )


def _assert_empty_leaf_is_managed(optimizer, optimizer_kind: str) -> None:
    empty_leaves = [
        leaf
        for leaf in optimizer.chained_optimizers
        if getattr(leaf.optimizer, "optimizer_kind", None) == optimizer_kind
        and not leaf.param_groups
    ]
    assert len(empty_leaves) == 1
    inner = empty_leaves[0].optimizer
    assert isinstance(inner, GPUStagedEmptyOptimizer)
    assert inner.cpu_slabs is None
    assert inner.units == ()
    assert inner.gpu_staging_state_numel == 0
    assert not inner._slots
    assert inner.cuda_state_numel == 0
    assert inner.residency == "CPU_RESIDENT"
    assert not inner.state
    assert inner.step() is None
    inner.drain()
    inner.offload_to_cpu()
    inner.restore_from_cpu()
    assert inner.state_dict() == {"state": {}, "param_groups": []}


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_official_mcore_pure_scalar_empty_muon_leaf_order_and_only_muon_dp1(
    tmp_path,
) -> None:
    """Official DP=1 edge classifications remain trainable for three steps."""
    pytest.importorskip("emerging_optimizers")
    import torch.distributed as dist
    from megatron.core import parallel_state

    if dist.is_initialized():
        pytest.skip("test requires ownership of the process-global DP=1 group")
    dist.init_process_group(
        "nccl",
        init_method=f"file://{tmp_path / 'dp1_empty_leaf_pg'}",
        rank=0,
        world_size=1,
    )
    torch.cuda.set_device(0)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    try:
        for model_cls, expected_by_kind in (
            (_OnlyMuonModel, {"muon": {"matrix"}, "scalar_adamw": set()}),
            (
                _OnlyScalarModel,
                {"muon": set(), "scalar_adamw": {"embedding", "bias"}},
            ),
        ):
            model = model_cls()
            optimizer = get_megatron_optimizer_with_gpu_staged_muon(
                _official_muon_config(),
                [model],
                _official_staged_config(),
            )
            names = {id(param): name for name, param in model.named_parameters()}
            actual_by_kind = {
                kind: {
                    names[id(param)]
                    for leaf in optimizer.chained_optimizers
                    if getattr(leaf.optimizer, "optimizer_kind", None) == kind
                    for group in leaf.param_groups
                    for param in group["params"]
                }
                for kind in ("muon", "scalar_adamw")
            }
            assert actual_by_kind == expected_by_kind
            if model_cls is _OnlyMuonModel:
                _assert_empty_leaf_is_managed(optimizer, "scalar_adamw")
            else:
                _assert_empty_leaf_is_managed(optimizer, "muon")
            assert [
                leaf.optimizer.optimizer_kind for leaf in optimizer.chained_optimizers
            ] == ["muon", "scalar_adamw"]

            nondefault_stream = torch.cuda.Stream()
            for step in range(3):
                for param_index, param in enumerate(model.parameters()):
                    accumulated = torch.zeros_like(param, dtype=torch.float32)
                    for accumulation in range(2):
                        accumulated.add_(
                            0.005 * (step + 1)
                            + 0.003 * accumulation
                            + 0.0001 * param_index
                        )
                    param.main_grad = accumulated
                if step == 1:
                    with torch.cuda.stream(nondefault_stream):
                        optimizer.step()
                    nondefault_stream.synchronize()
                else:
                    optimizer.step()
                optimizer.drain()
                assert optimizer.residency == "CPU_RESIDENT"
                assert optimizer.cuda_state_numel == 0

            scalar_groups = [
                group
                for leaf in optimizer.chained_optimizers
                if getattr(leaf.optimizer, "optimizer_kind", None) == "scalar_adamw"
                for group in leaf.param_groups
                if group["params"]
            ]
            if model_cls is _OnlyScalarModel:
                _assert_empty_leaf_is_managed(optimizer, "muon")
                assert scalar_groups
                assert all(group["step"] == 3 for group in scalar_groups)
            else:
                _assert_empty_leaf_is_managed(optimizer, "scalar_adamw")
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_official_mcore_muon_dp1_matches_staged_for_three_steps(tmp_path) -> None:
    """Official classification and DP=1 baseline match all staged state."""
    pytest.importorskip("emerging_optimizers")
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.optimizer.muon import get_megatron_muon_optimizer

    if dist.is_initialized():
        pytest.skip("test requires ownership of the process-global DP=1 group")
    dist.init_process_group(
        "nccl",
        init_method=f"file://{tmp_path / 'dp1_pg'}",
        rank=0,
        world_size=1,
    )
    torch.cuda.set_device(0)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    try:
        torch.manual_seed(7)
        staged_model = _TinyOfficialMuonModel()
        baseline_model = _TinyOfficialMuonModel()
        baseline_model.load_state_dict(staged_model.state_dict())

        staged = get_megatron_optimizer_with_gpu_staged_muon(
            _official_muon_config(),
            [staged_model],
            GPUStagedMuonConfig(
                buffer_count=1,
                slot_size_mb=1,
                split_qkv=False,
                fp32_matmul_prec="highest",
                num_ns_steps=3,
                use_nesterov=True,
            ),
        )
        baseline = get_megatron_muon_optimizer(
            _official_muon_config(),
            [baseline_model],
            use_gloo_process_groups=False,
            layer_wise_distributed_optimizer=True,
        )
        names_by_id = {
            id(param): name for name, param in staged_model.named_parameters()
        }
        classified = {
            kind: {
                names_by_id[id(param)]
                for leaf in staged.chained_optimizers
                if getattr(leaf.optimizer, "optimizer_kind", None) == kind
                for group in leaf.param_groups
                for param in group["params"]
            }
            for kind in ("muon", "scalar_adamw")
        }
        assert classified["muon"] == {"linear.weight"}
        assert classified["scalar_adamw"] == {
            "embedding",
            "linear.bias",
            "norm.weight",
            "norm.bias",
        }

        for step in range(3):
            for staged_param, baseline_param in zip(
                staged_model.parameters(), baseline_model.parameters(), strict=True
            ):
                accumulated = torch.zeros_like(staged_param, dtype=torch.float32)
                for accumulation in range(2):
                    accumulated.add_(0.005 * (step + 1) + 0.003 * accumulation)
                staged_param.main_grad = accumulated.clone()
                baseline_param.main_grad = accumulated.clone()
            staged.step()
            baseline.step()
            staged.drain()

        for staged_param, baseline_param in zip(
            staged_model.parameters(), baseline_model.parameters(), strict=True
        ):
            torch.testing.assert_close(
                staged_param,
                baseline_param,
                rtol=2e-3,
                atol=2e-3,
            )

        max_errors = {
            "model": 0.0,
            "master": 0.0,
            "momentum": 0.0,
            "exp_avg": 0.0,
            "exp_avg_sq": 0.0,
        }
        for staged_leaf, baseline_leaf in zip(
            staged.chained_optimizers,
            baseline.chained_optimizers,
            strict=True,
        ):
            for staged_group, baseline_group, model_group, main_group in zip(
                staged_leaf.param_groups,
                baseline_leaf.param_groups,
                baseline_leaf.float16_groups,
                baseline_leaf.fp32_from_float16_groups,
                strict=True,
            ):
                assert staged_group["lr"] == baseline_group["lr"]
                if (
                    getattr(staged_leaf.optimizer, "optimizer_kind", None)
                    == "scalar_adamw"
                ):
                    assert staged_group["step"] == baseline_group["step"] == 3
                for staged_param, baseline_param, baseline_main in zip(
                    staged_group["params"], model_group, main_group, strict=True
                ):
                    staged_state = staged_leaf.state[staged_param]
                    baseline_state = baseline_leaf.state[baseline_main]
                    max_errors["model"] = max(
                        max_errors["model"],
                        (staged_param.float() - baseline_param.float())
                        .abs()
                        .max()
                        .item(),
                    )
                    max_errors["master"] = max(
                        max_errors["master"],
                        (staged_state["master_param"].cuda() - baseline_main)
                        .abs()
                        .max()
                        .item(),
                    )
                    if "momentum_buffer" in staged_state:
                        max_errors["momentum"] = max(
                            max_errors["momentum"],
                            (
                                staged_state["momentum_buffer"].cuda()
                                - baseline_state["momentum_buffer"]
                            )
                            .abs()
                            .max()
                            .item(),
                        )
                    else:
                        for key in ("exp_avg", "exp_avg_sq"):
                            max_errors[key] = max(
                                max_errors[key],
                                (staged_state[key].cuda() - baseline_state[key])
                                .abs()
                                .max()
                                .item(),
                            )
        assert max_errors["model"] <= 2e-3
        assert max_errors["master"] <= 2e-3
        assert max_errors["momentum"] <= 2e-3
        assert max_errors["exp_avg"] <= 2e-6
        assert max_errors["exp_avg_sq"] <= 2e-6
        assert staged.residency == "CPU_RESIDENT"
        assert staged.cuda_state_numel == 0
        (tmp_path / "dp1_errors.json").write_text(
            json.dumps(max_errors, sort_keys=True) + "\n"
        )

        for model_cls, expected_by_kind in (
            (_OnlyMuonModel, {"muon": {"matrix"}, "scalar_adamw": set()}),
            (
                _OnlyScalarModel,
                {"muon": set(), "scalar_adamw": {"embedding", "bias"}},
            ),
        ):
            owner_model = model_cls()
            owner_optimizer = get_megatron_optimizer_with_gpu_staged_muon(
                _official_muon_config(),
                [owner_model],
                GPUStagedMuonConfig(
                    buffer_count=1,
                    slot_size_mb=1,
                    split_qkv=False,
                    fp32_matmul_prec="highest",
                    num_ns_steps=3,
                    use_nesterov=True,
                ),
            )
            owner_names = {
                id(param): name for name, param in owner_model.named_parameters()
            }
            actual_by_kind = {
                kind: {
                    owner_names[id(param)]
                    for leaf in owner_optimizer.chained_optimizers
                    if getattr(leaf.optimizer, "optimizer_kind", None) == kind
                    for group in leaf.param_groups
                    for param in group["params"]
                }
                for kind in ("muon", "scalar_adamw")
            }
            assert actual_by_kind == expected_by_kind
            empty_kind = "scalar_adamw" if model_cls is _OnlyMuonModel else "muon"
            _assert_empty_leaf_is_managed(owner_optimizer, empty_kind)
            assert [
                leaf.optimizer.optimizer_kind
                for leaf in owner_optimizer.chained_optimizers
            ] == ["muon", "scalar_adamw"]
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_official_mcore_muon_dp2_owner_update_and_sync(tmp_path: Path) -> None:
    """Real NCCL verifies one owner per parameter and official all-gather."""
    if torch.cuda.device_count() < 2:
        pytest.skip("real staged Muon DP=2 requires two CUDA devices")
    if importlib.util.find_spec("emerging_optimizers") is None:
        pytest.skip("MCore's optional emerging-optimizers backend is unavailable")

    output_dir = tmp_path / "muon-dp2-results"
    env = os.environ.copy()
    env["ACCEPTANCE_OUTPUT_DIR"] = str(output_dir)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--nnodes=1",
        "--master_addr=localhost",
        f"--master_port={find_free_ports(1)[0]}",
        "tests/torchrun/run_gpu_staged_muon_mcore.py",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    results = [
        json.loads((output_dir / f"rank_{rank}.json").read_text()) for rank in range(2)
    ]
    assert {result["residency"] for result in results} == {"CPU_RESIDENT"}
    assert {result["cuda_state_numel"] for result in results} == {0}
    assert {result["steps"] for result in results} == {3}
    assert {result["accumulation"] for result in results} == {2}
    assert {result["nondefault_stream_checked"] for result in results} == {True}
    assert all(set(result["owner_counts"].values()) == {1} for result in results)
    assert sorted(result["muon_cpu_slab_numel"] for result in results) == [0, 64]
    assert sorted(result["muon_unit_numels"] for result in results) == [[], [64]]
    for result in results:
        assert result["only_muon"]["residency"] == "CPU_RESIDENT"
        assert result["only_muon"]["cuda_state_numel"] == 0
        assert result["only_muon"]["empty_scalar_leaves"] == 1
        assert result["only_muon"]["empty_muon_leaves"] == (result["rank"] == 1)
        assert result["only_muon"]["leaf_order"] == ["muon", "scalar_adamw"]
        assert result["only_scalar"]["residency"] == "CPU_RESIDENT"
        assert result["only_scalar"]["cuda_state_numel"] == 0
        assert result["only_scalar"]["empty_scalar_leaves"] == 0
        assert result["only_scalar"]["empty_muon_leaves"] == 1
        assert result["only_scalar"]["leaf_order"] == ["muon", "scalar_adamw"]
        errors = result["max_errors"]
        assert errors["model"] <= 2e-3
        assert errors["master"] <= 2e-3
        assert errors["momentum"] <= 2e-3
        assert errors["exp_avg"] <= 2e-6
        assert errors["exp_avg_sq"] <= 2e-6


def _run_muon_topology(
    tmp_path: Path,
    *,
    topology: str,
    world_size: int,
    extra_env: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"{topology} requires {world_size} CUDA devices")
    if importlib.util.find_spec("emerging_optimizers") is None:
        pytest.skip("MCore's optional emerging-optimizers backend is unavailable")
    suffix = "" if not extra_env else "-" + "-".join(extra_env.values())
    output_dir = tmp_path / f"{topology}{suffix}"
    env = os.environ.copy()
    env["ACCEPTANCE_OUTPUT_DIR"] = str(output_dir)
    env["MUON_TOPOLOGY"] = topology
    if extra_env:
        env.update(extra_env)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={world_size}",
        "--nnodes=1",
        "--master_addr=localhost",
        f"--master_port={find_free_ports(1)[0]}",
        "tests/torchrun/run_gpu_staged_muon_topology.py",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=240)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=10)
        raise AssertionError(
            f"{topology} torchrun timed out and was reaped\n{stdout}{stderr}"
        ) from error
    assert process.returncode == 0, stdout + stderr
    return [
        json.loads((output_dir / f"rank_{rank}.json").read_text())
        for rank in range(world_size)
    ]


def _run_muon_allgather(
    tmp_path: Path, *, scenario: str, world_size: int
) -> list[dict[str, object]]:
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"{scenario} requires {world_size} CUDA devices")
    output_dir = tmp_path / f"allgather-{scenario}"
    env = os.environ.copy()
    env["ACCEPTANCE_OUTPUT_DIR"] = str(output_dir)
    env["MUON_ALLGATHER_SCENARIO"] = scenario
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={world_size}",
        "--nnodes=1",
        "--master_addr=localhost",
        f"--master_port={find_free_ports(1)[0]}",
        "tests/torchrun/run_gpu_staged_muon_allgather.py",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return [
        json.loads((output_dir / f"rank_{rank}.json").read_text())
        for rank in range(world_size)
    ]


def _assert_muon_topology_results(results: list[dict[str, object]]) -> None:
    assert {result["residency"] for result in results} == {"CPU_RESIDENT"}
    assert {result["cuda_state_numel"] for result in results} == {0}
    assert {result["nondefault_stream_checked"] for result in results} == {True}
    leaf_orders = [result["leaf_order"] for result in results]
    assert all(order == leaf_orders[0] for order in leaf_orders[1:])
    assert leaf_orders[0][0] == "muon"
    assert set(leaf_orders[0][1:]) <= {"scalar_adamw"}
    assert any(result["communication_trace"] for result in results)
    for result in results:
        assert set(result["owner_counts"].values()) == {1}
        errors = result["max_errors"]
        assert errors["model"] <= 2e-3
        assert errors["master"] <= 2e-3
        assert errors["momentum"] <= 2e-3
        assert errors["exp_avg"] <= 2e-6
        assert errors["exp_avg_sq"] <= 2e-6


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_official_mcore_muon_tp2_dp1_uses_tp_orthogonalization(
    tmp_path: Path,
) -> None:
    """TP=2 uses the official EO all-gather path on complete local shards."""
    results = _run_muon_topology(tmp_path, topology="tp2_dp1", world_size=2)
    _assert_muon_topology_results(results)
    assert {tuple(result["tp_group"]) for result in results} == {(0, 1)}
    assert all(
        all(trace["mode"] == "duplicated" for trace in result["communication_trace"])
        for result in results
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_official_mcore_muon_tp2_dp2_separates_tp_and_owner_groups(
    tmp_path: Path,
) -> None:
    """TP collectives stay inside TP groups and owner sync stays inside DP groups."""
    results = _run_muon_topology(tmp_path, topology="tp2_dp2", world_size=4)
    _assert_muon_topology_results(results)
    assert {tuple(result["tp_group"]) for result in results} == {(0, 1), (2, 3)}
    assert {tuple(result["group_members"]["dp_cp"]) for result in results} == {
        (0, 2),
        (1, 3),
    }


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_official_mcore_muon_dense_expert_uses_distinct_owner_domains(
    tmp_path: Path,
) -> None:
    """A real EP=2 topology keeps dense and expert ownership separate."""
    results = _run_muon_topology(tmp_path, topology="dense_expert", world_size=4)
    _assert_muon_topology_results(results)
    dense_groups = {tuple(result["group_members"]["dp_cp"]) for result in results}
    expert_groups = {tuple(result["group_members"]["expt_dp"]) for result in results}
    assert dense_groups == {(0, 1, 2, 3)}
    assert all(len(group) == 2 for group in expert_groups)
    assert dense_groups.isdisjoint(expert_groups)
    assert {tuple(result["expert_tp_group"]) for result in results} == {
        (0,),
        (1,),
        (2,),
        (3,),
    }


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_official_mcore_muon_expert_only_skips_all_empty_dense_domain(
    tmp_path: Path,
) -> None:
    """Expert-only EP=2 keeps dense [[], []] as a communication no-op."""
    results = _run_muon_topology(tmp_path, topology="expert_only", world_size=2)
    _assert_muon_topology_results(results)
    assert all(
        not any(name.startswith("dense.") for name in result["owner_counts"])
        for result in results
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    ("scenario", "world_size", "collectives"),
    [
        ("all_empty", 2, 0),
        ("first_empty", 2, 1),
        ("middle_empty", 3, 1),
        ("two_domains", 4, 2),
    ],
)
def test_staged_muon_empty_owner_allgather_matrix(
    tmp_path: Path, scenario: str, world_size: int, collectives: int
) -> None:
    """All-empty and arbitrary partial-empty owner layouts are NCCL-safe."""
    results = _run_muon_allgather(tmp_path, scenario=scenario, world_size=world_size)
    assert {result["collective_count"] for result in results} == {collectives}
    assert {result["health"] for result in results} == {world_size}


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    "participation", ["all_none", "partial", "muon_only", "scalar_only"]
)
def test_staged_muon_tp_gradient_participation_is_collective_safe(
    tmp_path: Path, participation: str
) -> None:
    """TP peers either skip together or reject a partial gradient before NS."""
    results = _run_muon_topology(
        tmp_path,
        topology="tp2_dp1",
        world_size=2,
        extra_env={"MUON_PARTICIPATION": participation},
    )
    assert {result["health"] for result in results} == {2}
    assert {result["cuda_state_numel"] for result in results} == {0}
    assert {result["residency"] for result in results} == {"CPU_RESIDENT"}
    if participation == "all_none":
        assert {result["failure"] for result in results} == {None}
        assert all(not result["communication_trace"] for result in results)
    else:
        if participation == "partial":
            assert all(
                "inconsistent gradient participation" in result["failure"]
                for result in results
            )
            assert all(not result["communication_trace"] for result in results)
        else:
            assert {result["failure"] for result in results} == {None}


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_staged_muon_tp_gradient_failure_is_global_across_dp_subgroups(
    tmp_path: Path,
) -> None:
    """One inconsistent TP subgroup stops every DP replica before mutation."""
    results = _run_muon_topology(
        tmp_path,
        topology="tp2_dp2",
        world_size=4,
        extra_env={"MUON_PARTICIPATION": "partial"},
    )
    assert {result["health"] for result in results} == {4}
    assert {result["residency"] for result in results} == {"CPU_RESIDENT"}
    assert {result["cuda_state_numel"] for result in results} == {0}
    partial_groups = {tuple(result["partial_tp_group"]) for result in results}
    assert len(partial_groups) == 1
    partial_group = partial_groups.pop()
    for result in results:
        if result["rank"] in partial_group:
            assert "inconsistent gradient participation" in result["failure"]
        else:
            assert "step preflight failed on another rank" in result["failure"]
    assert all(not result["communication_trace"] for result in results)


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_staged_muon_owner_storage_drift_is_collectively_rejected_dp2(
    tmp_path: Path,
) -> None:
    """One DP rank moving storage fails globally before parameter all-gather."""
    results = _run_muon_topology(
        tmp_path,
        topology="dp2",
        world_size=2,
        extra_env={"MUON_PARTICIPATION": "storage_drift"},
    )
    assert {result["health"] for result in results} == {2}
    assert {result["residency"] for result in results} == {"CPU_RESIDENT"}
    assert {result["cuda_state_numel"] for result in results} == {0}
    assert {result["data_allgather_count"] for result in results} == {0}
    assert all(result["failure"] is not None for result in results)
    assert all(not result["communication_trace"] for result in results)
