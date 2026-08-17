# SPDX-License-Identifier: Apache-2.0

"""Real DP=2 fixed-topology torch_dist checkpoint acceptance for staged Muon."""

from __future__ import annotations

import copy
import json
import os
import random
import resource
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

from areal.engine.megatron_utils.checkpointer import MegatronCheckpointManager
from areal.engine.megatron_utils.gpu_staged_muon import (
    GPUStagedMuonConfig,
    get_megatron_optimizer_with_gpu_staged_muon,
)
from areal.engine.megatron_utils.gpu_staged_optimizer_checkpoint import (
    abort_managed_checkpoint_load,
    begin_managed_checkpoint_load,
    build_managed_optimizer_tensor_manifest,
    commit_managed_checkpoint_load,
    create_managed_checkpoint_load_transaction,
    decide_managed_checkpoint_commit,
    merge_managed_optimizer_tensor_manifests,
    prepare_managed_checkpoint_commit,
    prepare_managed_checkpoint_load,
    prepare_managed_checkpoint_recovery,
    retry_managed_checkpoint_cleanup,
    validate_managed_optimizer_source_tensor_metadata,
)


class _TinyMuonModel(torch.nn.Module):
    def __init__(
        self,
        *,
        tp_size: int,
        pg_collection: ProcessGroupCollection,
    ) -> None:
        super().__init__()
        if tp_size == 1:
            self.linear = torch.nn.Linear(
                8, 8, bias=True, device="cuda", dtype=torch.bfloat16
            )
        else:
            layer_config = ModelParallelConfig(
                tensor_model_parallel_size=tp_size,
                bf16=True,
                params_dtype=torch.bfloat16,
                perform_initialization=True,
                use_cpu_initialization=False,
                gradient_accumulation_fusion=False,
            )

            def init_method(tensor: torch.Tensor) -> None:
                torch.nn.init.uniform_(tensor, -0.1, 0.1)

            self.linear = ColumnParallelLinear(
                8,
                8,
                config=layer_config,
                init_method=init_method,
                bias=True,
                gather_output=False,
                tp_group=pg_collection.tp,
            )
        self.norm = torch.nn.LayerNorm(8, device="cuda", dtype=torch.bfloat16)
        self.config = SimpleNamespace(
            num_attention_heads=1,
            num_query_groups=1,
            kv_channels=8,
        )
        self.ddp_config = DistributedDataParallelConfig()
        self._checkpoint_pg_collection = pg_collection

    def sharded_state_dict(self):
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(),
            "model.",
            tp_group=self._checkpoint_pg_collection.tp,
            dp_cp_group=self._checkpoint_pg_collection.dp_cp,
        )

    def load_state_dict(self, state_dict, *args, **kwargs):
        if state_dict and all(key.startswith("model.") for key in state_dict):
            state_dict = {
                key.removeprefix("model."): value for key, value in state_dict.items()
            }
        return super().load_state_dict(state_dict, *args, **kwargs)


class _CheckpointScheduler:
    """Small scheduler whose metadata and group LRs are checkpoint authority."""

    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.epoch = 0

    def _groups(self):
        return [
            group
            for leaf in self.optimizer.chained_optimizers
            for group in leaf.optimizer.param_groups
        ]

    def step(self) -> None:
        self.epoch += 1
        for group in self._groups():
            group["lr"] *= 0.9

    def state_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "lrs": [group["lr"] for group in self._groups()],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if set(state) != {"epoch", "lrs"}:
            raise ValueError("test scheduler checkpoint schema changed")
        lrs = state["lrs"]
        groups = self._groups()
        if not isinstance(lrs, list) or len(lrs) != len(groups):
            raise ValueError("test scheduler checkpoint group count changed")
        self.epoch = int(state["epoch"])
        for group, lr in zip(groups, lrs, strict=True):
            group["lr"] = float(lr)


class _TinyMuonOnlyModel(torch.nn.Module):
    """One owner-sharded matrix, leaving one DP rank with an empty manifest."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(8, 8, device="cuda", dtype=torch.bfloat16)
        )
        torch.nn.init.uniform_(self.weight, -0.1, 0.1)
        self.config = SimpleNamespace(
            num_attention_heads=1,
            num_query_groups=1,
            kv_channels=8,
        )
        self.ddp_config = DistributedDataParallelConfig()


class _TinyDenseExpertMuonModel(torch.nn.Module):
    """Enough dense and expert parameters to exercise new DP/EDP owners."""

    def __init__(self, *, pg_collection: ProcessGroupCollection) -> None:
        super().__init__()
        config = ModelParallelConfig(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
            bf16=True,
            params_dtype=torch.bfloat16,
            perform_initialization=True,
            use_cpu_initialization=False,
            gradient_accumulation_fusion=False,
        )

        def init_method(tensor: torch.Tensor) -> None:
            torch.nn.init.uniform_(tensor, -0.1, 0.1)

        self.dense = torch.nn.ModuleList(
            [
                ColumnParallelLinear(
                    8,
                    8,
                    config=config,
                    init_method=init_method,
                    bias=True,
                    gather_output=False,
                    tp_group=pg_collection.tp,
                )
                for _ in range(4)
            ]
        )
        self.experts = torch.nn.ModuleList(
            [
                ColumnParallelLinear(
                    8,
                    8,
                    config=config,
                    init_method=init_method,
                    bias=True,
                    gather_output=False,
                    is_expert=True,
                    tp_group=pg_collection.expt_tp,
                )
                for _ in range(2)
            ]
        )
        self.config = SimpleNamespace(
            num_attention_heads=1,
            num_query_groups=1,
            kv_channels=8,
        )
        self.ddp_config = DistributedDataParallelConfig()


def _optimizer_config() -> OptimizerConfig:
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


def _staged_config(snapshot_root: Path) -> GPUStagedMuonConfig:
    return GPUStagedMuonConfig(
        buffer_count=1,
        slot_size_mb=1,
        split_qkv=False,
        fp32_matmul_prec="highest",
        num_ns_steps=3,
        use_nesterov=True,
        checkpoint_snapshot_root=str(snapshot_root),
        checkpoint_snapshot_chunk_mb=1,
    )


def _set_gradients(model: torch.nn.Module, step: int) -> None:
    for parameter_index, parameter in enumerate(model.parameters()):
        accumulated = torch.zeros_like(parameter, dtype=torch.float32)
        for accumulation in range(2):
            accumulated.add_(
                0.004 * (step + 1) + 0.002 * accumulation + 0.0001 * parameter_index
            )
        parameter.main_grad = accumulated


def _local_state(
    optimizer, model: torch.nn.Module
) -> dict[str, dict[str, torch.Tensor]]:
    names = {parameter: name for name, parameter in model.named_parameters()}
    result = {}
    for leaf in optimizer.chained_optimizers:
        base = leaf.optimizer
        for group in base.param_groups:
            for parameter in group["params"]:
                result[names[parameter]] = {
                    key: value.clone()
                    for key, value in base.state[parameter].items()
                    if isinstance(value, torch.Tensor)
                }
    return result


def _logical_parameter_key(
    name: str,
    parameter: torch.Tensor,
    pg_collection: ProcessGroupCollection,
) -> str:
    if getattr(parameter, "allreduce", True):
        coordinate = (
            dist.get_rank(pg_collection.pp),
            dist.get_rank(pg_collection.tp),
        )
        return f"dense|pp={coordinate[0]}|tp={coordinate[1]}|{name}"
    coordinate = (
        dist.get_rank(pg_collection.pp),
        dist.get_rank(pg_collection.ep),
        dist.get_rank(pg_collection.expt_tp),
    )
    return (
        f"expert|pp={coordinate[0]}|ep={coordinate[1]}|expt_tp={coordinate[2]}|{name}"
    )


def _global_model_state(
    model: torch.nn.Module,
    pg_collection: ProcessGroupCollection,
    control_group,
) -> dict[str, torch.Tensor]:
    local = {
        _logical_parameter_key(name, parameter, pg_collection): parameter.detach()
        .cpu()
        .clone()
        for name, parameter in model.named_parameters()
    }
    participants = [None for _ in range(dist.get_world_size(control_group))]
    dist.all_gather_object(participants, local, group=control_group)
    merged: dict[str, torch.Tensor] = {}
    for state in participants:
        for key, value in state.items():
            previous = merged.setdefault(key, value)
            torch.testing.assert_close(previous, value, rtol=0.0, atol=0.0)
    return merged


def _load_global_model_state(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    pg_collection: ProcessGroupCollection,
) -> None:
    for name, parameter in model.named_parameters():
        key = _logical_parameter_key(name, parameter, pg_collection)
        if key not in state:
            raise KeyError(f"global model state is missing {key!r}")
        parameter.data.copy_(state[key])


def _assert_local_state_equal(left, right) -> None:
    if set(left) != set(right):
        raise AssertionError(f"owner state names differ: {set(left)} != {set(right)}")
    for name in left:
        if set(left[name]) != set(right[name]):
            raise AssertionError(f"state kinds differ for {name}")
        for state_kind in left[name]:
            torch.testing.assert_close(
                left[name][state_kind],
                right[name][state_kind],
                rtol=0.0,
                atol=0.0,
            )


def _local_group_metadata(optimizer) -> list[dict[str, object]]:
    return [
        {key: value for key, value in group.items() if key != "params"}
        for leaf in optimizer.chained_optimizers
        for group in leaf.optimizer.param_groups
    ]


def _slab_pointers(optimizer) -> dict[tuple[int, str], int]:
    result = {}
    for leaf_index, leaf in enumerate(optimizer.chained_optimizers):
        slabs = leaf.optimizer.cpu_slabs
        if slabs is None:
            continue
        for slab_name, slab in vars(slabs).items():
            result[(leaf_index, slab_name)] = slab.data_ptr()
    return result


def _assert_cpu_slab_contract(optimizer) -> None:
    """Every owned tensor remains an FP32 pinned view of an authoritative slab."""
    for leaf in optimizer.chained_optimizers:
        base = leaf.optimizer
        slabs = base.cpu_slabs
        if slabs is None:
            if base.state:
                raise AssertionError("an empty leaf created phantom optimizer state")
            continue
        slab_tensors = tuple(vars(slabs).values())
        if not slab_tensors:
            raise AssertionError("a non-empty leaf has no authoritative CPU slab")
        slab_storage_ids = {
            slab.untyped_storage()._cdata
            for slab in slab_tensors  # noqa: SLF001
        }
        for slab in slab_tensors:
            if slab.device.type != "cpu" or slab.dtype is not torch.float32:
                raise AssertionError("optimizer slab is not CPU FP32")
            if not slab.is_pinned():
                raise AssertionError("optimizer slab is not pinned")
        for state in base.state.values():
            for value in state.values():
                if not isinstance(value, torch.Tensor):
                    continue
                if value.untyped_storage()._cdata not in slab_storage_ids:  # noqa: SLF001
                    raise AssertionError("optimizer state view lost its slab alias")


def _record_phase(output_dir: Path, rank: int, phase: str) -> None:
    with (output_dir / f"rank_{rank}.phases").open("a") as stream:
        stream.write(f"{phase}\n")
        stream.flush()


def _directory_file_bytes(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _global_owner_state(
    optimizer,
    model,
    pg_collection: ProcessGroupCollection,
    control_group,
) -> dict[str, torch.Tensor]:
    names = {parameter: name for name, parameter in model.named_parameters()}
    local: dict[str, dict[str, torch.Tensor]] = {}
    for leaf in optimizer.chained_optimizers:
        base = leaf.optimizer
        for group in base.param_groups:
            for parameter in group["params"]:
                key = _logical_parameter_key(names[parameter], parameter, pg_collection)
                local[key] = {
                    state_kind: value.clone()
                    for state_kind, value in base.state[parameter].items()
                    if isinstance(value, torch.Tensor)
                }
    participants = [None for _ in range(dist.get_world_size(control_group))]
    dist.all_gather_object(participants, local, group=control_group)
    merged: dict[str, torch.Tensor] = {}
    for source_rank, state in enumerate(participants):
        for name, values in state.items():
            for state_kind, value in values.items():
                key = f"{name}|{state_kind}"
                if key in merged:
                    raise AssertionError(
                        f"logical optimizer state {key!r} has duplicate source "
                        f"owner on rank {source_rank}"
                    )
                merged[key] = value
    return merged


def _global_manifest(sharded, control_group):
    local = build_managed_optimizer_tensor_manifest(sharded)
    manifests = [None for _ in range(dist.get_world_size(control_group))]
    dist.all_gather_object(manifests, local, group=control_group)
    return merge_managed_optimizer_tensor_manifests(manifests)


def _reshard_group_metadata(optimizer) -> list[list[dict[str, object]]]:
    return [
        copy.deepcopy(leaf["param_groups"])
        for leaf in optimizer._checkpoint_metadata()["leaf_tree"]
    ]


def _run_metadata_authority_fault(
    *,
    output_dir: Path,
    snapshot_root: Path,
    checkpoint_dir: Path,
    pg_collection: ProcessGroupCollection,
    control_group,
) -> None:
    """Corrupt one contributor rank and prove preflight rejects without mutation."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model = _TinyMuonModel(tp_size=1, pg_collection=pg_collection)
    optimizer = get_megatron_optimizer_with_gpu_staged_muon(
        _optimizer_config(),
        [model],
        _staged_config(snapshot_root),
        pg_collection=pg_collection,
    )
    optimizer.bind_managed_checkpoint_process_group(control_group)
    before_state = _local_state(optimizer, model)
    before_groups = _local_group_metadata(optimizer)
    before_pointers = _slab_pointers(optimizer)
    before_model = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    original_local_metadata = optimizer._checkpoint_local_metadata
    if rank == 1:

        def corrupt_parameter_coordinate(self):
            del self
            metadata = original_local_metadata()
            for leaf in metadata["leaf_tree"]:
                if leaf["parameters"]:
                    coordinate = leaf["parameters"][0]["coordinate"]
                    coordinate[next(iter(coordinate))] = True
                    break
            else:
                raise RuntimeError("rank 1 has no payload coordinate to corrupt")
            return metadata

        optimizer._checkpoint_local_metadata = MethodType(
            corrupt_parameter_coordinate, optimizer
        )

    error = None
    try:
        optimizer._checkpoint_metadata()
    except BaseException as caught:
        error = f"{type(caught).__name__}: {caught}"
    errors = [None for _ in range(world_size)]
    dist.all_gather_object(errors, error, group=control_group)
    if any(value is None for value in errors) or len(set(errors)) != 1:
        raise AssertionError(f"metadata authority rejection diverged: {errors!r}")
    if "coordinate" not in errors[0]:
        raise AssertionError(f"unexpected metadata authority error: {errors[0]}")
    _assert_local_state_equal(before_state, _local_state(optimizer, model))
    if _local_group_metadata(optimizer) != before_groups:
        raise AssertionError("metadata rejection changed optimizer groups")
    if _slab_pointers(optimizer) != before_pointers:
        raise AssertionError("metadata rejection replaced an optimizer slab")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before_model[name], rtol=0.0, atol=0.0)
    if any(snapshot_root.iterdir()):
        raise AssertionError("metadata rejection created a rollback snapshot")
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise AssertionError("metadata rejection reached DCP mutation")
    if optimizer.residency != "CPU_RESIDENT" or optimizer.cuda_state_numel != 0:
        raise AssertionError("metadata rejection changed optimizer residency")

    nccl_health = torch.tensor([rank + 1.0], device="cuda")
    dist.all_reduce(nccl_health, group=dist.group.WORLD)
    cpu_health = torch.tensor([rank + 1], dtype=torch.int64)
    dist.all_reduce(cpu_health, group=control_group)
    expected = world_size * (world_size + 1) // 2
    if nccl_health.item() != expected or cpu_health.item() != expected:
        raise AssertionError("post-metadata-fault collective health probe failed")
    result = {
        "rank": rank,
        "error": error,
        "residency": optimizer.residency,
        "cuda_state_numel": optimizer.cuda_state_numel,
        "snapshot_files": len(list(snapshot_root.iterdir())),
        "dcp_files": (
            len(list(checkpoint_dir.iterdir())) if checkpoint_dir.exists() else 0
        ),
        "nccl_health": nccl_health.item(),
        "gloo_health": cpu_health.item(),
    }
    (output_dir / f"metadata_fault_rank_{rank}.json").write_text(
        json.dumps(result, sort_keys=True) + "\n"
    )


def _run_reshard_fault_cycles(
    *,
    optimizer,
    model: torch.nn.Module,
    template,
    checkpoint_dir: Path,
    snapshot_root: Path,
    output_dir: Path,
    control_group,
    failure_rank: int,
) -> int:
    """Exercise destination-owner rollback, recovery, and committed cleanup."""
    rank = dist.get_rank()
    rollback_disk_peak_bytes = 0

    def gather_error(local_error: str | None) -> list[str | None]:
        errors = [None for _ in range(dist.get_world_size(control_group))]
        dist.all_gather_object(errors, local_error, group=control_group)
        return errors

    def load_and_apply():
        nonlocal rollback_disk_peak_bytes
        transaction = begin_managed_checkpoint_load(optimizer)
        dist.barrier(group=control_group)
        rollback_disk_peak_bytes = max(
            rollback_disk_peak_bytes, _directory_file_bytes(snapshot_root)
        )
        loaded = dist_checkpointing.load(template, str(checkpoint_dir))
        optimizer.load_state_dict(loaded["optimizer"])
        return transaction

    before_state = _local_state(optimizer, model)
    before_groups = _local_group_metadata(optimizer)
    before_pointers = _slab_pointers(optimizer)

    transaction = load_and_apply()
    errors = gather_error(
        "injected destination local validate failure" if rank == failure_rank else None
    )
    if sum(error is not None for error in errors) != 1:
        raise AssertionError(f"local validate fault did not reach consensus: {errors}")
    abort_managed_checkpoint_load(
        transaction,
        RuntimeError(repr(errors)),
        poison=False,
    )
    dist.barrier(group=control_group)
    verify_error = None
    try:
        _assert_local_state_equal(before_state, _local_state(optimizer, model))
        if _local_group_metadata(optimizer) != before_groups:
            raise AssertionError("destination rollback changed param-group metadata")
        if _slab_pointers(optimizer) != before_pointers:
            raise AssertionError("destination rollback replaced CPU slabs")
        if any(snapshot_root.iterdir()):
            raise AssertionError("destination rollback retained snapshot files")
    except BaseException as error:
        verify_error = f"{type(error).__name__}: {error}"
    verify_errors = gather_error(verify_error)
    if any(error is not None for error in verify_errors):
        raise AssertionError(
            f"destination rollback verification failed: {verify_errors}"
        )
    _record_phase(output_dir, rank, "fault_local_validate_rolled_back")

    transaction = load_and_apply()
    injected_leaf = None
    original_abort = None
    if rank == failure_rank:
        injected_leaf = next(
            leaf.optimizer
            for leaf in optimizer.chained_optimizers
            if leaf.optimizer.state
        )
        original_abort = injected_leaf.abort_checkpoint_load
        pending = True

        def abort_once(self, error, *, poison=True, attempt_token=None):
            nonlocal pending
            del self, error, poison, attempt_token
            if pending:
                pending = False
                raise RuntimeError("injected rollback action failure")
            raise AssertionError("completed rollback action was replayed")

        injected_leaf.abort_checkpoint_load = MethodType(abort_once, injected_leaf)
    abort_managed_checkpoint_load(
        transaction,
        RuntimeError("injected post-DCP failure"),
        poison=True,
    )
    if injected_leaf is not None:
        injected_leaf.abort_checkpoint_load = original_abort
    rollback_pending = gather_error("rollback pending" if transaction.begun else None)
    if sum(error is not None for error in rollback_pending) != 1:
        raise AssertionError(
            f"rollback pending authority diverged: {rollback_pending!r}"
        )
    recovery = create_managed_checkpoint_load_transaction(optimizer)
    recovery_error = None
    try:
        prepare_managed_checkpoint_recovery(recovery)
    except BaseException as error:
        recovery_error = f"{type(error).__name__}: {error}"
    recovery_errors = gather_error(recovery_error)
    if any(error is not None for error in recovery_errors):
        raise AssertionError(f"rollback recovery failed: {recovery_errors!r}")
    dist.barrier(group=control_group)
    _assert_local_state_equal(before_state, _local_state(optimizer, model))
    if any(snapshot_root.iterdir()):
        raise AssertionError("rollback recovery retained snapshot files")
    _record_phase(output_dir, rank, "fault_rollback_recovered")

    transaction = load_and_apply()
    prepare_managed_checkpoint_load(transaction)
    prepare_managed_checkpoint_commit(transaction)
    decide_managed_checkpoint_commit(transaction)
    injected_leaf = None
    original_discard = None
    injection_setup_error = None
    if rank == failure_rank:
        try:
            assert transaction.cleanup_journal is not None
            injected_leaf = next(
                entry.leaf
                for entry in transaction.cleanup_journal.entries
                if callable(getattr(entry.leaf, "discard_checkpoint_snapshot", None))
            )
            original_discard = injected_leaf.discard_checkpoint_snapshot
            pending = True

            def discard_then_fail(self):
                nonlocal pending
                del self
                original_discard()
                if pending:
                    pending = False
                    raise RuntimeError("injected cleanup discard after-effect")

            injected_leaf.discard_checkpoint_snapshot = MethodType(
                discard_then_fail, injected_leaf
            )
        except BaseException as error:
            injection_setup_error = f"{type(error).__name__}: {error}"
    injection_setup_errors = gather_error(injection_setup_error)
    if any(error is not None for error in injection_setup_errors):
        raise AssertionError(
            f"cleanup injection setup failed: {injection_setup_errors!r}"
        )
    cleanup_error = None
    try:
        retry_managed_checkpoint_cleanup(transaction)
    except BaseException as error:
        cleanup_error = f"{type(error).__name__}: {error}"
    cleanup_errors = gather_error(cleanup_error)
    (output_dir / f"fault_cleanup_rank_{rank}.json").write_text(
        json.dumps(
            {
                "local_error": cleanup_error,
                "all_errors": cleanup_errors,
                "phase": transaction.phase.name,
                "pending_entries": (
                    len(transaction.cleanup_journal.entries)
                    if transaction.cleanup_journal is not None
                    else 0
                ),
            },
            sort_keys=True,
        )
        + "\n"
    )
    if sum(error is not None for error in cleanup_errors) != 1:
        raise AssertionError(
            f"cleanup failure did not reach consensus: {cleanup_errors}"
        )
    if injected_leaf is not None:
        injected_leaf.discard_checkpoint_snapshot = original_discard
    retry_error = None
    try:
        retry_managed_checkpoint_cleanup(transaction)
    except BaseException as error:
        retry_error = f"{type(error).__name__}: {error}"
    retry_errors = gather_error(retry_error)
    if any(error is not None for error in retry_errors):
        raise AssertionError(f"cleanup retry failed: {retry_errors!r}")
    dist.barrier(group=control_group)
    if any(snapshot_root.iterdir()):
        raise AssertionError("committed cleanup retained snapshot files")
    _record_phase(output_dir, rank, "fault_cleanup_after_effect_reconciled")
    return rollback_disk_peak_bytes


def _run_reshard_phase(
    *,
    phase: str,
    output_dir: Path,
    snapshot_root: Path,
    checkpoint_dir: Path,
    pg_collection: ProcessGroupCollection,
    control_group,
    model_kind: str,
) -> None:
    """Save or load one side of a real DP ownership migration."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(20260816)
    torch.cuda.manual_seed(20260816)
    model_parallel_cuda_manual_seed(20260816, force_reset_rng=True)
    if model_kind == "mixed":
        model = _TinyMuonModel(tp_size=1, pg_collection=pg_collection)
    elif model_kind == "pure_muon":
        model = _TinyMuonOnlyModel()
    elif model_kind == "dense_expert":
        model = _TinyDenseExpertMuonModel(pg_collection=pg_collection)
    else:
        raise RuntimeError(f"unsupported reshard model kind {model_kind!r}")

    if phase == "load":
        model_state = torch.load(
            output_dir / "checkpoint_model.pt", map_location="cuda", weights_only=True
        )
        _load_global_model_state(model, model_state, pg_collection)
    optimizer = get_megatron_optimizer_with_gpu_staged_muon(
        _optimizer_config(),
        [model],
        _staged_config(snapshot_root),
        pg_collection=pg_collection,
    )
    optimizer.bind_managed_checkpoint_process_group(control_group)
    load_cuda_peak_bytes = 0
    max_model_error = 0.0
    max_optimizer_errors: dict[str, float] = {}
    rollback_disk_peak_bytes = 0
    _record_phase(output_dir, rank, f"reshard_{phase}_optimizer_built")

    if phase == "save":
        for step in range(2):
            _set_gradients(model, step)
            optimizer.step()
            optimizer.drain()
        checkpoint_model_state = _global_model_state(
            model, pg_collection, control_group
        )
        if rank == 0:
            torch.save(checkpoint_model_state, output_dir / "checkpoint_model.pt")
        dist.barrier(group=dist.group.WORLD)
        sharded = optimizer.sharded_state_dict({})
        _record_phase(output_dir, rank, "reshard_save_template")
        dist_checkpointing.save(
            {"optimizer": sharded},
            str(checkpoint_dir),
            async_sharded_save=False,
        )
        validate_managed_optimizer_source_tensor_metadata(
            str(checkpoint_dir), _global_manifest(sharded, control_group)
        )
        _record_phase(output_dir, rank, "reshard_save_manifest")
        for step in range(2, 5):
            _set_gradients(model, step)
            optimizer.step()
            optimizer.drain()
            _record_phase(output_dir, rank, f"reshard_save_step_{step}")
        expected = {
            "model": {
                name: value.detach().cpu()
                for name, value in _global_model_state(
                    model, pg_collection, control_group
                ).items()
            },
            "optimizer": _global_owner_state(
                optimizer, model, pg_collection, control_group
            ),
            "param_groups": _reshard_group_metadata(optimizer),
        }
        if rank == 0:
            torch.save(expected, output_dir / "expected_after_three_steps.pt")
    elif phase == "load":
        outer = dist_checkpointing.load(
            {"optimizer": optimizer.managed_checkpoint_outer_template()},
            str(checkpoint_dir),
        )
        _record_phase(output_dir, rank, "reshard_load_outer")
        optimizer.validate_managed_checkpoint_outer_state(outer["optimizer"])
        _record_phase(output_dir, rank, "reshard_load_outer_validated")
        template = {"optimizer": optimizer.sharded_state_dict({})}
        _record_phase(output_dir, rank, "reshard_load_template")
        validate_managed_optimizer_source_tensor_metadata(
            str(checkpoint_dir),
            _global_manifest(template["optimizer"], control_group),
        )
        _record_phase(output_dir, rank, "reshard_load_manifest")
        if os.environ.get("MUON_RESHARD_FAULT_CYCLES") == "1":
            rollback_disk_peak_bytes = _run_reshard_fault_cycles(
                optimizer=optimizer,
                model=model,
                template=template,
                checkpoint_dir=checkpoint_dir,
                snapshot_root=snapshot_root,
                output_dir=output_dir,
                control_group=control_group,
                failure_rank=1,
            )
        pointers = _slab_pointers(optimizer)
        torch.cuda.reset_peak_memory_stats()
        load_start = torch.cuda.memory_allocated()
        transaction = begin_managed_checkpoint_load(optimizer)
        dist.barrier(group=control_group)
        rollback_disk_peak_bytes = max(
            rollback_disk_peak_bytes, _directory_file_bytes(snapshot_root)
        )
        _record_phase(output_dir, rank, "reshard_load_snapshot")
        loaded = dist_checkpointing.load(template, str(checkpoint_dir))
        _record_phase(output_dir, rank, "reshard_load_dcp")
        optimizer.load_state_dict(loaded["optimizer"])
        _record_phase(output_dir, rank, "reshard_load_applied")
        commit_error = None
        try:
            prepare_managed_checkpoint_load(transaction)
            commit_managed_checkpoint_load(transaction)
        except BaseException as error:
            commit_error = f"{type(error).__name__}: {error}"
        commit_errors = [None for _ in range(world_size)]
        dist.all_gather_object(commit_errors, commit_error, group=control_group)
        if any(error is not None for error in commit_errors):
            raise RuntimeError(f"DP reshard commit failed: {commit_errors!r}")
        _record_phase(output_dir, rank, "reshard_load_committed")
        if _slab_pointers(optimizer) != pointers:
            raise AssertionError("DP reshard replaced destination CPU slabs")
        load_cuda_peak_bytes = torch.cuda.max_memory_allocated() - load_start
        for step in range(2, 5):
            _set_gradients(model, step)
            optimizer.step()
            optimizer.drain()
            _record_phase(output_dir, rank, f"reshard_load_step_{step}")
        expected = torch.load(
            output_dir / "expected_after_three_steps.pt",
            map_location="cpu",
            weights_only=True,
        )
        actual_model = _global_model_state(model, pg_collection, control_group)
        for name, value in actual_model.items():
            max_model_error = max(
                max_model_error,
                float((value - expected["model"][name]).abs().max().item()),
            )
            torch.testing.assert_close(
                value, expected["model"][name], rtol=0.0, atol=0.0
            )
        actual_state = _global_owner_state(
            optimizer, model, pg_collection, control_group
        )
        if set(actual_state) != set(expected["optimizer"]):
            raise AssertionError("DP reshard optimizer state key mismatch")
        for key, value in actual_state.items():
            state_kind = key.rsplit("|", 1)[-1]
            error = float((value - expected["optimizer"][key]).abs().max().item())
            max_optimizer_errors[state_kind] = max(
                max_optimizer_errors.get(state_kind, 0.0), error
            )
            torch.testing.assert_close(
                value,
                expected["optimizer"][key],
                rtol=0.0,
                atol=1e-6 if model_kind == "dense_expert" else 0.0,
            )
        if _reshard_group_metadata(optimizer) != expected["param_groups"]:
            raise AssertionError("DP reshard param-group metadata mismatch")
    else:
        raise RuntimeError(f"unsupported Muon checkpoint phase {phase!r}")

    _assert_cpu_slab_contract(optimizer)
    if optimizer.residency != "CPU_RESIDENT" or optimizer.cuda_state_numel != 0:
        raise AssertionError("DP reshard violated staged residency")
    if any(snapshot_root.iterdir()):
        raise AssertionError("DP reshard retained a rollback snapshot after commit")
    checkpoint_bytes = (
        sum(path.stat().st_size for path in checkpoint_dir.rglob("*") if path.is_file())
        if rank == 0
        else 0
    )
    checkpoint_sizes = [checkpoint_bytes]
    dist.broadcast_object_list(checkpoint_sizes, src=0, group=dist.group.WORLD)
    owned_parameters = {
        parameter
        for leaf in optimizer.chained_optimizers
        for group in leaf.optimizer.param_groups
        for parameter in group["params"]
    }
    result = {
        "phase": phase,
        "rank": rank,
        "world_size": world_size,
        "model_kind": model_kind,
        "owned_parameters": sorted(
            [
                _logical_parameter_key(name, parameter, pg_collection)
                for name, parameter in model.named_parameters()
                if parameter in owned_parameters
            ]
        ),
        "groups": {
            name: list(dist.get_process_group_ranks(getattr(pg_collection, name)))
            for name in ("tp", "ep", "expt_tp", "dp_cp", "expt_dp")
        },
        "residency": optimizer.residency,
        "cuda_state_numel": optimizer.cuda_state_numel,
        "load_cuda_peak_bytes": load_cuda_peak_bytes,
        "checkpoint_bytes": checkpoint_sizes[0],
        "rollback_disk_peak_bytes": rollback_disk_peak_bytes,
        "rollback_disk_final_bytes": _directory_file_bytes(snapshot_root),
        "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "max_model_error": max_model_error,
        "max_optimizer_errors": max_optimizer_errors,
    }
    (output_dir / f"{phase}_rank_{rank}.json").write_text(
        json.dumps(result, sort_keys=True) + "\n"
    )
    health = torch.tensor([rank + 1.0], device="cuda")
    dist.all_reduce(health, group=dist.group.WORLD)
    if health.item() != world_size * (world_size + 1) / 2:
        raise AssertionError("post-reshard NCCL health probe failed")


def _seed_manager_checkpoint() -> None:
    random.seed(20260816)
    np.random.seed(20260816)
    torch.manual_seed(20260816)
    torch.cuda.manual_seed_all(20260816)
    model_parallel_cuda_manual_seed(20260816, force_reset_rng=True)


def _rng_draws() -> dict[str, object]:
    from megatron.core import tensor_parallel

    with tensor_parallel.get_cuda_rng_tracker().fork():
        tracker = torch.rand(4, device="cuda").cpu()
    return {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "cpu": torch.rand(4),
        "cuda": torch.rand(4, device="cuda").cpu(),
        "tracker": tracker,
    }


def _assert_rng_draws_equal(actual, expected) -> None:
    if actual["python"] != expected["python"]:
        raise AssertionError("Python RNG was not restored")
    if actual["numpy"] != expected["numpy"]:
        raise AssertionError("NumPy RNG was not restored")
    for key in ("cpu", "cuda", "tracker"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=0.0)


def _manager_for_muon(
    *, model, optimizer, scheduler, control_group, snapshot_root: Path
) -> MegatronCheckpointManager:
    return MegatronCheckpointManager(
        model=torch.nn.ModuleList([model]),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        use_distributed_optimizer=False,
        use_checkpoint_opt_param_scheduler=True,
        use_dist_checkpointing=True,
        async_save=False,
        checkpoint_process_group=control_group,
        managed_checkpoint_enabled=True,
        managed_checkpoint_snapshot_root=str(snapshot_root),
    )


def _assert_manager_clean(manager, optimizer, snapshot_root: Path) -> None:
    for field in (
        "_managed_checkpoint_poisoned_error",
        "_managed_checkpoint_recovery_transaction",
        "_managed_checkpoint_cleanup_recovery",
        "_managed_checkpoint_control_error",
    ):
        if getattr(manager, field) is not None:
            raise AssertionError(f"manager retained checkpoint authority in {field}")
    for leaf in optimizer.chained_optimizers:
        base = leaf.optimizer
        effective = getattr(base, "_effective_checkpoint_lifecycle", None)
        if callable(effective) and effective().name != "CLEAN":
            raise AssertionError("managed Muon leaf did not return to CLEAN")
        if getattr(base, "_checkpoint_active", False):
            raise AssertionError("empty managed leaf retained an active transaction")
        for field in (
            "_checkpoint_rollback",
            "_checkpoint_cleanup",
            "_checkpoint_prepared_cleanup",
            "_checkpoint_attempt_token",
            "_checkpoint_commit_token",
        ):
            if getattr(base, field, None) is not None:
                raise AssertionError(f"managed leaf retained {field}")
    if any(snapshot_root.iterdir()):
        raise AssertionError("managed manager retained rollback snapshot artifacts")


def _manager_checkpoint_expected(
    optimizer,
    model,
    scheduler,
    pg_collection: ProcessGroupCollection,
    control_group,
) -> dict[str, object]:
    return {
        "model": _global_model_state(model, pg_collection, control_group),
        "optimizer": _global_owner_state(
            optimizer, model, pg_collection, control_group
        ),
        "param_groups": _reshard_group_metadata(optimizer),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
    }


def _assert_manager_checkpoint_expected(
    expected,
    optimizer,
    model,
    scheduler,
    pg_collection: ProcessGroupCollection,
    control_group,
) -> tuple[float, dict[str, float]]:
    max_model_error = 0.0
    actual_model = _global_model_state(model, pg_collection, control_group)
    if set(actual_model) != set(expected["model"]):
        raise AssertionError("public manager model keys changed after DP reshard")
    for key, actual in actual_model.items():
        error = float((actual - expected["model"][key]).abs().max().item())
        max_model_error = max(max_model_error, error)
        torch.testing.assert_close(actual, expected["model"][key], rtol=0.0, atol=0.0)
    actual_optimizer = _global_owner_state(
        optimizer, model, pg_collection, control_group
    )
    if set(actual_optimizer) != set(expected["optimizer"]):
        raise AssertionError("public manager optimizer keys changed after DP reshard")
    max_optimizer_errors: dict[str, float] = {}
    for key, actual in actual_optimizer.items():
        state_kind = key.rsplit("|", 1)[-1]
        error = float((actual - expected["optimizer"][key]).abs().max().item())
        max_optimizer_errors[state_kind] = max(
            max_optimizer_errors.get(state_kind, 0.0), error
        )
        torch.testing.assert_close(
            actual, expected["optimizer"][key], rtol=0.0, atol=0.0
        )
    if _reshard_group_metadata(optimizer) != expected["param_groups"]:
        raise AssertionError("public manager param-group metadata was not restored")
    if scheduler.state_dict() != expected["scheduler"]:
        raise AssertionError("public manager scheduler state was not restored")
    return max_model_error, max_optimizer_errors


def _manager_state_tensors(optimizer, model) -> dict[str, torch.Tensor]:
    names = {parameter: name for name, parameter in model.named_parameters()}
    return {
        f"{names[parameter]}|{state_name}": value
        for leaf in optimizer.chained_optimizers
        for group in leaf.optimizer.param_groups
        for parameter in group["params"]
        for state_name, value in leaf.optimizer.state[parameter].items()
        if isinstance(value, torch.Tensor)
    }


def _install_partial_dcp_fault(*, optimizer, model, fault_evidence: dict[str, object]):
    from megatron.core.dist_checkpointing.strategies.torch import MCoreLoadPlanner

    original_commit = MCoreLoadPlanner.commit_tensor
    before = {
        key: value.clone()
        for key, value in _manager_state_tensors(optimizer, model).items()
    }
    if len(before) < 2:
        raise AssertionError(
            "DCP partial-write fault requires at least two local states"
        )
    restore_counts: dict[str, int] = {}
    injected_rollback = False

    def commit_then_fail(self, read_item, tensor):
        nonlocal injected_rollback
        original_commit(self, read_item, tensor)
        fqn = read_item.dest_index.fqn
        if injected_rollback or "optimizer.gpu_staged_muon.v2" not in fqn:
            return
        after = _manager_state_tensors(optimizer, model)
        changed = [key for key in before if not torch.equal(before[key], after[key])]
        unchanged = [key for key in before if torch.equal(before[key], after[key])]
        if not changed or not unchanged:
            raise AssertionError(
                "DCP fault did not observe one written and one unwritten managed state"
            )
        rollback = next(
            (
                leaf.optimizer._checkpoint_rollback
                for leaf in optimizer.chained_optimizers
                if getattr(leaf.optimizer, "_checkpoint_rollback", None) is not None
                and any(
                    action.name.startswith("slab.")
                    for action in leaf.optimizer._checkpoint_rollback.actions
                )
            ),
            None,
        )
        if rollback is None:
            raise AssertionError("DCP fault could not find a live rollback journal")
        failing_action = next(
            action
            for action in rollback.actions
            if action.name in {"slab.momentum", "slab.exp_avg"}
        )
        fail_pending = True
        for action in rollback.actions:
            original_restore = action.restore

            def restore(
                target, snapshot, *, _name=action.name, _restore=original_restore
            ):
                nonlocal fail_pending
                restore_counts[_name] = restore_counts.get(_name, 0) + 1
                if _name == failing_action.name and fail_pending:
                    fail_pending = False
                    raise RuntimeError("injected rollback action failure")
                _restore(target, snapshot)

            action.restore = restore
        injected_rollback = True
        fault_evidence.update(
            {
                "fqn": fqn,
                "changed": changed,
                "unchanged": unchanged,
                "failing_action": failing_action.name,
                "restore_counts": restore_counts,
                "before": before,
            }
        )
        raise RuntimeError("injected DCP managed tensor write after-effect")

    MCoreLoadPlanner.commit_tensor = commit_then_fail
    return MCoreLoadPlanner, original_commit


def _run_manager_reshard_phase(
    *,
    phase: str,
    output_dir: Path,
    snapshot_root: Path,
    checkpoint_dir: Path,
    pg_collection: ProcessGroupCollection,
    control_group,
) -> None:
    """Exercise DP reshard only through public MegatronCheckpointManager APIs."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    _seed_manager_checkpoint()
    model = _TinyMuonModel(tp_size=1, pg_collection=pg_collection)
    optimizer = get_megatron_optimizer_with_gpu_staged_muon(
        _optimizer_config(),
        [model],
        _staged_config(snapshot_root),
        pg_collection=pg_collection,
    )
    scheduler = _CheckpointScheduler(optimizer)
    manager = _manager_for_muon(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        control_group=control_group,
        snapshot_root=snapshot_root,
    )
    # The public manager performs this bind during its first checkpoint
    # request.  Bind once here as well because the acceptance fixture snapshots
    # global param-group metadata before the destination's first request.
    optimizer.bind_managed_checkpoint_process_group(control_group)
    for step in range(2):
        _set_gradients(model, step)
        optimizer.step()
        optimizer.drain()
        scheduler.step()

    expected_path = output_dir / "manager_expected.pt"
    rng_path = output_dir / "manager_rng.pt"
    fault_enabled = os.environ.get("MUON_MANAGER_DCP_FAULT") == "1"
    fault_rank = int(os.environ.get("MUON_MANAGER_DCP_FAULT_RANK", "1"))
    phase_trace: list[str] = []
    original_vote = manager._vote_managed_phase

    def traced_vote(self, phase_name, *args, **kwargs):
        del self
        phase_trace.append(phase_name)
        return original_vote(phase_name, *args, **kwargs)

    manager._vote_managed_phase = MethodType(traced_vote, manager)
    max_model_error = 0.0
    max_optimizer_errors: dict[str, float] = {}
    partial_fault: dict[str, object] = {}
    first_load_failed = False
    replacement_configure_failed = False
    second_cycle = False
    load_cuda_peak_bytes = 0

    if phase == "save":
        manager.save_checkpoint(
            str(checkpoint_dir), with_model=True, with_optimizer=True, with_rng=True
        )
        expected_rng = _rng_draws()
        for step in range(2, 5):
            _set_gradients(model, step)
            optimizer.step()
            optimizer.drain()
            scheduler.step()
        expected = _manager_checkpoint_expected(
            optimizer, model, scheduler, pg_collection, control_group
        )
        if rank == 0:
            torch.save(expected_rng, rng_path)
            torch.save(expected, expected_path)
        dist.barrier(group=control_group)
    elif phase == "load":
        # Keep model values equal to the source checkpoint while making every
        # optimizer state, scheduler field, and RNG stream observably different.
        for state in _manager_state_tensors(optimizer, model).values():
            state.add_(7.0)
        scheduler.epoch = 77
        for group in scheduler._groups():
            group["lr"] = 0.123
        random.seed(9)
        np.random.seed(9)
        torch.manual_seed(9)
        torch.cuda.manual_seed_all(9)
        before_model = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        before_state = _local_state(optimizer, model)
        before_groups = copy.deepcopy(_reshard_group_metadata(optimizer))
        before_scheduler = copy.deepcopy(scheduler.state_dict())
        before_pointers = _slab_pointers(optimizer)
        load_calls = 0
        original_load_data = manager._load_checkpoint_data

        def checked_load_data(self, *args, **kwargs):
            nonlocal load_calls
            del self
            load_calls += 1
            if fault_enabled and load_calls == 2:
                _assert_local_state_equal(before_state, _local_state(optimizer, model))
                if _reshard_group_metadata(optimizer) != before_groups:
                    raise AssertionError(
                        "recovery did not restore param-group metadata"
                    )
                if scheduler.state_dict() != before_scheduler:
                    raise AssertionError("recovery did not restore scheduler metadata")
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(
                        value, before_model[name], rtol=0.0, atol=0.0
                    )
                partial_fault["recovery_prefix_verified"] = True
            return original_load_data(*args, **kwargs)

        manager._load_checkpoint_data = MethodType(checked_load_data, manager)
        planner_type = None
        original_commit = None
        if fault_enabled and rank == fault_rank:
            planner_type, original_commit = _install_partial_dcp_fault(
                optimizer=optimizer,
                model=model,
                fault_evidence=partial_fault,
            )
        torch.cuda.reset_peak_memory_stats()
        load_start = torch.cuda.memory_allocated()
        first_error = None
        try:
            manager.load_checkpoint(
                str(checkpoint_dir),
                with_model=True,
                with_optimizer=True,
                with_rng=True,
            )
        except BaseException as error:
            first_error = f"{type(error).__name__}: {error}"
        finally:
            if planner_type is not None:
                planner_type.commit_tensor = original_commit
        first_errors = [None for _ in range(world_size)]
        dist.all_gather_object(first_errors, first_error, group=control_group)
        if fault_enabled:
            if not all(error is not None for error in first_errors):
                raise AssertionError(
                    f"partial DCP fault lacked consensus: {first_errors}"
                )
            first_load_failed = True
            if rank == fault_rank and not partial_fault.get("changed"):
                raise AssertionError("partial DCP fault hook was not reached")
            if manager._managed_checkpoint_poisoned_error is None:
                raise AssertionError("manager did not retain the failed load poison")
            if rank == fault_rank and not any(
                getattr(leaf.optimizer, "_checkpoint_rollback", None) is not None
                for leaf in optimizer.chained_optimizers
            ):
                raise AssertionError("manager lost retained leaf rollback authority")

            configured_leaf = optimizer.chained_optimizers[
                1 if fault_rank == 0 else 0
            ].optimizer
            original_configure = configured_leaf.configure_checkpoint_snapshot
            configure_failure_pending = rank == fault_rank

            def configure_then_fail(*args, **kwargs):
                nonlocal configure_failure_pending
                original_configure(*args, **kwargs)
                if configure_failure_pending:
                    configure_failure_pending = False
                    raise RuntimeError(
                        "injected replacement snapshot configure after-effect"
                    )

            configured_leaf.configure_checkpoint_snapshot = configure_then_fail
            replacement_error = None
            try:
                manager.load_checkpoint(
                    str(checkpoint_dir),
                    with_model=True,
                    with_optimizer=True,
                    with_rng=True,
                )
            except BaseException as error:
                replacement_error = f"{type(error).__name__}: {error}"
            finally:
                configured_leaf.configure_checkpoint_snapshot = original_configure
            replacement_errors = [None for _ in range(world_size)]
            dist.all_gather_object(
                replacement_errors, replacement_error, group=control_group
            )
            if not all(error is not None for error in replacement_errors):
                raise AssertionError(
                    "replacement configure fault lacked consensus: "
                    f"{replacement_errors}"
                )
            retained_recovery = manager._managed_checkpoint_recovery_transaction
            if retained_recovery is None:
                raise AssertionError("manager lost the failed replacement generation")
            if retained_recovery.reload_generation is None:
                raise AssertionError("replacement failure lost its reload generation")
            if retained_recovery.reload_generation.active_attempt is not None:
                raise AssertionError("replacement failure retained an active attempt")
            if any(
                leaf.optimizer.checkpoint_lifecycle != "RELOAD_REQUIRED"
                for leaf in optimizer.chained_optimizers
            ):
                raise AssertionError(
                    "replacement configure failure did not preserve RELOAD_REQUIRED"
                )
            replacement_configure_failed = True
            manager.load_checkpoint(
                str(checkpoint_dir),
                with_model=True,
                with_optimizer=True,
                with_rng=True,
            )
            if not partial_fault.get("recovery_prefix_verified"):
                raise AssertionError("public retry skipped retained rollback recovery")
            if rank == fault_rank:
                counts = partial_fault["restore_counts"]
                failing_action = partial_fault["failing_action"]
                if counts[failing_action] != 2:
                    raise AssertionError("pending rollback action was not retried once")
                replayed = {
                    name: count
                    for name, count in counts.items()
                    if name != failing_action and count != 1
                }
                if replayed:
                    raise AssertionError(
                        f"completed rollback actions were replayed: {replayed}"
                    )
        else:
            if any(error is not None for error in first_errors):
                raise AssertionError(f"public manager reshard failed: {first_errors}")

        if _slab_pointers(optimizer) != before_pointers:
            raise AssertionError("public manager load replaced authoritative CPU slabs")
        expected_rng = torch.load(rng_path, map_location="cpu", weights_only=False)
        _assert_rng_draws_equal(_rng_draws(), expected_rng)
        for step in range(2, 5):
            _set_gradients(model, step)
            optimizer.step()
            optimizer.drain()
            scheduler.step()
        expected = torch.load(expected_path, map_location="cpu", weights_only=False)
        max_model_error, max_optimizer_errors = _assert_manager_checkpoint_expected(
            expected,
            optimizer,
            model,
            scheduler,
            pg_collection,
            control_group,
        )
        if fault_enabled:
            second_checkpoint = output_dir / "manager_checkpoint_second"
            manager.save_checkpoint(
                str(second_checkpoint),
                with_model=True,
                with_optimizer=True,
                with_rng=True,
            )
            manager.load_checkpoint(
                str(second_checkpoint),
                with_model=True,
                with_optimizer=True,
                with_rng=True,
            )
            second_cycle = True
        load_cuda_peak_bytes = torch.cuda.max_memory_allocated() - load_start
    else:
        raise RuntimeError(f"unsupported public manager phase {phase!r}")

    _assert_cpu_slab_contract(optimizer)
    if optimizer.residency != "CPU_RESIDENT" or optimizer.cuda_state_numel != 0:
        raise AssertionError("public manager reshard violated CPU residency")
    _assert_manager_clean(manager, optimizer, snapshot_root)
    checkpoint_bytes = (
        sum(path.stat().st_size for path in checkpoint_dir.rglob("*") if path.is_file())
        if rank == 0
        else 0
    )
    checkpoint_sizes = [checkpoint_bytes]
    dist.broadcast_object_list(checkpoint_sizes, src=0, group=dist.group.WORLD)
    result = {
        "phase": phase,
        "rank": rank,
        "world_size": world_size,
        "fault_enabled": fault_enabled,
        "first_load_failed": first_load_failed,
        "fault_rank": fault_rank,
        "replacement_configure_failed": replacement_configure_failed,
        "second_cycle": second_cycle,
        "phase_trace": phase_trace,
        "partial_changed": partial_fault.get("changed", []),
        "partial_unchanged": partial_fault.get("unchanged", []),
        "recovery_prefix_verified": partial_fault.get(
            "recovery_prefix_verified", False
        ),
        "checkpoint_bytes": checkpoint_sizes[0],
        "load_cuda_peak_bytes": load_cuda_peak_bytes,
        "rollback_disk_final_bytes": _directory_file_bytes(snapshot_root),
        "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "residency": optimizer.residency,
        "cuda_state_numel": optimizer.cuda_state_numel,
        "max_model_error": max_model_error,
        "max_optimizer_errors": max_optimizer_errors,
    }
    (output_dir / f"manager_{phase}_rank_{rank}.json").write_text(
        json.dumps(result, sort_keys=True) + "\n"
    )
    health = torch.tensor([rank + 1.0], device="cuda")
    dist.all_reduce(health, group=dist.group.WORLD)
    if health.item() != world_size * (world_size + 1) / 2:
        raise AssertionError("public manager post-load NCCL health probe failed")


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    checkpoint_phase = os.environ.get("MUON_CHECKPOINT_PHASE", "fixed")
    topology = os.environ.get("MUON_CHECKPOINT_TOPOLOGY", "dp2")
    checkpoint_model = os.environ.get("MUON_CHECKPOINT_MODEL", "mixed")
    tp_size = 2 if topology == "tp2_dp2" else 1
    ep_size = 2 if checkpoint_model == "dense_expert" else 1
    expert_tp_size = 1 if checkpoint_model == "dense_expert" else tp_size
    expected_world_size = (
        world_size
        if topology in {"dp_reshard", "manager_reshard"}
        else (4 if topology == "tp2_dp2" else 2)
    )
    if topology not in {
        "dp2",
        "dp2_empty_owner",
        "tp2_dp2",
        "dp_reshard",
        "manager_reshard",
    }:
        raise RuntimeError(f"unsupported checkpoint topology {topology!r}")
    if world_size != expected_world_size:
        raise RuntimeError(
            f"checkpoint topology {topology} requires {expected_world_size} ranks, "
            f"got {world_size}"
        )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    output_dir = Path(os.environ["ACCEPTANCE_OUTPUT_DIR"])
    snapshot_root = output_dir / f"rollback_{checkpoint_phase}"
    checkpoint_dir = output_dir / "checkpoint"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_root.mkdir()
        if checkpoint_phase in {"fixed", "save"}:
            checkpoint_dir.mkdir()
    dist.barrier()
    _record_phase(output_dir, rank, "world_initialized")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=expert_tp_size,
    )
    try:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        if checkpoint_phase == "metadata_fault":
            control_group = dist.new_group(
                ranks=list(range(world_size)), backend="gloo"
            )
            _run_metadata_authority_fault(
                output_dir=output_dir,
                snapshot_root=snapshot_root,
                checkpoint_dir=checkpoint_dir,
                pg_collection=pg_collection,
                control_group=control_group,
            )
            return
        if checkpoint_phase in {"save", "load"}:
            control_group = dist.new_group(
                ranks=list(range(world_size)), backend="gloo"
            )
            if topology == "manager_reshard":
                _run_manager_reshard_phase(
                    phase=checkpoint_phase,
                    output_dir=output_dir,
                    snapshot_root=snapshot_root,
                    checkpoint_dir=checkpoint_dir,
                    pg_collection=pg_collection,
                    control_group=control_group,
                )
                return
            _run_reshard_phase(
                phase=checkpoint_phase,
                output_dir=output_dir,
                snapshot_root=snapshot_root,
                checkpoint_dir=checkpoint_dir,
                pg_collection=pg_collection,
                control_group=control_group,
                model_kind=checkpoint_model,
            )
            return
        model_parallel_cuda_manual_seed(20260816, force_reset_rng=True)
        baseline_model = (
            _TinyMuonOnlyModel()
            if topology == "dp2_empty_owner"
            else _TinyMuonModel(tp_size=tp_size, pg_collection=pg_collection)
        )
        baseline = get_megatron_optimizer_with_gpu_staged_muon(
            _optimizer_config(),
            [baseline_model],
            _staged_config(snapshot_root),
            pg_collection=pg_collection,
        )
        baseline.bind_managed_checkpoint_process_group(dist.group.WORLD)
        _record_phase(output_dir, rank, "baseline_built")
        for step in range(2):
            _set_gradients(baseline_model, step)
            baseline.step()
            baseline.drain()
        _record_phase(output_dir, rank, "baseline_stepped")

        checkpoint_model_state = {
            name: (
                value.detach().clone()
                if isinstance(value, torch.Tensor)
                else copy.deepcopy(value)
            )
            for name, value in baseline_model.state_dict().items()
        }
        checkpoint_optimizer_state = _local_state(baseline, baseline_model)
        checkpoint_group_metadata = _local_group_metadata(baseline)
        sharded = baseline.sharded_state_dict({})
        _record_phase(output_dir, rank, "save_template_built")
        dist_checkpointing.save(
            {"optimizer": sharded},
            str(checkpoint_dir),
            async_sharded_save=False,
        )
        _record_phase(output_dir, rank, "dcp_saved")
        local_manifest = build_managed_optimizer_tensor_manifest(sharded)
        manifests = [None for _ in range(world_size)]
        dist.all_gather_object(manifests, local_manifest)
        manifest = merge_managed_optimizer_tensor_manifests(manifests)
        validate_managed_optimizer_source_tensor_metadata(str(checkpoint_dir), manifest)
        _record_phase(output_dir, rank, "manifest_validated")

        resumed_model = (
            _TinyMuonOnlyModel()
            if topology == "dp2_empty_owner"
            else _TinyMuonModel(tp_size=tp_size, pg_collection=pg_collection)
        )
        resumed_model.load_state_dict(checkpoint_model_state)
        resumed = get_megatron_optimizer_with_gpu_staged_muon(
            _optimizer_config(),
            [resumed_model],
            _staged_config(snapshot_root),
            pg_collection=pg_collection,
        )
        resumed.bind_managed_checkpoint_process_group(dist.group.WORLD)
        _record_phase(output_dir, rank, "resumed_built")
        before_pointers = _slab_pointers(resumed)
        torch.cuda.reset_peak_memory_stats()
        load_start = torch.cuda.memory_allocated()
        transaction = begin_managed_checkpoint_load(resumed)
        _record_phase(output_dir, rank, "rollback_snapshots_built")
        template = {"optimizer": resumed.sharded_state_dict({})}
        _record_phase(output_dir, rank, "load_template_built")
        loaded = dist_checkpointing.load(template, str(checkpoint_dir))
        _record_phase(output_dir, rank, "dcp_loaded")
        resumed.load_state_dict(loaded["optimizer"])
        _record_phase(output_dir, rank, "state_applied")
        prepare_managed_checkpoint_load(transaction)
        commit_managed_checkpoint_load(transaction)
        _record_phase(output_dir, rank, "load_committed")
        load_peak = torch.cuda.max_memory_allocated() - load_start

        _assert_local_state_equal(
            checkpoint_optimizer_state, _local_state(resumed, resumed_model)
        )
        if _local_group_metadata(resumed) != checkpoint_group_metadata:
            raise AssertionError("checkpoint param-group metadata was not restored")
        after_pointers = _slab_pointers(resumed)
        if after_pointers != before_pointers:
            raise AssertionError("checkpoint load replaced an authoritative CPU slab")
        _assert_cpu_slab_contract(resumed)

        for step in range(2, 5):
            _set_gradients(baseline_model, step)
            baseline.step()
            baseline.drain()
            _set_gradients(resumed_model, step)
            resumed.step()
            resumed.drain()
        _record_phase(output_dir, rank, "continued_three_steps")
        for (name, baseline_parameter), (_, resumed_parameter) in zip(
            baseline_model.named_parameters(),
            resumed_model.named_parameters(),
            strict=True,
        ):
            torch.testing.assert_close(
                resumed_parameter, baseline_parameter, rtol=0.0, atol=0.0
            )
        _assert_local_state_equal(
            _local_state(baseline, baseline_model),
            _local_state(resumed, resumed_model),
        )
        if _local_group_metadata(resumed) != _local_group_metadata(baseline):
            raise AssertionError("resumed param-group metadata diverged from baseline")
        if resumed.residency != "CPU_RESIDENT" or resumed.cuda_state_numel != 0:
            raise AssertionError("resumed optimizer retained CUDA optimizer state")
        _assert_cpu_slab_contract(resumed)

        checkpoint_bytes = (
            sum(
                path.stat().st_size
                for path in checkpoint_dir.rglob("*")
                if path.is_file()
            )
            if rank == 0
            else 0
        )
        checkpoint_size = [checkpoint_bytes]
        dist.broadcast_object_list(checkpoint_size, src=0)
        result = {
            "rank": rank,
            "world_size": world_size,
            "topology": topology,
            "residency": resumed.residency,
            "cuda_state_numel": resumed.cuda_state_numel,
            "load_cuda_peak_bytes": load_peak,
            "checkpoint_bytes": checkpoint_size[0],
            "owned_parameters": sorted(_local_state(resumed, resumed_model)),
            "continued_steps": 3,
        }
        (output_dir / f"rank_{rank}.json").write_text(
            json.dumps(result, sort_keys=True) + "\n"
        )
        health = torch.tensor([rank + 1.0], device="cuda")
        dist.all_reduce(health)
        expected_health = world_size * (world_size + 1) / 2
        if health.item() != expected_health:
            raise AssertionError("post-checkpoint NCCL health probe failed")
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
