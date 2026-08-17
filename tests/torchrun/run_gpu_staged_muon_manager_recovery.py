# SPDX-License-Identifier: Apache-2.0

"""DP=2 public-manager recovery fault matrix for empty staged Muon leaves."""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

import areal.engine.megatron_utils.checkpointer as checkpointer_module
from areal.engine.megatron_utils.checkpointer import MegatronCheckpointManager
from areal.engine.megatron_utils.gpu_staged_muon import GPUStagedEmptyOptimizer
from areal.engine.megatron_utils.gpu_staged_optimizer_checkpoint import (
    abort_managed_checkpoint_load,
    apply_begin_managed_checkpoint_load,
    create_managed_checkpoint_load_transaction,
    poison_managed_checkpoint_transaction,
)


def _expect_consensus_failure(operation, group) -> None:
    local = None
    try:
        operation()
    except BaseException as error:
        local = f"{type(error).__name__}: {error}"
    reports = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(reports, local, group=group)
    if any(report is None for report in reports):
        raise AssertionError(f"checkpoint failure was not rank-consistent: {reports}")


def _configure_public_load(manager, leaf: GPUStagedEmptyOptimizer) -> None:
    manager._managed_optimizer_identities = lambda: {(): {}}
    manager._build_checkpoint_load_template = lambda **kwargs: {}
    manager._load_checkpoint_data = lambda *args, **kwargs: {}
    manager._apply_checkpoint_state = lambda *args, **kwargs: leaf.load_state_dict(
        {"state": {}, "param_groups": []}
    )
    manager.get_rng_state = lambda **kwargs: {"rng": "old"}
    manager.load_rng_states = lambda state: None


def _run_case(
    *,
    output_dir: Path,
    control_group,
    optimizer_kind: str,
    failure_timing: str,
    failure_rank: int,
) -> dict[str, object]:
    rank = dist.get_rank()
    case_name = f"{optimizer_kind}-{failure_timing}-rank{failure_rank}"
    common_path = output_dir / case_name / "common"
    mismatch_paths = [
        output_dir / case_name / "mismatch-rank0",
        output_dir / case_name / "mismatch-rank1",
    ]
    if rank == 0:
        common_path.mkdir(parents=True)
        for path in mismatch_paths:
            path.mkdir()
    dist.barrier(group=control_group)

    leaf = GPUStagedEmptyOptimizer(optimizer_kind)
    root = SimpleNamespace(optimizer=leaf)
    poisoned = create_managed_checkpoint_load_transaction(root)
    apply_begin_managed_checkpoint_load(poisoned)
    poison = RuntimeError(f"{case_name} poison")
    poison_managed_checkpoint_transaction(poisoned, poison)
    manager = MegatronCheckpointManager(
        model=[],
        optimizer=root,
        lr_scheduler=None,
        use_distributed_optimizer=True,
        checkpoint_process_group=control_group,
        managed_checkpoint_enabled=True,
    )
    manager._managed_checkpoint_poisoned_error = poison
    _configure_public_load(manager, leaf)

    phase_trace: list[str] = []
    original_vote = manager._vote_managed_phase

    def traced_vote(phase, local_error, transaction, **kwargs):
        phase_trace.append(phase)
        return original_vote(phase, local_error, transaction, **kwargs)

    manager._vote_managed_phase = traced_vote
    original_recover = leaf.prepare_checkpoint_recovery
    recovery_calls = 0
    fault_pending = rank == failure_rank

    def injected_recovery(
        *,
        attempt_token=None,
        recovery_action_token=None,
        reload_generation=None,
    ) -> None:
        nonlocal fault_pending, recovery_calls
        recovery_calls += 1
        if fault_pending and failure_timing == "pre_effect":
            fault_pending = False
            raise RuntimeError(f"{case_name} recovery pre-effect")
        original_recover(
            attempt_token=attempt_token,
            recovery_action_token=recovery_action_token,
            reload_generation=reload_generation,
        )
        if fault_pending and failure_timing == "post_effect":
            fault_pending = False
            raise RuntimeError(f"{case_name} recovery post-effect")

    leaf.prepare_checkpoint_recovery = injected_recovery
    _expect_consensus_failure(
        lambda: manager.load_checkpoint(str(common_path)), control_group
    )
    retained = manager._managed_checkpoint_recovery_transaction
    if retained is None:
        raise AssertionError("manager lost the recovery transaction")
    retained_attempt = retained.attempt_token
    retained_actions = tuple(
        authority.recovery_action_token for authority in retained.recovery_authorities
    )
    retained_completed = tuple(retained.recovery_completed)

    _expect_consensus_failure(
        lambda: manager.load_checkpoint(str(mismatch_paths[rank])), control_group
    )
    if manager._managed_checkpoint_recovery_transaction is not retained:
        raise AssertionError("request mismatch replaced the recovery transaction")
    if retained.attempt_token is not retained_attempt:
        raise AssertionError("request mismatch replaced the recovery attempt token")

    manager.load_checkpoint(str(common_path))
    if manager._managed_checkpoint_recovery_transaction is not None:
        raise AssertionError("successful replacement load retained recovery authority")
    if manager._managed_checkpoint_poisoned_error is not None:
        raise AssertionError("successful replacement load retained poison state")
    if manager._managed_checkpoint_cleanup_recovery is not None:
        raise AssertionError("successful replacement load retained cleanup state")
    if leaf.checkpoint_lifecycle != "CLEAN":
        raise AssertionError(f"empty leaf remained {leaf.checkpoint_lifecycle}")
    if leaf._checkpoint_recovery_journal is not None:
        raise AssertionError("empty leaf retained a recovery journal")
    if leaf._checkpoint_recovery_terminal_receipt is not None:
        raise AssertionError("replacement load retained the recovery receipt")

    if leaf._checkpoint_cleanup_terminal_receipt is not None:
        raise AssertionError("successful replacement load retained a cleanup receipt")
    manager.load_checkpoint(str(common_path))
    if leaf._checkpoint_cleanup_terminal_receipt is not None:
        raise AssertionError("second checkpoint cycle retained a cleanup receipt")
    if manager._managed_checkpoint_recovery_transaction is not None:
        raise AssertionError("second checkpoint cycle retained recovery authority")
    leaf.step()

    gathered_traces = [None for _ in range(dist.get_world_size(control_group))]
    dist.all_gather_object(gathered_traces, phase_trace, group=control_group)
    if any(trace != gathered_traces[0] for trace in gathered_traces[1:]):
        raise AssertionError(f"manager phase traces diverged: {gathered_traces}")
    required_phases = {
        "recovery_required",
        "recovery_discover",
        "recovery_rollback",
        "recovery_acknowledge",
    }
    if not required_phases.issubset(phase_trace):
        raise AssertionError(f"recovery phases are incomplete: {phase_trace}")

    expected_calls = 1
    if rank == failure_rank and failure_timing == "pre_effect":
        expected_calls = 2
    if recovery_calls != expected_calls:
        raise AssertionError(
            f"completed recovery action was replayed: {recovery_calls=} "
            f"{expected_calls=}"
        )
    return {
        "case": case_name,
        "rank": rank,
        "phase_trace": phase_trace,
        "recovery_calls": recovery_calls,
        "retained_action_count": len(retained_actions),
        "retained_completed_count": len(retained_completed),
        "lifecycle": leaf.checkpoint_lifecycle,
    }


def _run_existing_authority_case(
    *,
    output_dir: Path,
    control_group,
    optimizer_kind: str,
    authority_rank: int,
) -> dict[str, object]:
    rank = dist.get_rank()
    case_name = f"{optimizer_kind}-existing-authority-rank{authority_rank}"
    checkpoint_path = output_dir / case_name / "checkpoint"
    if rank == 0:
        checkpoint_path.mkdir(parents=True)
    dist.barrier(group=control_group)
    leaf = GPUStagedEmptyOptimizer(optimizer_kind)
    root = SimpleNamespace(optimizer=leaf)
    existing = None
    if rank == authority_rank:
        existing = create_managed_checkpoint_load_transaction(root)
        apply_begin_managed_checkpoint_load(existing)
    manager = MegatronCheckpointManager(
        model=[],
        optimizer=root,
        lr_scheduler=None,
        use_distributed_optimizer=True,
        checkpoint_process_group=control_group,
        managed_checkpoint_enabled=True,
    )
    _configure_public_load(manager, leaf)
    phase_trace: list[str] = []
    original_vote = manager._vote_managed_phase

    def traced_vote(phase, local_error, transaction, **kwargs):
        phase_trace.append(phase)
        return original_vote(phase, local_error, transaction, **kwargs)

    manager._vote_managed_phase = traced_vote
    _expect_consensus_failure(
        lambda: manager.load_checkpoint(str(checkpoint_path)), control_group
    )
    if rank == authority_rank:
        if leaf.checkpoint_load_attempt_token is not existing.attempt_token:
            raise AssertionError("manager failure damaged the pre-existing authority")
        abort_managed_checkpoint_load(
            existing, RuntimeError("original owner released authority"), poison=False
        )
    elif leaf.checkpoint_lifecycle != "CLEAN":
        raise AssertionError("a peer acquired authority after global preflight failure")
    dist.barrier(group=control_group)
    manager.load_checkpoint(str(checkpoint_path))
    if leaf.checkpoint_lifecycle != "CLEAN":
        raise AssertionError("replacement load did not finish cleanly")
    gathered_traces = [None for _ in range(dist.get_world_size(control_group))]
    dist.all_gather_object(gathered_traces, phase_trace, group=control_group)
    if any(trace != gathered_traces[0] for trace in gathered_traces[1:]):
        raise AssertionError(f"existing-authority phases diverged: {gathered_traces}")
    return {
        "case": case_name,
        "rank": rank,
        "phase_trace": phase_trace,
        "lifecycle": leaf.checkpoint_lifecycle,
    }


def _run_cleanup_after_effect_case(
    *,
    output_dir: Path,
    control_group,
    optimizer_kind: str,
    failure_rank: int,
) -> dict[str, object]:
    rank = dist.get_rank()
    case_name = f"{optimizer_kind}-cleanup-after-rank{failure_rank}"
    checkpoint_path = output_dir / case_name / "checkpoint"
    if rank == 0:
        checkpoint_path.mkdir(parents=True)
    dist.barrier(group=control_group)
    leaf = GPUStagedEmptyOptimizer(optimizer_kind)
    root = SimpleNamespace(optimizer=leaf)
    manager = MegatronCheckpointManager(
        model=[],
        optimizer=root,
        lr_scheduler=None,
        use_distributed_optimizer=True,
        checkpoint_process_group=control_group,
        managed_checkpoint_enabled=True,
    )
    _configure_public_load(manager, leaf)
    phase_trace: list[str] = []
    original_vote = manager._vote_managed_phase

    def traced_vote(phase, local_error, transaction, **kwargs):
        phase_trace.append(phase)
        return original_vote(phase, local_error, transaction, **kwargs)

    manager._vote_managed_phase = traced_vote
    original_acknowledge = leaf.acknowledge_checkpoint_cleanup
    failure_pending = rank == failure_rank
    acknowledge_calls = 0
    acknowledge_effects = 0

    def acknowledge_then_fail(commit_token, action_token) -> None:
        nonlocal acknowledge_calls, acknowledge_effects, failure_pending
        acknowledge_calls += 1
        terminal_before = leaf._checkpoint_cleanup_terminal_receipt
        original_acknowledge(commit_token, action_token)
        if terminal_before is None:
            acknowledge_effects += 1
        if failure_pending:
            failure_pending = False
            raise RuntimeError(f"{case_name} acknowledgement after-effect")

    leaf.acknowledge_checkpoint_cleanup = acknowledge_then_fail
    _expect_consensus_failure(
        lambda: manager.load_checkpoint(str(checkpoint_path)), control_group
    )
    if manager._managed_checkpoint_cleanup_recovery is None:
        raise AssertionError("manager lost committed cleanup authority")
    leaf.step()
    manager.load_checkpoint(str(checkpoint_path))
    if manager._managed_checkpoint_cleanup_recovery is not None:
        raise AssertionError("cleanup retry retained its transaction")
    if manager._managed_checkpoint_poisoned_error is not None:
        raise AssertionError("post-commit cleanup failure poisoned rollback state")
    if leaf.checkpoint_lifecycle != "CLEAN":
        raise AssertionError("cleanup retry did not return the leaf to CLEAN")
    if acknowledge_effects != 2:
        # One acknowledgement effect belongs to each complete checkpoint
        # cycle. Receipt reconciliation may re-enter the method, but cannot
        # replay the completed effect.
        raise AssertionError(
            f"cleanup acknowledgement effect replayed: {acknowledge_effects=}"
        )
    gathered_traces = [None for _ in range(dist.get_world_size(control_group))]
    dist.all_gather_object(gathered_traces, phase_trace, group=control_group)
    if any(trace != gathered_traces[0] for trace in gathered_traces[1:]):
        raise AssertionError(f"cleanup phases diverged: {gathered_traces}")
    return {
        "case": case_name,
        "rank": rank,
        "phase_trace": phase_trace,
        "acknowledge_calls": acknowledge_calls,
        "acknowledge_effects": acknowledge_effects,
        "lifecycle": leaf.checkpoint_lifecycle,
    }


def _run_save_directory_case(
    *,
    output_dir: Path,
    control_group,
    mode: str,
) -> dict[str, object]:
    """Exercise directory creation consensus before the first DCP operation."""
    rank = dist.get_rank()
    case_name = f"save-directory-{mode}"
    case_root = output_dir / case_name
    common_path = case_root / "checkpoint"
    if rank == 0:
        case_root.mkdir(parents=True)
        if mode == "existing":
            common_path.mkdir()
    dist.barrier(group=control_group)

    leaf = GPUStagedEmptyOptimizer("muon")
    manager = MegatronCheckpointManager(
        model=[],
        optimizer=SimpleNamespace(optimizer=leaf),
        lr_scheduler=None,
        use_distributed_optimizer=True,
        checkpoint_process_group=control_group,
        managed_checkpoint_enabled=True,
    )
    manager._managed_optimizer_identities = lambda: {(): {}}
    manager.generate_state_dict = lambda *args, **kwargs: {}
    phase_trace: list[str] = []
    original_vote = manager._vote_managed_phase

    def traced_vote(phase, local_error, transaction, **kwargs):
        phase_trace.append(phase)
        return original_vote(phase, local_error, transaction, **kwargs)

    manager._vote_managed_phase = traced_vote
    dcp_calls = 0
    original_save = checkpointer_module.save_dist_checkpointing
    original_makedirs = checkpointer_module.os.makedirs

    def record_save(*args, **kwargs):
        nonlocal dcp_calls
        del args, kwargs
        dcp_calls += 1
        return None

    def injected_makedirs(path, *args, **kwargs):
        if mode == "mkdir_failure" and rank == 1 and Path(path) == common_path:
            raise OSError("injected rank-local directory creation failure")
        return original_makedirs(path, *args, **kwargs)

    checkpointer_module.save_dist_checkpointing = record_save
    checkpointer_module.os.makedirs = injected_makedirs
    try:
        path = common_path
        expect_failure = mode in {"path_mismatch", "mkdir_failure"}
        if mode == "path_mismatch":
            path = case_root / f"checkpoint-rank{rank}"
        if expect_failure:
            _expect_consensus_failure(
                lambda: manager.save_checkpoint(
                    str(path), with_model=False, with_optimizer=True, with_rng=False
                ),
                control_group,
            )
        else:
            manager.save_checkpoint(
                str(path), with_model=False, with_optimizer=True, with_rng=False
            )
    finally:
        checkpointer_module.save_dist_checkpointing = original_save
        checkpointer_module.os.makedirs = original_makedirs

    expected_calls = 0 if mode in {"path_mismatch", "mkdir_failure"} else 1
    if dcp_calls != expected_calls:
        raise AssertionError(
            f"DCP entered unexpectedly for {mode}: {dcp_calls=} {expected_calls=}"
        )
    gathered_traces = [None for _ in range(dist.get_world_size(control_group))]
    dist.all_gather_object(gathered_traces, phase_trace, group=control_group)
    if any(trace != gathered_traces[0] for trace in gathered_traces[1:]):
        raise AssertionError(f"save-directory phases diverged: {gathered_traces}")
    if mode == "path_mismatch" and "save_directory" in phase_trace:
        raise AssertionError("path mismatch reached directory creation")
    if mode == "mkdir_failure" and "save_dcp" in phase_trace:
        raise AssertionError("directory failure reached DCP")
    return {
        "case": case_name,
        "rank": rank,
        "phase_trace": phase_trace,
        "dcp_calls": dcp_calls,
        "lifecycle": leaf.checkpoint_lifecycle,
    }


def main() -> None:
    dist.init_process_group("nccl", timeout=timedelta(seconds=45))
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise RuntimeError("Muon manager recovery acceptance requires DP=2")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    control_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=45))
    output_dir = Path(os.environ["ACCEPTANCE_OUTPUT_DIR"])
    if rank == 0:
        output_dir.mkdir(parents=True)
    dist.barrier(group=control_group)
    try:
        results = []
        for optimizer_kind in ("muon", "scalar_adamw"):
            for failure_timing in ("pre_effect", "post_effect"):
                for failure_rank in (0, 1):
                    results.append(
                        _run_case(
                            output_dir=output_dir,
                            control_group=control_group,
                            optimizer_kind=optimizer_kind,
                            failure_timing=failure_timing,
                            failure_rank=failure_rank,
                        )
                    )
            for authority_rank in (0, 1):
                results.append(
                    _run_existing_authority_case(
                        output_dir=output_dir,
                        control_group=control_group,
                        optimizer_kind=optimizer_kind,
                        authority_rank=authority_rank,
                    )
                )
            for failure_rank in (0, 1):
                results.append(
                    _run_cleanup_after_effect_case(
                        output_dir=output_dir,
                        control_group=control_group,
                        optimizer_kind=optimizer_kind,
                        failure_rank=failure_rank,
                    )
                )
        for mode in ("fresh", "existing", "path_mismatch", "mkdir_failure"):
            results.append(
                _run_save_directory_case(
                    output_dir=output_dir,
                    control_group=control_group,
                    mode=mode,
                )
            )
        gloo_health = torch.tensor([rank + 1], dtype=torch.int64)
        dist.all_reduce(gloo_health, group=control_group)
        if gloo_health.item() != 3:
            raise AssertionError("post-recovery Gloo health probe failed")
        nccl_health = torch.tensor([rank + 1], device="cuda", dtype=torch.int64)
        dist.all_reduce(nccl_health)
        if nccl_health.item() != 3:
            raise AssertionError("post-recovery NCCL health probe failed")
        (output_dir / f"rank_{rank}.json").write_text(
            json.dumps({"rank": rank, "cases": results}, sort_keys=True) + "\n"
        )
    finally:
        dist.destroy_process_group(control_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
