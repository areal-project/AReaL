# SPDX-License-Identifier: Apache-2.0

"""Real DP managed-async checkpoint/fence acceptance without a model forward."""

from __future__ import annotations

import argparse
import json
import os
import resource
import stat
import time
from datetime import timedelta
from pathlib import Path
from types import CodeType, FunctionType, SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.dist_checkpointing import serialization
from megatron.core.dist_checkpointing.strategies.async_utils import (
    AsyncCallsQueue,
    AsyncRequest,
)
from megatron.core.dist_checkpointing.strategies.state_dict_saver import (
    save_state_dict_async_finalize,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistSaveShardedStrategy,
)
from run_gpu_staged_optimizer_checkpoint import (
    _build_optimizer,
    _step,
)

from areal.engine.megatron_utils import managed_async_finalize as managed_finalize
from areal.engine.megatron_utils.checkpointer import MegatronCheckpointManager
from areal.engine.megatron_utils.managed_async_checkpoint import (
    ManagedAsyncSaveTransaction,
)
from areal.engine.megatron_utils.managed_async_finalize import (
    abort_managed_async_calls,
    finalize_managed_async_calls,
    get_managed_async_worker_recovery,
)
from areal.engine.megatron_utils.managed_async_marker import (
    MANAGED_ASYNC_COMPLETE,
    MANAGED_ASYNC_INCOMPLETE,
)


def _fd_targets() -> dict[int, str]:
    targets = {}
    for name in os.listdir("/proc/self/fd"):
        try:
            targets[int(name)] = os.readlink(f"/proc/self/fd/{name}")
        except FileNotFoundError:
            continue
    return targets


def _manager(optimizer, scheduler, checkpoint_group, model_param, *, async_save):
    manager = MegatronCheckpointManager(
        model=torch.nn.ModuleList(),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        async_save=async_save,
        checkpoint_process_group=checkpoint_group,
        managed_checkpoint_enabled=True,
    )
    manager._managed_model_parameter_names = lambda: {model_param: "model.parameter"}
    return manager


def _run_rank_local_finalize_failure_probe(
    checkpoint_group,
    checkpoint_dir: str,
    output_dir: str,
    rank: int,
    *,
    inject_callback_detail_failure: bool = False,
    inject_unreaped_worker: bool = False,
    partial_schedule_owner: int | None = None,
    failure_pop_owner: int | None = None,
    failure_pop_mode: str = "pre",
) -> None:
    """Exercise the MCore queue boundary without model/DCP startup in the timeout."""

    marker_dir = Path(checkpoint_dir)
    if (
        partial_schedule_owner is not None or failure_pop_owner is not None
    ) and rank == 0:
        marker_dir.mkdir(parents=True, exist_ok=True)
        (marker_dir / MANAGED_ASYNC_INCOMPLETE).write_text("probe")
    if partial_schedule_owner is not None or failure_pop_owner is not None:
        dist.barrier(group=checkpoint_group)

    phase_round = "initial"
    phase_trace: dict[str, list[tuple[int | None, str]]] = {phase_round: []}
    original_vote = managed_finalize._vote
    publication_write_context = ""

    def traced_vote(group, phase, error, *, phase_id=None, **kwargs):
        nonlocal publication_write_context
        phase_trace.setdefault(phase_round, []).append((phase_id, phase))
        reports = original_vote(
            group,
            phase,
            error,
            phase_id=phase_id,
            **kwargs,
        )
        if phase.endswith("_prepare"):
            publication_write_context = phase.removesuffix("_prepare")
        return reports

    managed_finalize._vote = traced_vote

    class ProbeWriter:
        checkpoint_dir = "/managed-finalize-probe"
        results_queue = None

        def retrieve_write_results(self):
            if rank == 1 and not inject_callback_detail_failure:
                raise RuntimeError("injected rank-local MCore finalize result failure")
            return []

        def finish(self, _metadata, _all_results):
            return None

    writer = ProbeWriter()
    global_metadata = object()
    dist_wrapper = SimpleNamespace(group=None, coordinator_rank=0)

    def closure_cell(value):
        return (lambda: value).__closure__[0]

    def nested_code(producer, name, freevars):
        matches = [
            value
            for value in producer.__code__.co_consts
            if isinstance(value, CodeType)
            and value.co_name == name
            and value.co_freevars == freevars
        ]
        assert len(matches) == 1
        return matches[0]

    save_state_dict_ret = (writer, global_metadata, dist_wrapper)
    finalize_fn = FunctionType(
        nested_code(
            TorchDistSaveShardedStrategy._get_save_and_finalize_callbacks,
            "finalize_fn",
            ("save_state_dict_async_finalize", "save_state_dict_ret"),
        ),
        TorchDistSaveShardedStrategy._get_save_and_finalize_callbacks.__globals__,
        closure=(
            closure_cell(save_state_dict_async_finalize),
            closure_cell(save_state_dict_ret),
        ),
    )

    checkpoint_dir = writer.checkpoint_dir
    if inject_callback_detail_failure and rank == 1:

        class InvalidVersion:
            def __int__(self):
                raise RuntimeError("injected callback detail conversion failure")

        sharded_strategy = SimpleNamespace(
            backend="torch_dist", version=InvalidVersion()
        )
    else:
        sharded_strategy = SimpleNamespace(backend="torch_dist", version=1)

    metadata_finalize_fn = FunctionType(
        nested_code(
            serialization.save,
            "metadata_finalize_fn",
            ("checkpoint_dir", "sharded_strategy"),
        ),
        serialization.save.__globals__,
        closure=(closure_cell(checkpoint_dir), closure_cell(sharded_strategy)),
    )

    queue = AsyncCallsQueue()
    request = AsyncRequest(None, (), [finalize_fn, metadata_finalize_fn])
    if partial_schedule_owner is None or rank == partial_schedule_owner:
        queue.schedule_async_request(request)

    pop_attempt_count = 0
    pop_remove_count = 0
    remaining_pre_pop_failures = (
        2 if rank == failure_pop_owner and failure_pop_mode in ("pre", "hold") else 0
    )
    remaining_post_pop_failures = (
        1 if rank == failure_pop_owner and failure_pop_mode == "post" else 0
    )
    remaining_clear_failures = int(
        rank == failure_pop_owner
        and failure_pop_mode == "clear"
        or failure_pop_owner is not None
        and rank == 1 - failure_pop_owner
        and failure_pop_mode == "republish"
    )
    remaining_publication_failures = 2
    original_pop_all = managed_finalize._pop_all_calls
    original_remove_active = managed_finalize._remove_active_call
    original_clear_publication = managed_finalize._clear_worker_recovery_publication
    original_write_publication = managed_finalize._write_worker_recovery_publication

    def fail_pre_pop(async_queue, *, expected_call_idx=None):
        nonlocal pop_attempt_count, remaining_pre_pop_failures
        pop_attempt_count += 1
        if remaining_pre_pop_failures:
            remaining_pre_pop_failures -= 1
            raise RuntimeError(
                f"injected rank {rank} failure before transaction queue pop"
            )
        return original_pop_all(
            async_queue,
            expected_call_idx=expected_call_idx,
        )

    def counted_remove(async_queue, active):
        nonlocal pop_remove_count, remaining_post_pop_failures
        pop_remove_count += 1
        result = original_remove_active(async_queue, active)
        if remaining_post_pop_failures:
            remaining_post_pop_failures -= 1
            raise RuntimeError(
                f"injected rank {rank} failure after transaction queue pop"
            )
        return result

    def fail_clear_publication(async_queue):
        nonlocal remaining_clear_failures
        if remaining_clear_failures:
            remaining_clear_failures -= 1
            raise RuntimeError(
                f"injected rank {rank} failure before recovery publication clear"
            )
        return original_clear_publication(async_queue)

    def fail_publication_write(async_queue, publication):
        nonlocal remaining_publication_failures
        if rank != failure_pop_owner or remaining_publication_failures == 0:
            return original_write_publication(async_queue, publication)
        if (
            failure_pop_mode == "publish-after"
            and publication_write_context == "failure_worker_recovery_publish"
        ):
            remaining_publication_failures = 0
            original_write_publication(async_queue, publication)
            raise RuntimeError("injected publication write failure after effect")
        expected_phase = {
            "hold": "failure_worker_recovery_hold",
            "republish": "failure_worker_recovery_republish",
        }.get(failure_pop_mode)
        if publication_write_context == expected_phase:
            remaining_publication_failures -= 1
            raise RuntimeError(
                f"injected {publication_write_context} failure before effect"
            )
        return original_write_publication(async_queue, publication)

    if failure_pop_owner is not None:
        managed_finalize._pop_all_calls = fail_pre_pop
        managed_finalize._remove_active_call = counted_remove
        managed_finalize._clear_worker_recovery_publication = fail_clear_publication
        managed_finalize._write_worker_recovery_publication = fail_publication_write

    class ProbeProcess:
        def __init__(self, alive: bool):
            self.alive = alive
            self.exitcode = None if alive else 0
            self.terminate_count = 0
            self.kill_count = 0
            self.join_count = 0
            self.close_count = 0

        def is_alive(self):
            return self.alive

        def join(self, _timeout=None):
            self.join_count += 1

        def terminate(self):
            self.terminate_count += 1

        def kill(self):
            self.kill_count += 1

        def close(self):
            self.close_count += 1

    probe_process = None
    if inject_unreaped_worker or (
        partial_schedule_owner is not None and rank == partial_schedule_owner
    ):
        probe_process = ProbeProcess(
            alive=(
                rank == 1
                if partial_schedule_owner is None
                else rank == partial_schedule_owner
            )
        )
        assert queue.async_calls
        queue.async_calls[0].async_caller.process = probe_process

    failure = None
    probe_recovery_token = managed_finalize.ManagedAsyncRecoveryToken()
    try:
        if partial_schedule_owner is None:
            finalize_managed_async_calls(
                queue,
                checkpoint_group,
                expected_call_idx=0,
                bound_call_idx=0,
                blocking=True,
                timeout_seconds=10.0,
                recovery_token=(
                    probe_recovery_token
                    if failure_pop_owner is not None
                    or partial_schedule_owner is not None
                    or inject_unreaped_worker
                    else None
                ),
            )
        else:
            abort_managed_async_calls(
                queue,
                checkpoint_group,
                recovery_token=probe_recovery_token,
            )
    except BaseException as error:
        failure = error
    assert failure is not None
    first_queue_depth = queue.get_num_unfinalized_calls()
    recovery_visible_first = get_managed_async_worker_recovery(queue) is not None
    manager_fence_retained = True
    manager_lease_retained = True
    manager_marker_retained = True
    manager_release_count = 0
    manager_leaf_fail_count = 0
    manager_retention: list[dict[str, bool]] = []
    manager = None
    manager_transaction = None
    manager_probe = partial_schedule_owner is not None or failure_pop_owner is not None
    if manager_probe:

        class ProbeLeaf:
            async_save_state = "SAVE_IN_FLIGHT"

            def fail_async_checkpoint_save(self, _error):
                nonlocal manager_leaf_fail_count
                manager_leaf_fail_count += 1
                self.async_save_state = "FAILED"

        class ProbeMarker:
            def __init__(self):
                self.close_count = 0

            def close(self):
                self.close_count += 1

        leaf = ProbeLeaf()
        marker = ProbeMarker()

        def release_lease():
            nonlocal manager_release_count
            manager_release_count += 1

        manager_transaction = ManagedAsyncSaveTransaction(
            checkpoint_id="partial-schedule-checkpoint",
            path="/managed-finalize-probe",
            leaves=(leaf,),
            control_group=checkpoint_group,
            logical_call_id=1,
            expected_call_idx=0,
            marker_leaves=[],
            marker_leaves_digest="probe",
        )
        manager_transaction.request = request
        manager_transaction.recovery_token = probe_recovery_token
        manager_transaction.marker_authority = marker
        manager_transaction.completion_callbacks.append(release_lease)
        manager = object.__new__(MegatronCheckpointManager)
        manager._async_queue = queue
        manager._managed_async_save = manager_transaction
        manager._managed_async_save_error = None
        manager._managed_async_marker_precommit_cleanup = None
        manager._managed_async_last_state = manager_transaction.state
        manager._record_managed_async_failure(manager_transaction, failure)
        manager_fence_retained = leaf.async_save_state == "SAVE_IN_FLIGHT"
        manager_lease_retained = manager_release_count == 0
        manager_marker_retained = manager_transaction.marker_authority is marker
        if failure_pop_mode == "publish-after":
            assert manager_transaction.worker_recovery is None
            assert not manager_fence_retained
            assert not manager_lease_retained
            assert not manager_marker_retained
        else:
            assert manager_transaction.worker_recovery is not None
            assert manager_fence_retained
            assert manager_lease_retained
            assert manager_marker_retained
        manager_retention.append(
            {
                "fence": manager_fence_retained,
                "lease": manager_lease_retained,
                "marker": manager_marker_retained,
            }
        )
    queue_depths = [first_queue_depth]
    recovery_visibility = [recovery_visible_first]
    recovery_required = recovery_visible_first or (
        failure_pop_owner is not None
        and probe_recovery_token.state
        is managed_finalize.ManagedAsyncRecoveryState.RECOVERY_REQUIRED
    )
    if inject_unreaped_worker or (manager_probe and recovery_required):
        if failure_pop_owner is not None:
            expected_first_depth = int(
                failure_pop_mode in ("pre", "hold") and rank == failure_pop_owner
            )
        elif inject_unreaped_worker:
            expected_first_depth = int(rank == 1)
        else:
            expected_first_depth = int(
                partial_schedule_owner is None or rank == partial_schedule_owner
            )
        assert first_queue_depth == expected_first_depth
        if failure_pop_mode != "republish":
            assert recovery_visible_first
        recovery_process_identity_preserved = True
        if probe_process is not None:
            owns_pending_process = not inject_unreaped_worker or rank == 1
            if owns_pending_process:
                assert probe_process.is_alive()
                recovery = get_managed_async_worker_recovery(queue)
                recovery_process_identity_preserved = (
                    getattr(recovery, "process", None) is probe_process
                )
                probe_process.alive = False
                probe_process.exitcode = -9
            else:
                assert not probe_process.is_alive()
                assert probe_process.close_count == 1
        recovery_attempts = (
            2
            if failure_pop_owner is not None and failure_pop_mode in ("pre", "hold")
            else 1
        )
        recovery_failure = None
        for attempt in range(recovery_attempts):
            phase_round = f"recovery_{attempt}"
            phase_trace[phase_round] = []
            try:
                finalize_managed_async_calls(
                    queue,
                    checkpoint_group,
                    expected_call_idx=0,
                    bound_call_idx=(0 if partial_schedule_owner is None else None),
                    blocking=True,
                    timeout_seconds=10.0,
                    recovery_token=(
                        probe_recovery_token
                        if failure_pop_owner is not None
                        or partial_schedule_owner is not None
                        or inject_unreaped_worker
                        else None
                    ),
                )
            except BaseException as error:
                recovery_failure = error
            assert recovery_failure is not None
            queue_depths.append(queue.get_num_unfinalized_calls())
            recovery_visibility.append(
                get_managed_async_worker_recovery(queue) is not None
            )
            if manager_probe:
                assert manager is not None
                assert manager_transaction is not None
                manager._record_managed_async_failure(
                    manager_transaction,
                    recovery_failure,
                )
                manager_retention.append(
                    {
                        "fence": leaf.async_save_state == "SAVE_IN_FLIGHT",
                        "lease": manager_release_count == 0,
                        "marker": manager_transaction.marker_authority is marker,
                    }
                )
        assert queue.get_num_unfinalized_calls() == 0
        assert get_managed_async_worker_recovery(queue) is None
        if manager_probe:
            assert manager is not None
            assert manager_transaction is not None
            assert manager_transaction.worker_recovery is None
            assert manager_release_count == 1
            assert manager_leaf_fail_count == 1
            assert manager_transaction.marker_authority is None
    else:
        recovery_process_identity_preserved = True
    errors = [None] * dist.get_world_size(checkpoint_group)
    dist.all_gather_object(
        errors,
        f"{type(failure).__name__}: {failure}",
        group=checkpoint_group,
    )
    assert all("finalize" in error.lower() for error in errors)
    probe = torch.tensor([rank + 1], device="cuda", dtype=torch.int64)
    dist.all_reduce(probe)
    assert probe.item() == sum(range(1, dist.get_world_size() + 1))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank{rank}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "async_state": "FAILED",
                "error": errors[rank],
                "collective_healthy": True,
                "queue_depth": queue.get_num_unfinalized_calls(),
                "first_queue_depth": first_queue_depth,
                "recovery_visible_first": recovery_visible_first,
                "recovery_process_identity_preserved": (
                    recovery_process_identity_preserved
                ),
                "partial_schedule_owner": partial_schedule_owner,
                "failure_pop_owner": failure_pop_owner,
                "failure_pop_mode": failure_pop_mode,
                "phase_trace": phase_trace,
                "queue_depths": queue_depths,
                "recovery_visibility": recovery_visibility,
                "recovery_token_state": probe_recovery_token.state.name,
                "pop_attempt_count": pop_attempt_count,
                "pop_remove_count": pop_remove_count,
                "manager_fence_retained": manager_fence_retained,
                "manager_lease_retained": manager_lease_retained,
                "manager_marker_retained": manager_marker_retained,
                "manager_retention": manager_retention,
                "manager_release_count": manager_release_count,
                "manager_leaf_fail_count": manager_leaf_fail_count,
                "incomplete": (marker_dir / MANAGED_ASYNC_INCOMPLETE).is_file(),
                "complete": (marker_dir / MANAGED_ASYNC_COMPLETE).exists(),
                "worker_close_count": (
                    probe_process.close_count if probe_process is not None else 0
                ),
                "worker_kill_count": (
                    probe_process.kill_count if probe_process is not None else 0
                ),
            },
            sort_keys=True,
        )
    )
    dist.barrier(group=checkpoint_group)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--numel", type=int, default=96)
    parser.add_argument("--inject-marker-publish-failure", action="store_true")
    parser.add_argument("--inject-rank-local-finalize-failure", action="store_true")
    parser.add_argument("--inject-callback-detail-failure", action="store_true")
    parser.add_argument("--inject-unreaped-worker", action="store_true")
    parser.add_argument(
        "--inject-partial-unbound-schedule-owner", type=int, choices=(0, 1)
    )
    parser.add_argument("--inject-failure-pop-owner", type=int, choices=(0, 1))
    parser.add_argument(
        "--inject-failure-pop-mode",
        choices=("pre", "post", "clear", "publish-after", "hold", "republish"),
        default="pre",
    )
    parser.add_argument(
        "--inject-marker-postcommit-fault",
        choices=("unlink-after-effect", "authority-close"),
    )
    args = parser.parse_args()

    timeout_seconds = (
        20
        if args.inject_rank_local_finalize_failure
        or args.inject_callback_detail_failure
        or args.inject_unreaped_worker
        or args.inject_partial_unbound_schedule_owner is not None
        or args.inject_failure_pop_owner is not None
        else 120
    )
    dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_seconds))
    checkpoint_group = dist.new_group(
        backend="gloo", timeout=timedelta(seconds=timeout_seconds)
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank()
    if (
        args.inject_rank_local_finalize_failure
        or args.inject_callback_detail_failure
        or args.inject_unreaped_worker
        or args.inject_partial_unbound_schedule_owner is not None
        or args.inject_failure_pop_owner is not None
    ):
        _run_rank_local_finalize_failure_probe(
            checkpoint_group,
            args.checkpoint_dir,
            args.output_dir,
            rank,
            inject_callback_detail_failure=args.inject_callback_detail_failure,
            inject_unreaped_worker=args.inject_unreaped_worker,
            partial_schedule_owner=args.inject_partial_unbound_schedule_owner,
            failure_pop_owner=args.inject_failure_pop_owner,
            failure_pop_mode=args.inject_failure_pop_mode,
        )
        dist.destroy_process_group()
        return
    parallel_state.initialize_model_parallel()

    (
        _initial,
        model_param,
        optimizer,
        inner,
        _owned_shard,
        _model_chunk,
        scheduler,
        _kwargs,
    ) = _build_optimizer(args.numel)
    for step in range(2):
        _step(model_param, optimizer, scheduler, step, args.numel)
    inner.drain()
    slab_pointers = tuple(
        slab.untyped_storage().data_ptr()
        for slab in (
            inner.cpu_slabs.master,
            inner.cpu_slabs.exp_avg,
            inner.cpu_slabs.exp_avg_sq,
        )
    )
    expected = {
        "master": inner.cpu_slabs.master.clone(),
        "exp_avg": inner.cpu_slabs.exp_avg.clone(),
        "exp_avg_sq": inner.cpu_slabs.exp_avg_sq.clone(),
        "step": inner.param_groups[0]["step"],
    }
    manager = _manager(
        optimizer, scheduler, checkpoint_group, model_param, async_save=True
    )
    if args.inject_marker_publish_failure and rank == 0:

        def fail_marker_publish(_transaction):
            raise RuntimeError("injected managed async marker publish failure")

        manager._prepare_managed_async_complete = fail_marker_publish
    if args.inject_marker_postcommit_fault == "unlink-after-effect" and rank == 0:
        from areal.engine.megatron_utils import managed_async_marker as marker_module

        real_unlink = marker_module.os.unlink
        injected = False

        def unlink_then_raise(name, *, dir_fd):
            nonlocal injected
            real_unlink(name, dir_fd=dir_fd)
            if name == marker_module.MANAGED_ASYNC_INCOMPLETE and not injected:
                injected = True
                raise OSError("injected incomplete unlink after-effect")

        marker_module.os.unlink = unlink_then_raise
    if args.inject_marker_postcommit_fault == "authority-close" and rank == 0:
        from areal.engine.megatron_utils import checkpoint_snapshot as snapshot_module

        real_prepare_close = snapshot_module._prepare_fd_close
        injected = False

        def fail_first_directory_close(owner):
            nonlocal injected
            if owner.file_type == stat.S_IFDIR and not injected:
                injected = True
                raise OSError("injected post-commit authority close failure")
            return real_prepare_close(owner)

        snapshot_module._prepare_fd_close = fail_first_directory_close
    fd_targets_before = _fd_targets()
    torch.cuda.reset_peak_memory_stats()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    save_started = time.perf_counter()
    manager.save_checkpoint(
        args.checkpoint_dir,
        with_model=False,
        with_optimizer=True,
        with_rng=False,
    )
    save_returned = time.perf_counter()
    rss_after_schedule = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    assert inner.async_save_state == "SAVE_IN_FLIGHT"

    if args.inject_marker_publish_failure:
        failure = None
        try:
            manager.wait_async_saves()
        except BaseException as error:
            failure = error
        assert failure is not None
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(
            errors,
            f"{type(failure).__name__}: {failure}",
            group=checkpoint_group,
        )
        assert all("async_complete_marker" in error for error in errors)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "async_state": manager.managed_async_save_state,
                    "incomplete": (
                        Path(args.checkpoint_dir)
                        / ".areal-managed-async-incomplete.json"
                    ).is_file(),
                    "complete": (
                        Path(args.checkpoint_dir) / ".areal-managed-async-complete.json"
                    ).is_file(),
                    "error": errors[rank],
                },
                sort_keys=True,
            )
        )
        dist.barrier(group=checkpoint_group)
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
        return

    # This step is the mutation fence test: it must first finalize the save,
    # then update from the checkpoint-time state.
    for continuation_step in range(2, 5):
        _step(model_param, optimizer, scheduler, continuation_step, args.numel)
    step_returned = time.perf_counter()
    inner.drain()
    assert inner.async_save_state == "COMPLETE"
    assert slab_pointers == tuple(
        slab.untyped_storage().data_ptr()
        for slab in (
            inner.cpu_slabs.master,
            inner.cpu_slabs.exp_avg,
            inner.cpu_slabs.exp_avg_sq,
        )
    )
    after_save_step = {
        "master": inner.cpu_slabs.master.clone(),
        "exp_avg": inner.cpu_slabs.exp_avg.clone(),
        "exp_avg_sq": inner.cpu_slabs.exp_avg_sq.clone(),
    }
    queue_depth_after_fence = manager._async_queue.get_num_unfinalized_calls()
    manager.close()
    fd_targets_after_save_close = _fd_targets()
    save_close_added_fd_targets = {
        fd: target
        for fd, target in fd_targets_after_save_close.items()
        if fd_targets_before.get(fd) != target
    }

    (
        _initial2,
        restored_model_param,
        restored_optimizer,
        restored_inner,
        _restored_owned_shard,
        _restored_model_chunk,
        restored_scheduler,
        _kwargs2,
    ) = _build_optimizer(args.numel)
    restored_manager = _manager(
        restored_optimizer,
        restored_scheduler,
        checkpoint_group,
        restored_model_param,
        async_save=False,
    )
    restored_manager.load_checkpoint(
        args.checkpoint_dir,
        with_model=False,
        with_optimizer=True,
        with_rng=False,
    )
    for name in ("master", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            getattr(restored_inner.cpu_slabs, name),
            expected[name],
            rtol=0.0,
            atol=0.0,
        )
    assert restored_inner.param_groups[0]["step"] == expected["step"]

    for continuation_step in range(2, 5):
        _step(
            restored_model_param,
            restored_optimizer,
            restored_scheduler,
            continuation_step,
            args.numel,
        )
    restored_inner.drain()
    errors = {}
    for name in ("master", "exp_avg", "exp_avg_sq"):
        errors[name] = (
            getattr(restored_inner.cpu_slabs, name)
            .sub(after_save_step[name])
            .abs()
            .max()
            .item()
        )
        torch.testing.assert_close(
            getattr(restored_inner.cpu_slabs, name),
            after_save_step[name],
            rtol=0.0,
            atol=0.0,
        )
    errors["model"] = (
        restored_model_param.float().sub(model_param.float()).abs().max().item()
    )
    torch.testing.assert_close(
        restored_model_param,
        model_param,
        rtol=0.0,
        atol=0.0,
    )
    assert restored_inner.cuda_state_numel == 0
    assert all(
        slab.is_pinned()
        for slab in (
            restored_inner.cpu_slabs.master,
            restored_inner.cpu_slabs.exp_avg,
            restored_inner.cpu_slabs.exp_avg_sq,
        )
    )
    restored_manager.close()
    fd_targets_end = _fd_targets()
    added_fd_targets = {
        fd: target
        for fd, target in fd_targets_end.items()
        if fd_targets_before.get(fd) != target
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank{rank}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "async_state": inner.async_save_state,
                "cuda_state_numel": restored_inner.cuda_state_numel,
                "step": restored_inner.param_groups[0]["step"],
                "scheduler_last_epoch": restored_scheduler.last_epoch,
                "scheduler_lr": restored_scheduler.get_last_lr(),
                "errors": errors,
                "save_schedule_seconds": save_returned - save_started,
                "step_fence_seconds": step_returned - save_returned,
                "rss_before": rss_before,
                "rss_after_schedule": rss_after_schedule,
                "rss_peak": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "cuda_peak": torch.cuda.max_memory_allocated(),
                "checkpoint_bytes": sum(
                    item.stat().st_size
                    for item in Path(args.checkpoint_dir).rglob("*")
                    if item.is_file()
                ),
                "queue_depth": queue_depth_after_fence,
                "fd_delta": len(fd_targets_end) - len(fd_targets_before),
                "added_fd_targets": added_fd_targets,
                "save_close_fd_delta": len(fd_targets_after_save_close)
                - len(fd_targets_before),
                "save_close_added_fd_targets": save_close_added_fd_targets,
                "checkpoint_complete": (
                    Path(args.checkpoint_dir) / ".areal-managed-async-complete.json"
                ).is_file(),
                "slab_storage_preserved": True,
            },
            sort_keys=True,
        )
    )
    dist.barrier(group=checkpoint_group)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
