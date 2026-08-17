# SPDX-License-Identifier: Apache-2.0

"""Fail-closed MCore 0.17 adapter for managed async checkpoint finalization.

MCore 0.17's public ``AsyncCallsQueue.maybe_finalize_async_calls`` performs
default-WORLD CUDA collectives both before and after running finalizers.  Its
two torch-dist finalizers also contain default-WORLD collectives.  A local
exception before one of those collectives can therefore strand the other
ranks.  Managed optimizer checkpoints use this pinned compatibility adapter
instead: every fallible local operation is followed by a vote on the explicit
WORLD-sized Gloo checkpoint group.

This is deliberately not a general AsyncCallsQueue implementation.  It only
accepts the exact non-persistent, single-request MCore 0.17 layout audited
below.  Ordinary (non-managed) checkpoints continue to use MCore directly.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import traceback
from dataclasses import dataclass
from enum import Enum, auto
from types import CodeType
from typing import Any

import torch.distributed as dist

_MCORE_VERSION = "0.17.0"
_SOURCE_HASHES = {
    "AsyncCallsQueue.schedule_async_request": (
        "6161e2c9beb10fb5668627494f4b5469b57015da1bacc4a314d8f1263473c64d"
    ),
    "AsyncCallsQueue.maybe_finalize_async_calls": (
        "b62acda6ad457972cd672d1f98cc9dabfe7c859c0fc2908a22fa23ef78e6e7ac"
    ),
    "TemporalAsyncCaller.is_current_async_call_done": (
        "750e9c16d36bf34cee16434fbeee6cb3bbce3736d78acc54c3321f7ece279df3"
    ),
    "TemporalAsyncCaller.close": (
        "2266c12dac500c8e88a80929fa384824d1e9e0d5e050b1c8b38fca8241d7bb01"
    ),
    "save_state_dict_async_finalize": (
        "e833e30e9e4f946b36b047ae15bd532f05fbae2bfca8110f4d81b84dcb576f0b"
    ),
    "TorchDistSaveShardedStrategy._get_save_and_finalize_callbacks": (
        "1f061d88803b230f8428b15f5f74e97a88f3c10809a390b42052d12257b8960f"
    ),
    "serialization.save": (
        "3c7befc783133ce7348cc91614b29384ad96ffd4d247d8b39558918dd6eaaa0f"
    ),
}

_CALLBACK_CODE_HASHES = {
    "finalize_fn": "04756eedbcb6cbe97942eabc9a0fc385e0e7fcb9f228b0b04397194c3f893950",
    "metadata_finalize_fn": (
        "993c66f0678e96ab7a33d90f08b0363bd0ff67e4f2adfe1cdd31fbc1a56b4c1e"
    ),
}


class ManagedAsyncFinalizeError(RuntimeError):
    """A rank-consistent managed async finalization failure."""

    def __init__(self, phase: str, reports: list[dict[str, Any]]):
        failures = [report for report in reports if report.get("error") is not None]
        summaries = "; ".join(
            f"rank {report['global_rank']}: {report['error']}" for report in failures
        )
        super().__init__(
            f"managed async finalize phase {phase!r} failed"
            + (f": {summaries}" if summaries else "")
        )
        self.phase = phase
        self.reports = reports


@dataclass(frozen=True)
class _MCoreFinalizeCallbacks:
    writer: Any
    global_metadata: Any
    dist_wrapper: Any
    checkpoint_dir: Any
    sharded_strategy: Any


@dataclass(frozen=True)
class _MCore017Contract:
    temporal_caller_type: type
    expected_finalize_fn: Any
    active_request_type: type
    dcp_callback_code: CodeType
    dcp_callback_globals: dict[str, Any]
    metadata_callback_code: CodeType
    metadata_callback_globals: dict[str, Any]


class _WorkerReapStage(Enum):
    WAIT_PENDING = auto()
    TERMINATE_PENDING = auto()
    TERMINATE_JOIN_PENDING = auto()
    KILL_PENDING = auto()
    KILL_JOIN_PENDING = auto()
    CLOSE_PENDING = auto()
    REAPED = auto()


@dataclass
class _WorkerRecoveryJournal:
    """Unique process authority retained until its handle is safely consumed."""

    active: Any | None
    caller: Any | None
    process: Any | None
    call_idx: int
    stage: _WorkerReapStage = _WorkerReapStage.WAIT_PENDING
    exitcode: int | None = None
    worker_outcome_error: BaseException | None = None
    original_failure: BaseException | None = None
    diagnostics: dict[str, str] | None = None

    def record_error(self, stage: _WorkerReapStage, error: BaseException) -> None:
        if self.diagnostics is None:
            self.diagnostics = {}
        self.diagnostics[stage.name] = f"{type(error).__name__}: {error}"


class _WorkerTerminalCleanupStage(Enum):
    RECORD_PENDING = auto()
    RECOVERY_ATTR_PENDING = auto()
    REFERENCES_PENDING = auto()
    COMPLETE = auto()


@dataclass
class _WorkerTerminalCleanupJournal:
    """Idempotent cleanup after worker authority is already safely consumed."""

    active: Any | None
    journal: _WorkerRecoveryJournal | None
    call_idx: int
    stage: _WorkerTerminalCleanupStage = _WorkerTerminalCleanupStage.RECORD_PENDING
    diagnostic: str | None = None


@dataclass
class _WorkerRecoveryPublication:
    """Global recovery-pending token, including ranks with no local worker."""

    original_failure: BaseException
    local_authority: _WorkerRecoveryJournal | _WorkerTerminalCleanupJournal | None


class ManagedAsyncRecoveryState(Enum):
    """Manager-owned authority state for one failed async request."""

    IDLE = auto()
    RECOVERY_REQUIRED = auto()
    CLEARED = auto()


@dataclass
class ManagedAsyncRecoveryToken:
    """Shared manager/adapter sentinel; publications are only local payloads."""

    state: ManagedAsyncRecoveryState = ManagedAsyncRecoveryState.IDLE

    def require_recovery(self) -> None:
        if self.state is not ManagedAsyncRecoveryState.CLEARED:
            self.state = ManagedAsyncRecoveryState.RECOVERY_REQUIRED

    def mark_cleared(self) -> None:
        self.state = ManagedAsyncRecoveryState.CLEARED


@dataclass(frozen=True)
class _PublicationPrepareStatus:
    candidate: _WorkerRecoveryPublication | None
    error: dict[str, Any] | None

    def details(self) -> dict[str, Any]:
        return {
            "prepared": self.candidate is not None,
            "error": self.error,
        }


@dataclass(frozen=True)
class _PublicationMutationStatus:
    committed: bool
    present: bool
    error: dict[str, Any] | None

    def details(self) -> dict[str, Any]:
        return {
            "committed": self.committed,
            "present": self.present,
            "error": self.error,
        }


@dataclass(frozen=True)
class _FailureQueueCleanupStatus:
    """Rank-local progress for one failed managed checkpoint transaction."""

    call_idx: int | None
    record_removed: bool
    terminal_cleanup_pending: bool
    worker_recovery_pending: bool
    error: dict[str, Any] | None

    @property
    def complete(self) -> bool:
        return (
            self.error is None
            and self.record_removed
            and not self.terminal_cleanup_pending
            and not self.worker_recovery_pending
        )

    def details(self) -> dict[str, Any]:
        return {
            "call_idx": self.call_idx,
            "record_removed": self.record_removed,
            "terminal_cleanup_pending": self.terminal_cleanup_pending,
            "worker_recovery_pending": self.worker_recovery_pending,
            "error": self.error,
        }


_WORKER_RECOVERY_ATTR = "_areal_managed_worker_recovery"
_WORKER_TERMINAL_CLEANUP_ATTR = "_areal_managed_worker_terminal_cleanup"
_WORKER_RECOVERY_PUBLICATION_ATTR = "_areal_managed_worker_recovery_publication"


def _source_hash(value: Any) -> str:
    return hashlib.sha256(inspect.getsource(value).encode()).hexdigest()


def _nested_callback_code(
    generator: Any, name: str, expected_freevars: tuple[str, ...]
) -> CodeType:
    matches = [
        value
        for value in generator.__code__.co_consts
        if isinstance(value, CodeType)
        and value.co_name == name
        and value.co_freevars == expected_freevars
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "MCore 0.17 callback generator structure changed: "
            f"callback={name!r}, matches={len(matches)}"
        )
    code = matches[0]
    actual_hash = hashlib.sha256(code.co_code).hexdigest()
    if actual_hash != _CALLBACK_CODE_HASHES[name]:
        raise RuntimeError(
            "MCore 0.17 callback code fingerprint changed: "
            f"callback={name!r}, expected={_CALLBACK_CODE_HASHES[name]}, "
            f"actual={actual_hash}"
        )
    return code


def _validate_mcore_017_contract(queue: Any) -> _MCore017Contract:
    if importlib.metadata.version("megatron-core") != _MCORE_VERSION:
        raise RuntimeError("managed async finalize requires megatron-core==0.17.0")

    from megatron.core.dist_checkpointing import serialization
    from megatron.core.dist_checkpointing.strategies import async_utils
    from megatron.core.dist_checkpointing.strategies.state_dict_saver import (
        save_state_dict_async_finalize,
    )
    from megatron.core.dist_checkpointing.strategies.torch import (
        TorchDistSaveShardedStrategy,
    )

    checks = {
        "AsyncCallsQueue.schedule_async_request": (
            async_utils.AsyncCallsQueue.schedule_async_request
        ),
        "AsyncCallsQueue.maybe_finalize_async_calls": (
            async_utils.AsyncCallsQueue.maybe_finalize_async_calls
        ),
        "TemporalAsyncCaller.is_current_async_call_done": (
            async_utils.TemporalAsyncCaller.is_current_async_call_done
        ),
        "TemporalAsyncCaller.close": async_utils.TemporalAsyncCaller.close,
        "save_state_dict_async_finalize": save_state_dict_async_finalize,
        "TorchDistSaveShardedStrategy._get_save_and_finalize_callbacks": (
            TorchDistSaveShardedStrategy._get_save_and_finalize_callbacks
        ),
        "serialization.save": serialization.save,
    }
    mismatches = {
        name: (_SOURCE_HASHES[name], _source_hash(value))
        for name, value in checks.items()
        if _source_hash(value) != _SOURCE_HASHES[name]
    }
    if mismatches:
        raise RuntimeError(
            "MCore 0.17 managed async private contract changed: "
            f"source_hash_mismatches={mismatches}"
        )
    if type(queue) is not async_utils.AsyncCallsQueue:
        raise TypeError(
            "managed async finalize requires the exact MCore 0.17 "
            f"AsyncCallsQueue, got {type(queue)!r}"
        )
    if queue.persistent is not False:
        raise RuntimeError(
            "managed async finalize does not support MCore persistent workers"
        )
    if not hasattr(queue, "async_calls") or not hasattr(queue, "call_idx"):
        raise RuntimeError("MCore 0.17 AsyncCallsQueue private layout changed")
    if async_utils.AsyncRequest._fields != (
        "async_fn",
        "async_fn_args",
        "finalize_fns",
        "async_fn_kwargs",
        "preload_fn",
        "is_frozen",
        "call_idx",
    ):
        raise RuntimeError("MCore 0.17 AsyncRequest field layout changed")
    if async_utils._ActiveAsyncRequest._fields != (
        "idx",
        "async_caller",
        "async_request",
    ):
        raise RuntimeError("MCore 0.17 active-request field layout changed")
    dcp_callback_code = _nested_callback_code(
        TorchDistSaveShardedStrategy._get_save_and_finalize_callbacks,
        "finalize_fn",
        ("save_state_dict_async_finalize", "save_state_dict_ret"),
    )
    metadata_callback_code = _nested_callback_code(
        serialization.save,
        "metadata_finalize_fn",
        ("checkpoint_dir", "sharded_strategy"),
    )
    return _MCore017Contract(
        temporal_caller_type=async_utils.TemporalAsyncCaller,
        expected_finalize_fn=save_state_dict_async_finalize,
        active_request_type=async_utils._ActiveAsyncRequest,
        dcp_callback_code=dcp_callback_code,
        dcp_callback_globals=(
            TorchDistSaveShardedStrategy._get_save_and_finalize_callbacks.__globals__
        ),
        metadata_callback_code=metadata_callback_code,
        metadata_callback_globals=serialization.save.__globals__,
    )


def _group_manifest(group: Any) -> tuple[int, tuple[int, ...]]:
    if group is None:
        raise RuntimeError(
            "managed async finalize requires an explicit WORLD-sized Gloo group"
        )
    backend = str(dist.get_backend(group)).lower()
    if backend != "gloo":
        raise RuntimeError(
            f"managed async finalize requires an explicit Gloo group, got {backend!r}"
        )
    ranks = tuple(dist.get_process_group_ranks(group))
    size = dist.get_world_size(group)
    if len(ranks) != size:
        raise RuntimeError(
            "managed async finalize process-group membership is inconsistent: "
            f"size={size}, ranks={ranks}"
        )
    group_rank = dist.get_rank(group)
    if group_rank < 0 or group_rank >= len(ranks):
        raise RuntimeError(f"invalid managed async checkpoint group rank {group_rank}")
    return ranks[group_rank], ranks


def _error_summary(error: BaseException | None) -> dict[str, Any] | None:
    if error is None:
        return None
    return {
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )[-16384:],
    }


def _vote(
    group: Any,
    phase: str,
    local_error: BaseException | None,
    *,
    phase_id: int | None = None,
    details: Any = None,
    require_equal_details: bool = False,
) -> list[dict[str, Any]]:
    global_rank, ranks = _group_manifest(group)
    report = {
        "global_rank": global_rank,
        "phase": phase,
        "phase_id": phase_id,
        "error": _error_summary(local_error),
        "details": details,
    }
    reports: list[dict[str, Any] | None] = [None] * len(ranks)
    try:
        dist.all_gather_object(reports, report, group=group)
    except BaseException as error:
        raise ManagedAsyncFinalizeError(
            phase,
            [
                {
                    "global_rank": global_rank,
                    "phase": phase,
                    "phase_id": phase_id,
                    "error": _error_summary(error),
                    "details": details,
                }
            ],
        ) from error
    typed_reports = [item for item in reports if item is not None]
    if len(typed_reports) != len(ranks):
        raise ManagedAsyncFinalizeError(phase, typed_reports)
    if any(
        item.get("phase") != phase or item.get("phase_id") != phase_id
        for item in typed_reports
    ):
        mismatch = RuntimeError(
            "managed async finalize collective phase sequence differs: "
            f"expected=({phase_id}, {phase!r}), "
            f"actual={[(item.get('phase_id'), item.get('phase')) for item in typed_reports]}"
        )
        typed_reports[0] = {
            **typed_reports[0],
            "error": _error_summary(mismatch),
        }
    if require_equal_details and any(
        item["details"] != typed_reports[0]["details"] for item in typed_reports[1:]
    ):
        mismatch = RuntimeError(
            f"managed async finalize phase {phase!r} rank details differ"
        )
        typed_reports[0] = {
            **typed_reports[0],
            "error": _error_summary(mismatch),
        }
    if any(item["error"] is not None for item in typed_reports):
        raise ManagedAsyncFinalizeError(phase, typed_reports) from local_error
    return typed_reports


def _closure_values(callback: Any) -> dict[str, Any]:
    closure = callback.__closure__
    freevars = callback.__code__.co_freevars
    if closure is None or len(closure) != len(freevars):
        raise RuntimeError(
            f"managed async finalize callback {callback!r} has invalid closure"
        )
    return {name: cell.cell_contents for name, cell in zip(freevars, closure)}


def _parse_mcore_callbacks(
    async_request: Any, contract: _MCore017Contract
) -> _MCoreFinalizeCallbacks:
    finalize_fns = async_request.finalize_fns
    if type(finalize_fns) is not list or len(finalize_fns) != 2:
        raise RuntimeError(
            "managed async finalize expected exactly the two audited MCore 0.17 "
            f"callbacks, got {len(finalize_fns) if isinstance(finalize_fns, list) else type(finalize_fns)!r}"
        )
    dcp_callback, metadata_callback = finalize_fns
    if (
        not inspect.isfunction(dcp_callback)
        or dcp_callback.__code__ is not contract.dcp_callback_code
        or dcp_callback.__globals__ is not contract.dcp_callback_globals
    ):
        raise RuntimeError("unrecognized MCore torch-dist finalize callback")
    dcp_closure = _closure_values(dcp_callback)
    if set(dcp_closure) != {
        "save_state_dict_async_finalize",
        "save_state_dict_ret",
    }:
        raise RuntimeError(
            f"MCore torch-dist finalize closure changed: freevars={sorted(dcp_closure)}"
        )
    if (
        dcp_closure["save_state_dict_async_finalize"]
        is not contract.expected_finalize_fn
    ):
        raise RuntimeError("MCore torch-dist finalize function identity changed")
    save_ret = dcp_closure["save_state_dict_ret"]
    if not isinstance(save_ret, (tuple, list)) or len(save_ret) != 3:
        raise RuntimeError("MCore torch-dist finalize payload shape changed")

    if (
        not inspect.isfunction(metadata_callback)
        or metadata_callback.__code__ is not contract.metadata_callback_code
        or metadata_callback.__globals__ is not contract.metadata_callback_globals
    ):
        raise RuntimeError("unrecognized MCore metadata finalize callback")
    metadata_closure = _closure_values(metadata_callback)
    if set(metadata_closure) != {"checkpoint_dir", "sharded_strategy"}:
        raise RuntimeError(
            "MCore metadata finalize closure changed: "
            f"freevars={sorted(metadata_closure)}"
        )
    writer, global_metadata, dist_wrapper = save_ret
    if getattr(dist_wrapper, "group", object()) is not None:
        raise RuntimeError(
            "MCore DCP finalize unexpectedly uses a non-WORLD planning group"
        )
    if getattr(dist_wrapper, "coordinator_rank", None) != 0:
        raise RuntimeError("MCore DCP finalize coordinator changed from rank 0")
    checkpoint_dir = metadata_closure["checkpoint_dir"]
    writer_path = getattr(writer, "checkpoint_dir", checkpoint_dir)
    if str(writer_path) != str(checkpoint_dir):
        raise RuntimeError(
            "MCore DCP writer and metadata callback checkpoint paths differ"
        )
    if async_request.async_fn is not None:
        if len(async_request.async_fn_args) != 3:
            raise RuntimeError("MCore async writer argument layout changed")
        if async_request.async_fn_args[2] is not getattr(writer, "results_queue", None):
            raise RuntimeError(
                "MCore async writer result queue is not bound to the request"
            )
    return _MCoreFinalizeCallbacks(
        writer=writer,
        global_metadata=global_metadata,
        dist_wrapper=dist_wrapper,
        checkpoint_dir=checkpoint_dir,
        sharded_strategy=metadata_closure["sharded_strategy"],
    )


def _is_wrapped_exception(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], BaseException)
    )


def _worker_journal(queue: Any) -> _WorkerRecoveryJournal | None:
    journal = vars(queue).get(_WORKER_RECOVERY_ATTR)
    if journal is not None and not isinstance(journal, _WorkerRecoveryJournal):
        raise RuntimeError(
            f"managed async worker recovery journal type changed: got={type(journal)!r}"
        )
    return journal


def _worker_terminal_cleanup(
    queue: Any,
) -> _WorkerTerminalCleanupJournal | None:
    cleanup = vars(queue).get(_WORKER_TERMINAL_CLEANUP_ATTR)
    if cleanup is not None and not isinstance(cleanup, _WorkerTerminalCleanupJournal):
        raise RuntimeError(
            "managed async worker terminal cleanup journal type changed: "
            f"got={type(cleanup)!r}"
        )
    return cleanup


def _worker_recovery_publication(
    queue: Any,
) -> _WorkerRecoveryPublication | None:
    publication = vars(queue).get(_WORKER_RECOVERY_PUBLICATION_ATTR)
    if publication is not None and not isinstance(
        publication, _WorkerRecoveryPublication
    ):
        raise RuntimeError(
            "managed async worker recovery publication type changed: "
            f"got={type(publication)!r}"
        )
    return publication


def has_managed_async_worker_recovery(queue: Any) -> bool:
    """Return whether an active record still owns worker recovery authority."""

    journal = _worker_journal(queue)
    return (
        (journal is not None and journal.original_failure is not None)
        or _worker_terminal_cleanup(queue) is not None
        or _worker_recovery_publication(queue) is not None
    )


def get_managed_async_worker_recovery(
    queue: Any,
) -> (
    _WorkerRecoveryJournal
    | _WorkerTerminalCleanupJournal
    | _WorkerRecoveryPublication
    | None
):
    """Return the unique retained journal for manager-owned diagnostics."""

    journal = _worker_journal(queue)
    if journal is not None and journal.original_failure is not None:
        return journal
    cleanup = _worker_terminal_cleanup(queue)
    if cleanup is not None:
        return cleanup
    publication = _worker_recovery_publication(queue)
    if publication is not None:
        return publication.local_authority or publication
    return None


def _ensure_worker_journal(queue: Any, active: Any) -> _WorkerRecoveryJournal:
    existing = _worker_journal(queue)
    caller = active.async_caller
    process = caller.process
    if existing is not None:
        if (
            existing.active is not active
            or existing.caller is not caller
            or existing.call_idx != active.idx
        ):
            raise RuntimeError(
                "managed async worker recovery authority changed while pending"
            )
        if (
            existing.stage is not _WorkerReapStage.REAPED
            and caller.process is not existing.process
        ):
            raise RuntimeError(
                "managed async worker process identity changed while recovery was pending"
            )
        return existing
    journal = _WorkerRecoveryJournal(
        active=active,
        caller=caller,
        process=process,
        call_idx=active.idx,
    )
    setattr(queue, _WORKER_RECOVERY_ATTR, journal)
    return journal


def _process_is_alive(journal: _WorkerRecoveryJournal) -> bool:
    if journal.process is None:
        return False
    return bool(journal.process.is_alive())


def _mark_worker_reaped(journal: _WorkerRecoveryJournal) -> None:
    journal.stage = _WorkerReapStage.REAPED
    journal.caller.process = None
    journal.caller.start_time = None
    journal.caller.preloaded_holder = None


def _process_handle_consumed(process: Any) -> bool:
    # multiprocessing.process.BaseProcess.close sets this private flag before
    # returning.  It is the only audited MCore 0.17 signal that a close which
    # raised after effect has nevertheless consumed the handle.
    return getattr(process, "_closed", False) is True


def _advance_worker_recovery(
    journal: _WorkerRecoveryJournal,
    *,
    timeout_seconds: float,
    abort: bool,
) -> None:
    """Advance only pending worker actions; never discard live authority."""

    while journal.stage is not _WorkerReapStage.REAPED:
        stage = journal.stage
        try:
            if stage is _WorkerReapStage.WAIT_PENDING:
                if journal.process is None:
                    _mark_worker_reaped(journal)
                    continue
                if abort:
                    if _process_is_alive(journal):
                        journal.stage = _WorkerReapStage.TERMINATE_PENDING
                        continue
                    journal.process.join(timeout_seconds)
                    journal.exitcode = journal.process.exitcode
                    journal.stage = _WorkerReapStage.CLOSE_PENDING
                    continue
                journal.process.join(timeout_seconds)
                if _process_is_alive(journal):
                    journal.worker_outcome_error = TimeoutError(
                        "managed async checkpoint worker exceeded finalize timeout "
                        f"{timeout_seconds:.3f}s"
                    )
                    journal.stage = _WorkerReapStage.TERMINATE_PENDING
                    continue
                journal.exitcode = journal.process.exitcode
                journal.stage = _WorkerReapStage.CLOSE_PENDING
                continue

            if stage is _WorkerReapStage.TERMINATE_PENDING:
                if not _process_is_alive(journal):
                    journal.stage = _WorkerReapStage.TERMINATE_JOIN_PENDING
                    continue
                terminate = getattr(journal.process, "terminate", None)
                if callable(terminate):
                    terminate()
                    journal.stage = _WorkerReapStage.TERMINATE_JOIN_PENDING
                else:
                    journal.stage = _WorkerReapStage.KILL_PENDING
                continue

            if stage is _WorkerReapStage.TERMINATE_JOIN_PENDING:
                journal.process.join(timeout_seconds)
                if _process_is_alive(journal):
                    journal.stage = _WorkerReapStage.KILL_PENDING
                else:
                    journal.exitcode = journal.process.exitcode
                    journal.stage = _WorkerReapStage.CLOSE_PENDING
                continue

            if stage is _WorkerReapStage.KILL_PENDING:
                if not _process_is_alive(journal):
                    journal.stage = _WorkerReapStage.KILL_JOIN_PENDING
                    continue
                journal.process.kill()
                journal.stage = _WorkerReapStage.KILL_JOIN_PENDING
                continue

            if stage is _WorkerReapStage.KILL_JOIN_PENDING:
                journal.process.join(timeout_seconds)
                if _process_is_alive(journal):
                    raise TimeoutError(
                        "managed async checkpoint worker remained alive after kill"
                    )
                journal.exitcode = journal.process.exitcode
                journal.stage = _WorkerReapStage.CLOSE_PENDING
                continue

            if stage is _WorkerReapStage.CLOSE_PENDING:
                if _process_is_alive(journal):
                    raise RuntimeError(
                        "managed async worker cannot close while process remains alive"
                    )
                close = getattr(journal.process, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException:
                        if _process_handle_consumed(journal.process):
                            _mark_worker_reaped(journal)
                        raise
                _mark_worker_reaped(journal)
                continue

            raise RuntimeError(f"unknown managed worker reap stage {stage!r}")
        except BaseException as error:
            journal.record_error(stage, error)
            raise


def _reap_temporal_worker(
    journal: _WorkerRecoveryJournal, *, timeout_seconds: float
) -> None:
    _advance_worker_recovery(journal, timeout_seconds=timeout_seconds, abort=False)
    if journal.worker_outcome_error is not None:
        raise journal.worker_outcome_error
    if journal.exitcode not in (0, None):
        raise RuntimeError(
            f"managed async checkpoint worker exited with code {journal.exitcode}"
        )


def _remove_active_call(queue: Any, active: Any) -> None:
    calls = queue.async_calls
    if not calls or calls[0] is not active:
        raise RuntimeError("MCore async queue head changed during managed finalization")
    calls.popleft()


def _ensure_worker_terminal_cleanup(
    queue: Any,
    journal: _WorkerRecoveryJournal,
    *,
    expected_call_idx: int | None,
) -> _WorkerTerminalCleanupJournal:
    existing = _worker_terminal_cleanup(queue)
    if existing is not None:
        if existing.journal is not journal or existing.call_idx != journal.call_idx:
            raise RuntimeError(
                "managed async worker terminal cleanup authority changed"
            )
        if expected_call_idx is not None and existing.call_idx != expected_call_idx:
            raise RuntimeError(
                "managed async terminal cleanup expected call index changed: "
                f"cleanup={existing.call_idx}, expected={expected_call_idx}"
            )
        return existing
    if journal.stage is not _WorkerReapStage.REAPED:
        raise RuntimeError(
            "managed async worker cannot enter terminal cleanup before REAPED: "
            f"stage={journal.stage.name}, call_idx={journal.call_idx}"
        )
    if expected_call_idx is not None and journal.call_idx != expected_call_idx:
        raise RuntimeError(
            "managed async terminal cleanup call index mismatch: "
            f"journal={journal.call_idx}, expected={expected_call_idx}"
        )
    cleanup = _WorkerTerminalCleanupJournal(
        active=journal.active,
        journal=journal,
        call_idx=journal.call_idx,
    )
    setattr(queue, _WORKER_TERMINAL_CLEANUP_ATTR, cleanup)
    return cleanup


def _advance_worker_terminal_cleanup(
    queue: Any, cleanup: _WorkerTerminalCleanupJournal
) -> None:
    """Finish queue removal and reference release without worker operations."""

    while cleanup.stage is not _WorkerTerminalCleanupStage.COMPLETE:
        journal = cleanup.journal
        active = cleanup.active
        if journal is None:
            raise RuntimeError("managed async terminal cleanup lost its journal")
        if cleanup.stage is _WorkerTerminalCleanupStage.RECORD_PENDING:
            if journal.stage is not _WorkerReapStage.REAPED:
                raise RuntimeError(
                    "managed async terminal cleanup journal is not REAPED"
                )
            if (
                active is None
                or journal.active is not active
                or journal.caller is not active.async_caller
                or journal.call_idx != active.idx
            ):
                raise RuntimeError(
                    "managed async terminal cleanup transaction binding changed: "
                    f"journal={journal.call_idx}, active="
                    f"{getattr(active, 'idx', None)}"
                )
            calls = queue.async_calls
            if calls:
                if calls[0] is not active or len(calls) != 1:
                    raise RuntimeError(
                        "managed async queue changed before terminal cleanup"
                    )
                _remove_active_call(queue, active)
            # If remove raised after popleft, an identical retry observes the
            # empty queue and reconciles the already-effective operation.
            if queue.async_calls:
                raise RuntimeError(
                    "managed async queue record removal did not complete"
                )
            cleanup.stage = _WorkerTerminalCleanupStage.RECOVERY_ATTR_PENDING
            continue

        if cleanup.stage is _WorkerTerminalCleanupStage.RECOVERY_ATTR_PENDING:
            current = _worker_journal(queue)
            if current is journal:
                setattr(queue, _WORKER_RECOVERY_ATTR, None)
            elif current is not None:
                raise RuntimeError(
                    "managed async recovery journal changed during terminal cleanup"
                )
            cleanup.stage = _WorkerTerminalCleanupStage.REFERENCES_PENDING
            continue

        if cleanup.stage is _WorkerTerminalCleanupStage.REFERENCES_PENDING:
            journal.active = None
            journal.caller = None
            journal.process = None
            journal.original_failure = None
            journal.worker_outcome_error = None
            journal.diagnostics = None
            cleanup.active = None
            cleanup.journal = None
            cleanup.diagnostic = None
            cleanup.stage = _WorkerTerminalCleanupStage.COMPLETE
            continue

        raise RuntimeError(f"unknown worker terminal cleanup stage {cleanup.stage!r}")

    setattr(queue, _WORKER_TERMINAL_CLEANUP_ATTR, None)


class _FinalizePhaseRunner:
    """Run one local operation followed by one ordered collective vote."""

    def __init__(self, group: Any):
        self.group = group
        self.next_phase_id = 0

    def run(
        self,
        phase: str,
        operation,
        *,
        details_fn=None,
        require_equal_details: bool = False,
    ) -> tuple[Any, list[dict[str, Any]]]:
        phase_id = self.next_phase_id
        self.next_phase_id += 1
        result = None
        details = None
        local_error = None
        try:
            result = operation()
            details = details_fn(result) if details_fn is not None else None
        except BaseException as error:
            local_error = error
        reports = _vote(
            self.group,
            phase,
            local_error,
            phase_id=phase_id,
            details=details,
            require_equal_details=require_equal_details,
        )
        return result, reports


def _abort_all_workers(queue: Any) -> None:
    errors: list[BaseException] = []
    calls = tuple(queue.async_calls)
    journal = _worker_journal(queue)
    if not calls and journal is not None:
        try:
            _advance_worker_recovery(journal, timeout_seconds=5.0, abort=True)
        except BaseException as error:
            errors.append(error)
    for active in calls:
        try:
            journal = _ensure_worker_journal(queue, active)
            _advance_worker_recovery(journal, timeout_seconds=5.0, abort=True)
        except BaseException as error:
            errors.append(error)
    if errors:
        primary = errors[0]
        for error in errors[1:]:
            primary.add_note(f"another worker abort failed: {error!r}")
        raise primary


def _pop_all_calls(queue: Any, *, expected_call_idx: int | None = None) -> None:
    cleanup = _worker_terminal_cleanup(queue)
    if cleanup is not None:
        _advance_worker_terminal_cleanup(queue, cleanup)
        return

    if len(queue.async_calls) > 1:
        raise RuntimeError(
            "managed async terminal cleanup supports one outstanding request"
        )
    journal = _worker_journal(queue)
    if journal is None and queue.async_calls:
        journal = _ensure_worker_journal(queue, queue.async_calls[0])
    if journal is None:
        return
    if journal.stage is not _WorkerReapStage.REAPED:
        raise RuntimeError(
            "managed async queue record retains unreaped worker authority: "
            f"stage={journal.stage.name}, call_idx={journal.call_idx}"
        )
    cleanup = _ensure_worker_terminal_cleanup(
        queue, journal, expected_call_idx=expected_call_idx
    )
    _advance_worker_terminal_cleanup(queue, cleanup)


def _publish_worker_recovery(
    queue: Any,
    runner: _FinalizePhaseRunner,
    original: BaseException,
) -> bool:
    """Publish through prepare/commit/result phases on every rank."""

    return _run_worker_recovery_publication_protocol(
        queue,
        runner,
        original,
        phase="failure_worker_recovery_publish",
    )


def _prepare_worker_recovery_publication(
    queue: Any,
    original: BaseException,
) -> _PublicationPrepareStatus:
    try:
        journal = _worker_journal(queue)
        cleanup = _worker_terminal_cleanup(queue)
        if journal is not None and journal.original_failure is None:
            journal.original_failure = original
        candidate = _WorkerRecoveryPublication(
            original_failure=original,
            local_authority=journal or cleanup,
        )
        return _PublicationPrepareStatus(candidate=candidate, error=None)
    except BaseException as error:
        return _PublicationPrepareStatus(candidate=None, error=_error_summary(error))


def _commit_worker_recovery_publication(
    queue: Any,
    candidate: _WorkerRecoveryPublication | None,
) -> _PublicationMutationStatus:
    local_error = None
    if candidate is None:
        local_error = RuntimeError(
            "managed async worker recovery publication was not prepared"
        )
    else:
        try:
            _write_worker_recovery_publication(queue, candidate)
        except BaseException as error:
            local_error = error
    actual = None
    try:
        actual = _worker_recovery_publication(queue)
    except BaseException as error:
        if local_error is None:
            local_error = error
        else:
            local_error.add_note(f"publication reconcile failed: {error!r}")
    return _PublicationMutationStatus(
        committed=candidate is not None and actual is candidate,
        present=actual is not None,
        error=_error_summary(local_error),
    )


def _write_worker_recovery_publication(
    queue: Any,
    publication: _WorkerRecoveryPublication,
) -> None:
    setattr(queue, _WORKER_RECOVERY_PUBLICATION_ATTR, publication)


def _run_worker_recovery_publication_protocol(
    queue: Any,
    runner: _FinalizePhaseRunner,
    original: BaseException,
    *,
    phase: str,
) -> bool:
    prepared, _ = runner.run(
        f"{phase}_prepare",
        lambda: _prepare_worker_recovery_publication(queue, original),
        details_fn=lambda status: status.details(),
    )
    _, reports = runner.run(
        phase,
        lambda: _commit_worker_recovery_publication(queue, prepared.candidate),
        details_fn=lambda status: status.details(),
    )
    all_committed = all(report["details"]["committed"] for report in reports)
    runner.run(
        f"{phase}_result",
        lambda: all_committed,
        details_fn=lambda committed: {"all_committed": committed},
        require_equal_details=True,
    )
    return all_committed


def _clear_worker_recovery_publication(queue: Any) -> None:
    if _worker_journal(queue) is not None:
        raise RuntimeError(
            "managed async worker publication cannot clear with a recovery journal"
        )
    if _worker_terminal_cleanup(queue) is not None:
        raise RuntimeError(
            "managed async worker publication cannot clear with terminal cleanup"
        )
    if queue.async_calls:
        raise RuntimeError(
            "managed async worker publication cannot clear with active queue records"
        )
    setattr(queue, _WORKER_RECOVERY_PUBLICATION_ATTR, None)


def _prepare_worker_recovery_publication_clear(
    queue: Any,
) -> _PublicationMutationStatus:
    local_error = None
    try:
        if _worker_journal(queue) is not None:
            raise RuntimeError(
                "managed async worker publication cannot clear with a recovery journal"
            )
        if _worker_terminal_cleanup(queue) is not None:
            raise RuntimeError(
                "managed async worker publication cannot clear with terminal cleanup"
            )
        if queue.async_calls:
            raise RuntimeError(
                "managed async worker publication cannot clear with active records"
            )
    except BaseException as error:
        local_error = error
    publication = None
    try:
        publication = _worker_recovery_publication(queue)
    except BaseException as error:
        if local_error is None:
            local_error = error
        else:
            local_error.add_note(f"publication prepare reconcile failed: {error!r}")
    return _PublicationMutationStatus(
        committed=local_error is None,
        present=publication is not None,
        error=_error_summary(local_error),
    )


def _commit_worker_recovery_publication_clear(
    queue: Any,
) -> _PublicationMutationStatus:
    local_error = None
    try:
        _clear_worker_recovery_publication(queue)
    except BaseException as error:
        local_error = error
    publication = object()
    try:
        publication = _worker_recovery_publication(queue)
    except BaseException as error:
        if local_error is None:
            local_error = error
        else:
            local_error.add_note(f"publication clear reconcile failed: {error!r}")
    return _PublicationMutationStatus(
        committed=publication is None,
        present=publication is not None,
        error=_error_summary(local_error),
    )


def _publication_not_committed_status(
    queue: Any,
    message: str,
) -> _PublicationMutationStatus:
    local_error: BaseException = RuntimeError(message)
    publication = None
    try:
        publication = _worker_recovery_publication(queue)
    except BaseException as error:
        local_error.add_note(f"publication reconcile failed: {error!r}")
    return _PublicationMutationStatus(
        committed=False,
        present=publication is not None,
        error=_error_summary(local_error),
    )


def _run_worker_recovery_publication_clear_protocol(
    queue: Any,
    runner: _FinalizePhaseRunner,
    original: BaseException,
) -> bool:
    _, prepare_reports = runner.run(
        "failure_worker_recovery_clear_prepare",
        lambda: _prepare_worker_recovery_publication_clear(queue),
        details_fn=lambda status: status.details(),
    )
    all_prepared = all(report["details"]["committed"] for report in prepare_reports)
    _, commit_reports = runner.run(
        "failure_worker_recovery_clear",
        lambda: (
            _commit_worker_recovery_publication_clear(queue)
            if all_prepared
            else _publication_not_committed_status(
                queue,
                "publication clear prepare did not reach consensus",
            )
        ),
        details_fn=lambda status: status.details(),
    )
    all_cleared = all(report["details"]["committed"] for report in commit_reports)
    runner.run(
        "failure_worker_recovery_clear_result",
        lambda: all_cleared,
        details_fn=lambda cleared: {"all_cleared": cleared},
        require_equal_details=True,
    )
    if all_cleared:
        return True
    _run_worker_recovery_publication_protocol(
        queue,
        runner,
        original,
        phase="failure_worker_recovery_republish",
    )
    return False


def _failure_queue_cleanup_status(
    queue: Any,
    *,
    transaction_call_idx: int | None,
    local_error: BaseException | None,
) -> _FailureQueueCleanupStatus:
    """Describe cleanup for one transaction without using global queue depth."""

    journal = _worker_journal(queue)
    cleanup = _worker_terminal_cleanup(queue)
    local_call_idx = (
        cleanup.call_idx
        if cleanup is not None
        else journal.call_idx
        if journal is not None
        else transaction_call_idx
    )
    if (
        journal is not None
        and cleanup is not None
        and journal.call_idx != cleanup.call_idx
    ):
        binding_error = RuntimeError(
            "managed async failure cleanup journals refer to different calls: "
            f"worker={journal.call_idx}, terminal={cleanup.call_idx}"
        )
        if local_error is None:
            local_error = binding_error
        else:
            local_error.add_note(str(binding_error))

    matching_records = [
        active
        for active in queue.async_calls
        if local_call_idx is None or getattr(active, "idx", None) == local_call_idx
    ]
    unrelated_records = [
        active
        for active in queue.async_calls
        if not any(active is matching for matching in matching_records)
    ]
    if unrelated_records:
        binding_error = RuntimeError(
            "managed async failure cleanup found an unrelated queue record: "
            f"transaction={local_call_idx}, "
            f"records={[getattr(active, 'idx', None) for active in queue.async_calls]}"
        )
        if local_error is None:
            local_error = binding_error
        else:
            local_error.add_note(str(binding_error))

    worker_pending = (
        journal is not None and journal.stage is not _WorkerReapStage.REAPED
    )
    terminal_pending = (
        cleanup is not None
        or bool(matching_records)
        or (journal is not None and journal.stage is _WorkerReapStage.REAPED)
    )
    return _FailureQueueCleanupStatus(
        call_idx=local_call_idx,
        record_removed=not matching_records,
        terminal_cleanup_pending=terminal_pending,
        worker_recovery_pending=worker_pending,
        error=_error_summary(local_error),
    )


def _attempt_failure_queue_cleanup(
    queue: Any,
    *,
    transaction_call_idx: int | None,
) -> _FailureQueueCleanupStatus:
    """Attempt local pop/reconcile and always return a vote-safe status."""

    local_error = None
    try:
        # A call-index mismatch can itself be the triggering failure.  Once a
        # journal owns the sole active record, cleanup follows that concrete
        # authority rather than refusing to reap it because the declared
        # transaction index was wrong.  ``transaction_call_idx`` still scopes
        # ranks which never acquired a local record.
        _pop_all_calls(queue)
    except BaseException as error:
        local_error = error
    return _failure_queue_cleanup_status(
        queue,
        transaction_call_idx=transaction_call_idx,
        local_error=local_error,
    )


def _cleanup_after_failure(
    runner: _FinalizePhaseRunner,
    queue: Any,
    original: BaseException,
    *,
    transaction_call_idx: int | None,
    recovery_token: ManagedAsyncRecoveryToken,
) -> bool:
    recovery_token.require_recovery()
    try:
        runner.run("failure_worker_reap", lambda: _abort_all_workers(queue))
    except BaseException as cleanup_error:
        original.add_note(f"failure_worker_reap failed: {cleanup_error!r}")

    # Re-read after reap: an early phase failure may not have created the
    # authoritative journal until _abort_all_workers inspected the active
    # record.  Never publish a stale pre-reap None.
    try:
        publication_committed = _publish_worker_recovery(queue, runner, original)
    except BaseException as cleanup_error:
        original.add_note(f"failure_worker_recovery_publish failed: {cleanup_error!r}")
        return False

    # Every rank attempts only its transaction-local record, then reports the
    # result in the same phase.  A rank that already popped is an idempotent
    # no-op; it still retains the publication/fence until all peers are clean.
    try:
        _, reports = runner.run(
            "failure_queue_pop",
            lambda: _attempt_failure_queue_cleanup(
                queue,
                transaction_call_idx=transaction_call_idx,
            ),
            details_fn=lambda status: status.details(),
        )
    except BaseException as cleanup_error:
        original.add_note(f"failure_queue_pop failed: {cleanup_error!r}")
        return False

    all_cleanup_complete = all(
        report["details"] is not None
        and report["details"]["error"] is None
        and report["details"]["record_removed"]
        and not report["details"]["terminal_cleanup_pending"]
        and not report["details"]["worker_recovery_pending"]
        for report in reports
    )
    for report in reports:
        diagnostic = report["details"].get("error") if report["details"] else None
        if diagnostic is not None:
            original.add_note(
                "failure_queue_pop rank "
                f"{report['global_rank']} retained cleanup: {diagnostic}"
            )

    if all_cleanup_complete and publication_committed:
        try:
            cleared = _run_worker_recovery_publication_clear_protocol(
                queue,
                runner,
                original,
            )
        except BaseException as cleanup_error:
            original.add_note(
                f"failure_worker_recovery_clear failed: {cleanup_error!r}"
            )
            return False
        if cleared:
            recovery_token.mark_cleared()
            return True
        return False

    try:
        retained = _run_worker_recovery_publication_protocol(
            queue,
            runner,
            original,
            phase="failure_worker_recovery_hold",
        )
    except BaseException as cleanup_error:
        original.add_note(f"failure_worker_recovery_hold failed: {cleanup_error!r}")
        return False
    if not retained:
        original.add_note("failure_worker_recovery_hold did not commit on every rank")
    return False


def abort_managed_async_calls(
    queue: Any,
    control_group: Any,
    *,
    recovery_token: ManagedAsyncRecoveryToken | None = None,
) -> None:
    """Abort and reap the sole managed request without MCore WORLD collectives."""

    if recovery_token is None:
        recovery_token = ManagedAsyncRecoveryToken()
    recovery_token.require_recovery()
    runner = _FinalizePhaseRunner(control_group)
    contract, _ = runner.run(
        "abort_compatibility", lambda: _validate_mcore_017_contract(queue)
    )

    def validate_shape():
        calls = queue.async_calls
        if len(calls) > 1:
            raise RuntimeError(
                "managed async checkpoint supports one outstanding request"
            )
        if calls:
            active = calls[0]
            if type(active) is not contract.active_request_type:
                raise TypeError(
                    "managed async checkpoint active-request record type changed"
                )
            if type(active.async_caller) is not contract.temporal_caller_type:
                raise TypeError("managed async checkpoint requires TemporalAsyncCaller")

    runner.run("abort_queue_shape", validate_shape)
    worker_error = None
    try:
        runner.run("abort_worker_reap", lambda: _abort_all_workers(queue))
    except BaseException as error:
        worker_error = error
    pop_phase = "abort_queue_hold" if worker_error is not None else "abort_queue_pop"
    pop_operation = (
        (lambda: None) if worker_error is not None else lambda: _pop_all_calls(queue)
    )
    try:
        runner.run(pop_phase, pop_operation)
    except BaseException as error:
        if worker_error is None:
            worker_error = error
        else:
            worker_error.add_note(f"abort queue pop failed: {error!r}")
    if worker_error is not None:
        journal = _worker_journal(queue)
        if journal is not None and journal.original_failure is None:
            journal.original_failure = worker_error
        try:
            _publish_worker_recovery(queue, runner, worker_error)
        except BaseException as publication_error:
            worker_error.add_note(
                f"abort worker recovery publication failed: {publication_error!r}"
            )
            setattr(
                queue,
                _WORKER_RECOVERY_PUBLICATION_ATTR,
                _WorkerRecoveryPublication(
                    original_failure=worker_error,
                    local_authority=journal or _worker_terminal_cleanup(queue),
                ),
            )
        raise worker_error
    runner.run(
        "abort_worker_recovery_clear",
        lambda: _clear_worker_recovery_publication(queue),
    )
    recovery_token.mark_cleared()


def preflight_managed_async_finalize(
    queue: Any,
    async_request: Any,
    control_group: Any,
    *,
    expected_call_idx: int,
) -> None:
    """Validate the pinned queue/callback plan before MCore forks a worker."""

    local_error = None
    details = None
    try:
        contract = _validate_mcore_017_contract(queue)
        _group_manifest(control_group)
        if queue.async_calls:
            raise RuntimeError(
                "managed async checkpoint supports one outstanding request"
            )
        if queue.call_idx + 1 != expected_call_idx:
            raise RuntimeError(
                "managed async checkpoint call index changed before schedule: "
                f"expected={expected_call_idx}, queue_next={queue.call_idx + 1}"
            )
        callbacks = _parse_mcore_callbacks(async_request, contract)
        details = {
            "expected_call_idx": expected_call_idx,
            "checkpoint_dir": str(callbacks.checkpoint_dir),
            "backend": str(callbacks.sharded_strategy.backend),
            "version": int(callbacks.sharded_strategy.version),
            "writer_type": (
                f"{type(callbacks.writer).__module__}."
                f"{type(callbacks.writer).__qualname__}"
            ),
        }
    except BaseException as error:
        local_error = error
    _vote(
        control_group,
        "schedule_preflight",
        local_error,
        details=details,
        require_equal_details=True,
    )


def finalize_managed_async_calls(
    queue: Any,
    control_group: Any,
    *,
    expected_call_idx: int,
    bound_call_idx: int | None,
    blocking: bool,
    timeout_seconds: float = 120.0,
    recovery_token: ManagedAsyncRecoveryToken | None = None,
) -> list[int]:
    """Finalize one managed MCore request with explicit Gloo consensus only."""

    runner = _FinalizePhaseRunner(control_group)
    if recovery_token is None:
        recovery_token = ManagedAsyncRecoveryToken()
    recovery_token.require_recovery()
    active = None
    failure_cleanup_performed = False
    try:

        def validate_compatibility():
            if type(blocking) is not bool:
                raise TypeError("managed async finalize blocking flag must be bool")
            if isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
                raise ValueError("managed async finalize timeout must be positive")
            contract = _validate_mcore_017_contract(queue)
            _group_manifest(control_group)
            return contract

        contract, _ = runner.run(
            "compatibility",
            validate_compatibility,
            details_fn=lambda _value: {
                "blocking": blocking,
                "expected_call_idx": expected_call_idx,
                "bound_call_idx": bound_call_idx,
                "timeout_seconds": float(timeout_seconds),
            },
            require_equal_details=True,
        )

        recovery_present, _ = runner.run(
            "worker_recovery_presence",
            lambda: has_managed_async_worker_recovery(queue),
            details_fn=lambda value: {"recovery_pending": value},
            require_equal_details=True,
        )
        if recovery_present:
            retained = _worker_journal(queue)
            terminal_cleanup = _worker_terminal_cleanup(queue)
            publication = _worker_recovery_publication(queue)

            def validate_recovery_binding():
                if isinstance(expected_call_idx, bool) or not isinstance(
                    expected_call_idx, int
                ):
                    raise TypeError(
                        "managed async expected call index must be an integer"
                    )
                if bound_call_idx is not None and (
                    isinstance(bound_call_idx, bool)
                    or not isinstance(bound_call_idx, int)
                ):
                    raise TypeError(
                        "managed async bound call index must be an integer or None "
                        "during worker recovery"
                    )
                local_call_idx = None
                if retained is not None:
                    if retained.active is None or retained.caller is None:
                        raise RuntimeError(
                            "managed async worker recovery authority was released "
                            "before terminal cleanup"
                        )
                    active_idx = retained.active.idx
                    if not (retained.call_idx == active_idx == queue.call_idx):
                        raise RuntimeError(
                            "managed async worker recovery call index mismatch: "
                            f"journal={retained.call_idx}, active={active_idx}, "
                            f"queue={queue.call_idx}"
                        )
                    if bound_call_idx is not None and not (
                        bound_call_idx == retained.call_idx == expected_call_idx
                    ):
                        raise RuntimeError(
                            "managed async worker recovery bound index mismatch: "
                            f"journal={retained.call_idx}, bound={bound_call_idx}, "
                            f"expected={expected_call_idx}"
                        )
                    local_call_idx = retained.call_idx
                elif terminal_cleanup is not None:
                    if bound_call_idx is not None and not (
                        terminal_cleanup.call_idx == bound_call_idx == expected_call_idx
                    ):
                        raise RuntimeError(
                            "managed async terminal cleanup call index mismatch: "
                            f"cleanup={terminal_cleanup.call_idx}, "
                            f"bound={bound_call_idx}, expected={expected_call_idx}"
                        )
                    local_call_idx = terminal_cleanup.call_idx
                elif queue.async_calls:
                    raise RuntimeError(
                        "managed async recovery publication lost local queue authority"
                    )
                return local_call_idx

            runner.run(
                "worker_recovery_binding",
                validate_recovery_binding,
                details_fn=lambda value: {"local_call_idx": value},
            )
            retained_failure = (
                (retained.original_failure if retained is not None else None)
                or (publication.original_failure if publication is not None else None)
                or RuntimeError(
                    "managed async worker recovery remained pending after a prior failure"
                )
            )
            _cleanup_after_failure(
                runner,
                queue,
                retained_failure,
                transaction_call_idx=(
                    expected_call_idx if bound_call_idx is not None else None
                ),
                recovery_token=recovery_token,
            )
            failure_cleanup_performed = True

            def report_retained_failure():
                raise retained_failure

            runner.run("worker_recovery_terminal", report_retained_failure)
            raise AssertionError(
                "worker recovery terminal phase unexpectedly succeeded"
            )

        def validate_transaction_indices():
            for name, value in (
                ("expected", expected_call_idx),
                ("bound", bound_call_idx),
            ):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(
                        f"managed async {name} call index must be an integer"
                    )
            return expected_call_idx, bound_call_idx

        runner.run(
            "transaction_call_indices",
            validate_transaction_indices,
            details_fn=lambda value: {
                "expected_call_idx": value[0],
                "bound_call_idx": value[1],
            },
            require_equal_details=True,
        )

        def validate_queue():
            calls = queue.async_calls
            if len(calls) != 1:
                raise RuntimeError(
                    "managed async checkpoint transaction requires exactly one "
                    f"outstanding request, got {len(calls)}"
                )
            candidate = calls[0]
            if type(candidate) is not contract.active_request_type:
                raise TypeError(
                    "managed async checkpoint active-request record type changed"
                )
            if type(candidate.async_caller) is not contract.temporal_caller_type:
                raise TypeError(
                    "managed async checkpoint requires the exact MCore 0.17 "
                    f"TemporalAsyncCaller, got {type(candidate.async_caller)!r}"
                )
            _ensure_worker_journal(queue, candidate)
            if not (
                queue.call_idx == candidate.idx == expected_call_idx == bound_call_idx
            ):
                raise RuntimeError(
                    "managed async queue/call index mismatch: "
                    f"queue={queue.call_idx}, active={candidate.idx}, "
                    f"expected={expected_call_idx}, bound={bound_call_idx}"
                )
            return candidate

        active, _ = runner.run(
            "queue_binding",
            validate_queue,
            details_fn=lambda value: {"call_idx": value.idx},
            require_equal_details=True,
        )

        def audit_callbacks():
            parsed = _parse_mcore_callbacks(active.async_request, contract)
            details = {
                "call_idx": active.idx,
                "checkpoint_dir": str(parsed.checkpoint_dir),
                "backend": str(parsed.sharded_strategy.backend),
                "version": int(parsed.sharded_strategy.version),
                "writer_type": (
                    f"{type(parsed.writer).__module__}."
                    f"{type(parsed.writer).__qualname__}"
                ),
            }
            return parsed, details

        audited, _ = runner.run(
            "callback_audit",
            audit_callbacks,
            details_fn=lambda value: value[1],
            require_equal_details=True,
        )
        callbacks = audited[0]

        if not blocking:

            def poll_worker():
                process = active.async_caller.process
                return bool(process is not None and process.is_alive())

            _, reports = runner.run(
                "worker_poll",
                poll_worker,
                details_fn=lambda alive: {"alive": alive},
            )
            if any(report["details"]["alive"] for report in reports):
                return []

        journal = _ensure_worker_journal(queue, active)
        runner.run(
            "worker_reap",
            lambda: _reap_temporal_worker(
                journal,
                timeout_seconds=timeout_seconds if blocking else 0.0,
            ),
        )

        def retrieve_results():
            value = callbacks.writer.retrieve_write_results()
            if _is_wrapped_exception(value):
                wrapped_error, wrapped_traceback = value
                wrapped_error.add_note(f"async worker traceback: {wrapped_traceback}")
                raise wrapped_error
            return value

        _, reports = runner.run(
            "worker_result",
            retrieve_results,
            details_fn=lambda value: {"write_results": value},
        )
        all_results = [report["details"]["write_results"] for report in reports]

        def validate_call_index():
            if not (
                queue.call_idx == active.idx == expected_call_idx == bound_call_idx
            ):
                raise RuntimeError(
                    "managed async queue/call index changed during finalize: "
                    f"queue={queue.call_idx}, active={active.idx}, "
                    f"expected={expected_call_idx}, bound={bound_call_idx}"
                )
            return active.idx

        runner.run(
            "call_index_before_finish",
            validate_call_index,
            details_fn=lambda value: {"call_idx": value},
            require_equal_details=True,
        )

        def finish_metadata():
            global_rank, _ = _group_manifest(control_group)
            coordinator_rank = callbacks.dist_wrapper.coordinator_rank
            if coordinator_rank != 0 or callbacks.dist_wrapper.group is not None:
                raise RuntimeError(
                    "MCore DCP finalize wrapper changed: managed adapter only supports "
                    "default coordinator rank 0 and a WORLD wrapper"
                )
            if global_rank == coordinator_rank:
                if callbacks.global_metadata is None:
                    raise RuntimeError("MCore DCP global metadata is missing")
                callbacks.writer.finish(callbacks.global_metadata, all_results)

        runner.run("dcp_metadata_finish", finish_metadata)
        runner.run(
            "call_index_before_config",
            validate_call_index,
            details_fn=lambda value: {"call_idx": value},
            require_equal_details=True,
        )

        def save_mcore_config():
            global_rank, _ = _group_manifest(control_group)
            if global_rank == 0:
                from megatron.core.dist_checkpointing.serialization import (
                    CheckpointingConfig,
                    save_config,
                )

                save_config(
                    CheckpointingConfig(
                        callbacks.sharded_strategy.backend,
                        callbacks.sharded_strategy.version,
                    ),
                    callbacks.checkpoint_dir,
                )

        runner.run("mcore_config_finish", save_mcore_config)
        runner.run(
            "call_index_before_pop",
            validate_call_index,
            details_fn=lambda value: {"call_idx": value},
            require_equal_details=True,
        )
        runner.run(
            "queue_pop",
            lambda: _pop_all_calls(queue, expected_call_idx=expected_call_idx),
            details_fn=lambda _value: {"call_idx": expected_call_idx},
            require_equal_details=True,
        )
        recovery_token.mark_cleared()
        return [expected_call_idx]
    except BaseException as error:
        if not failure_cleanup_performed:
            transaction_call_idx = (
                bound_call_idx
                if isinstance(bound_call_idx, int)
                and not isinstance(bound_call_idx, bool)
                else None
            )
            _cleanup_after_failure(
                runner,
                queue,
                error,
                transaction_call_idx=transaction_call_idx,
                recovery_token=recovery_token,
            )
        raise
