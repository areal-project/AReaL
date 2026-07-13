# SPDX-License-Identifier: Apache-2.0

"""Tests for the in-memory Memory Service evidence store."""

from __future__ import annotations

import faulthandler
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from hashlib import sha256 as calculate_sha256
from threading import Barrier, Event
from time import sleep

import pytest

import areal.v2.memory_service as memory_service
from areal.v2.memory_service import store as store_module
from areal.v2.memory_service._atomic import _atomic_publish
from areal.v2.memory_service.errors import (
    CandidateConflictError,
    CandidateNotFoundError,
    EvidenceConflictError,
    EvidenceNotFoundError,
    MemoryServiceError,
    ReleaseConflictError,
    ReleaseNotFoundError,
    RevisionConflictError,
    RevisionNotFoundError,
)
from areal.v2.memory_service.history_store import (
    InMemoryMemoryHistoryStore,
    MemoryHistoryStore,
)
from areal.v2.memory_service.history_types import (
    CandidateProposal,
    MemoryCandidate,
    MemoryRevision,
    RevisionOperation,
    RevisionProposal,
)
from areal.v2.memory_service.release_store import (
    InMemoryMemoryReleaseStore,
    MemoryReleaseStore,
)
from areal.v2.memory_service.release_types import MemoryRelease, ReleaseManifest
from areal.v2.memory_service.store import EvidenceStore, InMemoryEvidenceStore
from areal.v2.memory_service.types import (
    EvidenceEvent,
    EvidenceKind,
    EvidenceRecord,
    MemoryScope,
)

UTC_INSTANT = datetime(2026, 7, 7, 4, 5, 6, 789000, tzinfo=UTC)
CONCURRENCY_TIMEOUT_SECONDS = 10.0
HARD_CONCURRENCY_TIMEOUT_SECONDS = CONCURRENCY_TIMEOUT_SECONDS + 5.0


@contextmanager
def _hard_bounded_executor(*, max_workers: int) -> Iterator[ThreadPoolExecutor]:
    faulthandler.dump_traceback_later(
        HARD_CONCURRENCY_TIMEOUT_SECONDS,
        exit=True,
    )
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            yield executor
    finally:
        faulthandler.cancel_dump_traceback_later()


def test_hard_bounded_executor_waits_for_workers_before_cancel_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bounded_executor = _hard_bounded_executor
    watchdog_events: list[object] = []
    worker_started = Event()
    shutdown_started = Event()
    shutdown_finished = Event()
    worker_finished = Event()
    shutdown_wait_values: list[bool] = []
    original_shutdown = ThreadPoolExecutor.shutdown
    original_error = RuntimeError("body failed")

    def arm_watchdog(timeout: float, *, exit: bool) -> None:
        watchdog_events.append(("arm", timeout, exit))

    def cancel_watchdog() -> None:
        watchdog_events.append(
            (
                "cancel",
                worker_finished.is_set(),
                shutdown_finished.is_set(),
            )
        )

    def observe_shutdown(
        executor: ThreadPoolExecutor,
        wait: bool = True,
        *,
        cancel_futures: bool = False,
    ) -> None:
        shutdown_wait_values.append(wait)
        shutdown_started.set()
        original_shutdown(
            executor,
            wait=wait,
            cancel_futures=cancel_futures,
        )
        shutdown_finished.set()

    def slow_worker() -> None:
        worker_started.set()
        assert shutdown_started.wait(timeout=CONCURRENCY_TIMEOUT_SECONDS)
        worker_finished.set()

    monkeypatch.setattr(faulthandler, "dump_traceback_later", arm_watchdog)
    monkeypatch.setattr(faulthandler, "cancel_dump_traceback_later", cancel_watchdog)
    monkeypatch.setattr(ThreadPoolExecutor, "shutdown", observe_shutdown)

    with pytest.raises(RuntimeError) as raised:
        with bounded_executor(max_workers=1) as executor:
            future = executor.submit(slow_worker)
            assert worker_started.wait(timeout=CONCURRENCY_TIMEOUT_SECONDS)
            raise original_error

    assert raised.value is original_error
    assert future.result(timeout=CONCURRENCY_TIMEOUT_SECONDS) is None
    assert watchdog_events == [
        ("arm", CONCURRENCY_TIMEOUT_SECONDS + 5.0, True),
        ("cancel", True, True),
    ]
    assert shutdown_wait_values == [True]


class YieldingMissingDict(dict[object, object]):
    """Yield after a missing lookup to expose an unprotected check/write race."""

    def get(self, key: object, default: object = None) -> object:
        value = super().get(key, default)
        if value is default:
            sleep(0.05)
        return value


class FoldAwareTimezone(tzinfo):
    """Minimal repeated-hour timezone independent of the system tz database."""

    def utcoffset(self, value: datetime | None) -> timedelta:
        return timedelta(hours=-4 if value is not None and value.fold == 0 else -5)

    def dst(self, value: datetime | None) -> timedelta:
        return timedelta(hours=1 if value is not None and value.fold == 0 else 0)

    def tzname(self, value: datetime | None) -> str:
        return "FOLD-DST" if value is not None and value.fold == 0 else "FOLD-STD"


class MemoryScopeSubclass(MemoryScope):
    def __hash__(self) -> int:
        return 0


class EvidenceEventSubclass(EvidenceEvent):
    def canonical_bytes(self) -> bytes:
        return b"overridden"


class MutableHashStr(str):
    hash_calls: int
    hash_salt: int

    def __new__(cls, value: str) -> MutableHashStr:
        instance = str.__new__(cls, value)
        instance.hash_calls = 0
        instance.hash_salt = 0
        return instance

    def __hash__(self) -> int:
        self.hash_calls += 1
        return str.__hash__(self) + self.hash_salt


class AlwaysEqualStr(str):
    equality_calls: int

    def __new__(cls, value: str) -> AlwaysEqualStr:
        instance = str.__new__(cls, value)
        instance.equality_calls = 0
        return instance

    def __eq__(self, other: object) -> bool:
        self.equality_calls += 1
        return True

    __hash__ = str.__hash__


class SetThenInterruptDict(dict[object, object]):
    """Expose rollback bugs where a mapping mutates before interruption."""

    def __init__(self, values: dict[object, object] | None = None) -> None:
        super().__init__({} if values is None else values)
        self.should_interrupt = True

    def __setitem__(self, key: object, value: object) -> None:
        super().__setitem__(key, value)
        if self.should_interrupt:
            self.should_interrupt = False
            raise KeyboardInterrupt("injected publication interruption")


class InterruptingRollbackDict(SetThenInterruptDict):
    """Try to interrupt rollback through an overridden mapping method."""

    def pop(self, key: object, default: object = None) -> object:
        raise KeyboardInterrupt("rollback override must be bypassed")


def make_scope(**overrides: str) -> MemoryScope:
    values = {
        "tenant_id": "tenant-1",
        "namespace": "assistant-memory",
        "subject_id": "user-1",
    }
    values.update(overrides)
    return MemoryScope(**values)


def make_event(**overrides: object) -> EvidenceEvent:
    values: dict[str, object] = {
        "scope": make_scope(),
        "session_id": "session-1",
        "run_id": "run-1",
        "sequence_no": 0,
        "kind": EvidenceKind.USER_MESSAGE,
        "payload": "hello",
        "observed_at": UTC_INSTANT,
        "idempotency_key": "idempotency-1",
    }
    values.update(overrides)
    return EvidenceEvent(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("failing_index", ("idempotency", "scope"))
def test_append_rolls_back_every_index_after_base_exception(
    failing_index: str,
) -> None:
    store = InMemoryEvidenceStore()
    if failing_index == "idempotency":
        store._by_idempotency_key = SetThenInterruptDict()
    else:
        store._by_scope = SetThenInterruptDict()
    event = make_event()

    with pytest.raises(KeyboardInterrupt, match="publication interruption"):
        store.append(event)

    assert store._by_evidence_id == {}
    assert store._by_idempotency_key == {}
    assert store._by_scope == {}
    record = store.append(event)
    assert store.get(event.scope, record.evidence_id) == record
    assert store.list(event.scope) == (record,)


def test_rollback_bypasses_fault_injection_mapping_overrides() -> None:
    store = InMemoryEvidenceStore()
    store._by_idempotency_key = InterruptingRollbackDict()
    event = make_event()

    with pytest.raises(KeyboardInterrupt, match="publication interruption"):
        store.append(event)

    assert store._by_evidence_id == {}
    assert store._by_idempotency_key == {}
    assert store._by_scope == {}
    record = store.append(event)
    assert store.get(event.scope, record.evidence_id) == record


def test_atomic_publish_restores_an_existing_scope_list() -> None:
    old_record = object()
    new_record = object()
    scope_index: dict[object, list[object]] = {"scope": [old_record]}
    failing_index = SetThenInterruptDict()

    with pytest.raises(KeyboardInterrupt, match="publication interruption"):
        _atomic_publish(
            mapping_writes=(),
            sequence_appends=(
                (scope_index, "scope", new_record),
                (failing_index, "later-scope", new_record),
            ),
        )

    assert scope_index == {"scope": [old_record]}
    assert failing_index == {}


def test_error_types_share_memory_service_base_error() -> None:
    assert issubclass(EvidenceNotFoundError, MemoryServiceError)
    assert issubclass(EvidenceConflictError, MemoryServiceError)


def test_public_module_exports_stable_memory_contracts() -> None:
    """Expose only the intended immutable evidence, history, and release contracts."""
    from areal.v2.memory_service import (
        InMemoryMemoryReleaseStore as PublicInMemoryMemoryReleaseStore,
    )
    from areal.v2.memory_service import MemoryRelease as PublicMemoryRelease
    from areal.v2.memory_service import MemoryReleaseStore as PublicMemoryReleaseStore
    from areal.v2.memory_service import (
        ReleaseConflictError as PublicReleaseConflictError,
    )
    from areal.v2.memory_service import ReleaseManifest as PublicReleaseManifest
    from areal.v2.memory_service import (
        ReleaseNotFoundError as PublicReleaseNotFoundError,
    )

    assert memory_service.__doc__ == (
        "Public immutable Memory Service evidence, history, release, and runtime "
        "contracts."
    )
    assert memory_service.__all__ == [
        "CandidateConflictError",
        "CandidateNotFoundError",
        "CandidateProposal",
        "EvidenceConflictError",
        "EvidenceEvent",
        "EvidenceKind",
        "EvidenceNotFoundError",
        "EvidenceRecord",
        "EvidenceStore",
        "InMemoryEvidenceStore",
        "InMemoryMemoryHistoryStore",
        "InMemoryMemoryReleaseControlStore",
        "InMemoryMemoryReleaseStore",
        "InMemoryMemoryRuntimeStore",
        "MemoryBoundaryMismatchError",
        "MemoryCandidate",
        "MemoryConsumerAckConflictError",
        "MemoryConsumerAckNotFoundError",
        "MemoryConsumerAckV1",
        "MemoryConsumerCallV1",
        "MemoryConsumerKind",
        "MemoryDeliveryConflictError",
        "MemoryDeliveryNotFoundError",
        "MemoryDeliveryV1",
        "MemoryEvidenceRefV1",
        "MemoryExposureConflictError",
        "MemoryExposureNotFoundError",
        "MemoryExposureStatus",
        "MemoryExposureV1",
        "MemoryHistoryStore",
        "MemoryQueryAttemptV1",
        "MemoryQueryConflictError",
        "MemoryQueryItemV1",
        "MemoryQueryNotFoundError",
        "MemoryQueryResultV1",
        "MemoryQuerySpecV1",
        "MemoryRelease",
        "MemoryReleaseAssignmentConsumerKind",
        "MemoryReleaseAssignmentConflictError",
        "MemoryReleaseAssignmentNotFoundError",
        "MemoryReleaseAssignmentPolicy",
        "MemoryReleaseAssignmentV1",
        "MemoryReleaseAttestationConflictError",
        "MemoryReleaseAttestationNotFoundError",
        "MemoryReleaseAttestationRevoker",
        "MemoryReleaseAttestationRevocationV1",
        "MemoryReleaseAttestationV1",
        "MemoryReleaseAttestor",
        "MemoryReleaseControlStore",
        "MemoryReleaseRevocationConflictError",
        "MemoryReleaseRevocationNotFoundError",
        "MemoryReleaseRevocationReason",
        "MemoryReleaseStore",
        "MemoryRevision",
        "MemoryRevisionRefV1",
        "MemorySourceObjectKind",
        "MemorySourceObjectRefV1",
        "MemorySourceReadEventV1",
        "MemorySourceReadOperation",
        "MemorySourceReadPhase",
        "MemorySourceReadReceiptV1",
        "MemorySourceReadTranscriptV1",
        "MemoryRenderedRevisionRangeV1",
        "MemoryRenderedRevisionSpanV1",
        "MemoryRetrievalOutputV1",
        "MemoryRenderOutputV1",
        "MemoryRuntimeConsumer",
        "MemoryRuntimeRenderer",
        "MemoryRuntimeRetriever",
        "MemoryRuntimeStore",
        "MemoryScope",
        "MemoryServiceError",
        "ReleaseConflictError",
        "ReleaseManifest",
        "ReleaseNotFoundError",
        "RevisionConflictError",
        "RevisionNotFoundError",
        "RevisionOperation",
        "RevisionProposal",
    ]

    from areal.v2.memory_service import (
        CandidateConflictError as PublicCandidateConflictError,
    )
    from areal.v2.memory_service import (
        CandidateNotFoundError as PublicCandidateNotFoundError,
    )
    from areal.v2.memory_service import CandidateProposal as PublicCandidateProposal
    from areal.v2.memory_service import (
        EvidenceConflictError as PublicEvidenceConflictError,
    )
    from areal.v2.memory_service import EvidenceEvent as PublicEvidenceEvent
    from areal.v2.memory_service import EvidenceKind as PublicEvidenceKind
    from areal.v2.memory_service import (
        EvidenceNotFoundError as PublicEvidenceNotFoundError,
    )
    from areal.v2.memory_service import EvidenceRecord as PublicEvidenceRecord
    from areal.v2.memory_service import EvidenceStore as PublicEvidenceStore
    from areal.v2.memory_service import (
        InMemoryEvidenceStore as PublicInMemoryEvidenceStore,
    )
    from areal.v2.memory_service import (
        InMemoryMemoryHistoryStore as PublicInMemoryMemoryHistoryStore,
    )
    from areal.v2.memory_service import MemoryCandidate as PublicMemoryCandidate
    from areal.v2.memory_service import MemoryHistoryStore as PublicMemoryHistoryStore
    from areal.v2.memory_service import MemoryRevision as PublicMemoryRevision
    from areal.v2.memory_service import MemoryScope as PublicMemoryScope
    from areal.v2.memory_service import MemoryServiceError as PublicMemoryServiceError
    from areal.v2.memory_service import (
        RevisionConflictError as PublicRevisionConflictError,
    )
    from areal.v2.memory_service import (
        RevisionNotFoundError as PublicRevisionNotFoundError,
    )
    from areal.v2.memory_service import RevisionOperation as PublicRevisionOperation
    from areal.v2.memory_service import RevisionProposal as PublicRevisionProposal

    public_symbols = (
        PublicCandidateConflictError,
        PublicCandidateNotFoundError,
        PublicCandidateProposal,
        PublicEvidenceConflictError,
        PublicEvidenceEvent,
        PublicEvidenceKind,
        PublicEvidenceNotFoundError,
        PublicEvidenceRecord,
        PublicEvidenceStore,
        PublicInMemoryEvidenceStore,
        PublicInMemoryMemoryHistoryStore,
        PublicMemoryCandidate,
        PublicMemoryHistoryStore,
        PublicMemoryRevision,
        PublicMemoryScope,
        PublicMemoryServiceError,
        PublicRevisionConflictError,
        PublicRevisionNotFoundError,
        PublicRevisionOperation,
        PublicRevisionProposal,
    )
    defining_symbols = (
        CandidateConflictError,
        CandidateNotFoundError,
        CandidateProposal,
        EvidenceConflictError,
        EvidenceEvent,
        EvidenceKind,
        EvidenceNotFoundError,
        EvidenceRecord,
        EvidenceStore,
        InMemoryEvidenceStore,
        InMemoryMemoryHistoryStore,
        MemoryCandidate,
        MemoryHistoryStore,
        MemoryRevision,
        MemoryScope,
        MemoryServiceError,
        RevisionConflictError,
        RevisionNotFoundError,
        RevisionOperation,
        RevisionProposal,
    )
    for public_symbol, defining_symbol in zip(
        public_symbols,
        defining_symbols,
        strict=True,
    ):
        assert public_symbol is defining_symbol
    assert PublicInMemoryMemoryReleaseStore is InMemoryMemoryReleaseStore
    assert PublicMemoryRelease is MemoryRelease
    assert PublicMemoryReleaseStore is MemoryReleaseStore
    assert PublicReleaseConflictError is ReleaseConflictError
    assert PublicReleaseManifest is ReleaseManifest
    assert PublicReleaseNotFoundError is ReleaseNotFoundError


def test_evidence_store_contract_exposes_no_update_or_delete() -> None:
    """Keep both the protocol and implementation append-only."""
    store = InMemoryEvidenceStore()

    assert not hasattr(EvidenceStore, "update")
    assert not hasattr(EvidenceStore, "delete")
    assert not hasattr(store, "update")
    assert not hasattr(store, "delete")


def test_append_rejects_evidence_event_subclass() -> None:
    event = make_event()
    event_subclass = EvidenceEventSubclass(
        scope=event.scope,
        session_id=event.session_id,
        run_id=event.run_id,
        sequence_no=event.sequence_no,
        kind=event.kind,
        payload=event.payload,
        observed_at=event.observed_at,
        idempotency_key=event.idempotency_key,
    )

    with pytest.raises(TypeError, match="event must be an EvidenceEvent"):
        InMemoryEvidenceStore().append(event_subclass)


def test_get_rejects_memory_scope_subclass() -> None:
    scope = MemoryScopeSubclass("tenant-1", "assistant-memory", "user-1")

    with pytest.raises(TypeError, match="scope must be a MemoryScope"):
        InMemoryEvidenceStore().get(scope, "evd_missing")


def test_list_rejects_memory_scope_subclass() -> None:
    scope = MemoryScopeSubclass("tenant-1", "assistant-memory", "user-1")

    with pytest.raises(TypeError, match="scope must be a MemoryScope"):
        InMemoryEvidenceStore().list(scope)


def test_append_and_get_round_trip_returns_persisted_record() -> None:
    store = InMemoryEvidenceStore()
    event = make_event()

    record = store.append(event)

    assert record.event is event
    assert store.get(event.scope, record.evidence_id) is record


def test_append_identical_retry_returns_exact_original_record() -> None:
    store = InMemoryEvidenceStore()
    original_event = make_event()
    equivalent_retry = make_event()
    assert equivalent_retry is not original_event
    assert equivalent_retry.canonical_bytes() == original_event.canonical_bytes()

    original_record = store.append(original_event)
    retry_record = store.append(equivalent_retry)

    assert retry_record is original_record
    assert retry_record.event is original_event
    assert store.list(original_event.scope) == (original_record,)


def test_concurrent_identical_appends_return_exact_original_record() -> None:
    """Serialize same-event races into one record without duplicate writes."""
    store = InMemoryEvidenceStore()
    store._by_evidence_id = YieldingMissingDict()
    store._by_idempotency_key = YieldingMissingDict()
    event = make_event()
    caller_count = 32
    start = Barrier(caller_count, timeout=CONCURRENCY_TIMEOUT_SECONDS)

    def append_after_start() -> EvidenceRecord:
        start.wait()
        return store.append(event)

    with _hard_bounded_executor(max_workers=caller_count) as executor:
        futures = [executor.submit(append_after_start) for _ in range(caller_count)]
        records = tuple(
            future.result(timeout=CONCURRENCY_TIMEOUT_SECONDS) for future in futures
        )

    original = records[0]
    assert all(record is original for record in records)
    assert {record.evidence_id for record in records} == {original.evidence_id}
    assert store.list(event.scope) == (original,)


def test_append_rejects_changed_event_with_same_scoped_idempotency_key() -> None:
    store = InMemoryEvidenceStore()
    original = store.append(make_event(payload="first"))

    with pytest.raises(EvidenceConflictError, match="idempotency"):
        store.append(make_event(payload="changed"))

    assert store.get(original.event.scope, original.evidence_id) is original
    assert store.list(original.event.scope) == (original,)


def test_concurrent_scoped_idempotency_conflicts_leave_only_winner_indexed() -> None:
    """Commit one racing event atomically and reject every conflicting writer."""
    store = InMemoryEvidenceStore()
    store._by_evidence_id = YieldingMissingDict()
    store._by_idempotency_key = YieldingMissingDict()
    scope = make_scope()
    events = tuple(
        make_event(scope=scope, sequence_no=index, payload=f"payload-{index}")
        for index in range(32)
    )
    start = Barrier(len(events), timeout=CONCURRENCY_TIMEOUT_SECONDS)

    def append_after_start(event: EvidenceEvent) -> EvidenceRecord:
        start.wait()
        return store.append(event)

    with _hard_bounded_executor(max_workers=len(events)) as executor:
        futures = [executor.submit(append_after_start, event) for event in events]
        winners: list[EvidenceRecord] = []
        conflicts: list[EvidenceConflictError] = []
        for future in futures:
            try:
                winners.append(future.result(timeout=CONCURRENCY_TIMEOUT_SECONDS))
            except EvidenceConflictError as error:
                conflicts.append(error)

    assert len(winners) == 1
    assert len(conflicts) == len(events) - 1
    assert {type(error) for error in conflicts} == {EvidenceConflictError}

    winner = winners[0]
    losers = tuple(event for event in events if event is not winner.event)
    assert len(losers) == len(events) - 1
    assert store.list(scope) == (winner,)
    assert store.get(scope, winner.evidence_id) is winner
    assert store.append(winner.event) is winner

    for loser in losers:
        with pytest.raises(EvidenceConflictError, match="idempotency"):
            store.append(loser)
        loser_hash = calculate_sha256(loser.canonical_bytes()).hexdigest()
        with pytest.raises(EvidenceNotFoundError):
            store.get(scope, f"evd_{loser_hash[:24]}")

    assert store.list(scope) == (winner,)


def test_get_missing_evidence_raises_not_found() -> None:
    store = InMemoryEvidenceStore()

    with pytest.raises(EvidenceNotFoundError) as error:
        store.get(make_scope(), "evd_missing")

    assert type(error.value) is EvidenceNotFoundError


def test_get_snapshots_evidence_id_before_lookup() -> None:
    store = InMemoryEvidenceStore()
    record = store.append(make_event())
    query_id = MutableHashStr(record.evidence_id)
    query_id.hash_salt = 1_000_003

    assert store.get(record.event.scope, query_id) is record
    assert query_id.hash_calls == 0


def test_blank_query_values_remain_no_match() -> None:
    store = InMemoryEvidenceStore()
    record = store.append(make_event())

    with pytest.raises(EvidenceNotFoundError):
        store.get(record.event.scope, "")
    assert store.list(record.event.scope, session_id="") == ()
    assert store.list(record.event.scope, run_id="") == ()


def test_get_hides_record_that_belongs_to_another_scope() -> None:
    store = InMemoryEvidenceStore()
    record = store.append(make_event())
    other_scope = make_scope(subject_id="user-2")

    with pytest.raises(EvidenceNotFoundError) as missing_error:
        store.get(record.event.scope, "evd_missing")
    with pytest.raises(EvidenceNotFoundError) as wrong_scope_error:
        store.get(other_scope, record.evidence_id)

    assert type(wrong_scope_error.value) is EvidenceNotFoundError
    assert str(wrong_scope_error.value) == str(missing_error.value).replace(
        "evd_missing", record.evidence_id
    )


def test_list_filters_by_scope_session_and_run() -> None:
    store = InMemoryEvidenceStore()
    scope = make_scope()
    other_scope = make_scope(subject_id="user-2")
    session_one_run_one = store.append(
        make_event(
            scope=scope,
            session_id="session-1",
            run_id="run-1",
            idempotency_key="one-one",
        )
    )
    session_one_run_two = store.append(
        make_event(
            scope=scope,
            session_id="session-1",
            run_id="run-2",
            idempotency_key="one-two",
        )
    )
    session_two_run_one = store.append(
        make_event(
            scope=scope,
            session_id="session-2",
            run_id="run-1",
            idempotency_key="two-one",
        )
    )
    store.append(
        make_event(
            scope=other_scope,
            session_id="session-1",
            run_id="run-1",
            idempotency_key="one-one",
        )
    )

    assert store.list(scope) == (
        session_one_run_one,
        session_one_run_two,
        session_two_run_one,
    )
    assert store.list(scope, session_id="session-1") == (
        session_one_run_one,
        session_one_run_two,
    )
    assert store.list(scope, run_id="run-1") == (
        session_one_run_one,
        session_two_run_one,
    )
    assert store.list(scope, session_id="session-1", run_id="run-2") == (
        session_one_run_two,
    )
    assert store.list(scope, session_id="missing") == ()


@pytest.mark.parametrize("field", ["session_id", "run_id"])
def test_list_snapshots_filter_before_comparison(field: str) -> None:
    store = InMemoryEvidenceStore()
    record = store.append(make_event())
    query = AlwaysEqualStr(f"missing-{field}")

    assert store.list(record.event.scope, **{field: query}) == ()
    assert query.equality_calls == 0


def test_list_returns_deterministic_order_using_normalized_instants() -> None:
    store = InMemoryEvidenceStore()
    scope = make_scope()
    earlier_instant = store.append(
        make_event(
            scope=scope,
            session_id="session-a",
            run_id="run-a",
            sequence_no=1,
            observed_at=datetime(
                2026, 7, 7, 12, 0, tzinfo=timezone(timedelta(hours=8))
            ),
            idempotency_key="earlier-instant",
        )
    )
    later_instant = store.append(
        make_event(
            scope=scope,
            session_id="session-a",
            run_id="run-a",
            sequence_no=1,
            observed_at=datetime(2026, 7, 7, 4, 30, tzinfo=UTC),
            idempotency_key="later-instant",
        )
    )
    tied_a = store.append(
        make_event(
            scope=scope,
            session_id="session-a",
            run_id="run-a",
            sequence_no=2,
            payload="tie-a",
            idempotency_key="tie-a",
        )
    )
    tied_b = store.append(
        make_event(
            scope=scope,
            session_id="session-a",
            run_id="run-a",
            sequence_no=2,
            payload="tie-b",
            idempotency_key="tie-b",
        )
    )
    later_sequence = store.append(
        make_event(
            scope=scope,
            session_id="session-a",
            run_id="run-a",
            sequence_no=3,
            observed_at=datetime(2020, 1, 1, tzinfo=UTC),
            idempotency_key="later-sequence",
        )
    )
    later_run = store.append(
        make_event(
            scope=scope,
            session_id="session-a",
            run_id="run-b",
            sequence_no=0,
            idempotency_key="later-run",
        )
    )
    later_session = store.append(
        make_event(
            scope=scope,
            session_id="session-b",
            run_id="run-a",
            sequence_no=0,
            idempotency_key="later-session",
        )
    )
    tied_by_id = tuple(sorted((tied_a, tied_b), key=lambda item: item.evidence_id))

    assert store.list(scope) == (
        earlier_instant,
        later_instant,
        *tied_by_id,
        later_sequence,
        later_run,
        later_session,
    )


def test_list_orders_folded_datetimes_by_absolute_instant() -> None:
    """Snapshot and sort absolute instants when a wall-clock hour folds backward."""
    store = InMemoryEvidenceStore()
    scope = make_scope()
    fold_timezone = FoldAwareTimezone()
    earlier_source = datetime(2026, 11, 1, 1, 45, tzinfo=fold_timezone, fold=0)
    later_source = datetime(2026, 11, 1, 1, 15, tzinfo=fold_timezone, fold=1)
    earlier_instant = store.append(
        make_event(
            scope=scope,
            session_id="session-a",
            run_id="run-a",
            sequence_no=1,
            observed_at=earlier_source,
            idempotency_key="earlier-fold-instant",
        )
    )
    later_instant = store.append(
        make_event(
            scope=scope,
            session_id="session-a",
            run_id="run-a",
            sequence_no=1,
            observed_at=later_source,
            idempotency_key="later-fold-instant",
        )
    )

    assert earlier_instant.event.observed_at == datetime(2026, 11, 1, 5, 45, tzinfo=UTC)
    assert later_instant.event.observed_at == datetime(2026, 11, 1, 6, 15, tzinfo=UTC)
    assert earlier_instant.event.observed_at.tzinfo is UTC
    assert later_instant.event.observed_at.tzinfo is UTC
    assert store.list(scope) == (earlier_instant, later_instant)


def test_append_sets_aware_utc_created_at() -> None:
    record = InMemoryEvidenceStore().append(make_event())

    assert record.created_at.tzinfo is UTC
    assert record.created_at.utcoffset() == timedelta(0)


def test_append_uses_sha256_content_hash_and_derived_evidence_id() -> None:
    event = make_event()
    expected_hash = calculate_sha256(event.canonical_bytes()).hexdigest()

    record = InMemoryEvidenceStore().append(event)

    assert record.content_hash == expected_hash
    assert re.fullmatch(r"[0-9a-f]{64}", record.content_hash)
    assert record.evidence_id == f"evd_{expected_hash[:24]}"
    assert re.fullmatch(r"evd_[0-9a-f]{24}", record.evidence_id)


def test_same_idempotency_key_is_independent_across_scopes() -> None:
    store = InMemoryEvidenceStore()
    first_scope = make_scope(subject_id="user-1")
    second_scope = make_scope(subject_id="user-2")

    first = store.append(make_event(scope=first_scope, payload="first"))
    second = store.append(make_event(scope=second_scope, payload="second"))

    assert first is not second
    assert first.evidence_id != second.evidence_id
    assert store.list(first_scope) == (first,)
    assert store.list(second_scope) == (second,)


def test_full_hash_collision_fails_without_overwriting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ConstantHash:
        def hexdigest(self) -> str:
            return "a" * 64

    def constant_sha256(_: bytes) -> ConstantHash:
        return ConstantHash()

    monkeypatch.setattr(store_module.hashlib, "sha256", constant_sha256)
    store = InMemoryEvidenceStore()
    original = store.append(make_event(idempotency_key="first"))
    colliding_event = make_event(payload="different", idempotency_key="second")

    with pytest.raises(EvidenceConflictError, match="collision"):
        store.append(colliding_event)

    assert store.get(original.event.scope, original.evidence_id) is original
    assert store.list(original.event.scope) == (original,)


def test_hash_prefix_collision_fails_without_overwriting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject distinct full hashes that derive the same truncated evidence ID."""
    shared_prefix = "a" * 24
    original_digest = shared_prefix + "b" * 40
    colliding_digest = shared_prefix + "c" * 40
    original_event = make_event(idempotency_key="first")
    colliding_event = make_event(payload="different", idempotency_key="second")
    digest_by_canonical_bytes = {
        original_event.canonical_bytes(): original_digest,
        colliding_event.canonical_bytes(): colliding_digest,
    }

    class StubHash:
        def __init__(self, digest: str) -> None:
            self._digest = digest

        def hexdigest(self) -> str:
            return self._digest

    def prefix_colliding_sha256(canonical_bytes: bytes) -> StubHash:
        return StubHash(digest_by_canonical_bytes[canonical_bytes])

    monkeypatch.setattr(store_module.hashlib, "sha256", prefix_colliding_sha256)
    store = InMemoryEvidenceStore()
    original = store.append(original_event)

    with pytest.raises(EvidenceConflictError, match="collision"):
        store.append(colliding_event)

    assert original.content_hash == original_digest
    assert original.evidence_id == f"evd_{shared_prefix}"
    assert store.get(original.event.scope, original.evidence_id) is original
    assert store.list(original.event.scope) == (original,)


def test_hash_prefix_collision_is_isolated_across_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_prefix = "a" * 24
    first_digest = shared_prefix + "b" * 40
    second_digest = shared_prefix + "c" * 40
    first_scope = make_scope(subject_id="user-1")
    second_scope = make_scope(subject_id="user-2")
    missing_scope = make_scope(subject_id="user-3")
    first_event = make_event(
        scope=first_scope,
        payload="first",
        idempotency_key="shared-key",
    )
    second_event = make_event(
        scope=second_scope,
        payload="second",
        idempotency_key="shared-key",
    )
    digest_by_canonical_bytes = {
        first_event.canonical_bytes(): first_digest,
        second_event.canonical_bytes(): second_digest,
    }

    class StubHash:
        def __init__(self, digest: str) -> None:
            self._digest = digest

        def hexdigest(self) -> str:
            return self._digest

    def prefix_colliding_sha256(canonical_bytes: bytes) -> StubHash:
        return StubHash(digest_by_canonical_bytes[canonical_bytes])

    monkeypatch.setattr(store_module.hashlib, "sha256", prefix_colliding_sha256)
    store = InMemoryEvidenceStore()

    first = store.append(first_event)
    second = store.append(second_event)

    assert first is not second
    assert first.evidence_id == second.evidence_id == f"evd_{shared_prefix}"
    assert first.content_hash == first_digest
    assert second.content_hash == second_digest
    assert store.get(first_scope, first.evidence_id) is first
    assert store.get(second_scope, second.evidence_id) is second
    with pytest.raises(EvidenceNotFoundError):
        store.get(missing_scope, first.evidence_id)
    assert store.list(first_scope) == (first,)
    assert store.list(second_scope) == (second,)
    assert store.list(missing_scope) == ()
