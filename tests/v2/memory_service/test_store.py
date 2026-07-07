# SPDX-License-Identifier: Apache-2.0

"""Tests for the in-memory Memory Service evidence store."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from hashlib import sha256 as calculate_sha256
from threading import Barrier
from time import sleep

import pytest

import areal.v2.memory_service as memory_service
from areal.v2.memory_service import store as store_module
from areal.v2.memory_service.errors import (
    EvidenceConflictError,
    EvidenceNotFoundError,
    MemoryServiceError,
)
from areal.v2.memory_service.store import EvidenceStore, InMemoryEvidenceStore
from areal.v2.memory_service.types import (
    EvidenceEvent,
    EvidenceKind,
    EvidenceRecord,
    MemoryScope,
)

UTC_INSTANT = datetime(2026, 7, 7, 4, 5, 6, 789000, tzinfo=UTC)
CONCURRENCY_TIMEOUT_SECONDS = 10.0


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


def test_error_types_share_memory_service_base_error() -> None:
    assert issubclass(EvidenceNotFoundError, MemoryServiceError)
    assert issubclass(EvidenceConflictError, MemoryServiceError)


def test_public_module_exports_only_stable_evidence_contract() -> None:
    """Expose only the intended immutable evidence API at package level."""
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
    from areal.v2.memory_service import MemoryScope as PublicMemoryScope
    from areal.v2.memory_service import MemoryServiceError as PublicMemoryServiceError

    assert memory_service.__all__ == [
        "EvidenceConflictError",
        "EvidenceEvent",
        "EvidenceKind",
        "EvidenceNotFoundError",
        "EvidenceRecord",
        "EvidenceStore",
        "InMemoryEvidenceStore",
        "MemoryScope",
        "MemoryServiceError",
    ]
    assert (
        PublicEvidenceConflictError,
        PublicEvidenceEvent,
        PublicEvidenceKind,
        PublicEvidenceNotFoundError,
        PublicEvidenceRecord,
        PublicEvidenceStore,
        PublicInMemoryEvidenceStore,
        PublicMemoryScope,
        PublicMemoryServiceError,
    ) == (
        EvidenceConflictError,
        EvidenceEvent,
        EvidenceKind,
        EvidenceNotFoundError,
        EvidenceRecord,
        EvidenceStore,
        InMemoryEvidenceStore,
        MemoryScope,
        MemoryServiceError,
    )


def test_evidence_store_contract_exposes_no_update_or_delete() -> None:
    """Keep both the protocol and implementation append-only."""
    store = InMemoryEvidenceStore()

    assert not hasattr(EvidenceStore, "update")
    assert not hasattr(EvidenceStore, "delete")
    assert not hasattr(store, "update")
    assert not hasattr(store, "delete")


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

    with ThreadPoolExecutor(max_workers=caller_count) as executor:
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

    with ThreadPoolExecutor(max_workers=len(events)) as executor:
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
