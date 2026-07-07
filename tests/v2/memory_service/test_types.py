# SPDX-License-Identifier: Apache-2.0

"""Tests for immutable Memory Service evidence value objects."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone, tzinfo

import pytest

from areal.v2.memory_service.store import InMemoryEvidenceStore
from areal.v2.memory_service.types import (
    EvidenceEvent,
    EvidenceKind,
    EvidenceRecord,
    MemoryScope,
)

UTC_INSTANT = datetime(2026, 7, 7, 4, 5, 6, 789000, tzinfo=UTC)
LONE_SURROGATE = "\ud800"
PLUS_ONE_HOUR = timezone(timedelta(hours=1))
MINUS_ONE_HOUR = timezone(-timedelta(hours=1))
UTC_OVERFLOWING_DATETIMES = (
    datetime.min.replace(tzinfo=PLUS_ONE_HOUR),
    datetime.max.replace(tzinfo=MINUS_ONE_HOUR),
)
VALID_UTC_BOUNDARIES = (
    (
        (datetime.min + timedelta(hours=1)).replace(tzinfo=PLUS_ONE_HOUR),
        datetime.min.replace(tzinfo=UTC),
    ),
    (
        (datetime.max - timedelta(hours=1)).replace(tzinfo=MINUS_ONE_HOUR),
        datetime.max.replace(tzinfo=UTC),
    ),
)


class MutableTimezone(tzinfo):
    """Timezone whose offset can change after a datetime is constructed."""

    def __init__(self, offset: timedelta) -> None:
        self.offset = offset

    def utcoffset(self, value: datetime | None) -> timedelta:
        return self.offset

    def dst(self, value: datetime | None) -> timedelta:
        return timedelta(0)

    def tzname(self, value: datetime | None) -> str:
        return "MUTABLE"


class StatefulDatetime(datetime):
    """Datetime subclass with unsafe conversion and serialization overrides."""

    suffix: str = ""

    def astimezone(self, timezone: tzinfo | None = None) -> StatefulDatetime:
        return self

    def isoformat(self, sep: str = "T", timespec: str = "auto") -> str:
        return f"{datetime.isoformat(self, sep, timespec)}{self.suffix}"


class MutableHashStr(str):
    """String subclass whose hash can change after construction."""

    hash_salt: int

    def __new__(cls, value: str) -> MutableHashStr:
        instance = str.__new__(cls, value)
        instance.hash_salt = 0
        return instance

    def __hash__(self) -> int:
        return str.__hash__(self) + self.hash_salt

    def __str__(self) -> str:
        return "overridden"


class AdversarialStr(str):
    """String subclass overriding validation and conversion operations."""

    def __str__(self) -> str:
        return "overridden"

    def strip(self, chars: str | None = None) -> str:
        return "overridden"

    def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        return b"overridden"


class MemoryScopeSubclass(MemoryScope):
    def __hash__(self) -> int:
        return 0


class EvidenceEventSubclass(EvidenceEvent):
    def canonical_bytes(self) -> bytes:
        return b"overridden"


class OverriddenInt(int):
    def __int__(self) -> int:
        return 999


def make_scope(**overrides: object) -> MemoryScope:
    values: dict[str, object] = {
        "tenant_id": "tenant-1",
        "namespace": "assistant-memory",
        "subject_id": "user-1",
    }
    values.update(overrides)
    return MemoryScope(**values)  # type: ignore[arg-type]


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


def make_record(**overrides: object) -> EvidenceRecord:
    values: dict[str, object] = {
        "evidence_id": "evidence-1",
        "event": make_event(),
        "content_hash": "sha256:abc123",
        "created_at": UTC_INSTANT,
    }
    values.update(overrides)
    return EvidenceRecord(**values)  # type: ignore[arg-type]


def test_evidence_kind_values_are_lowercase_snake_case() -> None:
    assert {kind.name: kind.value for kind in EvidenceKind} == {
        "USER_MESSAGE": "user_message",
        "AGENT_MESSAGE": "agent_message",
        "TOOL_CALL": "tool_call",
        "TOOL_RESULT": "tool_result",
        "ENVIRONMENT": "environment",
        "FEEDBACK": "feedback",
        "OUTCOME": "outcome",
    }


@pytest.mark.parametrize("field", ["tenant_id", "namespace", "subject_id"])
@pytest.mark.parametrize("value", ["", " \t\n"])
def test_memory_scope_rejects_blank_identifiers(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=field):
        make_scope(**{field: value})


@pytest.mark.parametrize("field", ["tenant_id", "namespace", "subject_id"])
@pytest.mark.parametrize("value", [None, 7])
def test_memory_scope_rejects_non_string_identifiers(field: str, value: object) -> None:
    with pytest.raises(TypeError, match=field):
        make_scope(**{field: value})


def test_memory_scope_preserves_identifier_whitespace() -> None:
    scope = MemoryScope(
        tenant_id=" tenant-1 ",
        namespace=" assistant-memory ",
        subject_id=" user-1 ",
    )

    assert scope.tenant_id == " tenant-1 "
    assert scope.namespace == " assistant-memory "
    assert scope.subject_id == " user-1 "


def test_memory_scope_snapshots_mutable_hash_strings_for_store_indexes() -> None:
    tenant_id = MutableHashStr("tenant-1")
    namespace = MutableHashStr("assistant-memory")
    subject_id = MutableHashStr("user-1")
    scope = MemoryScope(
        tenant_id=tenant_id,
        namespace=namespace,
        subject_id=subject_id,
    )
    event = make_event(scope=scope)
    store = InMemoryEvidenceStore()
    scope_hash = hash(scope)
    original = store.append(event)

    tenant_id.hash_salt = 1_000_003
    namespace.hash_salt = 1_000_003
    subject_id.hash_salt = 1_000_003
    retry = store.append(event)

    assert retry is original
    assert hash(scope) == scope_hash
    assert type(scope.tenant_id) is str
    assert type(scope.namespace) is str
    assert type(scope.subject_id) is str
    assert scope == make_scope()
    assert store.get(scope, original.evidence_id) is original
    assert store.list(scope) == (original,)
    assert len(store._by_evidence_id) == 1
    assert len(store._by_idempotency_key) == 1
    assert len(store._by_scope) == 1


@pytest.mark.parametrize("field", ["tenant_id", "namespace", "subject_id"])
def test_memory_scope_rejects_invalid_unicode_identifier(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        make_scope(**{field: LONE_SURROGATE})


def test_string_snapshot_uses_base_blank_validation() -> None:
    with pytest.raises(ValueError, match="session_id"):
        make_event(session_id=AdversarialStr(" \t\n"))


def test_string_snapshot_uses_base_utf8_validation() -> None:
    with pytest.raises(ValueError, match="payload"):
        make_event(payload=AdversarialStr(LONE_SURROGATE))


@pytest.mark.parametrize("field", ["session_id", "run_id", "idempotency_key"])
@pytest.mark.parametrize("value", ["", " \t\n"])
def test_evidence_event_rejects_blank_identifiers(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=field):
        make_event(**{field: value})


@pytest.mark.parametrize("field", ["session_id", "run_id", "idempotency_key"])
@pytest.mark.parametrize("value", [None, 7])
def test_evidence_event_rejects_non_string_identifiers(
    field: str, value: object
) -> None:
    with pytest.raises(TypeError, match=field):
        make_event(**{field: value})


def test_evidence_event_preserves_identifier_whitespace() -> None:
    event = make_event(
        session_id=" session-1 ",
        run_id=" run-1 ",
        idempotency_key=" idempotency-1 ",
    )

    assert event.session_id == " session-1 "
    assert event.run_id == " run-1 "
    assert event.idempotency_key == " idempotency-1 "


@pytest.mark.parametrize(
    "field", ["session_id", "run_id", "idempotency_key", "payload"]
)
def test_evidence_event_rejects_invalid_unicode_string(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        make_event(**{field: LONE_SURROGATE})


@pytest.mark.parametrize(
    "scope",
    [
        {
            "tenant_id": "tenant-1",
            "namespace": "assistant-memory",
            "subject_id": "user-1",
        },
        object(),
    ],
    ids=["dict", "object"],
)
def test_evidence_event_requires_memory_scope(scope: object) -> None:
    with pytest.raises(TypeError, match="scope"):
        make_event(scope=scope)


def test_evidence_event_rejects_memory_scope_subclass() -> None:
    scope = MemoryScopeSubclass("tenant-1", "assistant-memory", "user-1")

    with pytest.raises(TypeError, match="scope must be a MemoryScope"):
        make_event(scope=scope)


def test_evidence_event_accepts_empty_string_payload() -> None:
    assert make_event(payload="").payload == ""


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_id", " session-1 "),
        ("run_id", " run-1 "),
        ("payload", " payload "),
        ("idempotency_key", " idempotency-1 "),
    ],
)
def test_evidence_event_snapshots_string_subclasses(field: str, value: str) -> None:
    event = make_event(**{field: AdversarialStr(value)})
    stored = getattr(event, field)

    assert type(stored) is str
    assert stored == value


@pytest.mark.parametrize("payload", [None, b"hello", 7])
def test_evidence_event_rejects_non_string_payload(payload: object) -> None:
    with pytest.raises(TypeError, match="payload"):
        make_event(payload=payload)


@pytest.mark.parametrize("sequence_no", [True, False, OverriddenInt(1), 1.0, "1", None])
def test_evidence_event_rejects_non_integer_sequence_numbers(
    sequence_no: object,
) -> None:
    with pytest.raises(TypeError, match="sequence_no"):
        make_event(sequence_no=sequence_no)


def test_evidence_event_rejects_negative_sequence_number() -> None:
    with pytest.raises(ValueError, match="sequence_no"):
        make_event(sequence_no=-1)


def test_evidence_event_accepts_zero_sequence_number() -> None:
    assert make_event(sequence_no=0).sequence_no == 0


def test_evidence_event_accepts_maximum_protocol_sequence_number() -> None:
    sequence_no = 2**63 - 1
    event = make_event(sequence_no=sequence_no)

    assert event.sequence_no == sequence_no
    assert json.loads(event.canonical_bytes())["sequence_no"] == sequence_no


@pytest.mark.parametrize(
    "sequence_no",
    [2**63, 10**4300],
    ids=["past-int64", "past-python-json-digit-limit"],
)
def test_evidence_event_rejects_sequence_number_above_protocol_maximum(
    sequence_no: int,
) -> None:
    with pytest.raises(ValueError, match="sequence_no"):
        make_event(sequence_no=sequence_no)


@pytest.mark.parametrize("kind", ["user_message", None, 7])
def test_evidence_event_requires_evidence_kind(kind: object) -> None:
    with pytest.raises(TypeError, match="kind"):
        make_event(kind=kind)


@pytest.mark.parametrize("observed_at", [None, "2026-07-07T04:05:06Z"])
def test_evidence_event_rejects_non_datetime_observed_at(
    observed_at: object,
) -> None:
    with pytest.raises(TypeError, match="observed_at"):
        make_event(observed_at=observed_at)


def test_evidence_event_rejects_naive_observed_at() -> None:
    with pytest.raises(ValueError, match="observed_at"):
        make_event(observed_at=datetime(2026, 7, 7, 4, 5, 6))


@pytest.mark.parametrize(
    "observed_at",
    UTC_OVERFLOWING_DATETIMES,
    ids=["minimum-at-plus-one", "maximum-at-minus-one"],
)
def test_evidence_event_rejects_datetime_that_cannot_normalize_to_utc(
    observed_at: datetime,
) -> None:
    with pytest.raises(ValueError, match="observed_at"):
        make_event(observed_at=observed_at)


def test_evidence_event_snapshots_mutable_observed_at_as_utc() -> None:
    source_timezone = MutableTimezone(timedelta(hours=1))
    source = datetime(2026, 7, 7, 5, 5, 6, 789000, tzinfo=source_timezone)
    event = make_event(observed_at=source)
    canonical_bytes = event.canonical_bytes()

    source_timezone.offset = timedelta(hours=2)

    assert event.canonical_bytes() == canonical_bytes
    assert event.observed_at == UTC_INSTANT
    assert event.observed_at.tzinfo is UTC


def test_evidence_event_snapshots_datetime_subclass_as_plain_utc() -> None:
    source = StatefulDatetime(
        2026,
        7,
        7,
        5,
        5,
        6,
        789000,
        tzinfo=timezone(timedelta(hours=1)),
    )
    source.suffix = "-before"
    event = make_event(observed_at=source)
    canonical_bytes = event.canonical_bytes()

    source.suffix = "-after"

    assert type(event.observed_at) is datetime
    assert event.observed_at == UTC_INSTANT
    assert event.canonical_bytes() == canonical_bytes
    assert b"-before" not in canonical_bytes


def test_canonical_bytes_are_sorted_compact_and_deterministic() -> None:
    event = make_event()

    expected = (
        b'{"idempotency_key":"idempotency-1","kind":"user_message",'
        b'"observed_at":"2026-07-07T04:05:06.789000+00:00",'
        b'"payload":"hello","run_id":"run-1",'
        b'"scope":{"namespace":"assistant-memory","subject_id":"user-1",'
        b'"tenant_id":"tenant-1"},"sequence_no":0,"session_id":"session-1"}'
    )

    assert event.canonical_bytes() == expected
    assert event.canonical_bytes() == make_event().canonical_bytes()


def test_canonical_bytes_preserve_valid_non_ascii_payload() -> None:
    payload = "你好，世界 🌍"
    canonical = make_event(payload=payload).canonical_bytes()

    assert payload.encode("utf-8") in canonical
    assert json.loads(canonical)["payload"] == payload


@pytest.mark.parametrize(
    ("observed_at", "expected_utc"),
    VALID_UTC_BOUNDARIES,
    ids=["minimum", "maximum"],
)
def test_canonical_bytes_support_valid_utc_datetime_boundaries(
    observed_at: datetime, expected_utc: datetime
) -> None:
    event = make_event(observed_at=observed_at)

    assert event.observed_at == expected_utc
    assert event.observed_at.tzinfo is UTC
    assert (
        json.loads(event.canonical_bytes())["observed_at"] == expected_utc.isoformat()
    )


def test_canonical_bytes_normalize_semantically_equal_instants_to_utc() -> None:
    utc_event = make_event(observed_at=UTC_INSTANT)
    utc_plus_eight_event = make_event(
        observed_at=UTC_INSTANT.astimezone(timezone(timedelta(hours=8)))
    )

    assert utc_event.observed_at == utc_plus_eight_event.observed_at == UTC_INSTANT
    assert utc_event.observed_at.tzinfo is UTC
    assert utc_plus_eight_event.observed_at.tzinfo is UTC
    assert utc_event.canonical_bytes() == utc_plus_eight_event.canonical_bytes()


@pytest.mark.parametrize(
    "event",
    ["not-an-event", {"payload": "hello"}, object()],
    ids=["string", "dict", "object"],
)
def test_evidence_record_requires_evidence_event(event: object) -> None:
    with pytest.raises(TypeError, match="event"):
        make_record(event=event)


def test_evidence_record_rejects_evidence_event_subclass() -> None:
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
        make_record(event=event_subclass)


@pytest.mark.parametrize("field", ["evidence_id", "content_hash"])
@pytest.mark.parametrize("value", ["", " \t\n"])
def test_evidence_record_rejects_blank_identifiers_and_hash(
    field: str, value: str
) -> None:
    with pytest.raises(ValueError, match=field):
        make_record(**{field: value})


@pytest.mark.parametrize("field", ["evidence_id", "content_hash"])
@pytest.mark.parametrize("value", [None, 7])
def test_evidence_record_rejects_non_string_identifiers_and_hash(
    field: str, value: object
) -> None:
    with pytest.raises(TypeError, match=field):
        make_record(**{field: value})


def test_evidence_record_preserves_identifier_and_hash_whitespace() -> None:
    record = make_record(evidence_id=" evidence-1 ", content_hash=" sha256:abc123 ")

    assert record.evidence_id == " evidence-1 "
    assert record.content_hash == " sha256:abc123 "


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evidence_id", " evidence-1 "),
        ("content_hash", " sha256:abc123 "),
    ],
)
def test_evidence_record_snapshots_string_subclasses(field: str, value: str) -> None:
    record = make_record(**{field: AdversarialStr(value)})
    stored = getattr(record, field)

    assert type(stored) is str
    assert stored == value


@pytest.mark.parametrize("field", ["evidence_id", "content_hash"])
def test_evidence_record_rejects_invalid_unicode_string(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        make_record(**{field: LONE_SURROGATE})


@pytest.mark.parametrize("created_at", [None, "2026-07-07T04:05:06Z"])
def test_evidence_record_rejects_non_datetime_created_at(created_at: object) -> None:
    with pytest.raises(TypeError, match="created_at"):
        make_record(created_at=created_at)


def test_evidence_record_rejects_naive_created_at() -> None:
    with pytest.raises(ValueError, match="created_at"):
        make_record(created_at=datetime(2026, 7, 7, 4, 5, 6))


@pytest.mark.parametrize(
    "created_at",
    UTC_OVERFLOWING_DATETIMES,
    ids=["minimum-at-plus-one", "maximum-at-minus-one"],
)
def test_evidence_record_rejects_datetime_that_cannot_normalize_to_utc(
    created_at: datetime,
) -> None:
    with pytest.raises(ValueError, match="created_at"):
        make_record(created_at=created_at)


def test_evidence_record_snapshots_mutable_created_at_as_utc() -> None:
    source_timezone = MutableTimezone(timedelta(hours=1))
    source = datetime(2026, 7, 7, 5, 5, 6, 789000, tzinfo=source_timezone)
    record = make_record(created_at=source)

    source_timezone.offset = timedelta(hours=2)

    assert record.created_at == UTC_INSTANT
    assert record.created_at.tzinfo is UTC


@pytest.mark.parametrize(
    ("created_at", "expected_utc"),
    VALID_UTC_BOUNDARIES,
    ids=["minimum", "maximum"],
)
def test_evidence_record_accepts_valid_utc_datetime_boundaries(
    created_at: datetime, expected_utc: datetime
) -> None:
    record = make_record(created_at=created_at)

    assert record.created_at == expected_utc
    assert record.created_at.tzinfo is UTC


@pytest.mark.parametrize(
    ("instance", "field", "new_value"),
    [
        (make_scope(), "tenant_id", "other-tenant"),
        (make_event(), "payload", "other payload"),
        (make_record(), "content_hash", "sha256:other"),
    ],
    ids=["memory-scope", "evidence-event", "evidence-record"],
)
def test_value_objects_are_frozen_and_slotted(
    instance: object, field: str, new_value: str
) -> None:
    assert not hasattr(instance, "__dict__")

    with pytest.raises(FrozenInstanceError):
        setattr(instance, field, new_value)
