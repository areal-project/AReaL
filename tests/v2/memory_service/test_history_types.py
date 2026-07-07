# SPDX-License-Identifier: Apache-2.0

"""Tests for immutable Memory Service candidate and revision values."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime, timedelta, timezone, tzinfo

import pytest

from areal.v2.memory_service.history_types import (
    CandidateProposal,
    MemoryCandidate,
    MemoryRevision,
    RevisionOperation,
    RevisionProposal,
)
from areal.v2.memory_service.types import MemoryScope


class StringSubclass(str):
    pass


class TupleSubclass(tuple[str, ...]):
    def __iter__(self):
        return iter(("overridden",))


class IntSubclass(int):
    pass


class MemoryScopeSubclass(MemoryScope):
    pass


class CandidateProposalSubclass(CandidateProposal):
    pass


class RevisionProposalSubclass(RevisionProposal):
    pass


class MutableTimezone(tzinfo):
    def __init__(self) -> None:
        self.offset = timedelta(hours=1)

    def utcoffset(self, value: datetime | None) -> timedelta:
        return self.offset

    def dst(self, value: datetime | None) -> timedelta:
        return timedelta(0)

    def tzname(self, value: datetime | None) -> str:
        return "MUTABLE"


class StatefulDatetime(datetime):
    def astimezone(self, timezone: tzinfo | None = None) -> StatefulDatetime:
        return self


LONE_SURROGATE = "\ud800"


def make_scope(**overrides: str) -> MemoryScope:
    values = {
        "tenant_id": "tenant-1",
        "namespace": "assistant-memory",
        "subject_id": "user-1",
    }
    values.update(overrides)
    return MemoryScope(**values)


def make_candidate_proposal(**overrides: object) -> CandidateProposal:
    values: dict[str, object] = {
        "scope": make_scope(),
        "content": "remember this",
        "evidence_ids": ("evd_a",),
        "idempotency_key": "candidate-1",
    }
    values.update(overrides)
    return CandidateProposal(**values)  # type: ignore[arg-type]


def make_revision_proposal(**overrides: object) -> RevisionProposal:
    values: dict[str, object] = {
        "scope": make_scope(),
        "candidate_id": "cand_a",
        "operation": RevisionOperation.ADD,
        "parent_revision_id": None,
        "idempotency_key": "revision-1",
    }
    values.update(overrides)
    return RevisionProposal(**values)  # type: ignore[arg-type]


def test_revision_operation_wire_values_are_stable() -> None:
    assert {item.name: item.value for item in RevisionOperation} == {
        "ADD": "add",
        "REFINE": "refine",
        "SUPERSEDE": "supersede",
        "CONTRADICT": "contradict",
    }


def test_candidate_proposal_canonical_schema_v1_is_frozen() -> None:
    proposal = CandidateProposal(
        scope=make_scope(),
        content="remember this",
        evidence_ids=("evd_a", "evd_b"),
        idempotency_key="candidate-1",
    )
    assert proposal.canonical_bytes() == (
        b'{"content":"remember this","evidence_ids":["evd_a","evd_b"],'
        b'"idempotency_key":"candidate-1","schema_version":1,'
        b'"scope":{"namespace":"assistant-memory","subject_id":"user-1",'
        b'"tenant_id":"tenant-1"}}'
    )


def test_candidate_proposal_canonical_schema_preserves_literal_utf8() -> None:
    proposal = CandidateProposal(
        scope=make_scope(),
        content="记住🙂",
        evidence_ids=("evd_a",),
        idempotency_key="candidate-1",
    )

    canonical_bytes = proposal.canonical_bytes()
    expected = (
        '{"content":"记住🙂","evidence_ids":["evd_a"],'
        '"idempotency_key":"candidate-1","schema_version":1,'
        '"scope":{"namespace":"assistant-memory","subject_id":"user-1",'
        '"tenant_id":"tenant-1"}}'
    ).encode()

    assert canonical_bytes == expected
    assert b"\\u" not in canonical_bytes


def test_revision_proposal_canonical_schema_v1_is_frozen() -> None:
    proposal = RevisionProposal(
        scope=make_scope(),
        candidate_id="cand_a",
        operation=RevisionOperation.ADD,
        parent_revision_id=None,
        idempotency_key="revision-1",
    )
    assert proposal.canonical_bytes() == (
        b'{"candidate_id":"cand_a","idempotency_key":"revision-1",'
        b'"operation":"add","parent_revision_id":null,"schema_version":1,'
        b'"scope":{"namespace":"assistant-memory","subject_id":"user-1",'
        b'"tenant_id":"tenant-1"}}'
    )


def test_non_add_canonical_schema_serializes_parent_as_exact_string() -> None:
    proposal = make_revision_proposal(
        operation=RevisionOperation.REFINE,
        parent_revision_id=StringSubclass("rev_parent"),
    )
    value = json.loads(proposal.canonical_bytes())

    assert type(proposal.parent_revision_id) is str
    assert value["parent_revision_id"] == "rev_parent"
    assert type(value["parent_revision_id"]) is str


@pytest.mark.parametrize(
    "operation",
    [
        RevisionOperation.REFINE,
        RevisionOperation.SUPERSEDE,
        RevisionOperation.CONTRADICT,
    ],
)
def test_non_add_revision_requires_one_parent(operation: RevisionOperation) -> None:
    with pytest.raises(ValueError, match="parent_revision_id"):
        make_revision_proposal(operation=operation, parent_revision_id=None)


def test_add_revision_rejects_parent() -> None:
    with pytest.raises(ValueError, match="parent_revision_id"):
        make_revision_proposal(
            operation=RevisionOperation.ADD,
            parent_revision_id="rev_parent",
        )


def test_candidate_proposal_snapshots_tuple_and_string_subclasses() -> None:
    proposal = make_candidate_proposal(
        content=StringSubclass(" remember this "),
        evidence_ids=TupleSubclass((StringSubclass("evd_a"), StringSubclass("evd_b"))),
        idempotency_key=StringSubclass(" candidate-1 "),
    )

    assert type(proposal.content) is str
    assert proposal.content == " remember this "
    assert type(proposal.evidence_ids) is tuple
    assert proposal.evidence_ids == ("evd_a", "evd_b")
    assert all(type(item) is str for item in proposal.evidence_ids)
    assert type(proposal.idempotency_key) is str
    assert proposal.idempotency_key == " candidate-1 "


def test_revision_proposal_snapshots_string_subclasses() -> None:
    proposal = make_revision_proposal(
        candidate_id=StringSubclass(" cand_a "),
        operation=RevisionOperation.REFINE,
        parent_revision_id=StringSubclass(" rev_parent "),
        idempotency_key=StringSubclass(" revision-1 "),
    )

    assert type(proposal.candidate_id) is str
    assert proposal.candidate_id == " cand_a "
    assert type(proposal.parent_revision_id) is str
    assert proposal.parent_revision_id == " rev_parent "
    assert type(proposal.idempotency_key) is str
    assert proposal.idempotency_key == " revision-1 "


@pytest.mark.parametrize("evidence_ids", [(), ("evd_a", "evd_a")])
def test_candidate_proposal_rejects_empty_or_duplicate_evidence(
    evidence_ids: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="evidence_ids"):
        make_candidate_proposal(evidence_ids=evidence_ids)


@pytest.mark.parametrize("evidence_ids", [["evd_a"], {"evd_a"}])
def test_candidate_proposal_rejects_non_tuple_evidence(evidence_ids: object) -> None:
    with pytest.raises(TypeError, match="evidence_ids"):
        make_candidate_proposal(evidence_ids=evidence_ids)


def test_candidate_proposal_rejects_blank_content_and_evidence_id() -> None:
    with pytest.raises(ValueError, match="content"):
        make_candidate_proposal(content=" \t\n")
    with pytest.raises(ValueError, match="evidence_ids"):
        make_candidate_proposal(evidence_ids=(" ",))


def test_proposals_require_exact_scope_and_revision_operation() -> None:
    scope = MemoryScopeSubclass("tenant-1", "assistant-memory", "user-1")
    with pytest.raises(TypeError, match="scope"):
        make_candidate_proposal(scope=scope)
    with pytest.raises(TypeError, match="operation"):
        make_revision_proposal(operation="add")


def test_persisted_records_reject_proposal_subclasses() -> None:
    candidate_proposal = CandidateProposalSubclass(
        make_scope(), "remember this", ("evd_a",), "candidate-1"
    )
    revision_proposal = RevisionProposalSubclass(
        make_scope(), "cand_a", RevisionOperation.ADD, None, "revision-1"
    )
    with pytest.raises(TypeError, match="CandidateProposal"):
        MemoryCandidate("cand_a", candidate_proposal, "a" * 64, datetime.now(UTC))
    with pytest.raises(TypeError, match="RevisionProposal"):
        MemoryRevision(
            "rev_a",
            "mem_a",
            0,
            revision_proposal,
            "b" * 64,
            datetime.now(UTC),
        )


def test_all_proposal_text_rejects_lone_surrogates() -> None:
    with pytest.raises(ValueError, match="content"):
        make_candidate_proposal(content=LONE_SURROGATE)
    with pytest.raises(ValueError, match="evidence_ids"):
        make_candidate_proposal(evidence_ids=(LONE_SURROGATE,))
    with pytest.raises(ValueError, match="idempotency_key"):
        make_candidate_proposal(idempotency_key=LONE_SURROGATE)
    with pytest.raises(ValueError, match="candidate_id"):
        make_revision_proposal(candidate_id=LONE_SURROGATE)
    with pytest.raises(ValueError, match="parent_revision_id"):
        make_revision_proposal(
            operation=RevisionOperation.REFINE,
            parent_revision_id=LONE_SURROGATE,
        )
    with pytest.raises(ValueError, match="idempotency_key"):
        make_revision_proposal(idempotency_key=LONE_SURROGATE)


@pytest.mark.parametrize("field", ["candidate_id", "content_hash"])
def test_memory_candidate_rejects_lone_surrogate_persisted_text(field: str) -> None:
    values: dict[str, object] = {
        "candidate_id": "cand_a",
        "proposal": make_candidate_proposal(),
        "content_hash": "a" * 64,
        "created_at": datetime.now(UTC),
    }
    values[field] = LONE_SURROGATE

    with pytest.raises(ValueError, match=rf"^{field} must be valid UTF-8$"):
        MemoryCandidate(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["revision_id", "memory_id", "content_hash"])
def test_memory_revision_rejects_lone_surrogate_persisted_text(field: str) -> None:
    values: dict[str, object] = {
        "revision_id": "rev_a",
        "memory_id": "mem_a",
        "generation": 0,
        "proposal": make_revision_proposal(),
        "content_hash": "b" * 64,
        "created_at": datetime.now(UTC),
    }
    values[field] = LONE_SURROGATE

    with pytest.raises(ValueError, match=rf"^{field} must be valid UTF-8$"):
        MemoryRevision(**values)  # type: ignore[arg-type]


def test_persisted_records_snapshot_identifiers_and_utc_datetime() -> None:
    created_at = datetime(2026, 7, 7, 12, tzinfo=timezone(timedelta(hours=8)))
    candidate = MemoryCandidate(
        candidate_id=StringSubclass("cand_a"),
        proposal=make_candidate_proposal(),
        content_hash=StringSubclass("a" * 64),
        created_at=created_at,
    )
    revision = MemoryRevision(
        revision_id=StringSubclass("rev_a"),
        memory_id=StringSubclass("mem_a"),
        generation=0,
        proposal=make_revision_proposal(),
        content_hash=StringSubclass("b" * 64),
        created_at=created_at,
    )

    assert type(candidate.candidate_id) is str
    assert type(candidate.content_hash) is str
    assert type(candidate.created_at) is datetime
    assert candidate.created_at == datetime(2026, 7, 7, 4, tzinfo=UTC)
    assert type(revision.revision_id) is str
    assert type(revision.memory_id) is str
    assert type(revision.content_hash) is str
    assert type(revision.created_at) is datetime
    assert revision.created_at.tzinfo is UTC


def test_persisted_records_detach_mutable_timezone_and_datetime_subclass() -> None:
    mutable_timezone = MutableTimezone()
    source = datetime(2026, 7, 7, 5, tzinfo=mutable_timezone)
    candidate = MemoryCandidate("cand_a", make_candidate_proposal(), "a" * 64, source)
    subclass_source = StatefulDatetime(2026, 7, 7, 4, tzinfo=UTC)
    revision = MemoryRevision(
        "rev_a",
        "mem_a",
        0,
        make_revision_proposal(),
        "b" * 64,
        subclass_source,
    )
    mutable_timezone.offset = timedelta(hours=2)

    assert type(candidate.created_at) is datetime
    assert candidate.created_at == datetime(2026, 7, 7, 4, tzinfo=UTC)
    assert type(revision.created_at) is datetime
    assert revision.created_at == datetime(2026, 7, 7, 4, tzinfo=UTC)


def test_persisted_records_reject_naive_or_non_normalizable_datetime() -> None:
    with pytest.raises(ValueError, match="created_at"):
        MemoryCandidate(
            "cand_a",
            make_candidate_proposal(),
            "a" * 64,
            datetime(2026, 7, 7),
        )
    with pytest.raises(ValueError, match="created_at"):
        MemoryRevision(
            "rev_a",
            "mem_a",
            0,
            make_revision_proposal(),
            "b" * 64,
            datetime.min.replace(tzinfo=timezone(timedelta(hours=1))),
        )


@pytest.mark.parametrize("generation", [True, IntSubclass(1), -1, 2**63])
def test_revision_rejects_invalid_generation(generation: object) -> None:
    with pytest.raises((TypeError, ValueError), match="generation"):
        MemoryRevision(
            revision_id="rev_a",
            memory_id="mem_a",
            generation=generation,  # type: ignore[arg-type]
            proposal=make_revision_proposal(),
            content_hash="a" * 64,
            created_at=datetime.now(UTC),
        )


@pytest.mark.parametrize("generation", [0, 2**63 - 1])
def test_revision_accepts_generation_boundaries(generation: int) -> None:
    revision = MemoryRevision(
        "rev_a",
        "mem_a",
        generation,
        make_revision_proposal(),
        "a" * 64,
        datetime.now(UTC),
    )
    assert revision.generation == generation


def test_values_are_frozen_and_slotted() -> None:
    values_and_fields = (
        (make_candidate_proposal(), "content"),
        (
            MemoryCandidate(
                "cand_a", make_candidate_proposal(), "a" * 64, datetime.now(UTC)
            ),
            "candidate_id",
        ),
        (make_revision_proposal(), "candidate_id"),
        (
            MemoryRevision(
                "rev_a",
                "mem_a",
                0,
                make_revision_proposal(),
                "b" * 64,
                datetime.now(UTC),
            ),
            "revision_id",
        ),
    )

    for value, field_name in values_and_fields:
        assert not hasattr(value, "__dict__")
        with pytest.raises(FrozenInstanceError):
            setattr(value, field_name, "changed")


def test_idempotency_key_is_part_of_candidate_identity() -> None:
    left = make_candidate_proposal(idempotency_key="attempt-a")
    right = make_candidate_proposal(idempotency_key="attempt-b")
    assert left.canonical_bytes() != right.canonical_bytes()


def test_exact_field_order_excludes_status_metadata_and_duplicated_evidence() -> None:
    assert tuple(field.name for field in fields(CandidateProposal)) == (
        "scope",
        "content",
        "evidence_ids",
        "idempotency_key",
    )
    assert tuple(field.name for field in fields(MemoryCandidate)) == (
        "candidate_id",
        "proposal",
        "content_hash",
        "created_at",
    )
    assert tuple(field.name for field in fields(RevisionProposal)) == (
        "scope",
        "candidate_id",
        "operation",
        "parent_revision_id",
        "idempotency_key",
    )
    assert tuple(field.name for field in fields(MemoryRevision)) == (
        "revision_id",
        "memory_id",
        "generation",
        "proposal",
        "content_hash",
        "created_at",
    )
