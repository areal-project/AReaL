# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, fields, replace
from datetime import UTC, datetime, timedelta, timezone
from hashlib import sha256

import pytest

from areal.v2.memory_service.release_control_types import (
    MemoryReleaseAssignmentConsumerKind,
    MemoryReleaseAssignmentV1,
    MemoryReleaseAttestationRevocationV1,
    MemoryReleaseAttestationV1,
    MemoryReleaseRevocationReason,
)
from areal.v2.memory_service.types import MemoryScope

_BASE = datetime(2026, 7, 13, 1, 2, 3, 456789, tzinfo=UTC)
_RELEASE_HASH = sha256(b"release").hexdigest()
_GRAPH_HASH = sha256(b"release-graph").hexdigest()
_COMPONENT_HASH = sha256(b"component-version").hexdigest()
_CONFIG_HASH = sha256(b"component-config").hexdigest()
_GROUP_INCARNATION = sha256(b"group-incarnation").hexdigest()
_TASK_HASH = sha256(b"task-policy").hexdigest()
_RETRIEVAL_HASH = sha256(b"retrieval-policy").hexdigest()
_TASK_CONFIG_HASH = sha256(b"task-config").hexdigest()
_RETRIEVAL_CONFIG_HASH = sha256(b"retrieval-config").hexdigest()
_RENDERER_HASH = sha256(b"renderer-version").hexdigest()
_RENDERER_CONFIG_HASH = sha256(b"renderer-config").hexdigest()
_CONSUMER_HASH = sha256(b"consumer-version").hexdigest()
_CONSUMER_CONFIG_HASH = sha256(b"consumer-config").hexdigest()
_EXPECTED_HASHES = (
    "7518345b895cf433c100f37d6f446d3f60b1d387bcd28a2308b74a1119a5a406",
    "abda486bdac41dfebbe12eb6e0583aeb672c3783d8079cf0d407159cb23d0fc5",
    "a335ddd2c980e9987df88d29d079e867505b67d3a035c99c5820310f16d72496",
)


def _records():
    scope = MemoryScope("tenant", "agent-memory", "subject")
    attestation = MemoryReleaseAttestationV1.create(
        scope=scope,
        release_id=f"rel_{_RELEASE_HASH[:24]}",
        release_content_sha256=_RELEASE_HASH,
        release_graph_sha256=_GRAPH_HASH,
        attestor_id="safety-gate",
        attestor_version_sha256=_COMPONENT_HASH,
        attestor_config_sha256=_CONFIG_HASH,
        valid_from=_BASE,
        valid_until=_BASE + timedelta(hours=1),
        evaluated_at=_BASE,
        attested_at=_BASE + timedelta(microseconds=1),
        idempotency_key="attest-1",
    )
    revocation = MemoryReleaseAttestationRevocationV1.create(
        scope=scope,
        attestation_id=attestation.attestation_id,
        attestation_content_sha256=attestation.content_hash,
        revoker_id="operator-gate",
        revoker_version_sha256=_COMPONENT_HASH,
        revoker_config_sha256=_CONFIG_HASH,
        reason=MemoryReleaseRevocationReason.POLICY_REGRESSION,
        reason_detail_sha256=sha256(b"incident-7").hexdigest(),
        evaluated_at=_BASE + timedelta(minutes=1),
        revoked_at=_BASE + timedelta(minutes=1, microseconds=1),
        idempotency_key="revoke-1",
    )
    assignment = MemoryReleaseAssignmentV1.create(
        scope=scope,
        rollout_group_id="group-7",
        rollout_group_incarnation_sha256=_GROUP_INCARNATION,
        attestation_id=attestation.attestation_id,
        attestation_content_sha256=attestation.content_hash,
        release_id=attestation.release_id,
        release_content_sha256=attestation.release_content_sha256,
        release_graph_sha256=attestation.release_graph_sha256,
        assignment_policy_id="stable-rollout",
        assignment_policy_version_sha256=_COMPONENT_HASH,
        assignment_policy_config_sha256=_CONFIG_HASH,
        task_policy_id="frozen-agent",
        task_policy_version_sha256=_TASK_HASH,
        task_policy_config_sha256=_TASK_CONFIG_HASH,
        retrieval_policy_id="release-order-v1",
        retrieval_policy_version_sha256=_RETRIEVAL_HASH,
        retrieval_policy_config_sha256=_RETRIEVAL_CONFIG_HASH,
        renderer_id="json-lines-v1",
        renderer_version_sha256=_RENDERER_HASH,
        renderer_config_sha256=_RENDERER_CONFIG_HASH,
        consumer_kind=MemoryReleaseAssignmentConsumerKind.MODEL_CALL,
        consumer_id="areal-openai-model-call",
        consumer_version_sha256=_CONSUMER_HASH,
        consumer_config_sha256=_CONSUMER_CONFIG_HASH,
        max_returned_items=4,
        max_context_utf8_bytes=4096,
        evaluated_at=_BASE + timedelta(seconds=1),
        assigned_at=_BASE + timedelta(seconds=1, microseconds=1),
        assignment_valid_until=_BASE + timedelta(minutes=30),
        idempotency_key="assign-1",
    )
    return attestation, revocation, assignment


def test_control_records_have_golden_canonical_hashes_and_ids() -> None:
    records = _records()
    assert tuple(item.content_hash for item in records) == _EXPECTED_HASHES
    assert records[0].attestation_id == f"mrat_{_EXPECTED_HASHES[0][:24]}"
    assert records[1].revocation_id == f"mrvk_{_EXPECTED_HASHES[1][:24]}"
    assert records[2].assignment_id == f"masn_{_EXPECTED_HASHES[2][:24]}"
    for record in records:
        assert sha256(record.canonical_bytes()).hexdigest() == record.content_hash


def test_assignment_commits_the_complete_query_policy_snapshot() -> None:
    _attestation, _revocation, assignment = _records()
    value = json.loads(assignment.canonical_bytes())

    assert value["release_graph_sha256"] == _GRAPH_HASH
    assert value["rollout_group_incarnation_sha256"] == _GROUP_INCARNATION
    assert value["task_policy_version_sha256"] == _TASK_HASH
    assert value["retrieval_policy_version_sha256"] == _RETRIEVAL_HASH
    assert value["task_policy_config_sha256"] == _TASK_CONFIG_HASH
    assert value["retrieval_policy_config_sha256"] == _RETRIEVAL_CONFIG_HASH
    assert value["renderer_version_sha256"] == _RENDERER_HASH
    assert value["consumer_kind"] == "model_call"
    assert value["max_returned_items"] == 4
    assert value["max_context_utf8_bytes"] == 4096
    assert value["evaluated_at"] < value["assigned_at"]
    assert value["assigned_at"] < value["assignment_valid_until"]


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("rollout_group_id", "other-group"),
        ("rollout_group_incarnation_sha256", sha256(b"other-group").hexdigest()),
        ("release_graph_sha256", sha256(b"other-graph").hexdigest()),
        ("task_policy_id", "other-task"),
        ("task_policy_version_sha256", sha256(b"other-task").hexdigest()),
        ("task_policy_config_sha256", sha256(b"other-task-config").hexdigest()),
        ("retrieval_policy_id", "other-retriever"),
        (
            "retrieval_policy_version_sha256",
            sha256(b"other-retriever").hexdigest(),
        ),
        (
            "retrieval_policy_config_sha256",
            sha256(b"other-retriever-config").hexdigest(),
        ),
        ("renderer_id", "other-renderer"),
        ("renderer_version_sha256", sha256(b"other-renderer").hexdigest()),
        ("consumer_kind", MemoryReleaseAssignmentConsumerKind.CONTEXT),
        ("consumer_id", "other-consumer"),
        ("consumer_version_sha256", sha256(b"other-consumer").hexdigest()),
        ("max_returned_items", 5),
        ("max_context_utf8_bytes", 8192),
    ),
)
def test_assignment_identity_changes_with_every_execution_binding(
    field_name: str,
    replacement: object,
) -> None:
    assignment = _records()[2]
    with pytest.raises(ValueError, match="content_hash disagrees"):
        replace(assignment, **{field_name: replacement})


def test_component_configuration_is_part_of_every_control_commitment() -> None:
    attestation, revocation, assignment = _records()
    assert json.loads(attestation.canonical_bytes())["attestor_config_sha256"] == (
        _CONFIG_HASH
    )
    assert json.loads(revocation.canonical_bytes())["revoker_config_sha256"] == (
        _CONFIG_HASH
    )
    assert (
        json.loads(assignment.canonical_bytes())["assignment_policy_config_sha256"]
        == _CONFIG_HASH
    )
    assert json.loads(assignment.canonical_bytes())["task_policy_config_sha256"] == (
        _TASK_CONFIG_HASH
    )
    assert (
        json.loads(assignment.canonical_bytes())["retrieval_policy_config_sha256"]
        == _RETRIEVAL_CONFIG_HASH
    )


def test_control_records_normalize_equivalent_timezone_values() -> None:
    attestation, _, _ = _records()
    shifted = MemoryReleaseAttestationV1.create(
        scope=attestation.scope,
        release_id=attestation.release_id,
        release_content_sha256=attestation.release_content_sha256,
        release_graph_sha256=attestation.release_graph_sha256,
        attestor_id=attestation.attestor_id,
        attestor_version_sha256=attestation.attestor_version_sha256,
        attestor_config_sha256=attestation.attestor_config_sha256,
        valid_from=attestation.valid_from.astimezone(timezone(timedelta(hours=8))),
        valid_until=attestation.valid_until.astimezone(timezone(timedelta(hours=8))),
        evaluated_at=attestation.evaluated_at.astimezone(timezone(timedelta(hours=8))),
        attested_at=attestation.attested_at.astimezone(timezone(timedelta(hours=8))),
        idempotency_key=attestation.idempotency_key,
    )
    assert shifted == attestation


@pytest.mark.parametrize("index", range(3))
def test_control_records_are_frozen_and_reject_identity_mutation(index: int) -> None:
    record = _records()[index]
    with pytest.raises(FrozenInstanceError):
        record.content_hash = "0" * 64
    with pytest.raises(ValueError, match="content_hash disagrees"):
        replace(record, content_hash="0" * 64)


def test_attestation_rejects_empty_or_inverted_validity_window() -> None:
    attestation, _, _ = _records()
    values = {
        "scope": attestation.scope,
        "release_id": attestation.release_id,
        "release_content_sha256": attestation.release_content_sha256,
        "release_graph_sha256": attestation.release_graph_sha256,
        "attestor_id": attestation.attestor_id,
        "attestor_version_sha256": attestation.attestor_version_sha256,
        "attestor_config_sha256": attestation.attestor_config_sha256,
        "evaluated_at": attestation.evaluated_at,
        "attested_at": attestation.attested_at,
        "idempotency_key": attestation.idempotency_key,
    }
    with pytest.raises(ValueError, match="non-empty"):
        MemoryReleaseAttestationV1.create(
            valid_from=_BASE,
            valid_until=_BASE,
            **values,
        )
    with pytest.raises(ValueError, match="non-empty"):
        MemoryReleaseAttestationV1.create(
            valid_from=_BASE + timedelta(seconds=1),
            valid_until=_BASE,
            **values,
        )


def test_control_references_require_matching_public_id_and_full_hash() -> None:
    attestation, revocation, assignment = _records()
    with pytest.raises(ValueError, match="release_id disagrees"):
        replace(attestation, release_id="rel_" + "0" * 24)
    with pytest.raises(ValueError, match="attestation_id disagrees"):
        replace(revocation, attestation_id="mrat_" + "0" * 24)
    with pytest.raises(ValueError, match="attestation_id disagrees"):
        replace(assignment, attestation_id="mrat_" + "0" * 24)
    with pytest.raises(ValueError, match="release_id disagrees"):
        replace(assignment, release_id="rel_" + "0" * 24)


@pytest.mark.parametrize("index", range(3))
def test_control_records_reject_commit_time_before_evaluation(index: int) -> None:
    record = _records()[index]
    if isinstance(record, MemoryReleaseAttestationV1):
        with pytest.raises(ValueError, match="evaluated_at"):
            replace(record, attested_at=record.evaluated_at - timedelta(microseconds=1))
    elif isinstance(record, MemoryReleaseAttestationRevocationV1):
        with pytest.raises(ValueError, match="evaluated_at"):
            replace(record, revoked_at=record.evaluated_at - timedelta(microseconds=1))
    else:
        with pytest.raises(ValueError, match="evaluated_at"):
            replace(record, assigned_at=record.evaluated_at - timedelta(microseconds=1))


def test_assignment_requires_a_non_empty_execution_window() -> None:
    assignment = _records()[2]
    with pytest.raises(ValueError, match="validity window"):
        replace(assignment, assignment_valid_until=assignment.assigned_at)


@pytest.mark.parametrize("field_name", ("max_returned_items", "max_context_utf8_bytes"))
@pytest.mark.parametrize("value", (True, -1, 2**63))
def test_assignment_rejects_invalid_budgets(field_name: str, value: object) -> None:
    assignment = _records()[2]
    expected = "must be an int" if value is True else "must be between"
    with pytest.raises((TypeError, ValueError), match=expected):
        replace(assignment, **{field_name: value})


def test_revocation_reason_is_a_bounded_exact_enum() -> None:
    _attestation, revocation, _assignment = _records()
    with pytest.raises(TypeError, match="MemoryReleaseRevocationReason"):
        replace(revocation, reason="policy_regression")
    with pytest.raises(ValueError, match="reason_detail_sha256"):
        replace(
            revocation,
            reason=MemoryReleaseRevocationReason.OTHER,
            reason_detail_sha256=None,
        )


@pytest.mark.parametrize("record", _records())
def test_control_records_reject_subclass_scope(record: object) -> None:
    class ScopeSubclass(MemoryScope):
        pass

    with pytest.raises(TypeError, match="scope must be a MemoryScope"):
        replace(record, scope=ScopeSubclass("tenant", "namespace", "subject"))


def test_control_schema_has_no_trusted_active_or_utility_flags() -> None:
    for record_type in (
        MemoryReleaseAttestationV1,
        MemoryReleaseAttestationRevocationV1,
        MemoryReleaseAssignmentV1,
    ):
        names = {field.name for field in fields(record_type)}
        assert names.isdisjoint({"trusted", "active", "promoted", "useful", "score"})


def test_every_non_derived_field_is_present_in_canonical_bytes() -> None:
    for record, id_field in zip(
        _records(),
        ("attestation_id", "revocation_id", "assignment_id"),
        strict=True,
    ):
        expected = {field.name for field in fields(record)} - {
            id_field,
            "content_hash",
        }
        actual = set(json.loads(record.canonical_bytes())) - {
            "record_kind",
            "schema_version",
        }
        assert actual == expected


def test_control_value_types_are_public_identity_exports() -> None:
    from areal.v2.memory_service import (
        MemoryReleaseAssignmentConsumerKind as PublicConsumerKind,
    )
    from areal.v2.memory_service import (
        MemoryReleaseAssignmentV1 as PublicAssignment,
    )
    from areal.v2.memory_service import (
        MemoryReleaseAttestationRevocationV1 as PublicRevocation,
    )
    from areal.v2.memory_service import (
        MemoryReleaseAttestationV1 as PublicAttestation,
    )
    from areal.v2.memory_service import (
        MemoryReleaseRevocationReason as PublicRevocationReason,
    )

    assert PublicConsumerKind is MemoryReleaseAssignmentConsumerKind
    assert PublicAssignment is MemoryReleaseAssignmentV1
    assert PublicRevocation is MemoryReleaseAttestationRevocationV1
    assert PublicAttestation is MemoryReleaseAttestationV1
    assert PublicRevocationReason is MemoryReleaseRevocationReason
