# SPDX-License-Identifier: Apache-2.0

"""Immutable control-plane records for graph-committed Memory releases.

The records deliberately live outside :mod:`release_types`: attesting,
revoking, or assigning a release never mutates its source graph or identity.
They are reproducible value objects, not bearer credentials.  Trust comes
from resolving an ID plus full hash in a trusted control store.  SHA-256 is an
integrity commitment, not a signature or proof that a release is useful.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256

from areal.v2.memory_service.types import MemoryScope

_SCHEMA_VERSION = 1
_MAX_INTEGER = 2**63 - 1


def _canonical_json_bytes(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _record_bytes(record_kind: str, value: dict[str, object]) -> bytes:
    return _canonical_json_bytes(
        {
            "record_kind": record_kind,
            "schema_version": _SCHEMA_VERSION,
            **value,
        }
    )


def _string(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a str")
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field_name} must be valid UTF-8") from error
    return value


def _digest(value: object, field_name: str) -> str:
    value = _string(value, field_name)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


def _integer(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an int")
    if not 0 <= value <= _MAX_INTEGER:
        raise ValueError(f"{field_name} must be between 0 and {_MAX_INTEGER}")
    return value


def _optional_digest(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _digest(value, field_name)


def _referenced_identity(
    record_id: object,
    content_hash: object,
    *,
    id_field: str,
    hash_field: str,
    prefix: str,
) -> tuple[str, str]:
    record_id = _string(record_id, id_field)
    content_hash = _digest(content_hash, hash_field)
    if record_id != f"{prefix}{content_hash[:24]}":
        raise ValueError(f"{id_field} disagrees with {hash_field}")
    return record_id, content_hash


def _scope(value: object) -> MemoryScope:
    if type(value) is not MemoryScope:
        raise TypeError("scope must be a MemoryScope")
    return value


def _aware_datetime(value: object, field_name: str) -> datetime:
    if type(value) is not datetime:
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    try:
        normalized = value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{field_name} must be normalizable to UTC") from error
    return datetime(
        normalized.year,
        normalized.month,
        normalized.day,
        normalized.hour,
        normalized.minute,
        normalized.second,
        normalized.microsecond,
        tzinfo=UTC,
    )


def _scope_value(scope: MemoryScope) -> dict[str, str]:
    scope = _scope(scope)
    return {
        "namespace": scope.namespace,
        "subject_id": scope.subject_id,
        "tenant_id": scope.tenant_id,
    }


def _datetime_value(value: datetime, field_name: str) -> str:
    return _aware_datetime(value, field_name).isoformat()


def _record_identity(
    record_id: object,
    content_hash: object,
    *,
    id_field: str,
    prefix: str,
    canonical_bytes: bytes,
) -> tuple[str, str]:
    content_hash = _digest(content_hash, "content_hash")
    expected_hash = sha256(canonical_bytes).hexdigest()
    if content_hash != expected_hash:
        raise ValueError("content_hash disagrees with canonical record bytes")
    record_id = _string(record_id, id_field)
    if record_id != f"{prefix}{expected_hash[:24]}":
        raise ValueError(f"{id_field} disagrees with canonical record bytes")
    return record_id, content_hash


class MemoryReleaseRevocationReason(StrEnum):
    """Bounded reason codes; sensitive free text stays outside the ledger."""

    POLICY_REGRESSION = "policy_regression"
    SAFETY = "safety"
    OPERATOR = "operator"
    SUPERSEDED = "superseded"
    OTHER = "other"


class MemoryReleaseAssignmentConsumerKind(StrEnum):
    """The trusted boundary at which assigned context must be acknowledged."""

    CONTEXT = "context"
    MODEL_CALL = "model_call"


@dataclass(frozen=True, slots=True)
class MemoryReleaseAttestationV1:
    """A trusted policy's time-bounded admission of one exact release graph."""

    attestation_id: str
    scope: MemoryScope
    release_id: str
    release_content_sha256: str
    release_graph_sha256: str
    attestor_id: str
    attestor_version_sha256: str
    attestor_config_sha256: str
    valid_from: datetime
    valid_until: datetime
    evaluated_at: datetime
    attested_at: datetime
    idempotency_key: str
    content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _scope(self.scope))
        for field_name in ("release_id", "attestor_id", "idempotency_key"):
            object.__setattr__(
                self,
                field_name,
                _string(getattr(self, field_name), field_name),
            )
        for field_name in (
            "release_content_sha256",
            "release_graph_sha256",
            "attestor_version_sha256",
            "attestor_config_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name),
            )
        _referenced_identity(
            self.release_id,
            self.release_content_sha256,
            id_field="release_id",
            hash_field="release_content_sha256",
            prefix="rel_",
        )
        for field_name in (
            "valid_from",
            "valid_until",
            "evaluated_at",
            "attested_at",
        ):
            object.__setattr__(
                self,
                field_name,
                _aware_datetime(getattr(self, field_name), field_name),
            )
        if self.valid_from >= self.valid_until:
            raise ValueError("attestation validity window must be non-empty")
        if self.evaluated_at > self.attested_at:
            raise ValueError("evaluated_at must not follow attested_at")
        record_id, content_hash = _record_identity(
            self.attestation_id,
            self.content_hash,
            id_field="attestation_id",
            prefix="mrat_",
            canonical_bytes=self.canonical_bytes(),
        )
        object.__setattr__(self, "attestation_id", record_id)
        object.__setattr__(self, "content_hash", content_hash)

    def _canonical_value(self) -> dict[str, object]:
        _referenced_identity(
            self.release_id,
            self.release_content_sha256,
            id_field="release_id",
            hash_field="release_content_sha256",
            prefix="rel_",
        )
        return {
            "attested_at": _datetime_value(self.attested_at, "attested_at"),
            "attestor_config_sha256": _digest(
                self.attestor_config_sha256,
                "attestor_config_sha256",
            ),
            "attestor_id": _string(self.attestor_id, "attestor_id"),
            "attestor_version_sha256": _digest(
                self.attestor_version_sha256,
                "attestor_version_sha256",
            ),
            "evaluated_at": _datetime_value(self.evaluated_at, "evaluated_at"),
            "idempotency_key": _string(self.idempotency_key, "idempotency_key"),
            "release_content_sha256": _digest(
                self.release_content_sha256,
                "release_content_sha256",
            ),
            "release_graph_sha256": _digest(
                self.release_graph_sha256,
                "release_graph_sha256",
            ),
            "release_id": _string(self.release_id, "release_id"),
            "scope": _scope_value(self.scope),
            "valid_from": _datetime_value(self.valid_from, "valid_from"),
            "valid_until": _datetime_value(self.valid_until, "valid_until"),
        }

    def canonical_bytes(self) -> bytes:
        return _record_bytes("memory_release_attestation", self._canonical_value())

    @classmethod
    def create(
        cls,
        *,
        scope: MemoryScope,
        release_id: str,
        release_content_sha256: str,
        release_graph_sha256: str,
        attestor_id: str,
        attestor_version_sha256: str,
        attestor_config_sha256: str,
        valid_from: datetime,
        valid_until: datetime,
        evaluated_at: datetime,
        attested_at: datetime,
        idempotency_key: str,
    ) -> MemoryReleaseAttestationV1:
        values = {
            "scope": scope,
            "release_id": release_id,
            "release_content_sha256": release_content_sha256,
            "release_graph_sha256": release_graph_sha256,
            "attestor_id": attestor_id,
            "attestor_version_sha256": attestor_version_sha256,
            "attestor_config_sha256": attestor_config_sha256,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "evaluated_at": evaluated_at,
            "attested_at": attested_at,
            "idempotency_key": idempotency_key,
        }
        canonical_value = {
            "attested_at": _datetime_value(attested_at, "attested_at"),
            "attestor_config_sha256": _digest(
                attestor_config_sha256,
                "attestor_config_sha256",
            ),
            "attestor_id": _string(attestor_id, "attestor_id"),
            "attestor_version_sha256": _digest(
                attestor_version_sha256,
                "attestor_version_sha256",
            ),
            "evaluated_at": _datetime_value(evaluated_at, "evaluated_at"),
            "idempotency_key": _string(idempotency_key, "idempotency_key"),
            "release_content_sha256": _digest(
                release_content_sha256,
                "release_content_sha256",
            ),
            "release_graph_sha256": _digest(
                release_graph_sha256,
                "release_graph_sha256",
            ),
            "release_id": _string(release_id, "release_id"),
            "scope": _scope_value(scope),
            "valid_from": _datetime_value(valid_from, "valid_from"),
            "valid_until": _datetime_value(valid_until, "valid_until"),
        }
        content_hash = sha256(
            _record_bytes("memory_release_attestation", canonical_value)
        ).hexdigest()
        return cls(
            attestation_id=f"mrat_{content_hash[:24]}",
            content_hash=content_hash,
            **values,
        )


@dataclass(frozen=True, slots=True)
class MemoryReleaseAttestationRevocationV1:
    """A trusted policy's irreversible revocation of one attestation."""

    revocation_id: str
    scope: MemoryScope
    attestation_id: str
    attestation_content_sha256: str
    revoker_id: str
    revoker_version_sha256: str
    revoker_config_sha256: str
    reason: MemoryReleaseRevocationReason
    reason_detail_sha256: str | None
    evaluated_at: datetime
    revoked_at: datetime
    idempotency_key: str
    content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _scope(self.scope))
        for field_name in ("attestation_id", "revoker_id", "idempotency_key"):
            object.__setattr__(
                self,
                field_name,
                _string(getattr(self, field_name), field_name),
            )
        for field_name in (
            "attestation_content_sha256",
            "revoker_version_sha256",
            "revoker_config_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name),
            )
        _referenced_identity(
            self.attestation_id,
            self.attestation_content_sha256,
            id_field="attestation_id",
            hash_field="attestation_content_sha256",
            prefix="mrat_",
        )
        if type(self.reason) is not MemoryReleaseRevocationReason:
            raise TypeError("reason must be a MemoryReleaseRevocationReason")
        object.__setattr__(
            self,
            "reason_detail_sha256",
            _optional_digest(self.reason_detail_sha256, "reason_detail_sha256"),
        )
        if (
            self.reason is MemoryReleaseRevocationReason.OTHER
            and self.reason_detail_sha256 is None
        ):
            raise ValueError("OTHER revocations require reason_detail_sha256")
        for field_name in ("evaluated_at", "revoked_at"):
            object.__setattr__(
                self,
                field_name,
                _aware_datetime(getattr(self, field_name), field_name),
            )
        if self.evaluated_at > self.revoked_at:
            raise ValueError("evaluated_at must not follow revoked_at")
        record_id, content_hash = _record_identity(
            self.revocation_id,
            self.content_hash,
            id_field="revocation_id",
            prefix="mrvk_",
            canonical_bytes=self.canonical_bytes(),
        )
        object.__setattr__(self, "revocation_id", record_id)
        object.__setattr__(self, "content_hash", content_hash)

    def _canonical_value(self) -> dict[str, object]:
        _referenced_identity(
            self.attestation_id,
            self.attestation_content_sha256,
            id_field="attestation_id",
            hash_field="attestation_content_sha256",
            prefix="mrat_",
        )
        return {
            "attestation_content_sha256": _digest(
                self.attestation_content_sha256,
                "attestation_content_sha256",
            ),
            "attestation_id": _string(self.attestation_id, "attestation_id"),
            "evaluated_at": _datetime_value(self.evaluated_at, "evaluated_at"),
            "idempotency_key": _string(self.idempotency_key, "idempotency_key"),
            "reason": self.reason.value,
            "reason_detail_sha256": _optional_digest(
                self.reason_detail_sha256,
                "reason_detail_sha256",
            ),
            "revoked_at": _datetime_value(self.revoked_at, "revoked_at"),
            "revoker_config_sha256": _digest(
                self.revoker_config_sha256,
                "revoker_config_sha256",
            ),
            "revoker_id": _string(self.revoker_id, "revoker_id"),
            "revoker_version_sha256": _digest(
                self.revoker_version_sha256,
                "revoker_version_sha256",
            ),
            "scope": _scope_value(self.scope),
        }

    def canonical_bytes(self) -> bytes:
        return _record_bytes(
            "memory_release_attestation_revocation",
            self._canonical_value(),
        )

    @classmethod
    def create(
        cls,
        *,
        scope: MemoryScope,
        attestation_id: str,
        attestation_content_sha256: str,
        revoker_id: str,
        revoker_version_sha256: str,
        revoker_config_sha256: str,
        reason: MemoryReleaseRevocationReason,
        reason_detail_sha256: str | None,
        evaluated_at: datetime,
        revoked_at: datetime,
        idempotency_key: str,
    ) -> MemoryReleaseAttestationRevocationV1:
        values = {
            "scope": scope,
            "attestation_id": attestation_id,
            "attestation_content_sha256": attestation_content_sha256,
            "revoker_id": revoker_id,
            "revoker_version_sha256": revoker_version_sha256,
            "revoker_config_sha256": revoker_config_sha256,
            "reason": reason,
            "reason_detail_sha256": reason_detail_sha256,
            "evaluated_at": evaluated_at,
            "revoked_at": revoked_at,
            "idempotency_key": idempotency_key,
        }
        if type(reason) is not MemoryReleaseRevocationReason:
            raise TypeError("reason must be a MemoryReleaseRevocationReason")
        canonical_value = {
            "attestation_content_sha256": _digest(
                attestation_content_sha256,
                "attestation_content_sha256",
            ),
            "attestation_id": _string(attestation_id, "attestation_id"),
            "evaluated_at": _datetime_value(evaluated_at, "evaluated_at"),
            "idempotency_key": _string(idempotency_key, "idempotency_key"),
            "reason": reason.value,
            "reason_detail_sha256": _optional_digest(
                reason_detail_sha256,
                "reason_detail_sha256",
            ),
            "revoked_at": _datetime_value(revoked_at, "revoked_at"),
            "revoker_config_sha256": _digest(
                revoker_config_sha256,
                "revoker_config_sha256",
            ),
            "revoker_id": _string(revoker_id, "revoker_id"),
            "revoker_version_sha256": _digest(
                revoker_version_sha256,
                "revoker_version_sha256",
            ),
            "scope": _scope_value(scope),
        }
        content_hash = sha256(
            _record_bytes(
                "memory_release_attestation_revocation",
                canonical_value,
            )
        ).hexdigest()
        return cls(
            revocation_id=f"mrvk_{content_hash[:24]}",
            content_hash=content_hash,
            **values,
        )


@dataclass(frozen=True, slots=True)
class MemoryReleaseAssignmentV1:
    """One exact rollout incarnation's historical execution authorization.

    A trusted store must still resolve this record as active immediately before
    every query, render, and consumer boundary.  The value is not a lease or a
    bearer token by itself.
    """

    assignment_id: str
    scope: MemoryScope
    rollout_group_id: str
    rollout_group_incarnation_sha256: str
    attestation_id: str
    attestation_content_sha256: str
    release_id: str
    release_content_sha256: str
    release_graph_sha256: str
    assignment_policy_id: str
    assignment_policy_version_sha256: str
    assignment_policy_config_sha256: str
    task_policy_id: str
    task_policy_version_sha256: str
    task_policy_config_sha256: str
    retrieval_policy_id: str
    retrieval_policy_version_sha256: str
    retrieval_policy_config_sha256: str
    renderer_id: str
    renderer_version_sha256: str
    renderer_config_sha256: str
    consumer_kind: MemoryReleaseAssignmentConsumerKind
    consumer_id: str
    consumer_version_sha256: str
    consumer_config_sha256: str
    max_returned_items: int
    max_context_utf8_bytes: int
    evaluated_at: datetime
    assigned_at: datetime
    assignment_valid_until: datetime
    idempotency_key: str
    content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _scope(self.scope))
        for field_name in (
            "rollout_group_id",
            "attestation_id",
            "release_id",
            "assignment_policy_id",
            "task_policy_id",
            "retrieval_policy_id",
            "renderer_id",
            "consumer_id",
            "idempotency_key",
        ):
            object.__setattr__(
                self,
                field_name,
                _string(getattr(self, field_name), field_name),
            )
        for field_name in (
            "rollout_group_incarnation_sha256",
            "attestation_content_sha256",
            "release_content_sha256",
            "release_graph_sha256",
            "assignment_policy_version_sha256",
            "assignment_policy_config_sha256",
            "task_policy_version_sha256",
            "task_policy_config_sha256",
            "retrieval_policy_version_sha256",
            "retrieval_policy_config_sha256",
            "renderer_version_sha256",
            "renderer_config_sha256",
            "consumer_version_sha256",
            "consumer_config_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name),
            )
        _referenced_identity(
            self.attestation_id,
            self.attestation_content_sha256,
            id_field="attestation_id",
            hash_field="attestation_content_sha256",
            prefix="mrat_",
        )
        _referenced_identity(
            self.release_id,
            self.release_content_sha256,
            id_field="release_id",
            hash_field="release_content_sha256",
            prefix="rel_",
        )
        if type(self.consumer_kind) is not MemoryReleaseAssignmentConsumerKind:
            raise TypeError(
                "consumer_kind must be a MemoryReleaseAssignmentConsumerKind"
            )
        for field_name in ("max_returned_items", "max_context_utf8_bytes"):
            object.__setattr__(
                self,
                field_name,
                _integer(getattr(self, field_name), field_name),
            )
        for field_name in (
            "evaluated_at",
            "assigned_at",
            "assignment_valid_until",
        ):
            object.__setattr__(
                self,
                field_name,
                _aware_datetime(getattr(self, field_name), field_name),
            )
        if self.evaluated_at > self.assigned_at:
            raise ValueError("evaluated_at must not follow assigned_at")
        if self.assigned_at >= self.assignment_valid_until:
            raise ValueError("assignment validity window must be non-empty")
        record_id, content_hash = _record_identity(
            self.assignment_id,
            self.content_hash,
            id_field="assignment_id",
            prefix="masn_",
            canonical_bytes=self.canonical_bytes(),
        )
        object.__setattr__(self, "assignment_id", record_id)
        object.__setattr__(self, "content_hash", content_hash)

    def _canonical_value(self) -> dict[str, object]:
        _referenced_identity(
            self.attestation_id,
            self.attestation_content_sha256,
            id_field="attestation_id",
            hash_field="attestation_content_sha256",
            prefix="mrat_",
        )
        _referenced_identity(
            self.release_id,
            self.release_content_sha256,
            id_field="release_id",
            hash_field="release_content_sha256",
            prefix="rel_",
        )
        return {
            "assigned_at": _datetime_value(self.assigned_at, "assigned_at"),
            "assignment_valid_until": _datetime_value(
                self.assignment_valid_until,
                "assignment_valid_until",
            ),
            "assignment_policy_config_sha256": _digest(
                self.assignment_policy_config_sha256,
                "assignment_policy_config_sha256",
            ),
            "assignment_policy_id": _string(
                self.assignment_policy_id,
                "assignment_policy_id",
            ),
            "assignment_policy_version_sha256": _digest(
                self.assignment_policy_version_sha256,
                "assignment_policy_version_sha256",
            ),
            "attestation_content_sha256": _digest(
                self.attestation_content_sha256,
                "attestation_content_sha256",
            ),
            "attestation_id": _string(self.attestation_id, "attestation_id"),
            "consumer_config_sha256": _digest(
                self.consumer_config_sha256,
                "consumer_config_sha256",
            ),
            "consumer_id": _string(self.consumer_id, "consumer_id"),
            "consumer_kind": self.consumer_kind.value,
            "consumer_version_sha256": _digest(
                self.consumer_version_sha256,
                "consumer_version_sha256",
            ),
            "evaluated_at": _datetime_value(self.evaluated_at, "evaluated_at"),
            "idempotency_key": _string(self.idempotency_key, "idempotency_key"),
            "max_context_utf8_bytes": _integer(
                self.max_context_utf8_bytes,
                "max_context_utf8_bytes",
            ),
            "max_returned_items": _integer(
                self.max_returned_items,
                "max_returned_items",
            ),
            "release_content_sha256": _digest(
                self.release_content_sha256,
                "release_content_sha256",
            ),
            "release_graph_sha256": _digest(
                self.release_graph_sha256,
                "release_graph_sha256",
            ),
            "release_id": _string(self.release_id, "release_id"),
            "renderer_config_sha256": _digest(
                self.renderer_config_sha256,
                "renderer_config_sha256",
            ),
            "renderer_id": _string(self.renderer_id, "renderer_id"),
            "renderer_version_sha256": _digest(
                self.renderer_version_sha256,
                "renderer_version_sha256",
            ),
            "retrieval_policy_config_sha256": _digest(
                self.retrieval_policy_config_sha256,
                "retrieval_policy_config_sha256",
            ),
            "retrieval_policy_id": _string(
                self.retrieval_policy_id,
                "retrieval_policy_id",
            ),
            "retrieval_policy_version_sha256": _digest(
                self.retrieval_policy_version_sha256,
                "retrieval_policy_version_sha256",
            ),
            "rollout_group_id": _string(
                self.rollout_group_id,
                "rollout_group_id",
            ),
            "rollout_group_incarnation_sha256": _digest(
                self.rollout_group_incarnation_sha256,
                "rollout_group_incarnation_sha256",
            ),
            "scope": _scope_value(self.scope),
            "task_policy_id": _string(self.task_policy_id, "task_policy_id"),
            "task_policy_config_sha256": _digest(
                self.task_policy_config_sha256,
                "task_policy_config_sha256",
            ),
            "task_policy_version_sha256": _digest(
                self.task_policy_version_sha256,
                "task_policy_version_sha256",
            ),
        }

    def canonical_bytes(self) -> bytes:
        return _record_bytes("memory_release_assignment", self._canonical_value())

    @classmethod
    def create(
        cls,
        *,
        scope: MemoryScope,
        rollout_group_id: str,
        rollout_group_incarnation_sha256: str,
        attestation_id: str,
        attestation_content_sha256: str,
        release_id: str,
        release_content_sha256: str,
        release_graph_sha256: str,
        assignment_policy_id: str,
        assignment_policy_version_sha256: str,
        assignment_policy_config_sha256: str,
        task_policy_id: str,
        task_policy_version_sha256: str,
        task_policy_config_sha256: str,
        retrieval_policy_id: str,
        retrieval_policy_version_sha256: str,
        retrieval_policy_config_sha256: str,
        renderer_id: str,
        renderer_version_sha256: str,
        renderer_config_sha256: str,
        consumer_kind: MemoryReleaseAssignmentConsumerKind,
        consumer_id: str,
        consumer_version_sha256: str,
        consumer_config_sha256: str,
        max_returned_items: int,
        max_context_utf8_bytes: int,
        evaluated_at: datetime,
        assigned_at: datetime,
        assignment_valid_until: datetime,
        idempotency_key: str,
    ) -> MemoryReleaseAssignmentV1:
        values = {
            "scope": scope,
            "rollout_group_id": rollout_group_id,
            "rollout_group_incarnation_sha256": rollout_group_incarnation_sha256,
            "attestation_id": attestation_id,
            "attestation_content_sha256": attestation_content_sha256,
            "release_id": release_id,
            "release_content_sha256": release_content_sha256,
            "release_graph_sha256": release_graph_sha256,
            "assignment_policy_id": assignment_policy_id,
            "assignment_policy_version_sha256": assignment_policy_version_sha256,
            "assignment_policy_config_sha256": assignment_policy_config_sha256,
            "task_policy_id": task_policy_id,
            "task_policy_version_sha256": task_policy_version_sha256,
            "task_policy_config_sha256": task_policy_config_sha256,
            "retrieval_policy_id": retrieval_policy_id,
            "retrieval_policy_version_sha256": retrieval_policy_version_sha256,
            "retrieval_policy_config_sha256": retrieval_policy_config_sha256,
            "renderer_id": renderer_id,
            "renderer_version_sha256": renderer_version_sha256,
            "renderer_config_sha256": renderer_config_sha256,
            "consumer_kind": consumer_kind,
            "consumer_id": consumer_id,
            "consumer_version_sha256": consumer_version_sha256,
            "consumer_config_sha256": consumer_config_sha256,
            "max_returned_items": max_returned_items,
            "max_context_utf8_bytes": max_context_utf8_bytes,
            "evaluated_at": evaluated_at,
            "assigned_at": assigned_at,
            "assignment_valid_until": assignment_valid_until,
            "idempotency_key": idempotency_key,
        }
        if type(consumer_kind) is not MemoryReleaseAssignmentConsumerKind:
            raise TypeError(
                "consumer_kind must be a MemoryReleaseAssignmentConsumerKind"
            )
        canonical_value = {
            "assigned_at": _datetime_value(assigned_at, "assigned_at"),
            "assignment_valid_until": _datetime_value(
                assignment_valid_until,
                "assignment_valid_until",
            ),
            "assignment_policy_config_sha256": _digest(
                assignment_policy_config_sha256,
                "assignment_policy_config_sha256",
            ),
            "assignment_policy_id": _string(
                assignment_policy_id,
                "assignment_policy_id",
            ),
            "assignment_policy_version_sha256": _digest(
                assignment_policy_version_sha256,
                "assignment_policy_version_sha256",
            ),
            "attestation_content_sha256": _digest(
                attestation_content_sha256,
                "attestation_content_sha256",
            ),
            "attestation_id": _string(attestation_id, "attestation_id"),
            "consumer_config_sha256": _digest(
                consumer_config_sha256,
                "consumer_config_sha256",
            ),
            "consumer_id": _string(consumer_id, "consumer_id"),
            "consumer_kind": consumer_kind.value,
            "consumer_version_sha256": _digest(
                consumer_version_sha256,
                "consumer_version_sha256",
            ),
            "evaluated_at": _datetime_value(evaluated_at, "evaluated_at"),
            "idempotency_key": _string(idempotency_key, "idempotency_key"),
            "max_context_utf8_bytes": _integer(
                max_context_utf8_bytes,
                "max_context_utf8_bytes",
            ),
            "max_returned_items": _integer(
                max_returned_items,
                "max_returned_items",
            ),
            "release_content_sha256": _digest(
                release_content_sha256,
                "release_content_sha256",
            ),
            "release_graph_sha256": _digest(
                release_graph_sha256,
                "release_graph_sha256",
            ),
            "release_id": _string(release_id, "release_id"),
            "renderer_config_sha256": _digest(
                renderer_config_sha256,
                "renderer_config_sha256",
            ),
            "renderer_id": _string(renderer_id, "renderer_id"),
            "renderer_version_sha256": _digest(
                renderer_version_sha256,
                "renderer_version_sha256",
            ),
            "retrieval_policy_config_sha256": _digest(
                retrieval_policy_config_sha256,
                "retrieval_policy_config_sha256",
            ),
            "retrieval_policy_id": _string(
                retrieval_policy_id,
                "retrieval_policy_id",
            ),
            "retrieval_policy_version_sha256": _digest(
                retrieval_policy_version_sha256,
                "retrieval_policy_version_sha256",
            ),
            "rollout_group_id": _string(rollout_group_id, "rollout_group_id"),
            "rollout_group_incarnation_sha256": _digest(
                rollout_group_incarnation_sha256,
                "rollout_group_incarnation_sha256",
            ),
            "scope": _scope_value(scope),
            "task_policy_id": _string(task_policy_id, "task_policy_id"),
            "task_policy_config_sha256": _digest(
                task_policy_config_sha256,
                "task_policy_config_sha256",
            ),
            "task_policy_version_sha256": _digest(
                task_policy_version_sha256,
                "task_policy_version_sha256",
            ),
        }
        content_hash = sha256(
            _record_bytes("memory_release_assignment", canonical_value)
        ).hexdigest()
        return cls(
            assignment_id=f"masn_{content_hash[:24]}",
            content_hash=content_hash,
            **values,
        )
