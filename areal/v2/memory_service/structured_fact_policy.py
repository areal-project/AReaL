# SPDX-License-Identifier: Apache-2.0

"""Fail-closed structured-fact updates over immutable Memory releases.

The core deliberately does not infer authority from ``EvidenceKind``.  A
deployment supplies an :class:`EvidenceTrustPolicy`; scope and observation
cutoff gates run before that policy is called.  Event timestamps are gates,
not conflict-resolution clocks: updates execute in the caller's explicit tuple
order and SUPERSEDE uses compare-and-swap on an exact parent revision.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from areal.v2.memory_service.errors import MemoryServiceError
from areal.v2.memory_service.history_store import MemoryHistoryStore
from areal.v2.memory_service.history_types import (
    CandidateProposal,
    MemoryRevision,
    RevisionOperation,
    RevisionProposal,
)
from areal.v2.memory_service.release_store import MemoryReleaseStore
from areal.v2.memory_service.release_types import MemoryRelease, ReleaseManifest
from areal.v2.memory_service.store import EvidenceStore
from areal.v2.memory_service.types import (
    EvidenceRecord,
    MemoryScope,
    _validate_aware_datetime,
    _validate_string,
)

_SCHEMA_VERSION = 1
_UPDATE_FIELDS = {
    "expected_parent_revision_id",
    "fact_key",
    "fact_value",
    "operation",
    "record_kind",
    "schema_version",
}


def _canonical_json(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()


def _digest(value: object, field_name: str) -> str:
    value = _validate_string(value, field_name)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


class StructuredFactOperation(StrEnum):
    ADD = "add"
    CONFIRM = "confirm"
    SUPERSEDE = "supersede"


@dataclass(frozen=True, slots=True)
class StructuredFactUpdateV1:
    """One closed-schema, parent-bound fact update command."""

    fact_key: str
    fact_value: str
    operation: StructuredFactOperation
    expected_parent_revision_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fact_key",
            _validate_string(self.fact_key, "fact_key"),
        )
        object.__setattr__(
            self,
            "fact_value",
            _validate_string(self.fact_value, "fact_value"),
        )
        if type(self.operation) is not StructuredFactOperation:
            raise TypeError("operation must be a StructuredFactOperation")
        parent = self.expected_parent_revision_id
        if parent is not None:
            parent = _validate_string(parent, "expected_parent_revision_id")
            object.__setattr__(self, "expected_parent_revision_id", parent)
        if self.operation is StructuredFactOperation.ADD and parent is not None:
            raise ValueError("ADD must not carry expected_parent_revision_id")
        if self.operation is not StructuredFactOperation.ADD and parent is None:
            raise ValueError(
                "CONFIRM and SUPERSEDE require expected_parent_revision_id"
            )

    def canonical_bytes(self) -> bytes:
        return _canonical_json(
            {
                "expected_parent_revision_id": self.expected_parent_revision_id,
                "fact_key": self.fact_key,
                "fact_value": self.fact_value,
                "operation": self.operation.value,
                "record_kind": "structured_fact_update",
                "schema_version": _SCHEMA_VERSION,
            }
        )

    def to_payload(self) -> str:
        return self.canonical_bytes().decode()

    @classmethod
    def from_payload(cls, payload: str) -> StructuredFactUpdateV1:
        payload = _validate_string(payload, "payload", allow_blank=True)
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("payload must be valid structured-update JSON") from error
        if type(value) is not dict or set(value) != _UPDATE_FIELDS:
            raise ValueError("payload must use the exact structured-update schema")
        if value["record_kind"] != "structured_fact_update":
            raise ValueError("payload record_kind is not structured_fact_update")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("payload schema_version is unsupported")
        try:
            operation = StructuredFactOperation(value["operation"])
        except (TypeError, ValueError) as error:
            raise ValueError("payload operation is unsupported") from error
        update = cls(
            fact_key=value["fact_key"],
            fact_value=value["fact_value"],
            operation=operation,
            expected_parent_revision_id=value["expected_parent_revision_id"],
        )
        if update.to_payload() != payload:
            raise ValueError("payload must use canonical structured-update JSON")
        return update


def structured_fact_state_payload(fact_key: str, fact_value: str) -> str:
    """Serialize one active fact state independently from its update command."""

    fact_key = _validate_string(fact_key, "fact_key")
    fact_value = _validate_string(fact_value, "fact_value")
    return _canonical_json({"fact_key": fact_key, "fact_value": fact_value}).decode()


def parse_structured_fact_state(payload: str) -> tuple[str, str]:
    payload = _validate_string(payload, "payload")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("fact state must be valid JSON") from error
    if type(value) is not dict or set(value) != {"fact_key", "fact_value"}:
        raise ValueError("fact state must use the exact state schema")
    fact_key = _validate_string(value["fact_key"], "fact_key")
    fact_value = _validate_string(value["fact_value"], "fact_value")
    if structured_fact_state_payload(fact_key, fact_value) != payload:
        raise ValueError("fact state must use canonical JSON")
    return fact_key, fact_value


class EvidenceAuthority(StrEnum):
    AUTHORITATIVE = "authoritative"
    EVALUATOR_ONLY = "evaluator_only"
    INELIGIBLE = "ineligible"
    UNTRUSTED = "untrusted"


@dataclass(frozen=True, slots=True)
class EvidenceTrustDecision:
    authority: EvidenceAuthority
    reason: str

    def __post_init__(self) -> None:
        if type(self.authority) is not EvidenceAuthority:
            raise TypeError("authority must be an EvidenceAuthority")
        object.__setattr__(self, "reason", _validate_string(self.reason, "reason"))


class EvidenceTrustPolicy(Protocol):
    """Deployment-owned authority decision; EvidenceKind alone is not identity."""

    trust_policy_id: str
    trust_policy_version_sha256: str
    trust_policy_config_sha256: str

    def evaluate(
        self,
        *,
        evidence: EvidenceRecord,
        update: StructuredFactUpdateV1,
    ) -> EvidenceTrustDecision: ...


class StructuredFactDecision(StrEnum):
    APPLIED = "applied"
    CONFIRMED = "confirmed"
    REPLAYED = "replayed"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class StructuredFactEvidenceDecisionV1:
    evidence_id: str
    decision: StructuredFactDecision
    reason: str
    authority: EvidenceAuthority | None
    operation: StructuredFactOperation | None
    fact_key: str | None
    expected_parent_revision_id: str | None
    active_parent_revision_id: str | None
    revision_id: str | None


@dataclass(frozen=True, slots=True)
class StructuredFactPolicyResultV1:
    release: MemoryRelease
    decisions: tuple[StructuredFactEvidenceDecisionV1, ...]


@dataclass(frozen=True, slots=True)
class _ActiveFact:
    value: str
    revision: MemoryRevision


class StructuredFactPolicy:
    """Apply explicit parent-CAS commands and publish only accepted fact tips."""

    def __init__(
        self,
        history_store: MemoryHistoryStore,
        release_store: MemoryReleaseStore,
        *,
        evidence_store: EvidenceStore,
        trust_policy: EvidenceTrustPolicy,
    ) -> None:
        self._evidence_store = evidence_store
        self._history_store = history_store
        self._release_store = release_store
        self._trust_policy = trust_policy
        self._trust_policy_id = _validate_string(
            getattr(trust_policy, "trust_policy_id", None),
            "trust_policy_id",
        )
        self._trust_policy_version = _digest(
            getattr(trust_policy, "trust_policy_version_sha256", None),
            "trust_policy_version_sha256",
        )
        self._trust_policy_config = _digest(
            getattr(trust_policy, "trust_policy_config_sha256", None),
            "trust_policy_config_sha256",
        )
        evaluate = getattr(trust_policy, "evaluate", None)
        if not callable(evaluate):
            raise TypeError("trust_policy must define evaluate")
        self._evaluate = evaluate
        self._evaluate_owner = getattr(evaluate, "__self__", None)
        self._evaluate_function = getattr(evaluate, "__func__", evaluate)

    def _policy_identity_is_unchanged(self) -> bool:
        try:
            evaluate = getattr(self._trust_policy, "evaluate", None)
            return (
                _validate_string(
                    getattr(self._trust_policy, "trust_policy_id", None),
                    "trust_policy_id",
                )
                == self._trust_policy_id
                and _digest(
                    getattr(
                        self._trust_policy,
                        "trust_policy_version_sha256",
                        None,
                    ),
                    "trust_policy_version_sha256",
                )
                == self._trust_policy_version
                and _digest(
                    getattr(
                        self._trust_policy,
                        "trust_policy_config_sha256",
                        None,
                    ),
                    "trust_policy_config_sha256",
                )
                == self._trust_policy_config
                and callable(evaluate)
                and getattr(evaluate, "__self__", None) is self._evaluate_owner
                and getattr(evaluate, "__func__", evaluate) is self._evaluate_function
            )
        except (TypeError, ValueError):
            return False

    def _artifact_idempotency_key(
        self,
        *,
        artifact: str,
        operation_namespace: str,
        record: EvidenceRecord,
        update: StructuredFactUpdateV1,
    ) -> str:
        """Bind one append-only artifact to the full policy incarnation.

        ``operation_namespace`` is the caller's apply-level idempotency key.  It
        prevents two intentional policy runs over the same immutable evidence
        from aliasing each other's candidate or revision.  The digest avoids
        delimiter ambiguity in caller-controlled identifiers.
        """

        material = _canonical_json(
            {
                "artifact": artifact,
                "evidence_id": record.evidence_id,
                "operation_namespace": operation_namespace,
                "schema_version": _SCHEMA_VERSION,
                "scope": {
                    "namespace": record.event.scope.namespace,
                    "subject_id": record.event.scope.subject_id,
                    "tenant_id": record.event.scope.tenant_id,
                },
                "trust_policy": {
                    "config_sha256": self._trust_policy_config,
                    "id": self._trust_policy_id,
                    "version_sha256": self._trust_policy_version,
                },
                "update_sha256": hashlib.sha256(update.canonical_bytes()).hexdigest(),
            }
        )
        return f"structured-fact-v1:{artifact}:{hashlib.sha256(material).hexdigest()}"

    def _trust(
        self,
        evidence: EvidenceRecord,
        update: StructuredFactUpdateV1,
    ) -> EvidenceTrustDecision | None:
        if not self._policy_identity_is_unchanged():
            return None
        try:
            decision = self._evaluate(evidence=evidence, update=update)
        except Exception:
            return None
        if (
            type(decision) is not EvidenceTrustDecision
            or not self._policy_identity_is_unchanged()
        ):
            return None
        return decision

    @staticmethod
    def _evidence_commitment_is_valid(record: EvidenceRecord) -> bool:
        canonical_bytes = record.event.canonical_bytes()
        content_hash = hashlib.sha256(canonical_bytes).hexdigest()
        return (
            record.content_hash == content_hash
            and record.evidence_id == f"evd_{content_hash[:24]}"
        )

    def _resolve_persisted_evidence(
        self,
        scope: MemoryScope,
        record: EvidenceRecord,
    ) -> tuple[EvidenceRecord | None, str | None]:
        """Bind caller input to the exact record held by the evidence store."""

        if not self._evidence_commitment_is_valid(record):
            return None, "invalid_evidence_commitment"
        try:
            persisted = self._evidence_store.get(scope, record.evidence_id)
        except MemoryServiceError:
            return None, "evidence_not_persisted"
        except Exception:
            return None, "evidence_store_failure"
        if type(persisted) is not EvidenceRecord or not (
            self._evidence_commitment_is_valid(persisted)
        ):
            return None, "evidence_store_contract_violation"
        if (
            persisted.evidence_id != record.evidence_id
            or persisted.event.canonical_bytes() != record.event.canonical_bytes()
            or persisted.content_hash != record.content_hash
            or persisted.created_at != record.created_at
        ):
            return None, "evidence_record_mismatch"
        return persisted, None

    def _active_facts(
        self,
        scope: MemoryScope,
        base_release: MemoryRelease,
    ) -> dict[str, _ActiveFact]:
        active: dict[str, _ActiveFact] = {}
        for revision in self._release_store.get_release_revisions(
            scope,
            base_release.release_id,
        ):
            candidate = self._history_store.get_candidate(
                scope,
                revision.proposal.candidate_id,
            )
            key, value = parse_structured_fact_state(candidate.proposal.content)
            if key in active:
                raise ValueError("base release contains duplicate structured-fact keys")
            active[key] = _ActiveFact(value=value, revision=revision)
        return active

    def _is_replay(
        self,
        scope: MemoryScope,
        active: _ActiveFact,
        record: EvidenceRecord,
        update: StructuredFactUpdateV1,
        operation_namespace: str,
    ) -> bool:
        revision = active.revision
        candidate = self._history_store.get_candidate(
            scope,
            revision.proposal.candidate_id,
        )
        expected_revision_operation = {
            StructuredFactOperation.ADD: RevisionOperation.ADD,
            StructuredFactOperation.SUPERSEDE: RevisionOperation.SUPERSEDE,
        }.get(update.operation)
        return (
            expected_revision_operation is not None
            and revision.proposal.operation is expected_revision_operation
            and revision.proposal.parent_revision_id
            == update.expected_parent_revision_id
            and revision.proposal.idempotency_key
            == self._artifact_idempotency_key(
                artifact="revision",
                operation_namespace=operation_namespace,
                record=record,
                update=update,
            )
            and candidate.proposal.evidence_ids == (record.evidence_id,)
            and candidate.proposal.idempotency_key
            == self._artifact_idempotency_key(
                artifact="candidate",
                operation_namespace=operation_namespace,
                record=record,
                update=update,
            )
            and parse_structured_fact_state(candidate.proposal.content)
            == (update.fact_key, update.fact_value)
        )

    @staticmethod
    def _decision(
        evidence_id: str,
        decision: StructuredFactDecision,
        reason: str,
        *,
        authority: EvidenceAuthority | None = None,
        update: StructuredFactUpdateV1 | None = None,
        active_parent_revision_id: str | None = None,
        revision_id: str | None = None,
    ) -> StructuredFactEvidenceDecisionV1:
        return StructuredFactEvidenceDecisionV1(
            evidence_id=evidence_id,
            decision=decision,
            reason=reason,
            authority=authority,
            operation=None if update is None else update.operation,
            fact_key=None if update is None else update.fact_key,
            expected_parent_revision_id=(
                None if update is None else update.expected_parent_revision_id
            ),
            active_parent_revision_id=active_parent_revision_id,
            revision_id=revision_id,
        )

    @staticmethod
    def _abort_trust_dependent_decisions(
        decisions: list[StructuredFactEvidenceDecisionV1],
    ) -> tuple[StructuredFactEvidenceDecisionV1, ...]:
        """Withdraw claims made under a policy incarnation that drifted.

        Core scope/cutoff gates and malformed-input quarantine do not depend on
        the deployment trust policy, so their decisions remain valid.  A
        revision ID is retained when an append-only orphan was already written;
        it is audit evidence, not a claim that the revision was published.
        """

        return tuple(
            replace(
                item,
                decision=StructuredFactDecision.QUARANTINED,
                reason="trust_policy_incarnation_changed_batch_aborted",
                authority=None,
            )
            if item.operation is not None
            else item
            for item in decisions
        )

    def apply(
        self,
        *,
        scope: MemoryScope,
        base_release: MemoryRelease,
        evidence: tuple[EvidenceRecord, ...],
        captured_through: datetime,
        idempotency_key: str,
    ) -> StructuredFactPolicyResultV1:
        """Apply evidence in explicit tuple order; never resolve by timestamp."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        if type(base_release) is not MemoryRelease:
            raise TypeError("base_release must be a MemoryRelease")
        if base_release.manifest.scope != scope:
            raise ValueError("base_release belongs to a different scope")
        if type(evidence) is not tuple or any(
            type(item) is not EvidenceRecord for item in tuple.__iter__(evidence)
        ):
            raise TypeError("evidence must be a tuple of EvidenceRecord values")
        if len({item.evidence_id for item in evidence}) != len(evidence):
            raise ValueError("evidence must not contain duplicate IDs")
        captured_through = _validate_aware_datetime(
            captured_through,
            "captured_through",
        )
        idempotency_key = _validate_string(idempotency_key, "idempotency_key")
        active = self._active_facts(scope, base_release)
        changed = False
        decisions: list[StructuredFactEvidenceDecisionV1] = []

        for record in evidence:
            event_id = record.evidence_id
            event = record.event
            if event.scope != scope:
                decisions.append(
                    self._decision(
                        record.evidence_id,
                        StructuredFactDecision.REJECTED,
                        "foreign_scope",
                    )
                )
                continue
            if event.observed_at > captured_through:
                decisions.append(
                    self._decision(
                        record.evidence_id,
                        StructuredFactDecision.REJECTED,
                        "after_capture_cutoff",
                    )
                )
                continue
            record, evidence_failure = self._resolve_persisted_evidence(scope, record)
            if record is None:
                decisions.append(
                    self._decision(
                        event_id,
                        StructuredFactDecision.QUARANTINED,
                        evidence_failure or "evidence_resolution_failure",
                    )
                )
                continue
            event = record.event
            try:
                update = StructuredFactUpdateV1.from_payload(event.payload)
            except (TypeError, ValueError):
                decisions.append(
                    self._decision(
                        record.evidence_id,
                        StructuredFactDecision.QUARANTINED,
                        "malformed_update",
                    )
                )
                continue

            trust = self._trust(record, update)
            if trust is None:
                decisions.append(
                    self._decision(
                        record.evidence_id,
                        StructuredFactDecision.QUARANTINED,
                        "trust_policy_failure",
                        update=update,
                    )
                )
                continue
            if trust.authority is not EvidenceAuthority.AUTHORITATIVE:
                decisions.append(
                    self._decision(
                        record.evidence_id,
                        StructuredFactDecision.REJECTED,
                        trust.reason,
                        authority=trust.authority,
                        update=update,
                    )
                )
                continue

            current = active.get(update.fact_key)
            current_revision_id = (
                None if current is None else current.revision.revision_id
            )
            if current is not None and self._is_replay(
                scope,
                current,
                record,
                update,
                idempotency_key,
            ):
                decisions.append(
                    self._decision(
                        record.evidence_id,
                        StructuredFactDecision.REPLAYED,
                        "idempotent_replay",
                        authority=trust.authority,
                        update=update,
                        active_parent_revision_id=current_revision_id,
                        revision_id=current_revision_id,
                    )
                )
                continue

            conflict_reason: str | None = None
            if update.operation is StructuredFactOperation.ADD:
                if current is not None:
                    conflict_reason = "add_requires_absent_fact"
            elif current is None:
                conflict_reason = "operation_requires_active_parent"
            elif update.expected_parent_revision_id != current_revision_id:
                conflict_reason = "expected_parent_mismatch"
            elif update.operation is StructuredFactOperation.CONFIRM:
                if update.fact_value != current.value:
                    conflict_reason = "confirm_value_mismatch"
                else:
                    decisions.append(
                        self._decision(
                            record.evidence_id,
                            StructuredFactDecision.CONFIRMED,
                            "confirmed_current_fact",
                            authority=trust.authority,
                            update=update,
                            active_parent_revision_id=current_revision_id,
                            revision_id=current_revision_id,
                        )
                    )
                    continue
            elif update.fact_value == current.value:
                conflict_reason = "supersede_same_value_requires_confirm"

            if conflict_reason is not None:
                decisions.append(
                    self._decision(
                        record.evidence_id,
                        StructuredFactDecision.QUARANTINED,
                        conflict_reason,
                        authority=trust.authority,
                        update=update,
                        active_parent_revision_id=current_revision_id,
                    )
                )
                continue

            candidate = CandidateProposal(
                scope=scope,
                content=structured_fact_state_payload(
                    update.fact_key,
                    update.fact_value,
                ),
                evidence_ids=(record.evidence_id,),
                idempotency_key=self._artifact_idempotency_key(
                    artifact="candidate",
                    operation_namespace=idempotency_key,
                    record=record,
                    update=update,
                ),
            )
            revision_operation = {
                StructuredFactOperation.ADD: RevisionOperation.ADD,
                StructuredFactOperation.SUPERSEDE: RevisionOperation.SUPERSEDE,
            }[update.operation]
            try:
                stored_candidate = self._history_store.append_candidate(candidate)
                revision = self._history_store.append_revision(
                    RevisionProposal(
                        scope=scope,
                        candidate_id=stored_candidate.candidate_id,
                        operation=revision_operation,
                        parent_revision_id=update.expected_parent_revision_id,
                        idempotency_key=self._artifact_idempotency_key(
                            artifact="revision",
                            operation_namespace=idempotency_key,
                            record=record,
                            update=update,
                        ),
                    )
                )
            except MemoryServiceError:
                decisions.append(
                    self._decision(
                        record.evidence_id,
                        StructuredFactDecision.QUARANTINED,
                        "history_persistence_conflict",
                        authority=trust.authority,
                        update=update,
                        active_parent_revision_id=current_revision_id,
                    )
                )
                continue
            active[update.fact_key] = _ActiveFact(update.fact_value, revision)
            changed = True
            decisions.append(
                self._decision(
                    record.evidence_id,
                    StructuredFactDecision.APPLIED,
                    "authoritative_update_applied",
                    authority=trust.authority,
                    update=update,
                    active_parent_revision_id=current_revision_id,
                    revision_id=revision.revision_id,
                )
            )

        if not self._policy_identity_is_unchanged():
            return StructuredFactPolicyResultV1(
                base_release,
                self._abort_trust_dependent_decisions(decisions),
            )
        if not changed:
            return StructuredFactPolicyResultV1(base_release, tuple(decisions))
        release = self._release_store.append_release(
            ReleaseManifest(
                scope=scope,
                revision_ids=tuple(
                    active[key].revision.revision_id for key in sorted(active)
                ),
            ),
            idempotency_key=idempotency_key,
        )
        if not self._policy_identity_is_unchanged():
            return StructuredFactPolicyResultV1(
                base_release,
                self._abort_trust_dependent_decisions(decisions),
            )
        return StructuredFactPolicyResultV1(release, tuple(decisions))


__all__ = [
    "EvidenceAuthority",
    "EvidenceTrustDecision",
    "EvidenceTrustPolicy",
    "StructuredFactDecision",
    "StructuredFactEvidenceDecisionV1",
    "StructuredFactOperation",
    "StructuredFactPolicy",
    "StructuredFactPolicyResultV1",
    "StructuredFactUpdateV1",
    "parse_structured_fact_state",
    "structured_fact_state_payload",
]
