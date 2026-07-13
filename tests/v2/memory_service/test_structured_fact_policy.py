# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from areal.v2.memory_service import (
    EvidenceAuthority,
    EvidenceEvent,
    EvidenceKind,
    EvidenceRecord,
    EvidenceTrustDecision,
    InMemoryEvidenceStore,
    InMemoryMemoryHistoryStore,
    InMemoryMemoryReleaseStore,
    MemoryScope,
    ReleaseManifest,
    RevisionOperation,
    StructuredFactDecision,
    StructuredFactOperation,
    StructuredFactPolicy,
    StructuredFactUpdateV1,
    parse_structured_fact_state,
)

_BASE = datetime(2026, 7, 13, tzinfo=UTC)
_VERSION = hashlib.sha256(b"test-trust-v1").hexdigest()
_CONFIG = hashlib.sha256(b"test-trust-config-v1").hexdigest()


class _TrustPolicy:
    trust_policy_id = "test-explicit-authority"

    def __init__(
        self,
        *,
        version: str = _VERSION,
        config: str = _CONFIG,
        mutate_on_call: int | None = None,
    ) -> None:
        self.trust_policy_version_sha256 = version
        self.trust_policy_config_sha256 = config
        self.calls = []
        self.untrusted_ids = set()
        self.fail = False
        self.mutate_on_call = mutate_on_call

    def evaluate(self, *, evidence, update):
        self.calls.append((evidence.evidence_id, update))
        if len(self.calls) == self.mutate_on_call:
            self.trust_policy_config_sha256 = hashlib.sha256(
                b"mutated-during-evaluation"
            ).hexdigest()
        if self.fail:
            raise RuntimeError("injected trust-policy failure")
        if evidence.evidence_id in self.untrusted_ids:
            return EvidenceTrustDecision(
                EvidenceAuthority.UNTRUSTED,
                "test_policy_untrusted",
            )
        return EvidenceTrustDecision(
            EvidenceAuthority.AUTHORITATIVE,
            "test_policy_authoritative",
        )


def _seed():
    evidence = InMemoryEvidenceStore()
    history = InMemoryMemoryHistoryStore(evidence)
    releases = InMemoryMemoryReleaseStore(history)
    scope = MemoryScope("tenant", "structured-facts", "subject")
    empty = releases.append_release(
        ReleaseManifest(scope=scope, revision_ids=()),
        idempotency_key="empty-release",
    )
    trust = _TrustPolicy()
    policy = StructuredFactPolicy(
        history,
        releases,
        evidence_store=evidence,
        trust_policy=trust,
    )
    return scope, evidence, history, releases, empty, trust, policy


def _append(
    store,
    *,
    scope,
    key,
    update=None,
    payload=None,
    observed_at=_BASE,
    kind=EvidenceKind.FEEDBACK,
):
    if payload is None:
        payload = update.to_payload()
    return store.append(
        EvidenceEvent(
            scope=scope,
            session_id="session",
            run_id="run",
            sequence_no=len(store.list(scope)),
            kind=kind,
            payload=payload,
            observed_at=observed_at,
            idempotency_key=key,
        )
    )


def _update(operation, value, parent=None):
    return StructuredFactUpdateV1(
        fact_key="project_access_code",
        fact_value=value,
        operation=operation,
        expected_parent_revision_id=parent,
    )


def test_update_v1_is_closed_schema_and_parent_bound() -> None:
    add = _update(StructuredFactOperation.ADD, "A")
    confirm = _update(StructuredFactOperation.CONFIRM, "A", "rev_parent")
    supersede = _update(StructuredFactOperation.SUPERSEDE, "B", "rev_parent")

    assert StructuredFactUpdateV1.from_payload(add.to_payload()) == add
    assert StructuredFactUpdateV1.from_payload(confirm.to_payload()) == confirm
    assert StructuredFactUpdateV1.from_payload(supersede.to_payload()) == supersede
    with pytest.raises(ValueError, match="ADD must not carry"):
        _update(StructuredFactOperation.ADD, "A", "rev_parent")
    with pytest.raises(ValueError, match="require expected_parent"):
        _update(StructuredFactOperation.CONFIRM, "A")
    with pytest.raises(ValueError, match="require expected_parent"):
        _update(StructuredFactOperation.SUPERSEDE, "B")

    extra = json.loads(add.to_payload())
    extra["source"] = "not-in-v1"
    with pytest.raises(ValueError, match="exact structured-update schema"):
        StructuredFactUpdateV1.from_payload(
            json.dumps(extra, separators=(",", ":"), sort_keys=True)
        )


def test_scope_and_cutoff_gate_before_external_trust_policy() -> None:
    scope, evidence, history, _releases, empty, trust, policy = _seed()
    foreign_scope = MemoryScope("tenant", "structured-facts", "foreign")
    foreign = _append(
        evidence,
        scope=foreign_scope,
        key="foreign",
        update=_update(StructuredFactOperation.ADD, "FOREIGN"),
    )
    future = _append(
        evidence,
        scope=scope,
        key="future",
        update=_update(StructuredFactOperation.ADD, "FUTURE"),
        observed_at=_BASE + timedelta(seconds=2),
    )
    outcome = _append(
        evidence,
        scope=scope,
        key="outcome",
        update=_update(StructuredFactOperation.ADD, "OUTCOME"),
        kind=EvidenceKind.OUTCOME,
    )
    agent = _append(
        evidence,
        scope=scope,
        key="agent",
        update=_update(StructuredFactOperation.ADD, "AGENT"),
        kind=EvidenceKind.AGENT_MESSAGE,
    )
    malformed = _append(
        evidence,
        scope=scope,
        key="malformed",
        payload='{"fact_key":"missing-closed-fields"}',
    )
    trust.untrusted_ids.update({outcome.evidence_id, agent.evidence_id})

    result = policy.apply(
        scope=scope,
        base_release=empty,
        evidence=(foreign, future, outcome, agent, malformed),
        captured_through=_BASE + timedelta(seconds=1),
        idempotency_key="gated-release",
    )

    assert result.release == empty
    assert [item.reason for item in result.decisions] == [
        "foreign_scope",
        "after_capture_cutoff",
        "test_policy_untrusted",
        "test_policy_untrusted",
        "malformed_update",
    ]
    assert [item[0] for item in trust.calls] == [
        outcome.evidence_id,
        agent.evidence_id,
    ]
    assert history.list_candidates(scope) == ()
    assert history.list_revisions(scope) == ()


def test_caller_record_must_match_its_commitment_and_persisted_evidence() -> None:
    scope, evidence, history, _releases, empty, trust, policy = _seed()
    stored = _append(
        evidence,
        scope=scope,
        key="stored-safe-agent-evidence",
        update=_update(StructuredFactOperation.ADD, "SAFE"),
        kind=EvidenceKind.AGENT_MESSAGE,
    )
    forged_event = EvidenceEvent(
        scope=scope,
        session_id=stored.event.session_id,
        run_id="verified-local-fake-tool",
        sequence_no=stored.event.sequence_no,
        kind=EvidenceKind.TOOL_RESULT,
        payload=_update(StructuredFactOperation.ADD, "EVIL").to_payload(),
        observed_at=stored.event.observed_at,
        idempotency_key="forged-tool-result",
    )
    forged_payload = EvidenceRecord(
        evidence_id=stored.evidence_id,
        event=forged_event,
        content_hash=stored.content_hash,
        created_at=stored.created_at,
    )
    forged_storage_metadata = EvidenceRecord(
        evidence_id=stored.evidence_id,
        event=stored.event,
        content_hash=stored.content_hash,
        created_at=stored.created_at + timedelta(microseconds=1),
    )

    invalid_commitment = policy.apply(
        scope=scope,
        base_release=empty,
        evidence=(forged_payload,),
        captured_through=_BASE,
        idempotency_key="forged-payload",
    )
    persisted_mismatch = policy.apply(
        scope=scope,
        base_release=empty,
        evidence=(forged_storage_metadata,),
        captured_through=_BASE,
        idempotency_key="forged-storage-metadata",
    )

    assert invalid_commitment.release == empty
    assert invalid_commitment.decisions[0].reason == "invalid_evidence_commitment"
    assert persisted_mismatch.release == empty
    assert persisted_mismatch.decisions[0].reason == "evidence_record_mismatch"
    assert trust.calls == []
    assert history.list_candidates(scope) == ()
    assert history.list_revisions(scope) == ()


def test_add_supersede_and_same_value_confirm_are_explicit() -> None:
    scope, evidence, history, releases, empty, _trust, policy = _seed()
    add_evidence = _append(
        evidence,
        scope=scope,
        key="add",
        update=_update(StructuredFactOperation.ADD, "OLD"),
    )
    added = policy.apply(
        scope=scope,
        base_release=empty,
        evidence=(add_evidence,),
        captured_through=_BASE,
        idempotency_key="added-release",
    )
    root_id = added.release.manifest.revision_ids[0]
    supersede_evidence = _append(
        evidence,
        scope=scope,
        key="supersede",
        update=_update(StructuredFactOperation.SUPERSEDE, "NEW", root_id),
    )
    superseded = policy.apply(
        scope=scope,
        base_release=added.release,
        evidence=(supersede_evidence,),
        captured_through=_BASE,
        idempotency_key="superseded-release",
    )
    tip_id = superseded.release.manifest.revision_ids[0]
    tip = history.get_revision(scope, tip_id)
    assert tip.proposal.operation is RevisionOperation.SUPERSEDE
    assert tip.proposal.parent_revision_id == root_id
    assert superseded.decisions[0].active_parent_revision_id == root_id
    assert superseded.decisions[0].revision_id == tip_id

    counts = (len(history.list_candidates(scope)), len(history.list_revisions(scope)))
    confirm_evidence = _append(
        evidence,
        scope=scope,
        key="confirm",
        update=_update(StructuredFactOperation.CONFIRM, "NEW", tip_id),
    )
    confirmed = policy.apply(
        scope=scope,
        base_release=superseded.release,
        evidence=(confirm_evidence,),
        captured_through=_BASE,
        idempotency_key="confirm-does-not-publish",
    )
    assert confirmed.release == superseded.release
    assert confirmed.decisions[0].decision is StructuredFactDecision.CONFIRMED
    assert confirmed.decisions[0].revision_id == tip_id
    assert counts == (
        len(history.list_candidates(scope)),
        len(history.list_revisions(scope)),
    )

    assert releases.get_release(scope, added.release.release_id) == added.release
    old_tip = releases.get_release_revisions(scope, added.release.release_id)[0]
    old_candidate = history.get_candidate(scope, old_tip.proposal.candidate_id)
    assert parse_structured_fact_state(old_candidate.proposal.content)[1] == "OLD"


def test_wrong_or_missing_parent_is_quarantined_without_history_writes() -> None:
    scope, evidence, history, _releases, empty, _trust, policy = _seed()
    add = _append(
        evidence,
        scope=scope,
        key="root",
        update=_update(StructuredFactOperation.ADD, "ROOT"),
    )
    seeded = policy.apply(
        scope=scope,
        base_release=empty,
        evidence=(add,),
        captured_through=_BASE,
        idempotency_key="root-release",
    )
    root_id = seeded.release.manifest.revision_ids[0]
    counts = (len(history.list_candidates(scope)), len(history.list_revisions(scope)))
    wrong = _append(
        evidence,
        scope=scope,
        key="wrong-parent",
        update=_update(StructuredFactOperation.SUPERSEDE, "WRONG", "rev_wrong"),
    )
    missing_value = json.loads(
        _update(StructuredFactOperation.SUPERSEDE, "MISSING", root_id).to_payload()
    )
    missing_value["expected_parent_revision_id"] = None
    missing = _append(
        evidence,
        scope=scope,
        key="missing-parent",
        payload=json.dumps(missing_value, separators=(",", ":"), sort_keys=True),
    )
    result = policy.apply(
        scope=scope,
        base_release=seeded.release,
        evidence=(wrong, missing),
        captured_through=_BASE,
        idempotency_key="quarantined-release",
    )

    assert result.release == seeded.release
    assert [item.decision for item in result.decisions] == [
        StructuredFactDecision.QUARANTINED,
        StructuredFactDecision.QUARANTINED,
    ]
    assert [item.reason for item in result.decisions] == [
        "expected_parent_mismatch",
        "malformed_update",
    ]
    assert result.decisions[0].expected_parent_revision_id == "rev_wrong"
    assert result.decisions[0].active_parent_revision_id == root_id
    assert counts == (
        len(history.list_candidates(scope)),
        len(history.list_revisions(scope)),
    )


def test_parent_cas_uses_tuple_order_not_timestamp_lww() -> None:
    scope, evidence, history, _releases, empty, _trust, policy = _seed()
    root = _append(
        evidence,
        scope=scope,
        key="root",
        update=_update(StructuredFactOperation.ADD, "ROOT"),
    )
    seeded = policy.apply(
        scope=scope,
        base_release=empty,
        evidence=(root,),
        captured_through=_BASE + timedelta(minutes=1),
        idempotency_key="root-release",
    )
    root_id = seeded.release.manifest.revision_ids[0]
    tuple_first_but_later_timestamp = _append(
        evidence,
        scope=scope,
        key="tuple-first",
        update=_update(StructuredFactOperation.SUPERSEDE, "FIRST", root_id),
        observed_at=_BASE + timedelta(seconds=20),
    )
    tuple_second_but_earlier_timestamp = _append(
        evidence,
        scope=scope,
        key="tuple-second",
        update=_update(StructuredFactOperation.SUPERSEDE, "SECOND", root_id),
        observed_at=_BASE + timedelta(seconds=10),
    )
    result = policy.apply(
        scope=scope,
        base_release=seeded.release,
        evidence=(tuple_first_but_later_timestamp, tuple_second_but_earlier_timestamp),
        captured_through=_BASE + timedelta(minutes=1),
        idempotency_key="cas-release",
    )

    assert [item.decision for item in result.decisions] == [
        StructuredFactDecision.APPLIED,
        StructuredFactDecision.QUARANTINED,
    ]
    tip = history.get_revision(scope, result.release.manifest.revision_ids[0])
    candidate = history.get_candidate(scope, tip.proposal.candidate_id)
    assert parse_structured_fact_state(candidate.proposal.content)[1] == "FIRST"


def test_replay_is_idempotent_and_trust_failure_is_fail_closed() -> None:
    scope, evidence, history, releases, empty, trust, policy = _seed()
    record = _append(
        evidence,
        scope=scope,
        key="replay",
        update=_update(StructuredFactOperation.ADD, "VALUE"),
    )
    first = policy.apply(
        scope=scope,
        base_release=empty,
        evidence=(record,),
        captured_through=_BASE,
        idempotency_key="replay-release",
    )
    counts = (len(history.list_candidates(scope)), len(history.list_revisions(scope)))
    exact_retry = policy.apply(
        scope=scope,
        base_release=empty,
        evidence=(record,),
        captured_through=_BASE,
        idempotency_key="replay-release",
    )
    replay_on_active = policy.apply(
        scope=scope,
        base_release=first.release,
        evidence=(record,),
        captured_through=_BASE,
        idempotency_key="replay-release",
    )
    different_namespace = policy.apply(
        scope=scope,
        base_release=first.release,
        evidence=(record,),
        captured_through=_BASE,
        idempotency_key="different-operation-namespace",
    )
    different_incarnation = StructuredFactPolicy(
        history,
        releases,
        evidence_store=evidence,
        trust_policy=_TrustPolicy(
            config=hashlib.sha256(b"different-replay-config").hexdigest()
        ),
    ).apply(
        scope=scope,
        base_release=first.release,
        evidence=(record,),
        captured_through=_BASE,
        idempotency_key="replay-release",
    )
    assert exact_retry.release == first.release
    assert replay_on_active.release == first.release
    assert replay_on_active.decisions[0].decision is StructuredFactDecision.REPLAYED
    assert different_namespace.release == first.release
    assert different_namespace.decisions[0].reason == "add_requires_absent_fact"
    assert different_incarnation.release == first.release
    assert different_incarnation.decisions[0].reason == "add_requires_absent_fact"
    assert counts == (
        len(history.list_candidates(scope)),
        len(history.list_revisions(scope)),
    )

    trust.fail = True
    confirm = _append(
        evidence,
        scope=scope,
        key="trust-fails",
        update=_update(
            StructuredFactOperation.CONFIRM,
            "VALUE",
            first.release.manifest.revision_ids[0],
        ),
    )
    failed = policy.apply(
        scope=scope,
        base_release=first.release,
        evidence=(confirm,),
        captured_through=_BASE,
        idempotency_key="trust-failure",
    )
    assert failed.release == first.release
    assert failed.decisions[0].reason == "trust_policy_failure"
    assert counts == (
        len(history.list_candidates(scope)),
        len(history.list_revisions(scope)),
    )


def test_policy_incarnation_drift_aborts_the_whole_batch() -> None:
    scope, evidence, history, releases, empty, trust, _policy = _seed()
    trust.mutate_on_call = 2
    policy = StructuredFactPolicy(
        history,
        releases,
        evidence_store=evidence,
        trust_policy=trust,
    )
    first = _append(
        evidence,
        scope=scope,
        key="batch-first",
        update=_update(StructuredFactOperation.ADD, "FIRST"),
    )
    second = _append(
        evidence,
        scope=scope,
        key="batch-mutates-policy",
        update=_update(StructuredFactOperation.ADD, "SECOND"),
    )

    result = policy.apply(
        scope=scope,
        base_release=empty,
        evidence=(first, second),
        captured_through=_BASE,
        idempotency_key="mutated-policy-batch",
    )

    assert result.release == empty
    assert [item.decision for item in result.decisions] == [
        StructuredFactDecision.QUARANTINED,
        StructuredFactDecision.QUARANTINED,
    ]
    assert {item.reason for item in result.decisions} == {
        "trust_policy_incarnation_changed_batch_aborted"
    }
    assert all(item.authority is None for item in result.decisions)
    assert result.decisions[0].revision_id is not None
    assert result.decisions[1].revision_id is None
    assert len(history.list_candidates(scope)) == 1
    assert len(history.list_revisions(scope)) == 1
    assert result.decisions[0].revision_id not in result.release.manifest.revision_ids


def test_policy_drift_during_release_append_returns_the_base_release() -> None:
    scope, evidence, history, releases, empty, trust, _policy = _seed()

    class _MutatingReleaseStore:
        def append_release(self, manifest, *, idempotency_key):
            release = releases.append_release(
                manifest,
                idempotency_key=idempotency_key,
            )
            if manifest.revision_ids:
                trust.trust_policy_config_sha256 = hashlib.sha256(
                    b"mutated-during-release-append"
                ).hexdigest()
            return release

        def get_release(self, requested_scope, release_id):
            return releases.get_release(requested_scope, release_id)

        def get_release_revisions(self, requested_scope, release_id):
            return releases.get_release_revisions(requested_scope, release_id)

        def list_releases(self, requested_scope):
            return releases.list_releases(requested_scope)

    policy = StructuredFactPolicy(
        history,
        _MutatingReleaseStore(),
        evidence_store=evidence,
        trust_policy=trust,
    )
    record = _append(
        evidence,
        scope=scope,
        key="drift-during-release-append",
        update=_update(StructuredFactOperation.ADD, "VALUE"),
    )

    result = policy.apply(
        scope=scope,
        base_release=empty,
        evidence=(record,),
        captured_through=_BASE,
        idempotency_key="release-append-drift",
    )

    assert result.release == empty
    assert result.decisions[0].decision is StructuredFactDecision.QUARANTINED
    assert result.decisions[0].reason == (
        "trust_policy_incarnation_changed_batch_aborted"
    )
    assert result.decisions[0].revision_id is not None
    assert len(history.list_candidates(scope)) == 1
    assert len(history.list_revisions(scope)) == 1
    orphan_releases = tuple(
        release for release in releases.list_releases(scope) if release != empty
    )
    assert len(orphan_releases) == 1
    assert orphan_releases[0].manifest.revision_ids == (
        result.decisions[0].revision_id,
    )


def test_policy_callable_identity_uses_is_without_hostile_equality() -> None:
    class _Owner:
        def __eq__(self, other):
            del other
            raise AssertionError("policy owner equality must never run")

        def evaluate(self, *, evidence, update):
            del evidence, update
            return EvidenceTrustDecision(
                EvidenceAuthority.AUTHORITATIVE,
                "forged_equal_owner",
            )

    class _SwappingPolicy:
        trust_policy_id = "swapping-policy"
        trust_policy_version_sha256 = _VERSION
        trust_policy_config_sha256 = _CONFIG

        def __init__(self):
            self._owners = (_Owner(), _Owner())
            self._reads = 0

        @property
        def evaluate(self):
            owner = self._owners[min(self._reads, 1)]
            self._reads += 1
            return owner.evaluate

    scope, evidence, history, releases, empty, _trust, _policy = _seed()
    policy = StructuredFactPolicy(
        history,
        releases,
        evidence_store=evidence,
        trust_policy=_SwappingPolicy(),
    )
    record = _append(
        evidence,
        scope=scope,
        key="swapped-owner",
        update=_update(StructuredFactOperation.ADD, "VALUE"),
    )

    result = policy.apply(
        scope=scope,
        base_release=empty,
        evidence=(record,),
        captured_through=_BASE,
        idempotency_key="swapped-owner-batch",
    )

    assert result.release == empty
    assert result.decisions[0].reason == (
        "trust_policy_incarnation_changed_batch_aborted"
    )
    assert history.list_candidates(scope) == ()
    assert history.list_revisions(scope) == ()


def test_artifact_idempotency_binds_operation_namespace() -> None:
    scope, evidence, history, _releases, empty, _trust, policy = _seed()
    record = _append(
        evidence,
        scope=scope,
        key="namespace-record",
        update=_update(StructuredFactOperation.ADD, "VALUE"),
    )

    first = policy.apply(
        scope=scope,
        base_release=empty,
        evidence=(record,),
        captured_through=_BASE,
        idempotency_key="operation-namespace-one",
    )
    retry = policy.apply(
        scope=scope,
        base_release=empty,
        evidence=(record,),
        captured_through=_BASE,
        idempotency_key="operation-namespace-one",
    )
    counts_after_retry = (
        len(history.list_candidates(scope)),
        len(history.list_revisions(scope)),
    )
    second = policy.apply(
        scope=scope,
        base_release=empty,
        evidence=(record,),
        captured_through=_BASE,
        idempotency_key="operation-namespace-two",
    )

    assert retry.release == first.release
    assert counts_after_retry == (1, 1)
    assert second.release != first.release
    assert (
        len({item.proposal.idempotency_key for item in history.list_candidates(scope)})
        == 2
    )
    assert (
        len({item.proposal.idempotency_key for item in history.list_revisions(scope)})
        == 2
    )


def test_artifact_idempotency_binds_policy_version_and_config() -> None:
    evidence = InMemoryEvidenceStore()
    history = InMemoryMemoryHistoryStore(evidence)
    scope = MemoryScope("tenant", "structured-facts", "incarnations")
    record = _append(
        evidence,
        scope=scope,
        key="incarnation-record",
        update=_update(StructuredFactOperation.ADD, "VALUE"),
    )
    incarnations = (
        (_VERSION, _CONFIG),
        (hashlib.sha256(b"test-trust-v2").hexdigest(), _CONFIG),
        (_VERSION, hashlib.sha256(b"test-trust-config-v2").hexdigest()),
    )

    for index, (version, config) in enumerate(incarnations):
        releases = InMemoryMemoryReleaseStore(history)
        empty = releases.append_release(
            ReleaseManifest(scope=scope, revision_ids=()),
            idempotency_key=f"incarnation-empty-{index}",
        )
        policy = StructuredFactPolicy(
            history,
            releases,
            evidence_store=evidence,
            trust_policy=_TrustPolicy(version=version, config=config),
        )
        result = policy.apply(
            scope=scope,
            base_release=empty,
            evidence=(record,),
            captured_through=_BASE,
            idempotency_key="shared-operation-namespace",
        )
        assert result.decisions[0].decision is StructuredFactDecision.APPLIED

    assert len(
        {item.proposal.idempotency_key for item in history.list_candidates(scope)}
    ) == len(incarnations)
    assert len(
        {item.proposal.idempotency_key for item in history.list_revisions(scope)}
    ) == len(incarnations)
