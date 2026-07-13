# SPDX-License-Identifier: Apache-2.0

"""Small causal smoke test for an adaptive structured-memory codebook.

The task is deliberately simple: an agent must deploy a project with its current
access code, while the future query contains the project name but never the code.
The three arms share one immutable evidence stream:

* ``no_memory`` uses an empty release;
* ``static`` freezes the release before later feedback arrives;
* ``adaptive`` applies trusted updates and publishes a new immutable release.

This is not a benchmark or a claim of statistical significance.  It is an
executable check that one concrete update rule can improve paired future tasks,
and that the improvement travels through assignment, retrieval, rendering,
consumer acknowledgement, and exposure records rather than an evaluator-side
shortcut.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

from areal.v2.memory_service import (
    CandidateProposal,
    EvidenceEvent,
    EvidenceKind,
    EvidenceRecord,
    InMemoryEvidenceStore,
    InMemoryMemoryHistoryStore,
    InMemoryMemoryReleaseControlStore,
    InMemoryMemoryReleaseStore,
    InMemoryMemoryRuntimeStore,
    MemoryConsumerCallV1,
    MemoryConsumerKind,
    MemoryQueryResultV1,
    MemoryQuerySpecV1,
    MemoryRelease,
    MemoryReleaseAssignmentConsumerKind,
    MemoryReleaseAssignmentV1,
    MemoryReleaseRevocationReason,
    MemoryRenderedRevisionRangeV1,
    MemoryRenderOutputV1,
    MemoryRetrievalOutputV1,
    MemoryScope,
    ReleaseManifest,
    RevisionOperation,
    RevisionProposal,
)

_BASE = datetime(2026, 7, 13, tzinfo=UTC)
_FACT_KEY = "project_access_code"
_BASELINE_CUTOFF = _BASE + timedelta(seconds=10)
_ADAPTIVE_CUTOFF = _BASE + timedelta(seconds=30)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


_TASK_VERSION = _digest("adaptive-codebook-task-v1")
_TASK_CONFIG = _digest("mutable-project-access-code-v1")
_RETRIEVER_VERSION = _digest("release-order-retriever-v1")
_RETRIEVER_CONFIG = _digest("return-all-within-budget-v1")
_RENDERER_VERSION = _digest("structured-fact-json-lines-v1")
_RENDERER_CONFIG = _digest("utf8-json-lines-v1")
_CONSUMER_VERSION = _digest("deterministic-project-agent-v1")
_CONSUMER_CONFIG = _digest("deploy-from-injected-facts-v1")
_ASSIGNER_VERSION = _digest("paired-arm-assigner-v1")
_ASSIGNER_CONFIG = _digest("one-release-per-subject-arm-v1")
_ATTESTOR_VERSION = _digest("local-smoke-attestor-v1")
_ATTESTOR_CONFIG = _digest("admit-exact-smoke-release-v1")
_REVOKER_VERSION = _digest("local-smoke-revoker-v1")
_REVOKER_CONFIG = _digest("operator-only-smoke-revocation-v1")


def fact_payload(fact_key: str, fact_value: str) -> str:
    """Return the exact JSON payload accepted by ``StructuredFactUpdater``."""

    return json.dumps(
        {"fact_key": fact_key, "fact_value": fact_value},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_fact(payload: str) -> tuple[str, str] | None:
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return None
    if type(value) is not dict or set(value) != {"fact_key", "fact_value"}:
        return None
    fact_key = value["fact_key"]
    fact_value = value["fact_value"]
    if (
        type(fact_key) is not str
        or not fact_key.strip()
        or type(fact_value) is not str
        or not fact_value.strip()
    ):
        return None
    return fact_key, fact_value


@dataclass(frozen=True, slots=True)
class FactUpdateDecision:
    evidence_id: str
    fact_key: str | None
    operation: RevisionOperation | None
    revision_id: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class FactUpdateResult:
    release: MemoryRelease
    decisions: tuple[FactUpdateDecision, ...]


class StructuredFactUpdater:
    """Apply a narrow, auditable fact-update policy to immutable history.

    Only FEEDBACK and TOOL_RESULT evidence may create facts.  A new key becomes
    an ADD revision, a trusted value change becomes SUPERSEDE, and an identical
    value is a no-op.  Agent messages and outcomes can remain in the evidence
    ledger for audit without silently becoming authoritative memory.
    """

    trusted_kinds = frozenset({EvidenceKind.FEEDBACK, EvidenceKind.TOOL_RESULT})

    def __init__(self, history_store, release_store) -> None:
        self._history_store = history_store
        self._release_store = release_store

    def update(
        self,
        *,
        scope: MemoryScope,
        base_release: MemoryRelease,
        evidence: Iterable[EvidenceRecord],
        captured_through: datetime,
        idempotency_prefix: str,
    ) -> FactUpdateResult:
        """Publish the current fact tips as a new immutable release."""

        if base_release.manifest.scope != scope:
            raise ValueError("base release belongs to a different scope")
        tips: dict[str, tuple[str, str]] = {}
        for revision in self._release_store.get_release_revisions(
            scope, base_release.release_id
        ):
            candidate = self._history_store.get_candidate(
                scope, revision.proposal.candidate_id
            )
            fact = _decode_fact(candidate.proposal.content)
            if fact is None:
                raise ValueError("base release contains a non-structured fact")
            fact_key, fact_value = fact
            if fact_key in tips:
                raise ValueError("base release contains duplicate fact keys")
            tips[fact_key] = (fact_value, revision.revision_id)

        ordered = sorted(
            tuple(evidence),
            key=lambda item: (
                item.event.observed_at,
                item.event.sequence_no,
                item.evidence_id,
            ),
        )
        decisions: list[FactUpdateDecision] = []
        for record in ordered:
            event = record.event
            if event.scope != scope:
                decisions.append(
                    FactUpdateDecision(
                        record.evidence_id, None, None, None, "foreign_scope"
                    )
                )
                continue
            if event.observed_at > captured_through:
                decisions.append(
                    FactUpdateDecision(
                        record.evidence_id, None, None, None, "after_capture_cutoff"
                    )
                )
                continue
            if event.kind not in self.trusted_kinds:
                decisions.append(
                    FactUpdateDecision(
                        record.evidence_id, None, None, None, "untrusted_kind"
                    )
                )
                continue
            fact = _decode_fact(event.payload)
            if fact is None:
                decisions.append(
                    FactUpdateDecision(
                        record.evidence_id, None, None, None, "malformed_fact"
                    )
                )
                continue
            fact_key, fact_value = fact
            current = tips.get(fact_key)
            if current is not None and current[0] == fact_value:
                decisions.append(
                    FactUpdateDecision(
                        record.evidence_id,
                        fact_key,
                        None,
                        current[1],
                        "same_value_noop",
                    )
                )
                continue

            candidate = self._history_store.append_candidate(
                CandidateProposal(
                    scope=scope,
                    content=fact_payload(fact_key, fact_value),
                    evidence_ids=(record.evidence_id,),
                    idempotency_key=(
                        f"{idempotency_prefix}:candidate:{record.evidence_id}"
                    ),
                )
            )
            operation = (
                RevisionOperation.ADD
                if current is None
                else RevisionOperation.SUPERSEDE
            )
            revision = self._history_store.append_revision(
                RevisionProposal(
                    scope=scope,
                    candidate_id=candidate.candidate_id,
                    operation=operation,
                    parent_revision_id=None if current is None else current[1],
                    idempotency_key=(
                        f"{idempotency_prefix}:revision:{record.evidence_id}"
                    ),
                )
            )
            tips[fact_key] = (fact_value, revision.revision_id)
            decisions.append(
                FactUpdateDecision(
                    record.evidence_id,
                    fact_key,
                    operation,
                    revision.revision_id,
                    "trusted_update",
                )
            )

        manifest = ReleaseManifest(
            scope=scope,
            revision_ids=tuple(tips[key][1] for key in sorted(tips)),
        )
        release = self._release_store.append_release(
            manifest,
            idempotency_key=f"{idempotency_prefix}:release",
        )
        return FactUpdateResult(release=release, decisions=tuple(decisions))


@dataclass(frozen=True, slots=True)
class ReleaseOrderRetriever:
    retrieval_policy_id: str = "adaptive-codebook-release-order"
    retrieval_policy_version_sha256: str = _RETRIEVER_VERSION
    retrieval_policy_config_sha256: str = _RETRIEVER_CONFIG

    def retrieve(self, *, attempt, query, eligible_items):
        del attempt, query
        revision_ids = tuple(item.revision.revision_id for item in eligible_items)
        return MemoryRetrievalOutputV1(
            retrieved_revision_ids=revision_ids,
            returned_revision_ids=revision_ids,
        )


@dataclass(frozen=True, slots=True)
class StructuredFactRenderer:
    renderer_id: str = "adaptive-codebook-json-lines"
    renderer_version_sha256: str = _RENDERER_VERSION
    renderer_config_sha256: str = _RENDERER_CONFIG

    def render(self, query_result: MemoryQueryResultV1) -> MemoryRenderOutputV1:
        context = bytearray()
        ranges: list[MemoryRenderedRevisionRangeV1] = []
        for item in query_result.returned_items:
            start = len(context)
            context.extend(item.content.encode())
            context.extend(b"\n")
            ranges.append(
                MemoryRenderedRevisionRangeV1(
                    revision_id=item.revision.revision_id,
                    rendered_start=start,
                    rendered_end=len(context),
                )
            )
        return MemoryRenderOutputV1(bytes(context), tuple(ranges))


@dataclass(frozen=True, slots=True)
class DeployRequest:
    project_id: str
    access_code: str | None


def _history_sha256(history: tuple[bytes, ...]) -> str:
    digest = hashlib.sha256(b"areal-memory-runtime-history-v1\0")
    digest.update(len(history).to_bytes(8, "big"))
    for item in history:
        digest.update(len(item).to_bytes(8, "big"))
        digest.update(item)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ProjectAgentConsumer:
    """Deterministic stand-in for an agent at the actual context boundary."""

    consumer_kind: MemoryConsumerKind = MemoryConsumerKind.CONTEXT
    consumer_id: str = "adaptive-codebook-project-agent"
    consumer_version_sha256: str = _CONSUMER_VERSION
    consumer_config_sha256: str = _CONSUMER_CONFIG

    def submit(
        self,
        *,
        delivery,
        rendered_context: bytes,
        query: bytes,
        history: tuple[bytes, ...],
        call_id: str,
    ) -> MemoryConsumerCallV1:
        facts: dict[str, str] = {}
        for line in rendered_context.splitlines():
            fact = _decode_fact(line.decode())
            if fact is not None:
                facts[fact[0]] = fact[1]
        output = DeployRequest(
            project_id=delivery.scope.subject_id,
            access_code=facts.get(_FACT_KEY),
        )
        prefix = b"system: use only the injected structured facts\n"
        suffix = b"user: " + query
        submitted_prompt = prefix + rendered_context + suffix
        return MemoryConsumerCallV1(
            delivery_id=delivery.delivery_id,
            delivery_content_sha256=delivery.content_hash,
            call_id=call_id,
            submitted_prompt=submitted_prompt,
            context_start=len(prefix),
            context_end=len(prefix) + len(rendered_context),
            observed_query_sha256=hashlib.sha256(query).hexdigest(),
            observed_history_sha256=_history_sha256(history),
            observed_history_length=len(history),
            input_token_ids=None,
            output=output,
        )


@dataclass(frozen=True, slots=True)
class FakeDeployTool:
    """Score the consumer's real output, never the release or evaluator label."""

    def reward(
        self,
        output: object,
        *,
        project_id: str,
        expected_access_code: str,
    ) -> int:
        if type(output) is not DeployRequest or output.project_id != project_id:
            return -1
        if output.access_code is None:
            return 0
        return 1 if output.access_code == expected_access_code else -1


@dataclass(frozen=True, slots=True)
class _LocalAttestor:
    attestor_id: str = "adaptive-codebook-local-attestor"
    attestor_version_sha256: str = _ATTESTOR_VERSION
    attestor_config_sha256: str = _ATTESTOR_CONFIG

    def attest(self, *, release, evaluated_at):
        del release, evaluated_at
        return _BASE - timedelta(minutes=1), _BASE + timedelta(days=366)


@dataclass(frozen=True, slots=True)
class _LocalRevoker:
    revoker_id: str = "adaptive-codebook-local-revoker"
    revoker_version_sha256: str = _REVOKER_VERSION
    revoker_config_sha256: str = _REVOKER_CONFIG

    def revoke(self, *, attestation, evaluated_at):
        del attestation, evaluated_at
        return MemoryReleaseRevocationReason.OPERATOR, None


@dataclass(frozen=True, slots=True)
class _LocalAssignmentPolicy:
    assignment_policy_id: str = "local-paired-arm-assigner"
    assignment_policy_version_sha256: str = _ASSIGNER_VERSION
    assignment_policy_config_sha256: str = _ASSIGNER_CONFIG

    def authorize(self, **arguments):
        del arguments
        return _BASE + timedelta(days=365)


def _assign_release(
    *,
    control: InMemoryMemoryReleaseControlStore,
    scope: MemoryScope,
    release: MemoryRelease,
    arm: str,
    retriever: ReleaseOrderRetriever,
    renderer: StructuredFactRenderer,
    consumer: ProjectAgentConsumer,
) -> MemoryReleaseAssignmentV1:
    """Attest and assign through the real trusted reference control store."""

    attestation = control.attest_release(
        scope,
        release.release_id,
        release_content_sha256=release.content_hash,
        idempotency_key=f"attestation:{scope.subject_id}:{arm}",
    )
    group_id = f"{scope.subject_id}:{arm}"
    return control.assign_release(
        scope,
        group_id,
        rollout_group_incarnation_sha256=_digest(f"incarnation:{group_id}"),
        attestation_id=attestation.attestation_id,
        attestation_content_sha256=attestation.content_hash,
        task_policy_id="mutable-project-access-code",
        task_policy_version_sha256=_TASK_VERSION,
        task_policy_config_sha256=_TASK_CONFIG,
        retrieval_policy_id=retriever.retrieval_policy_id,
        retrieval_policy_version_sha256=retriever.retrieval_policy_version_sha256,
        retrieval_policy_config_sha256=retriever.retrieval_policy_config_sha256,
        renderer_id=renderer.renderer_id,
        renderer_version_sha256=renderer.renderer_version_sha256,
        renderer_config_sha256=renderer.renderer_config_sha256,
        consumer_kind=MemoryReleaseAssignmentConsumerKind.CONTEXT,
        consumer_id=consumer.consumer_id,
        consumer_version_sha256=consumer.consumer_version_sha256,
        consumer_config_sha256=consumer.consumer_config_sha256,
        max_returned_items=4,
        max_context_utf8_bytes=4096,
        idempotency_key=f"assignment:{group_id}",
    )


@dataclass(frozen=True, slots=True)
class ArmObservation:
    arm: str
    query_text: str
    assignment_id: str
    release_id: str
    eligible_revision_ids: tuple[str, ...]
    retrieved_revision_ids: tuple[str, ...]
    returned_revision_ids: tuple[str, ...]
    injected_revision_ids: tuple[str, ...]
    exposure_id: str
    exposure_status: str
    consumer_output: DeployRequest
    reward: int


@dataclass(frozen=True, slots=True)
class SubjectResult:
    case_name: str
    subject_id: str
    expected_access_code: str
    captured_evidence_ids: tuple[str, ...]
    static_decisions: tuple[FactUpdateDecision, ...]
    adaptive_decisions: tuple[FactUpdateDecision, ...]
    arms: tuple[ArmObservation, ...]

    def arm(self, name: str) -> ArmObservation:
        return next(item for item in self.arms if item.arm == name)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    subjects: tuple[SubjectResult, ...]

    def paired_utility(self) -> tuple[dict[str, object], ...]:
        """Return inspectable per-subject rewards and adaptive deltas."""

        rows = []
        for subject in self.subjects:
            no_memory = subject.arm("no_memory").reward
            static = subject.arm("static").reward
            adaptive = subject.arm("adaptive").reward
            rows.append(
                {
                    "case_name": subject.case_name,
                    "subject_id": subject.subject_id,
                    "no_memory": no_memory,
                    "static": static,
                    "adaptive": adaptive,
                    "adaptive_minus_no_memory": adaptive - no_memory,
                    "adaptive_minus_static": adaptive - static,
                }
            )
        return tuple(rows)

    def to_json(self) -> str:
        return json.dumps(
            {
                "paired_utility": self.paired_utility(),
                "subjects": [asdict(subject) for subject in self.subjects],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class _Case:
    name: str
    expected_code: str
    baseline: tuple[tuple[EvidenceKind, str], ...]
    later: tuple[tuple[EvidenceKind, str], ...]
    later_is_foreign: bool = False
    later_is_future: bool = False


_CASES = (
    _Case("feedback_add", "AC-101", (), ((EvidenceKind.FEEDBACK, "AC-101"),)),
    _Case("tool_result_add", "AC-102", (), ((EvidenceKind.TOOL_RESULT, "AC-102"),)),
    _Case(
        "feedback_supersede",
        "AC-203",
        ((EvidenceKind.FEEDBACK, "OLD-203"),),
        ((EvidenceKind.FEEDBACK, "AC-203"),),
    ),
    _Case(
        "tool_result_supersede",
        "AC-204",
        ((EvidenceKind.TOOL_RESULT, "OLD-204"),),
        ((EvidenceKind.TOOL_RESULT, "AC-204"),),
    ),
    _Case(
        "same_value_noop",
        "AC-305",
        ((EvidenceKind.FEEDBACK, "AC-305"),),
        ((EvidenceKind.TOOL_RESULT, "AC-305"),),
    ),
    _Case(
        "agent_conflict_ignored",
        "AC-306",
        ((EvidenceKind.FEEDBACK, "AC-306"),),
        ((EvidenceKind.AGENT_MESSAGE, "HALLUCINATED-306"),),
    ),
    _Case(
        "foreign_scope_ignored",
        "AC-307",
        ((EvidenceKind.TOOL_RESULT, "AC-307"),),
        ((EvidenceKind.FEEDBACK, "FOREIGN-307"),),
        later_is_foreign=True,
    ),
    _Case(
        "future_outcome_ignored",
        "AC-308",
        ((EvidenceKind.TOOL_RESULT, "AC-308"),),
        (
            (EvidenceKind.OUTCOME, "LEAKED-308"),
            (EvidenceKind.FEEDBACK, "FUTURE-TRUSTED-308"),
        ),
        later_is_future=True,
    ),
)


def _append_event(
    evidence_store: InMemoryEvidenceStore,
    *,
    scope: MemoryScope,
    kind: EvidenceKind,
    value: str,
    sequence_no: int,
    observed_at: datetime,
    case_name: str,
) -> EvidenceRecord:
    return evidence_store.append(
        EvidenceEvent(
            scope=scope,
            session_id=f"capture:{case_name}",
            run_id="shared-capture-stream",
            sequence_no=sequence_no,
            kind=kind,
            payload=fact_payload(_FACT_KEY, value),
            observed_at=observed_at,
            idempotency_key=f"{case_name}:evidence:{sequence_no}:{scope.subject_id}",
        )
    )


def _run_arm(
    *,
    arm: str,
    scope: MemoryScope,
    assignment: MemoryReleaseAssignmentV1,
    runtime: InMemoryMemoryRuntimeStore,
    renderer: StructuredFactRenderer,
    consumer: ProjectAgentConsumer,
    expected_access_code: str,
) -> ArmObservation:
    query_text = f"Deploy project {scope.subject_id} using its current credential."
    if expected_access_code in query_text:
        raise AssertionError("future query must not contain the answer")
    query = query_text.encode()
    spec = MemoryQuerySpecV1(
        scope=scope,
        assignment_id=assignment.assignment_id,
        assignment_content_sha256=assignment.content_hash,
        release_id=assignment.release_id,
        trajectory_id=f"trajectory:{scope.subject_id}:{arm}",
        rollout_group_id=assignment.rollout_group_id,
        rollout_group_incarnation_sha256=(assignment.rollout_group_incarnation_sha256),
        query_sequence_no=0,
        query_sha256=hashlib.sha256(query).hexdigest(),
        task_policy_id=assignment.task_policy_id,
        task_policy_version_sha256=assignment.task_policy_version_sha256,
        task_policy_config_sha256=assignment.task_policy_config_sha256,
        retrieval_policy_id=assignment.retrieval_policy_id,
        retrieval_policy_version_sha256=(assignment.retrieval_policy_version_sha256),
        retrieval_policy_config_sha256=(assignment.retrieval_policy_config_sha256),
        max_returned_items=assignment.max_returned_items,
        max_context_utf8_bytes=assignment.max_context_utf8_bytes,
        idempotency_key=f"query:{scope.subject_id}:{arm}",
    )
    attempt = runtime.begin_query(spec)
    result = runtime.resolve_query(scope, attempt.attempt_id, query=query)
    delivery = runtime.prepare_delivery(
        scope,
        result.query_result_id,
        renderer_id=renderer.renderer_id,
        renderer_version_sha256=renderer.renderer_version_sha256,
    )
    exposure, output = runtime.submit_delivery(
        scope,
        delivery.delivery_id,
        consumer_id=consumer.consumer_id,
        consumer_version_sha256=consumer.consumer_version_sha256,
        call_id=f"consumer:{scope.subject_id}:{arm}",
        query=query,
        history=(),
    )
    if type(output) is not DeployRequest:
        raise TypeError("project consumer returned an unexpected output")
    reward = FakeDeployTool().reward(
        output,
        project_id=scope.subject_id,
        expected_access_code=expected_access_code,
    )
    return ArmObservation(
        arm=arm,
        query_text=query_text,
        assignment_id=exposure.assignment_id,
        release_id=exposure.release_id,
        eligible_revision_ids=tuple(
            item.revision_id for item in exposure.eligible_revisions
        ),
        retrieved_revision_ids=tuple(
            item.revision_id for item in exposure.retrieved_revisions
        ),
        returned_revision_ids=tuple(
            item.revision_id for item in exposure.returned_revisions
        ),
        injected_revision_ids=tuple(
            item.revision_id for item in exposure.injected_revisions
        ),
        exposure_id=exposure.exposure_id,
        exposure_status=exposure.status.value,
        consumer_output=output,
        reward=reward,
    )


def _evaluate_case(case: _Case) -> SubjectResult:
    evidence_store = InMemoryEvidenceStore()
    history_store = InMemoryMemoryHistoryStore(evidence_store)
    release_store = InMemoryMemoryReleaseStore(history_store)
    scope = MemoryScope("local-smoke", "adaptive-codebook", case.name)
    foreign_scope = MemoryScope("local-smoke", "adaptive-codebook", "foreign")

    captured: list[EvidenceRecord] = []
    sequence_no = 0
    for kind, value in case.baseline:
        captured.append(
            _append_event(
                evidence_store,
                scope=scope,
                kind=kind,
                value=value,
                sequence_no=sequence_no,
                observed_at=_BASE + timedelta(seconds=sequence_no),
                case_name=case.name,
            )
        )
        sequence_no += 1
    for kind, value in case.later:
        event_scope = foreign_scope if case.later_is_foreign else scope
        observed_at = _BASE + timedelta(seconds=20 + sequence_no)
        if case.later_is_future and kind is EvidenceKind.FEEDBACK:
            observed_at = _BASE + timedelta(seconds=40 + sequence_no)
        captured.append(
            _append_event(
                evidence_store,
                scope=event_scope,
                kind=kind,
                value=value,
                sequence_no=sequence_no,
                observed_at=observed_at,
                case_name=case.name,
            )
        )
        sequence_no += 1

    empty_release = release_store.append_release(
        ReleaseManifest(scope=scope, revision_ids=()),
        idempotency_key=f"{case.name}:empty-release",
    )
    updater = StructuredFactUpdater(history_store, release_store)
    static_update = updater.update(
        scope=scope,
        base_release=empty_release,
        evidence=captured,
        captured_through=_BASELINE_CUTOFF,
        idempotency_prefix=f"{case.name}:static",
    )
    adaptive_update = updater.update(
        scope=scope,
        base_release=static_update.release,
        evidence=captured,
        captured_through=_ADAPTIVE_CUTOFF,
        idempotency_prefix=f"{case.name}:adaptive",
    )

    retriever = ReleaseOrderRetriever()
    renderer = StructuredFactRenderer()
    consumer = ProjectAgentConsumer()
    control = InMemoryMemoryReleaseControlStore(
        release_store,
        attestor=_LocalAttestor(),
        revoker=_LocalRevoker(),
        assignment_policy=_LocalAssignmentPolicy(),
        clock=lambda: _BASE,
    )
    releases = {
        "no_memory": empty_release,
        "static": static_update.release,
        "adaptive": adaptive_update.release,
    }
    assignments = {
        arm: _assign_release(
            control=control,
            scope=scope,
            release=release,
            arm=arm,
            retriever=retriever,
            renderer=renderer,
            consumer=consumer,
        )
        for arm, release in releases.items()
    }
    runtime = InMemoryMemoryRuntimeStore(
        history_store,
        release_store,
        release_control_store=control,
        retrievers=(retriever,),
        renderers=(renderer,),
        consumers=(consumer,),
    )
    arms = tuple(
        _run_arm(
            arm=arm,
            scope=scope,
            assignment=assignments[arm],
            runtime=runtime,
            renderer=renderer,
            consumer=consumer,
            expected_access_code=case.expected_code,
        )
        for arm in ("no_memory", "static", "adaptive")
    )
    return SubjectResult(
        case_name=case.name,
        subject_id=scope.subject_id,
        expected_access_code=case.expected_code,
        captured_evidence_ids=tuple(item.evidence_id for item in captured),
        static_decisions=static_update.decisions,
        adaptive_decisions=adaptive_update.decisions,
        arms=arms,
    )


def run_smoke_evaluation() -> EvaluationReport:
    """Run eight paired subjects locally with no model or GPU dependency."""

    report = EvaluationReport(tuple(_evaluate_case(case) for case in _CASES))
    if len(report.subjects) < 8:
        raise AssertionError("the adaptive smoke evaluation requires at least 8 cases")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(run_smoke_evaluation().to_json())


if __name__ == "__main__":
    main()
