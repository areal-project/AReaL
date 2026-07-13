# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import UTC, datetime

from examples.memory_service.adaptive_codebook_eval import (
    DeployRequest,
    FakeDeployTool,
    FixedKindFixtureTrustPolicy,
    run_smoke_evaluation,
)

from areal.v2.memory_service import (
    EvidenceAuthority,
    EvidenceEvent,
    EvidenceKind,
    InMemoryEvidenceStore,
    MemoryExposureStatus,
    MemoryScope,
    StructuredFactOperation,
    StructuredFactUpdateV1,
)


def _by_case(report):
    return {subject.case_name: subject for subject in report.subjects}


def test_eight_subject_paired_smoke_makes_adaptive_rule_work() -> None:
    report = run_smoke_evaluation()

    assert len(report.subjects) == 8
    assert [row["case_name"] for row in report.paired_utility()] == [
        subject.case_name for subject in report.subjects
    ]
    assert [subject.arm("no_memory").reward for subject in report.subjects] == [0] * 8
    assert [subject.arm("static").reward for subject in report.subjects] == [
        0,
        0,
        -1,
        -1,
        1,
        1,
        1,
        1,
    ]
    assert [subject.arm("adaptive").reward for subject in report.subjects] == [1] * 8
    for subject in report.subjects:
        assert (
            tuple(decision.evidence_id for decision in subject.static_decisions)
            == subject.captured_evidence_ids
        )
        assert (
            tuple(decision.evidence_id for decision in subject.adaptive_decisions)
            == subject.captured_evidence_ids
        )


def test_reward_path_contains_real_assignment_retrieval_and_exposure() -> None:
    report = run_smoke_evaluation()

    for subject in report.subjects:
        assert len({item.assignment_id for item in subject.arms}) == len(subject.arms)
        assert len({item.exposure_id for item in subject.arms}) == len(subject.arms)
        for observation in subject.arms:
            assert subject.expected_access_code not in observation.query_text
            assert observation.assignment_id.startswith("masn_")
            assert observation.release_id.startswith("rel_")
            assert observation.exposure_id.startswith("mexp_")
            assert (
                observation.returned_revision_ids == observation.injected_revision_ids
            )
            assert set(observation.returned_revision_ids) <= set(
                observation.retrieved_revision_ids
            )
            assert set(observation.retrieved_revision_ids) <= set(
                observation.eligible_revision_ids
            )

        no_memory = subject.arm("no_memory")
        assert no_memory.eligible_revision_ids == ()
        assert no_memory.retrieved_revision_ids == ()
        assert no_memory.returned_revision_ids == ()
        assert no_memory.injected_revision_ids == ()
        assert no_memory.exposure_status == MemoryExposureStatus.MEMORY_OFF.value

        adaptive = subject.arm("adaptive")
        assert adaptive.exposure_status == MemoryExposureStatus.DELIVERED.value
        assert adaptive.consumer_output == DeployRequest(
            project_id=subject.subject_id,
            access_code=subject.expected_access_code,
        )


def test_update_policy_adds_supersedes_noops_and_rejects_pollution() -> None:
    cases = _by_case(run_smoke_evaluation())

    assert [
        decision.operation
        for decision in cases["feedback_add"].adaptive_decisions
        if decision.operation is not None
    ] == [StructuredFactOperation.ADD]
    assert [
        decision.operation
        for decision in cases["tool_result_add"].adaptive_decisions
        if decision.operation is not None
    ] == [StructuredFactOperation.ADD]
    for name in ("feedback_supersede", "tool_result_supersede"):
        assert StructuredFactOperation.SUPERSEDE in {
            decision.operation for decision in cases[name].adaptive_decisions
        }

    noop = cases["same_value_noop"]
    assert "confirmed_current_fact" in {
        decision.reason for decision in noop.adaptive_decisions
    }
    assert noop.arm("static").release_id == noop.arm("adaptive").release_id

    agent = cases["agent_conflict_ignored"]
    assert "fixture_untrusted_kind" in {
        decision.reason for decision in agent.adaptive_decisions
    }
    assert agent.arm("static").release_id == agent.arm("adaptive").release_id

    foreign = cases["foreign_scope_ignored"]
    assert "foreign_scope" in {
        decision.reason for decision in foreign.adaptive_decisions
    }
    assert foreign.arm("static").release_id == foreign.arm("adaptive").release_id

    future = cases["future_outcome_ignored"]
    assert "fixture_outcome_evaluator_only" in {
        decision.reason for decision in future.adaptive_decisions
    }
    assert EvidenceAuthority.EVALUATOR_ONLY in {
        decision.authority for decision in future.adaptive_decisions
    }
    assert "after_capture_cutoff" in {
        decision.reason for decision in future.adaptive_decisions
    }
    assert future.arm("static").release_id == future.arm("adaptive").release_id


def test_fixture_tool_result_requires_the_verified_local_run_marker() -> None:
    scope = MemoryScope("local-smoke", "adaptive-codebook", "tool-marker")
    evidence_store = InMemoryEvidenceStore()
    update = StructuredFactUpdateV1(
        fact_key="project_access_code",
        fact_value="VALUE",
        operation=StructuredFactOperation.ADD,
        expected_parent_revision_id=None,
    )

    def append_tool_result(*, sequence_no: int, run_id: str):
        return evidence_store.append(
            EvidenceEvent(
                scope=scope,
                session_id="fixture-tool-authority",
                run_id=run_id,
                sequence_no=sequence_no,
                kind=EvidenceKind.TOOL_RESULT,
                payload=update.to_payload(),
                observed_at=datetime(2026, 7, 13, tzinfo=UTC),
                idempotency_key=f"tool-result-{sequence_no}",
            )
        )

    verified = append_tool_result(
        sequence_no=0,
        run_id="verified-local-fake-tool",
    )
    unverified = append_tool_result(sequence_no=1, run_id="unverified-tool")
    trust = FixedKindFixtureTrustPolicy()

    assert (
        trust.evaluate(evidence=verified, update=update).authority
        is EvidenceAuthority.AUTHORITATIVE
    )
    assert (
        trust.evaluate(evidence=unverified, update=update).authority
        is EvidenceAuthority.INELIGIBLE
    )


def test_fake_deploy_tool_scores_only_the_actual_consumer_output() -> None:
    subject = run_smoke_evaluation().subjects[0]
    tool = FakeDeployTool()
    actual_output = subject.arm("adaptive").consumer_output

    assert (
        tool.reward(
            actual_output,
            project_id=subject.subject_id,
            expected_access_code=subject.expected_access_code,
        )
        == 1
    )
    assert (
        tool.reward(
            DeployRequest(subject.subject_id, "evaluator-side-answer"),
            project_id=subject.subject_id,
            expected_access_code=subject.expected_access_code,
        )
        == -1
    )
    assert (
        tool.reward(
            DeployRequest(subject.subject_id, None),
            project_id=subject.subject_id,
            expected_access_code=subject.expected_access_code,
        )
        == 0
    )


def test_json_report_keeps_subject_pairs_instead_of_claiming_significance() -> None:
    report = run_smoke_evaluation()
    value = json.loads(report.to_json())

    assert len(value["paired_utility"]) == 8
    assert {row["subject_id"] for row in value["paired_utility"]} == {
        subject.subject_id for subject in report.subjects
    }
    assert "p_value" not in value
    assert "significant" not in value
