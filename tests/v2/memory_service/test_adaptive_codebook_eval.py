# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from examples.memory_service.adaptive_codebook_eval import (
    DeployRequest,
    FakeDeployTool,
    run_smoke_evaluation,
)

from areal.v2.memory_service import MemoryExposureStatus, RevisionOperation


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
        assert len({item.assignment_id for item in subject.arms}) == 3
        assert len({item.exposure_id for item in subject.arms}) == 3
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
    ] == [RevisionOperation.ADD]
    assert [
        decision.operation
        for decision in cases["tool_result_add"].adaptive_decisions
        if decision.operation is not None
    ] == [RevisionOperation.ADD]
    for name in ("feedback_supersede", "tool_result_supersede"):
        assert RevisionOperation.SUPERSEDE in {
            decision.operation for decision in cases[name].adaptive_decisions
        }

    noop = cases["same_value_noop"]
    assert "same_value_noop" in {
        decision.reason for decision in noop.adaptive_decisions
    }
    assert noop.arm("static").release_id == noop.arm("adaptive").release_id

    agent = cases["agent_conflict_ignored"]
    assert "untrusted_kind" in {
        decision.reason for decision in agent.adaptive_decisions
    }
    assert agent.arm("static").release_id == agent.arm("adaptive").release_id

    foreign = cases["foreign_scope_ignored"]
    assert "foreign_scope" in {
        decision.reason for decision in foreign.adaptive_decisions
    }
    assert foreign.arm("static").release_id == foreign.arm("adaptive").release_id

    future = cases["future_outcome_ignored"]
    assert "untrusted_kind" in {
        decision.reason for decision in future.adaptive_decisions
    }
    assert "after_capture_cutoff" in {
        decision.reason for decision in future.adaptive_decisions
    }
    assert future.arm("static").release_id == future.arm("adaptive").release_id


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
