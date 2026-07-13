# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections import Counter

from examples.memory_service.adaptive_codebook_eval import (
    PREREGISTERED_RELATIONS,
    run_smoke_evaluation,
)


def _cases(report):
    return {subject.case_name: subject for subject in report.subjects}


def test_preregistered_counterfactual_relations_hold_subject_by_subject() -> None:
    report = run_smoke_evaluation()
    changed = [subject for subject in report.subjects if subject.changed_case]
    stable = [subject for subject in report.subjects if not subject.changed_case]

    assert len(changed) == 4
    assert len(stable) == 4
    for subject in report.subjects:
        assert subject.arm("stale").release_id == subject.arm("static").release_id
    for subject in changed:
        assert subject.arm("adaptive").reward > subject.arm("target_masked").reward
        assert subject.arm("adaptive").reward > subject.arm("stale").reward
    for subject in stable:
        assert subject.arm("adaptive").reward >= subject.arm("stale").reward
    cases = _cases(report)
    for name in ("feedback_add", "tool_result_add"):
        assert cases[name].change_kind == "add"
        assert cases[name].arm("stale").consumer_output.access_code is None
        assert cases[name].arm("stale").exposure_status == "memory_off"
    assert (
        cases["feedback_supersede"].arm("stale").consumer_output.access_code
        == "OLD-203"
    )
    assert (
        cases["tool_result_supersede"].arm("stale").consumer_output.access_code
        == "OLD-204"
    )
    assert [subject.arm("oracle").reward for subject in report.subjects] == [1] * 8
    assert [subject.arm("raw_history").reward for subject in report.subjects] == [
        1,
        1,
        1,
        1,
        1,
        -1,
        1,
        -1,
    ]


def test_target_mask_preserves_runtime_shape_and_rendered_utf8_length() -> None:
    report = run_smoke_evaluation()

    for subject in report.subjects:
        adaptive = subject.arm("adaptive")
        masked = subject.arm("target_masked")
        assert len(masked.eligible_revision_ids) == len(adaptive.eligible_revision_ids)
        assert len(masked.retrieved_revision_ids) == len(
            adaptive.retrieved_revision_ids
        )
        assert len(masked.returned_revision_ids) == len(adaptive.returned_revision_ids)
        assert len(masked.injected_revision_ids) == len(adaptive.injected_revision_ids)
        assert (
            masked.rendered_context_utf8_bytes == adaptive.rendered_context_utf8_bytes
        )
        assert masked.consumer_output.access_code != subject.expected_access_code
        assert set(masked.consumer_output.access_code) == {"*"}
        assert len(masked.consumer_output.access_code.encode()) == len(
            subject.expected_access_code.encode()
        )
        assert masked.reward == -1


def test_raw_last_write_wins_is_pollutable_but_still_scope_and_cutoff_bound() -> None:
    cases = _cases(run_smoke_evaluation())

    agent = cases["agent_conflict_ignored"]
    assert agent.raw_selected_evidence_id == agent.captured_evidence_ids[-1]
    assert agent.arm("raw_history").consumer_output.access_code == "HALLUCINATED-306"
    assert agent.arm("adaptive").consumer_output.access_code == "AC-306"
    assert agent.arm("raw_history").returned_evidence_ids == (
        agent.captured_evidence_ids
    )

    foreign = cases["foreign_scope_ignored"]
    assert foreign.raw_selected_evidence_id == foreign.captured_evidence_ids[0]
    assert foreign.arm("raw_history").consumer_output.access_code == "AC-307"
    assert foreign.arm("raw_history").consumer_output.access_code != "FOREIGN-307"
    assert foreign.arm("raw_history").returned_evidence_ids == (
        foreign.captured_evidence_ids[0],
    )

    future = cases["future_outcome_ignored"]
    assert future.raw_selected_evidence_id == future.captured_evidence_ids[1]
    assert future.arm("raw_history").consumer_output.access_code == "LEAKED-308"
    assert future.arm("raw_history").consumer_output.access_code != "FUTURE-TRUSTED-308"
    assert future.arm("adaptive").consumer_output.access_code == "AC-308"
    assert future.arm("raw_history").returned_evidence_ids == (
        future.captured_evidence_ids[0],
        future.captured_evidence_ids[1],
    )


def test_oracle_and_mask_truth_are_isolated_from_measured_evidence() -> None:
    report = run_smoke_evaluation()

    for subject in report.subjects:
        measured = set(subject.measured_scope_evidence_ids)
        captured_in_scope = {
            evidence_id
            for evidence_id in subject.captured_evidence_ids
            if evidence_id in measured
        }
        assert measured == captured_in_scope
        assert set(subject.arm("oracle").returned_evidence_ids).isdisjoint(measured)
        assert set(subject.arm("target_masked").returned_evidence_ids).isdisjoint(
            measured
        )
        assert subject.raw_selected_evidence_id in measured
        assert (
            subject.raw_selected_evidence_id
            in subject.arm("raw_history").returned_evidence_ids
        )


def test_frozen_cyclic_order_is_balanced_and_every_arm_has_real_exposure() -> None:
    report = run_smoke_evaluation()
    canonical = (
        "no_memory",
        "static",
        "adaptive",
        "target_masked",
        "stale",
        "raw_history",
        "oracle",
    )
    position_counts = {arm: Counter() for arm in canonical}

    for index, subject in enumerate(report.subjects):
        expected = canonical[index % len(canonical) :] + canonical[: index % 7]
        assert tuple(item.arm for item in subject.arms) == expected
        assert len({item.assignment_id for item in subject.arms}) == 7
        assert len({item.exposure_id for item in subject.arms}) == 7
        for position, observation in enumerate(subject.arms):
            position_counts[observation.arm][position] += 1
            assert observation.assignment_id.startswith("masn_")
            assert observation.release_id.startswith("rel_")
            assert observation.exposure_id.startswith("mexp_")
            assert observation.chain_bound
            assert observation.consumer_output.project_id == subject.subject_id
            assert observation.returned_revision_ids == (
                observation.injected_revision_ids
            )
            assert set(observation.returned_revision_ids) <= set(
                observation.retrieved_revision_ids
            )
            expected_status = (
                "delivered" if observation.eligible_revision_ids else "memory_off"
            )
            assert observation.exposure_status == expected_status
        for arm in ("adaptive", "target_masked", "raw_history", "oracle"):
            assert subject.arm(arm).exposure_status == "delivered"

    for arm in canonical:
        counts = [position_counts[arm][position] for position in range(7)]
        assert max(counts) - min(counts) <= 1


def test_report_exposes_counterfactual_utility_and_mechanism_coverage() -> None:
    report = run_smoke_evaluation()
    coverage = report.mechanism_coverage()
    value = json.loads(report.to_json())

    assert len(report.counterfactual_utility()) == 8
    assert coverage == {
        "subjects": 8,
        "actual_runtime_exposures": 56,
        "masked_equal_length_subjects": 8,
        "changed_adaptive_gt_masked": 4,
        "add_adaptive_gt_missing_stale": 2,
        "supersede_adaptive_gt_old_value_stale": 2,
        "stable_adaptive_no_regression": 4,
        "oracle_ceiling_subjects": 8,
        "raw_positive_subjects": 6,
        "raw_negative_subjects": 2,
    }
    assert value["mechanism_coverage"] == coverage
    assert len(value["counterfactual_utility"]) == 8
    assert tuple(value["preregistered_relations"]) == PREREGISTERED_RELATIONS
    assert "p_value" not in value
    assert "bootstrap" not in value
