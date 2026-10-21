"""CPU-only contract tests for private verifier telemetry and the Arena client."""

import copy
import json
import math
from pathlib import Path

import pytest

from examples.swe.arena_client import _parse_task_result
from examples.swe.arena_reward_breakdown import (
    arena_reward_metrics,
    build_fine64_reward_breakdown,
)


@pytest.fixture
def measured_result():
    """Numeric-only snapshot from the running mean80/min20 verifier."""
    path = Path(__file__).parent / "fixtures" / "arena_fine64_reward.json"
    return json.loads(path.read_text())


def test_measured_breakdown_distinguishes_weights_and_actual_contributions(
    measured_result,
):
    """A 10% visual weight contributes 67.46% of this actual total score."""
    breakdown = build_fine64_reward_breakdown(measured_result)
    root = breakdown["root"]
    rule, vlm = root["children"]
    quality, factor = rule["children"]

    assert root["value"] == pytest.approx(0.14824290384003877)
    assert rule["weight"] == 0.9
    assert vlm["weight"] == 0.1
    assert rule["contribution_to_parent"] == pytest.approx(0.04824290384003876)
    assert vlm["contribution_to_parent"] == pytest.approx(0.1)
    assert vlm["share_of_parent"] == pytest.approx(0.6745685453376226)
    assert sum(c["share_of_parent"] for c in root["children"]) == pytest.approx(1)
    assert quality["value"] == pytest.approx(0.14928802894519366)
    assert factor["value"] == pytest.approx(0.35905910787133954)
    assert "weight" not in factor
    assert "contribution_to_parent" not in factor
    assert "share_of_parent" not in factor
    assert "semantic_h" not in json.dumps(breakdown)


def test_client_preserves_sub_scores_and_only_top_level_reward(measured_result):
    """Grader telemetry survives parsing without becoming the scalar reward."""
    breakdown = build_fine64_reward_breakdown(measured_result)
    payload = {
        "task_id": "task:env:0",
        "status": "OK",
        "score": round(measured_result["score"], 4),
        "sub_scores": measured_result["rule_result"]["sub_scores"],
        "raw": {"reward_breakdown": breakdown, "score": 1.0},
    }

    result = _parse_task_result(payload)
    metrics = arena_reward_metrics(
        sub_scores=result.sub_scores, raw=result.raw, score=result.score
    )

    assert result.score == 0.1482
    assert result.sub_scores == payload["sub_scores"]
    assert metrics["reward_breakdown/valid"] == 1
    assert metrics["reward_sub_score/factor_F"] == pytest.approx(0.35905910787133954)
    prefix = "reward_breakdown/total/rule/quality/mean/cliff_context"
    assert metrics[f"{prefix}/layout/weight"] == 0.25
    assert metrics[f"{prefix}/surface/weight"] == 0.35
    assert metrics[f"{prefix}/view/weight"] == 0.4
    assert all(math.isfinite(value) for value in metrics.values())


def test_fallback_reports_effective_weights_and_omits_unknown_visual_score(
    measured_result,
):
    """VLM failure switches to rule=1, VLM=0 without inventing a visual zero."""
    measured_result.update(
        score_mode="RULE_ONLY_FALLBACK",
        score=measured_result["rule_score"],
        vlm_score=None,
        vlm_status="FAILED",
        visual=None,
    )
    breakdown = build_fine64_reward_breakdown(measured_result)
    rule, vlm = breakdown["root"]["children"]
    metrics = arena_reward_metrics(
        sub_scores=None,
        raw={"reward_breakdown": breakdown},
        score=measured_result["score"],
    )

    assert rule["weight"] == 1 and rule["declared_weight"] == 0.9
    assert vlm["weight"] == 0 and vlm["declared_weight"] == 0.1
    assert vlm["value"] is None and vlm["excluded"]
    assert "reward_breakdown/total/vlm/score" not in metrics
    assert metrics["reward_breakdown/total/vlm/computed"] == 0
    assert metrics["reward_breakdown/total/rule/share_of_parent_mean"] == 1
    assert metrics["reward_breakdown/valid"] == 1


@pytest.mark.parametrize(
    "mode,status,score",
    [
        ("HARD_INVALID", "INVALID_SUBMISSION", 0.0),
        ("RULE_UNSCORABLE", "EVAL_FAILED", None),
    ],
)
def test_unmeasured_failures_do_not_fabricate_component_zeros(mode, status, score):
    """Protocol zero and unscorable outcomes preserve missing measurements."""
    result = {
        "schema": "building-reward-v1-overall-90-10",
        "score_mode": mode,
        "status": status,
        "score": score,
    }
    breakdown = build_fine64_reward_breakdown(result)
    metrics = arena_reward_metrics(
        sub_scores=None, raw={"reward_breakdown": breakdown}, score=score
    )

    assert metrics["reward_breakdown/valid"] == 1
    assert not any("/rule/" in key or "/vlm/" in key for key in metrics)
    assert metrics["reward_breakdown/total/computed"] == float(score is not None)


def test_zero_total_has_null_shares_and_retains_all_tied_minimum_ids(measured_result):
    """All-zero quality is measured, while division by zero stays undefined."""
    rule = measured_result["rule_result"]
    for part in rule["required_role_diagnostics"]:
        part.update(layout=0, surface=0, view=0, quality=0)
    rule["aggregation"].update(mean=0, minimum=0)
    rule["sub_scores"].update(semantic_Q=0, minimum_q=0)
    rule["score"] = 0
    measured_result.update(rule_score=0, score=0, vlm_score=0)
    measured_result["visual"]["vlm_score"] = 0
    for item in measured_result["visual"]["items"]:
        item["value"] = 0
    breakdown = build_fine64_reward_breakdown(measured_result)
    root = breakdown["root"]
    minimum = root["children"][0]["children"][0]["children"][1]
    metrics = arena_reward_metrics(
        sub_scores=None, raw={"reward_breakdown": breakdown}, score=0
    )

    assert len(minimum["selected_part_ids"]) == len(rule["required_role_diagnostics"])
    assert all(child["share_of_parent"] is None for child in root["children"])
    assert metrics["reward_breakdown/total/rule/share_computed"] == 0
    assert metrics["reward_breakdown/total/rule/score"] == 0
    assert "reward_breakdown/total/rule/share_of_parent_mean" not in metrics


@pytest.mark.parametrize("field", ["score", "quality", "policy", "missing_part", "vlm"])
def test_producer_rejects_incomplete_or_disagreeing_measurements(
    measured_result, field
):
    """A different policy or incomplete artifact cannot be guessed from total."""
    if field == "score":
        measured_result["score"] += 0.1
    elif field == "quality":
        measured_result["rule_result"]["required_role_diagnostics"][0]["quality"] = 1
    elif field == "policy":
        measured_result["rule_result"]["aggregation"]["policy"] = "another-policy"
    elif field == "missing_part":
        measured_result["rule_result"].pop("required_role_diagnostics")
    else:
        measured_result["visual"]["items"].pop()

    with pytest.raises((KeyError, ValueError)):
        build_fine64_reward_breakdown(measured_result)


@pytest.mark.parametrize(
    "corruption",
    [
        "total",
        "share",
        "product_weight",
        "name",
        "duplicate",
        "nan",
        "infinite",
        "huge",
        "schema",
    ],
)
def test_invalid_optional_breakdown_is_omitted_without_changing_reward(
    measured_result, corruption
):
    """Bad telemetry has a validity flag and never changes reward admission."""
    breakdown = copy.deepcopy(build_fine64_reward_breakdown(measured_result))
    root = breakdown["root"]
    if corruption == "total":
        root["value"] = 1
    elif corruption == "share":
        root["children"][0]["share_of_parent"] = 0.99
    elif corruption == "product_weight":
        root["children"][0]["children"][1]["weight"] = 0.1
    elif corruption == "name":
        root["children"][0]["name"] = "path/injection"
    elif corruption == "duplicate":
        root["children"][1]["name"] = "rule"
    elif corruption in {"nan", "infinite"}:
        root["value"] = float("nan" if corruption == "nan" else "inf")
    elif corruption == "huge":
        root["children"] = root["children"] * 129
    else:
        breakdown["schema"] = "unknown"
    result = _parse_task_result(
        {
            "task_id": "task",
            "status": "OK",
            "score": 0.1482,
            "raw": {"reward_breakdown": breakdown},
        }
    )
    metrics = arena_reward_metrics(
        sub_scores=result.sub_scores, raw=result.raw, score=result.score
    )

    assert result.score == 0.1482
    assert metrics == {
        "reward_sub_score/invalid_entries": 0,
        "reward_breakdown/present": 1,
        "reward_breakdown/valid": 0,
    }


def test_legacy_results_and_invalid_sub_scores_preserve_authoritative_reward():
    """Older graders need no breakdown; malformed optional metrics are isolated."""
    result = _parse_task_result(
        {
            "task_id": "task",
            "status": "DONE",
            "score": 0.5,
            "sub_scores": {"good": 0.2, "boolean": True, "nan": float("nan"), "x/y": 1},
            "raw": {"nested": {"score": 1, "sub_scores": {"hidden": 1}}},
        }
    )
    metrics = arena_reward_metrics(
        sub_scores=result.sub_scores, raw=result.raw, score=result.score
    )

    assert result.score == 0.5
    assert metrics["reward_sub_score/good"] == 0.2
    assert metrics["reward_sub_score/invalid_entries"] == 3
    assert metrics["reward_breakdown/present"] == 0
    assert metrics["reward_breakdown/valid"] == 0
    assert not any("hidden" in key for key in metrics)


def test_part_weights_are_normalized_without_changing_declared_weights(measured_result):
    """Nonuniform part priorities retain both original and effective weights."""
    rule = measured_result["rule_result"]
    parts = rule["required_role_diagnostics"]
    parts[0]["weight"] = 10
    mean = sum(part["weight"] * part["quality"] for part in parts) / sum(
        part["weight"] for part in parts
    )
    quality = 0.8 * mean + 0.2 * min(part["quality"] for part in parts)
    rule["aggregation"]["mean"] = mean
    rule["sub_scores"]["semantic_Q"] = quality
    rule["score"] = quality * rule["sub_scores"]["factor_F"]
    measured_result["rule_score"] = rule["score"]
    measured_result["score"] = 0.9 * rule["score"] + 0.1

    breakdown = build_fine64_reward_breakdown(measured_result)
    mean_node = breakdown["root"]["children"][0]["children"][0]["children"][0]

    assert mean_node["children"][0]["declared_weight"] == 10
    assert mean_node["children"][0]["weight"] == pytest.approx(10 / 18)
    assert sum(child["weight"] for child in mean_node["children"]) == pytest.approx(1)


def test_minimum_ties_use_original_measurements_instead_of_recomputed_rounding(
    measured_result,
):
    """Equivalent verifier q values remain tied across floating-point sums."""
    rule = measured_result["rule_result"]
    parts = rule["required_role_diagnostics"][:2]
    rule["required_role_diagnostics"] = parts
    observations = [
        (0.48411875607914223, 0.19876821879897177, 0.5235035860014359),
        (0.45933765777727764, 0.235167486034618, 0.5071424136089107),
    ]
    for part, (layout, surface, view) in zip(parts, observations):
        part.update(layout=layout, surface=surface, view=view, quality=0.4)
    rule["aggregation"].update(mean=0.4, minimum=0.4)
    rule["sub_scores"].update(semantic_Q=0.4, minimum_q=0.4)
    rule["score"] = 0.4 * rule["sub_scores"]["factor_F"]
    measured_result["rule_score"] = rule["score"]
    measured_result["score"] = 0.9 * rule["score"] + 0.1

    breakdown = build_fine64_reward_breakdown(measured_result)
    quality = breakdown["root"]["children"][0]["children"][0]

    assert quality["children"][1]["selected_part_ids"] == [p["id"] for p in parts]
    assert [p["value"] for p in quality["children"][0]["children"]] == [0.4, 0.4]
    assert breakdown["root"]["value"] == measured_result["score"]
