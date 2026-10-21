"""Private reward telemetry shared by Arena graders and the training adapter.

This module has no AReaL/runtime dependencies, so a grader can bundle it and put
``build_fine64_reward_breakdown(result)`` in ``result['raw']['reward_breakdown']``.
The training adapter only uses the envelope's top-level score as its reward.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

SCHEMA = "arena-reward-breakdown/v1"
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,95}\Z")
_MODES = (
    "MIXED_90_10",
    "RULE_ONLY_FALLBACK",
    "HARD_INVALID",
    "RULE_UNSCORABLE",
)
_MAX_NODES = 256
_MAX_DEPTH = 10


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Reward telemetry values must be real numbers")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Reward telemetry values must be finite")
    return value


def _same(actual: Any, expected: float) -> None:
    if not math.isclose(_number(actual), expected, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError("Reward breakdown disagrees with measured scores")


def _leaf(name: str, value: Any, **extra: Any) -> dict[str, Any]:
    return {"name": name, "operator": "leaf", "value": _number(value), **extra}


def _weighted(
    name: str, terms: Sequence[tuple[float, dict[str, Any]]]
) -> dict[str, Any]:
    value = math.fsum(weight * child["value"] for weight, child in terms if weight)
    children = []
    for weight, child in terms:
        contribution = weight * child["value"] if weight else 0.0
        children.append(
            {
                **child,
                "weight": weight,
                "contribution_to_parent": contribution,
                "share_of_parent": contribution / value if value else None,
            }
        )
    return {
        "name": name,
        "operator": "weighted_sum",
        "value": value,
        "children": children,
    }


def _product(name: str, children: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": name,
        "operator": "product",
        "value": math.prod(child["value"] for child in children),
        "children": children,
    }


def _measured(node: dict[str, Any], value: Any) -> dict[str, Any]:
    """Check arithmetic, then preserve the authoritative measurement verbatim."""
    _same(value, node["value"])
    node["value"] = _number(value)
    if node["operator"] == "weighted_sum":
        for child in node["children"]:
            child["share_of_parent"] = (
                child["contribution_to_parent"] / node["value"]
                if node["value"]
                else None
            )
    return node


def build_fine64_reward_breakdown(result: Mapping[str, Any]) -> dict[str, Any]:
    """Export the measured mean80/min20 fine64 formula without rescoring it.

    Requires the full verifier result, including actual per-part and visual
    measurements. The seven API ``sub_scores`` alone are insufficient. Missing
    measurements or another scoring policy raise instead of fabricating values.
    All weights/contributions/shares refer to the immediate parent. Product
    factors have values, but no additive weights or shares.
    """
    if result.get("schema") != "building-reward-v1-overall-90-10":
        raise ValueError("Unsupported fine64 measurement schema")
    mode = result.get("score_mode")
    if mode not in _MODES:
        raise ValueError("Unsupported fine64 score mode")
    if mode in {"HARD_INVALID", "RULE_UNSCORABLE"}:
        expected_status = (
            "INVALID_SUBMISSION" if mode == "HARD_INVALID" else "EVAL_FAILED"
        )
        expected_score = 0.0 if mode == "HARD_INVALID" else None
        if (
            result.get("status") != expected_status
            or result.get("score") != expected_score
            or isinstance(result.get("score"), bool)
        ):
            raise ValueError("Inconsistent unscored/invalid fine64 outcome")
        return {
            "schema": SCHEMA,
            "score_mode": mode,
            "root": {
                "name": "total",
                "operator": "terminal",
                "value": expected_score,
                "status": expected_status,
            },
        }
    if result.get("status") != "OK":
        raise ValueError("Successful fine64 breakdown requires an OK result")
    rule = result["rule_result"]
    aggregation = rule["aggregation"]
    if aggregation.get("policy") != "mean80-min20-v1":
        raise ValueError("Unsupported fine64 aggregation policy")
    parts = rule["required_role_diagnostics"]
    if not parts or len(parts) > (_MAX_NODES - 16) // 4:
        raise ValueError("Expected bounded, nonempty measured fine64 parts")
    weights = [_number(part["weight"]) for part in parts]
    if any(weight <= 0 for weight in weights):
        raise ValueError("Part weights must be positive")
    weight_sum = math.fsum(weights)
    part_terms = []
    for part, weight in zip(parts, weights):
        node = _weighted(
            part["id"],
            [
                (0.25, _leaf("layout", part["layout"])),
                (0.35, _leaf("surface", part["surface"])),
                (0.40, _leaf("view", part["view"])),
            ],
        )
        _measured(node, part["quality"])
        node["declared_weight"] = weight
        part_terms.append((weight / weight_sum, node))
    mean = _measured(_weighted("mean", part_terms), aggregation["mean"])
    minimum = min(child["value"] for _, child in part_terms)
    minimum_node = _leaf(
        "minimum",
        minimum,
        selected_part_ids=[
            child["name"] for _, child in part_terms if child["value"] == minimum
        ],
    )
    quality = _weighted("quality", [(0.8, mean), (0.2, minimum_node)])
    _same(aggregation["minimum"], minimum)
    _same(aggregation["mean_weight"], 0.8)
    _same(aggregation["minimum_weight"], 0.2)
    subs = rule["sub_scores"]
    _measured(quality, subs["semantic_Q"])
    _same(subs["minimum_q"], minimum)
    anomaly = _number(subs["anomaly_A"])
    local_error = _number(subs["local_E"])
    excess = _number(rule["full_scene_geometry"]["excess_ratio"])
    if not 0 <= anomaly <= 1 or not 0 <= local_error <= 1 or excess < 0:
        raise ValueError("Fine64 penalties outside their declared ranges")
    x = max(0.0, local_error - 0.25)
    anomaly_penalty = 0.15 * anomaly
    local_penalty = 0.20 * x / (0.20 + x)
    retention = _leaf("retention", 1.0 - anomaly_penalty - local_penalty)
    retention["diagnostics"] = {
        "anomaly_A": anomaly,
        "local_E": local_error,
        "anomaly_penalty": anomaly_penalty,
        "local_penalty": local_penalty,
    }
    factor = _product(
        "factor", [retention, _leaf("scene_excess", 1.0 / (1.0 + excess))]
    )
    _same(subs["scene_excess_factor"], factor["children"][1]["value"])
    _measured(factor, subs["factor_F"])
    rule_node = _measured(_product("rule", [quality, factor]), rule["score"])
    rule_node["declared_weight"] = 0.9
    _same(result["rule_score"], rule_node["value"])
    if mode == "MIXED_90_10":
        visual = result["visual"]
        expected_items = {
            "overall_shape",
            "major_proportions",
            "mass_layout",
            "large_completeness",
        }
        items = visual["items"]
        if (
            result.get("vlm_status") != "COMPLETE"
            or visual.get("status") != "COMPLETE"
            or len(items) != 4
            or {item["id"] for item in items} != expected_items
        ):
            raise ValueError("Mixed fine64 reward requires four measured VLM items")
        for item in items:
            _same(item["weight"], 0.25)
            if type(item["value"]) is not int or item["value"] not in (0, 1):
                raise ValueError("VLM item must be an actual binary vote result")
        vlm = _weighted(
            "vlm", [(0.25, _leaf(item["id"], item["value"])) for item in items]
        )
        vlm["declared_weight"] = 0.1
        _measured(vlm, visual["vlm_score"])
        _same(result["vlm_score"], vlm["value"])
        root = _weighted("total", [(0.9, rule_node), (0.1, vlm)])
    else:
        if result.get("vlm_status") == "COMPLETE":
            raise ValueError("Complete VLM cannot be a rule-only fallback")
        root = _weighted(
            "total",
            [
                (1.0, rule_node),
                (
                    0.0,
                    {
                        "name": "vlm",
                        "operator": "leaf",
                        "value": None,
                        "declared_weight": 0.1,
                        "excluded": True,
                    },
                ),
            ],
        )
    _measured(root, result["score"])
    breakdown = {"schema": SCHEMA, "score_mode": mode, "root": root}
    _breakdown_metrics(breakdown, result["score"])
    return breakdown


def _breakdown_metrics(breakdown: Any, score: float | None) -> dict[str, float]:
    if not isinstance(breakdown, Mapping) or breakdown.get("schema") != SCHEMA:
        raise ValueError("Unknown reward breakdown schema")
    mode = breakdown.get("score_mode")
    if mode not in _MODES:
        raise ValueError("Unknown reward breakdown mode")
    metrics = {f"reward_breakdown/mode/{name}": float(mode == name) for name in _MODES}
    nodes = 0

    def visit(node: Any, path: str, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_NODES or depth > _MAX_DEPTH or not isinstance(node, Mapping):
            raise ValueError("Reward breakdown exceeds bounded tree contract")
        name = node.get("name")
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("Invalid reward component name")
        prefix = f"{path}/{name}"
        value = node.get("value")
        if value is not None:
            value = _number(value)
            if not 0 <= value <= 1:
                raise ValueError("Reward component score outside [0,1]")
            metrics[f"{prefix}/score"] = value
        metrics[f"{prefix}/computed"] = float(value is not None)
        for field in ("weight", "declared_weight", "contribution_to_parent"):
            if field in node:
                number = _number(node[field])
                if number < 0:
                    raise ValueError("Reward weights/contributions cannot be negative")
                metrics[f"{prefix}/{field}"] = number
        # These are means of per-episode ratios, not ratios of batch sums.
        share = node.get("share_of_parent")
        if share is not None:
            metrics[f"{prefix}/share_of_parent_mean"] = _number(share)
        if "share_of_parent" in node:
            metrics[f"{prefix}/share_computed"] = float(share is not None)
        diagnostics = node.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping) or len(diagnostics) > 16:
            raise ValueError("Invalid reward diagnostics")
        for key, diagnostic in diagnostics.items():
            if not isinstance(key, str) or not _NAME.fullmatch(key):
                raise ValueError("Invalid diagnostic name")
            metrics[f"{prefix}/diagnostic/{key}"] = _number(diagnostic)
        children = node.get("children", [])
        if not isinstance(children, list) or len(children) > 64:
            raise ValueError("Invalid reward children")
        operator = node.get("operator")
        if operator in {"leaf", "terminal"}:
            if children:
                raise ValueError("Leaf components cannot have children")
        elif operator in {"weighted_sum", "product"}:
            if not children or value is None:
                raise ValueError("Aggregation requires children and measured value")
            if any(not isinstance(child, Mapping) for child in children):
                raise ValueError("Invalid reward child")
            if operator == "weighted_sum":
                weights = [_number(child["weight"]) for child in children]
                _same(math.fsum(weights), 1.0)
                contributions = []
                for weight, child in zip(weights, children):
                    contribution = weight * _number(child["value"]) if weight else 0.0
                    _same(child["contribution_to_parent"], contribution)
                    share = child.get("share_of_parent")
                    if value:
                        _same(share, contribution / value)
                    elif share is not None:
                        raise ValueError("Zero-score share must be null")
                    contributions.append(contribution)
                _same(value, math.fsum(contributions))
            else:
                if any(
                    field in child
                    for child in children
                    for field in ("weight", "contribution_to_parent", "share_of_parent")
                ):
                    raise ValueError("Product factors cannot carry additive shares")
                _same(value, math.prod(_number(child["value"]) for child in children))
        else:
            raise ValueError("Unknown reward operator")
        names = [child.get("name") for child in children if isinstance(child, Mapping)]
        if len(set(names)) != len(children):
            raise ValueError("Reward siblings must have unique names")
        for child in children:
            visit(child, prefix, depth + 1)

    root = breakdown["root"]
    visit(root, "reward_breakdown", 0)
    if root["name"] != "total":
        raise ValueError("Reward root must be total")
    if mode in {"HARD_INVALID", "RULE_UNSCORABLE"}:
        expected = 0.0 if mode == "HARD_INVALID" else None
        if root.get("operator") != "terminal" or root.get("value") != expected:
            raise ValueError("Invalid terminal reward mode")
    elif root.get("operator") != "weighted_sum" or root.get("value") is None:
        raise ValueError("Successful reward mode requires a measured sum")
    if (score is None) != (root.get("value") is None):
        raise ValueError("Reward total missing or unexpectedly measured")
    if score is not None and not math.isclose(
        _number(score), root["value"], rel_tol=0.0, abs_tol=0.000051
    ):
        # Arena's current API rounds its score to four decimal places.
        raise ValueError("Reward breakdown total disagrees with Arena score")
    return metrics


def arena_reward_metrics(
    *, sub_scores: Any, raw: Any, score: float | None
) -> dict[str, float]:
    """Read only declared telemetry, never discover/replace the training reward.

    Bad optional telemetry is observable and omitted; it cannot reject an
    otherwise valid rollout. Names and tree size are bounded before publication.
    """
    metrics: dict[str, float] = {}
    invalid = 0
    if isinstance(sub_scores, Mapping) and len(sub_scores) <= 128:
        for key, value in sub_scores.items():
            try:
                if not isinstance(key, str) or not _NAME.fullmatch(key):
                    raise ValueError("Invalid sub-score name")
                metrics[f"reward_sub_score/{key}"] = _number(value)
            except (TypeError, ValueError, OverflowError):
                invalid += 1
    elif sub_scores is not None:
        invalid += 1
    metrics["reward_sub_score/invalid_entries"] = float(invalid)
    breakdown = raw.get("reward_breakdown") if isinstance(raw, Mapping) else None
    metrics["reward_breakdown/present"] = float(breakdown is not None)
    metrics["reward_breakdown/valid"] = 0.0
    if breakdown is not None:
        try:
            metrics.update(_breakdown_metrics(breakdown, score))
            metrics["reward_breakdown/valid"] = 1.0
        except (KeyError, TypeError, ValueError, OverflowError):
            pass
    return metrics
