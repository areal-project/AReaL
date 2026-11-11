# SPDX-License-Identifier: Apache-2.0

"""SLO triggers, role-history readiness and unchanged-search detection."""

from __future__ import annotations

import math

from areal.elastic.contracts import RollingSLO
from areal.elastic.reconfigurator.reference.forecast import HistoricalReplay
from areal.elastic.reconfigurator.reference.heuristic import POLICY
from areal.elastic.serving import rolling_slo


def search_state(owner, view, evaluator):
    active = [r for r in view.requests if r.completed_at is None]
    recent = [
        t
        for e in view.episodes
        if e.completed_at is not None
        and e.completed_at >= view.now - owner.config.slo_window_seconds
        for t in e.completed_turns
    ]
    statistics = rolling_slo(view, owner.slo, owner.config.slo_window_seconds)
    return (
        view.layout,
        view.leases,
        getattr(evaluator, "source_ids", ()),
        view.actor.waiting,
        owner.forecaster.regime_index,
        owner.forecaster.features,
        (
            len(active),
            sum(max(0, r.input_tokens - r.prefilled) for r in active),
            sum(t.output_tokens for t in recent) / max(1, len(recent)),
            sum(t.tool_seconds or 0 for t in recent) / max(1, len(recent)),
            getattr(evaluator, "delay", 0.0),
            statistics.ttft_quantile_seconds,
            statistics.tpot_quantile_seconds,
        ),
        (
            statistics.fraction is not None
            and statistics.fraction >= owner.slo.minimum_request_fraction,
            owner._trigger(statistics),
        ),
    )


def redundant_search(owner, view, evaluator):
    if owner.config.independent_role_history and owner.allowed_roles(
        view.now
    ) != owner.allowed_roles(owner.last_search):
        return False
    old = owner._last_search_state
    state = owner.search_state(view, evaluator)
    if old is None or state[:5] != old[:5] or state[7] != old[7]:
        return False
    if view.now - owner.last_search >= owner.config.slo_window_seconds:
        return False
    ratio = owner.forecaster.config.change_ratio
    if state[5] is not None and old[5] is not None:
        if state[5].distance(old[5]) >= ratio:
            return False
    return all(
        (
            abs(a - b) / max(abs(a), abs(b), 1e-09) < ratio
            for a, b in zip(state[6], old[6], strict=True)
        )
    )


def history_wait(owner, last_switch: float) -> float:
    history = (
        owner.forecaster.config.history_seconds
        if isinstance(owner.forecaster, HistoricalReplay)
        else 0.0
    )
    wait = owner.config.post_switch_wait_seconds
    # Keep startup warmup independent of the wait after later reconfigurations.
    return max(
        owner.config.cooldown_seconds,
        history if wait is None or not math.isfinite(last_switch) else wait,
    )


def allowed_roles(owner, when: float) -> dict[str, bool]:
    return {
        role: when - max(0.0, completed) >= history_wait(owner, completed)
        for role, completed in owner.last_role_switch.items()
    }


def trigger(owner, stats: RollingSLO) -> str:
    if stats.fraction is None:
        return "none"
    if stats.fraction < owner.config.pressure_request_fraction:
        return "pressure"
    if (
        stats.completed >= owner.config.minimum_completed_requests
        and stats.fraction > owner.config.slack_request_fraction
    ):
        return "slack"
    return "none"


def observed_evidence(view, window):
    """Summarize observed completed turns and current admission backlog."""
    completed = [e for e in view.episodes if e.completed_at is not None]
    recent = [e for e in completed if e.completed_at >= view.now - window]
    turns = [t for e in recent for t in e.completed_turns]
    return dict(
        completed=len(completed),
        waiting=sum(not e.admitted for e in view.episodes),
        shape=(
            sum(t.output_tokens for t in turns) / max(1, len(turns)),
            sum(t.tool_seconds or 0 for t in turns) / max(1, len(turns)),
            sum(len(e.completed_turns) for e in recent) / max(1, len(recent)),
        ),
    )


def evidence_redundant_search(owner, view, evaluator):
    old = owner._retry_evidence
    if old is None:
        owner.retry_reason = "first_search"
        return False
    if view.now - owner.last_search < POLICY["minimum_search_spacing_seconds"]:
        return True
    if (
        sum(
            t > view.now - POLICY["search_rate_window_seconds"]
            for t in owner._retry_times
        )
        >= POLICY["searches_per_rate_window"]
    ):
        return True
    now = observed_evidence(view, owner.config.slo_window_seconds)
    stats = rolling_slo(view, owner.slo, owner.config.slo_window_seconds)
    shape_changed = any(
        abs(a - b) / max(abs(a), abs(b), 1e-9) >= POLICY["shape_change_ratio"]
        for a, b in zip(now["shape"], old["shape"], strict=True)
    )
    reasons = []
    if owner.forecaster.regime_index != old["regime"]:
        reasons.append("detected_regime")
    if (
        shape_changed
        and now["completed"] - old["completed"]
        >= POLICY["minimum_new_completed_episodes"]
    ):
        reasons.append("observed_shape_change")
    if owner._trigger(stats) == "pressure" and old["trigger"] != "pressure":
        reasons.append("new_slo_pressure")
    if now["waiting"] - old["waiting"] >= max(
        evaluator.config.batch_size, old["waiting"] * 0.25
    ):
        reasons.append("admission_queue_growth")
    if (
        not old["target_met"]
        and view.now - owner.last_search >= POLICY["unresolved_retry_seconds"]
        and now["completed"] - old["completed"]
        >= POLICY["minimum_new_completed_episodes"]
    ):
        reasons.append("unresolved_with_new_completed_history")
    owner.retry_reason = "+".join(reasons) if reasons else "unchanged_evidence"
    return not reasons
