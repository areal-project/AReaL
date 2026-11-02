# SPDX-License-Identifier: Apache-2.0

"""SLO triggers, role-history readiness and unchanged-search detection."""

from __future__ import annotations

from areal.elastic.contracts import RollingSLO
from areal.elastic.reconfigurator.reference.forecast import HistoricalReplay
from areal.elastic.serving import rolling_slo


def search_state(owner, view, model):
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
        getattr(model, "source_ids", ()),
        view.actor.waiting,
        owner.forecaster.regime_index,
        owner.forecaster.features,
        (
            len(active),
            sum(max(0, r.input_tokens - r.prefilled) for r in active),
            sum(t.output_tokens for t in recent) / max(1, len(recent)),
            sum(t.tool_seconds or 0 for t in recent) / max(1, len(recent)),
            getattr(model, "delay", 0.0),
            statistics.ttft_quantile_seconds,
            statistics.tbt_quantile_seconds,
        ),
        (
            statistics.fraction is not None
            and statistics.fraction >= owner.slo.minimum_request_fraction,
            owner._trigger(statistics),
        ),
    )


def redundant_search(owner, view, model):
    if owner.config.independent_role_history and owner.allowed_roles(
        view.now
    ) != owner.allowed_roles(owner.last_search):
        return False
    old = owner._last_search_state
    state = owner.search_state(view, model)
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


def allowed_roles(owner, when: float) -> dict[str, bool]:
    seconds = max(
        owner.config.cooldown_seconds,
        owner.forecaster.config.history_seconds
        if isinstance(owner.forecaster, HistoricalReplay)
        else 0.0,
    )
    return {
        role: when - max(0.0, completed) >= seconds
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
