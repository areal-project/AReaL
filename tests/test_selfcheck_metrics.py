"""Selfcheck curves use the same completed-episode cohort at every check."""

import json
from types import SimpleNamespace

import httpx
import pytest

from examples.swe.arena_client import ArenaOpenAPIClient, ArenaTaskResult

from areal.utils.rollout_metrics import (
    ConsumedMetrics,
    capture_metrics,
    capture_scalar,
)


async def _read_checks(audit, budget):
    def handler(request):
        name = request.url.path.rsplit("/", 1)[-1]
        if name == "artifacts":
            names = ["checker-audit.json"]
            if budget is not None:
                names.append("checker-budget.json")
            return httpx.Response(
                200,
                json={
                    "items": [{"name": [], "role": "harness", "attempt": 1}]
                    + [
                        {"name": name, "role": "harness", "attempt": 1}
                        for name in names
                    ]
                },
            )
        assert request.url.params["attempt"] == "1"
        return httpx.Response(
            200,
            content=json.dumps(budget if name == "checker-budget.json" else audit),
            headers={"Content-Type": "application/json"},
        )

    api = ArenaOpenAPIClient(base_url="https://arena.example", api_token="test-token")
    result = ArenaTaskResult(
        task_id="job:env:0",
        status="DONE",
        score=1.0,
        artifacts_uri="tasks/job:env:0/attempts/1/",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await api.get_selfcheck_metrics(result, client=client)


async def _episode(scores, stream="a", trajectory=None, max_calls=3):
    telemetry = await _read_checks(
        [{"response": {"status": "OK", "score": score}} for score in scores],
        {"used_calls": len(scores), "max_calls": max_calls},
    )
    with capture_metrics() as metrics:
        for key, value in telemetry.items():
            assert capture_scalar(f"rollout/stream/{stream}/selfcheck/{key}", value)
    return SimpleNamespace(trajectory=trajectory, metrics=metrics)


@pytest.mark.asyncio
async def test_selfcheck_fills_early_stops_and_preserves_actual_regressions():
    """A one-check episode contributes to every later observed check slot."""
    consumed = ConsumedMetrics()
    for scores in ([1.0], [0.0, 1.0], [0.0, 1.0, 0.0]):
        consumed.consume(await _episode(scores))
    # Prefetch has completed, but its four-check history is not consumed yet.
    await _episode([0.0, 0.0, 0.0, 1.0], max_calls=4)

    stats = consumed.export()
    prefix = "rollout/stream/a/selfcheck"
    assert stats[f"{prefix}/calls"] == 2.0
    for index, score in [(1, 1 / 3), (2, 1), (3, 2 / 3)]:
        scope = f"{prefix}/check_{index}"
        assert stats[f"{scope}/score"] == pytest.approx(score)
        assert stats[f"{scope}/score__count"] == 3
    assert not any(
        "report_" in key
        or "score_filled" in key
        or "_last_score" in key
        or "check_4" in key
        or "actual_count" in key
        or "filled_count" in key
        for key in stats
    )
    assert consumed.export() == {}


@pytest.mark.asyncio
async def test_selfcheck_uses_harness_budget_even_when_no_episode_reaches_it():
    """Each stream keeps its configured curve length across consumption windows."""
    consumed = ConsumedMetrics()
    for scores, stream, maximum in [
        ([1], "a", 3),
        ([0, 1, 0], "a", 3),
        ([0.25], "org/b", 2),
    ]:
        consumed.consume(await _episode(scores, stream, max_calls=maximum))
    stats = consumed.export()
    assert stats["rollout/stream/a/selfcheck/check_3/score"] == 0.5
    assert stats["rollout/stream/org/b/selfcheck/check_1/score"] == 0.25
    assert stats["rollout/stream/org/b/selfcheck/check_2/score"] == 0.25
    assert stats["rollout/stream/org/b/selfcheck/check_2/score__count"] == 1
    assert not any("org/b/selfcheck/check_3" in key for key in stats)

    consumed.consume(await _episode([0.75]))
    stats = consumed.export()
    assert stats["rollout/stream/a/selfcheck/check_1/score"] == 0.75
    assert stats["rollout/stream/a/selfcheck/check_3/score"] == 0.75
    assert stats["rollout/stream/a/selfcheck/check_3/score__count"] == 1
    assert not any("org/b" in key for key in stats)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "audit,budget,expected",
    [
        ([], {"used_calls": 0, "max_calls": 3}, {"calls": 0.0}),
        ([], None, {}),
        ([{"response": {"status": "OK", "score": 1}}], None, {}),
        (
            [{"response": {"status": "OK", "score": 1}}],
            {"used_calls": 1},
            {"calls": 1.0},
        ),
        (
            [{"response": {"status": "OK", "score": 1}}],
            {"used_calls": 2, "max_calls": 3},
            {"calls": 2.0},
        ),
        (
            [{"response": {"status": "OK", "score": 1}}],
            {"used_calls": 1, "max_calls": 10**100},
            {"calls": 1.0},
        ),
        ([], {"used_calls": 10**400, "max_calls": 10**400}, {}),
        (
            [{"response": {"status": "EVAL_FAILED", "score": 0}}],
            {"used_calls": 1, "max_calls": 3},
            {"calls": 1.0},
        ),
        (
            [{"response": {"status": "OK", "score": None}}],
            {"used_calls": 1, "max_calls": 3},
            {"calls": 1.0},
        ),
        (
            [{"response": {"status": "OK", "score": float("inf")}}],
            {"used_calls": 1, "max_calls": 3},
            {"calls": 1.0},
        ),
    ],
)
async def test_selfcheck_missing_or_failed_history_does_not_become_an_early_stop(
    audit, budget, expected
):
    """Zero checks and incomplete telemetry do not fabricate curve scores."""
    assert await _read_checks(audit, budget) == expected
