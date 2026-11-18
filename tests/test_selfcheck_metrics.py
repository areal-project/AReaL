"""Selfcheck curves use the same completed-episode cohort at every check."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from examples.swe.arena_client import (
    SELFCHECK_BUDGET_KEY,
    SELFCHECK_CANDIDATE_CHANGED,
    ArenaOpenAPIClient,
    ArenaTaskResult,
)

from areal.utils.rollout_metrics import (
    ConsumedMetrics,
    capture_metrics,
    capture_scalar,
)


async def _read_checks(
    audit, budget, final=None, *, final_attempt=1, final_delay=0.0, timeout=None
):
    artifacts = {"checker-audit.json": audit}
    if budget is not None:
        artifacts["checker-budget.json"] = budget
    if final is not None:
        artifacts["checker-final.json"] = final

    async def handler(request):
        name = request.url.path.rsplit("/", 1)[-1]
        if name == "artifacts":
            return httpx.Response(
                200,
                json={
                    "items": [{"name": [], "role": "harness", "attempt": 1}]
                    + [
                        {
                            "name": name,
                            "role": "harness",
                            "attempt": final_attempt
                            if name == "checker-final.json"
                            else 1,
                        }
                        for name in artifacts
                    ]
                },
            )
        assert request.url.params["attempt"] == "1"
        if name == "checker-final.json" and final_delay:
            await asyncio.sleep(final_delay)
        return httpx.Response(
            200,
            content=json.dumps(artifacts[name]),
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
        return await api.get_selfcheck_metrics(result, client=client, timeout=timeout)


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
        (
            [],
            {"used_calls": 0, "max_calls": 3},
            {"calls": 0.0, SELFCHECK_BUDGET_KEY: 3.0},
        ),
        ([], None, {}),
        ([{"response": {"status": "OK", "score": 1}}], None, {}),
        (
            [{"response": {"status": "OK", "score": 1}}],
            {"used_calls": 1},
            {"calls": 1.0, SELFCHECK_BUDGET_KEY: 1.0},
        ),
        (
            [{"response": {"status": "OK", "score": 1}}],
            {"used_calls": 2, "max_calls": 3},
            {"calls": 2.0, SELFCHECK_BUDGET_KEY: 3.0},
        ),
        (
            [{"response": {"status": "OK", "score": 1}}],
            {"used_calls": 1, "max_calls": 10**100},
            {"calls": 1.0, SELFCHECK_BUDGET_KEY: 10.0},
        ),
        ([], {"used_calls": 10**400, "max_calls": 10**400}, {}),
        (
            [{"response": {"status": "EVAL_FAILED", "score": 0}}],
            {"used_calls": 1, "max_calls": 3},
            {"calls": 1.0, SELFCHECK_BUDGET_KEY: 3.0},
        ),
        (
            [{"response": {"status": "OK", "score": None}}],
            {"used_calls": 1, "max_calls": 3},
            {"calls": 1.0, SELFCHECK_BUDGET_KEY: 3.0},
        ),
        (
            [{"response": {"status": "OK", "score": float("inf")}}],
            {"used_calls": 1, "max_calls": 3},
            {"calls": 1.0, SELFCHECK_BUDGET_KEY: 3.0},
        ),
    ],
)
async def test_selfcheck_missing_or_failed_history_does_not_become_an_early_stop(
    audit, budget, expected
):
    """Zero checks and incomplete telemetry do not fabricate curve scores."""
    assert await _read_checks(audit, budget) == expected


@pytest.mark.asyncio
async def test_selfcheck_zero_without_budget_requires_final_and_empty_audit():
    final = {"phase": "agent", "hint_calls": 0}
    expected = {"calls": 0.0, SELFCHECK_BUDGET_KEY: 0.0}
    assert await _read_checks([], None, final) == expected
    assert await _read_checks([], None, {**final, "status": "agent_failed"}) == {}
    assert await _read_checks([{"response": {"score": 1}}], None, final) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "used_calls,last_hash,final_hash,final_attempt,expected",
    [
        (2, "b" * 64, "a" * 64, 1, 1.0),
        (2, "b" * 64, "B" * 64, 1, 0.0),
        (3, "b" * 64, "a" * 64, 1, None),
        (0, "b" * 64, "a" * 64, 1, None),
        (2, "invalid", "a" * 64, 1, None),
        (2, "b" * 64, "", 1, None),
        (2, "b" * 64, "a" * 64, 2, None),
    ],
)
async def test_selfcheck_candidate_comparison_uses_last_call_in_agent_attempt(
    used_calls, last_hash, final_hash, final_attempt, expected
):
    audit = (
        [
            {"sha256": "a" * 64, "response": {"status": "OK", "score": 1}},
            {"sha256": last_hash, "response": {"status": "OK", "score": 1}},
        ]
        if used_calls
        else []
    )
    metrics = await _read_checks(
        audit,
        {"used_calls": used_calls, "max_calls": 3},
        {"phase": "agent", "sha256": final_hash},
        final_attempt=final_attempt,
    )
    assert metrics.get(SELFCHECK_CANDIDATE_CHANGED) == expected


@pytest.mark.asyncio
async def test_selfcheck_slow_final_artifact_keeps_existing_metrics():
    audit = [{"sha256": "a" * 64, "response": {"status": "OK", "score": 1}}]
    metrics = await _read_checks(
        audit,
        {"used_calls": 1, "max_calls": 3},
        {"phase": "agent", "sha256": "b" * 64},
        final_delay=1,
        timeout=0.1,
    )
    assert SELFCHECK_CANDIDATE_CHANGED not in metrics
    assert metrics["calls"] == 1.0
    assert metrics["check_3/score"] == 1.0
