# SPDX-License-Identifier: Apache-2.0

"""Arena Task event parsing for rollout reward timing."""

import pytest

from examples.swe.arena_client import ArenaOpenAPIClient


def test_reward_phase_intervals_uses_reward_sandbox_to_evaluation_done():
    """The observed Arena reward pipeline supplies the reward boundaries."""
    trace = {
        "events": [
            {
                "attempt": 1,
                "phase": "AGENT_RUNNING",
                "status": "done",
                "timestamp": "2026-11-04T10:36:34.083598Z",
            },
            {
                "attempt": 1,
                "phase": "WAITING_REWARD_SBX_SLOT",
                "status": "started",
                "timestamp": "2026-11-04T10:36:34.643837Z",
            },
            {
                "attempt": 1,
                "phase": "CREATING_REWARD_SBX",
                "status": "started",
                "timestamp": "2026-11-04T10:36:34.761935Z",
            },
            {
                "attempt": 1,
                "phase": "EVALUATING",
                "status": "started",
                "timestamp": "2026-11-04T10:36:35.758531Z",
            },
            {
                "attempt": 1,
                "phase": "EVALUATING",
                "status": "done",
                "timestamp": "2026-11-04T10:36:42.162641Z",
            },
        ]
    }

    intervals = ArenaOpenAPIClient.reward_phase_intervals(trace)

    assert ArenaOpenAPIClient.latest_attempt_id(trace) == "1"
    assert len(intervals) == 1
    assert intervals[0]["phase"] == "reward"
    assert intervals[0]["attempt_id"] == "1"
    assert intervals[0]["ended_at"] - intervals[0]["started_at"] == pytest.approx(
        7.518804
    )


def test_reward_phase_intervals_missing_reward_events_returns_empty():
    """Tool activity is never inferred from an Arena AGENT_RUNNING event."""
    trace = {
        "events": [
            {
                "attempt": 1,
                "phase": "AGENT_RUNNING",
                "status": "tool_call",
                "timestamp": "2026-11-04T10:36:34Z",
            }
        ]
    }

    assert ArenaOpenAPIClient.reward_phase_intervals(trace) == []
