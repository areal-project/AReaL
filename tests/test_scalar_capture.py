# SPDX-License-Identifier: Apache-2.0
import asyncio

import pytest

from areal.utils.scalar_capture import capture_scalars
from areal.utils.stats_tracker import DistributedStatsTracker


@pytest.mark.asyncio
async def test_capture_isolates_concurrent_groups_and_inherits_into_children():
    """Async and thread children share only their parent's metrics context."""
    tracker = DistributedStatsTracker("custom")

    async def child(reward):
        await asyncio.sleep(0)
        tracker.scalar(reward=reward)

    async def group(reward):
        with capture_scalars(select=lambda key: key.startswith("custom/")) as metrics:
            await asyncio.gather(
                child(reward), asyncio.to_thread(tracker.scalar, reward=reward)
            )
        return metrics

    first, second = await asyncio.gather(group(0), group(1))
    assert first == {"custom/reward": [0.0, 0.0]}
    assert second == {"custom/reward": [1.0, 1.0]}
    tracker.scalar(reward=0.25)
    assert tracker.export()["custom/reward"] == 0.25


def test_capture_exception_restores_context_and_preserves_observations():
    """A failing attempt retains already recorded metrics without leaking scope."""
    tracker = DistributedStatsTracker("custom")
    with (
        pytest.raises(ValueError),
        capture_scalars(select=lambda key: key.startswith("custom/")) as metrics,
    ):
        tracker.scalar(reward=0)
        raise ValueError("failed")
    assert metrics == {"custom/reward": [0.0]}
    tracker.scalar(reward=1)
    assert tracker.export()["custom/reward"] == 1


def test_nested_capture_restores_selection_and_explicit_groups_bypass():
    tracker = DistributedStatsTracker("custom")
    group = object()
    with capture_scalars(select=lambda key: key == "custom/value") as outer:
        tracker.scalar(value=1)
        with capture_scalars(select=lambda key: True) as inner:
            tracker.scalar(value=2)
            tracker.scalar(reduce_group=group, value=3)
        tracker.scalar(value=4, immediate=5)
    assert outer == {"custom/value": [1.0, 4.0]}
    assert inner == {"custom/value": [2.0]}
    assert tracker.stats["custom/value"] == [3.0]
    assert tracker.reduce_groups["custom/value"] is group
    assert tracker.export(key="immediate")["custom/immediate"] == 5
