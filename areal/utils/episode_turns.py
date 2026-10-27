# SPDX-License-Identifier: Apache-2.0
"""Episode generation observations, independent of retained training rows."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from areal.utils.stats_tracker import DistributedStatsTracker

from areal.utils.harness_metrics import HARNESS_OUTCOME_CODES, classify_harness_outcome


@dataclass
class _Episode:
    count: int | None = None
    metadata: dict[str, Any] | None = None
    require_status: bool = False
    delegated: bool = False


_EPISODE: ContextVar[_Episode | None] = ContextVar("episode_turns", default=None)


def count_generated_responses(interactions: Mapping[str, Any]) -> int | None:
    """Count raw unique responses before export removes retries/ancestors.

    An empty cache is an observed zero. Non-token/external API responses without
    generation metadata are unknown, rather than fabricated zero-turn episodes.
    """
    responses = [v.model_response for v in interactions.values()]
    if any(response is None for response in responses):
        return None
    return sum(response.output_len > 0 for response in responses)


def observe_generated_response(output_len: int) -> None:
    episode = _EPISODE.get()
    if episode is not None:
        episode.count = (episode.count or 0) + int(output_len > 0)


def set_episode_turns(
    count: int | None,
    metadata: dict[str, Any] | None = None,
    *,
    require_status: bool = False,
) -> None:
    episode = _EPISODE.get()
    if episode is not None:
        episode.count = count if isinstance(count, int) and count >= 0 else None
        episode.metadata = metadata
        episode.require_status = require_status


def is_turn_distribution(key: str) -> bool:
    name = key.split("/", 1)[-1]
    return name in {
        "num_turns",
        "num_turns_no_harness_err",
        "num_turns_harness_err",
    } or name.startswith("num_turns_harness_err/")


def publish_turn_distribution(
    key: str, values: list[float], *, tracker: DistributedStatsTracker | None = None
) -> None:
    import torch

    from areal.utils import stats_tracker
    from areal.utils.stats_tracker import ReduceType

    if tracker is None:
        tracker = stats_tracker.get()
    value = torch.tensor(values, dtype=torch.float32)
    # stat() revalidates all previous tensor/denominator pairs on every call.
    # Consumption can accumulate many rejected attempts before one export, so
    # retain per-call summaries without revisiting the window's history.
    tracker.stat_compact(
        torch.ones_like(value, dtype=torch.bool),
        ReduceType.AVG_MIN_MAX,
        **{key: value},
    )
    tracker.stat_compact(
        torch.ones(1, dtype=torch.bool),
        ReduceType.SUM,
        **{f"{key}_count": torch.tensor([len(values)], dtype=torch.float64)},
    )


def record_episode_turns(
    count: int | None,
    metadata: dict[str, Any] | None = None,
    *,
    require_status: bool = False,
) -> None:
    """Publish one logical episode inside the enclosing task's metric capture."""
    import torch

    from areal.infra import workflow_context
    from areal.utils import stats_tracker
    from areal.utils.scalar_capture import capture_scalar
    from areal.utils.stats_tracker import ReduceType

    parent = _EPISODE.get()
    if parent is not None:
        parent.delegated = True
    scope = workflow_context.stat_scope()
    missing_key = f"{scope}/num_turns_missing_count"
    if not capture_scalar(missing_key, float(count is None)):
        stats_tracker.get().stat_compact(
            torch.ones(1, dtype=torch.bool),
            ReduceType.SUM,
            **{missing_key: torch.tensor([float(count is None)])},
        )
    if count is None:
        return
    names = ["num_turns"]
    outcome = classify_harness_outcome(metadata, require_status=require_status)
    if outcome > 0:
        code = HARNESS_OUTCOME_CODES[outcome - 1]
        names.extend(["num_turns_harness_err", f"num_turns_harness_err/{code}"])
    elif outcome == 0:
        names.append("num_turns_no_harness_err")
    for name in names:
        key = f"{scope}/{name}"
        if not capture_scalar(key, float(count)):
            publish_turn_distribution(key, [float(count)])


@contextmanager
def track_episode_turns() -> Iterator[None]:
    """Nested grouped members replace the outer task observation, never add to it."""
    parent = _EPISODE.get()
    if parent is not None:
        parent.delegated = True
    episode = _Episode()
    token = _EPISODE.set(episode)
    try:
        yield
    finally:
        _EPISODE.reset(token)
        if not episode.delegated:
            record_episode_turns(
                episode.count, episode.metadata, require_status=episode.require_status
            )
