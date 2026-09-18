# SPDX-License-Identifier: Apache-2.0

"""Branch-isolated scoring shared by the v1 and v2 trajectory exporters."""

from dataclasses import dataclass, field

from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.reward.prm.runner import PRMRunner, PRMTurnResult, record_prm_results
from areal.utils import stats_tracker


@dataclass
class PRMExportStats:
    """Committed session observations, independent of subsequent group filtering.

    Transport the existing typed observations, not averages of worker averages.
    Recording them at the consumer reuses the v1 metric names and reductions.
    Shared ancestor occurrences deliberately count once per exported branch.
    """

    turns: list[PRMTurnResult] = field(default_factory=list)
    trajectory_rewards: list[tuple[str, float]] = field(default_factory=list)

    def extend(self, other: "PRMExportStats") -> None:
        self.turns.extend(other.turns)
        self.trajectory_rewards.extend(other.trajectory_rewards)

    def record(self, *, is_eval: bool) -> None:
        record_prm_results(self.turns, is_eval=is_eval)
        tracker = stats_tracker.get("eval-rollout" if is_eval else "rollout")
        for scorer_name, reward in self.trajectory_rewards:
            tracker.scalar(**{f"prm_trajectory_reward/{scorer_name}": reward})


async def score_prm_branches(
    interactions: dict[str, InteractionWithTokenLogpReward],
    runner: PRMRunner,
    *,
    session_id: str,
    is_eval: bool,
) -> tuple[dict[str, InteractionWithTokenLogpReward], PRMExportStats]:
    """Score complete concat branches without mutating inputs or publishing stats.

    Return observations only after every branch of the session succeeds. A later
    group rejection can discard the trajectories without changing what these
    observations mean: successful session scoring, not completed training.
    """
    scored: dict[str, InteractionWithTokenLogpReward] = {}
    stats = PRMExportStats()
    for leaf_id, leaf in interactions.items():
        cache, cloned_leaf = InteractionCache.clone_chain(leaf, session_id=session_id)
        if cloned_leaf.interaction_id != leaf_id:
            raise ValueError(
                "PRM exported leaf ID mismatch: "
                f"mapping key {leaf_id!r}, interaction ID "
                f"{cloned_leaf.interaction_id!r}"
            )
        full_messages = list(cloned_leaf.messages or []) + list(
            cloned_leaf.output_message_list or []
        )
        results = await runner.run(
            cache,
            ctx={"messages": full_messages},
            is_eval=is_eval,
            record_metrics=False,
        )
        stats.turns.extend(results)
        totals: dict[str, float] = {}
        for result in results:
            totals[result.scorer_name] = (
                totals.get(result.scorer_name, 0.0) + result.reward
            )
        stats.trajectory_rewards.extend(totals.items())
        scored[leaf_id] = cloned_leaf
    return scored, stats
