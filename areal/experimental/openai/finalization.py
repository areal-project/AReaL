# SPDX-License-Identifier: Apache-2.0

"""Rollout response policy, applied after selection and before reward propagation."""

from collections.abc import Callable

import torch

from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.types import InteractionWithTokenLogpReward

ResponseValidator = Callable[[InteractionWithTokenLogpReward], str | None]


def final_response_error(interaction: InteractionWithTokenLogpReward) -> str | None:
    """Require usable text or tools in the retained final assistant response.

    The client records parsing facts, not task success. Reasoning alone is
    allowed by the API but does not satisfy this rollout termination policy.
    Older or external interactions without parsing facts remain unclassified;
    do not infer a model failure from missing metadata.
    """
    facts = (interaction.metadata or {}).get("response_parse")
    if facts is None:
        return None
    if facts["tool_call_parse_failed"]:
        return "invalid_tool_call"
    if not facts["has_content"] and not facts["has_tool_calls"]:
        return "empty_assistant_response"
    return None


def finalize_interactions(
    cache: InteractionCache,
    *,
    style: str,
    reward_discount: float | None = None,
    drop_retry_orphans: bool = False,
    response_validator: ResponseValidator | None = final_response_error,
    episode_reward: float | None = None,
) -> dict[str, InteractionWithTokenLogpReward]:
    """Select retained responses, apply task policy, then propagate rewards.

    Concat trajectories are validated only at their completed leaves. In
    individual style each exported response is validated before discounting.
    A caller with a different termination policy can replace or disable the
    validator without changing parsing or cache behavior.

    This must run before discounting, shaping, normalization, or serialization.
    Explicit discounting through ArealOpenAI also uses this entry point; later
    exports preserve its discounted returns instead of applying the gate again.
    """
    if episode_reward is not None and style != "concat":
        raise ValueError("Episode reward assignment requires concat export")
    terminal_id = getattr(cache, "_last_task_reward_id", None)
    interactions = cache.export_interactions(
        style=style, drop_retry_orphans=drop_retry_orphans
    )
    if cache.reward_discount_applied:
        return interactions

    episode_scope = False
    if episode_reward is not None and interactions:
        # One explicit task return may span independent compaction segments.
        # Identify its terminal response from reward assignment, not leaf order.
        if terminal_id not in interactions:
            scored = [
                key for key, value in interactions.items() if value.reward is not None
            ]
            terminal_id = (
                next(iter(interactions))
                if len(interactions) == 1
                else scored[0]
                if len(scored) == 1
                else None
            )
        cache = InteractionCache.from_dict(interactions)
        episode_scope = cache._can_assign_episode_reward(episode_reward)
        cache.assign_episode_reward(episode_reward)
        if episode_scope and terminal_id is None and response_validator is not None:
            raise ValueError("Episode response validation requires a final response ID")

    if response_validator is not None:
        if episode_scope:
            error = response_validator(interactions[terminal_id])
            failures = (
                [(key, value, error) for key, value in interactions.items()]
                if error is not None
                else []
            )
        else:
            failures = []
            for interaction_id, interaction in interactions.items():
                error = response_validator(interaction)
                if error is not None:
                    failures.append((interaction_id, interaction, error))
        for interaction_id, interaction, error in failures:
            metadata = dict(interaction.metadata or {})
            metadata["reward_validation_error"] = error
            metadata.setdefault(
                "reward_before_response_validation",
                interaction.original_reward
                if interaction.original_reward is not None
                else interaction.reward,
            )
            if episode_scope:
                metadata["reward_validation_scope"] = "episode"
                metadata["reward_validation_final_response_id"] = terminal_id
                interaction.rollout_reward = 0.0
            interaction.metadata = metadata
            interaction.original_reward = 0.0
            cache.set_reward(interaction_id, 0.0)
            if interaction._cache is not None:
                interaction._cache = {
                    **interaction._cache,
                    "rewards": torch.zeros_like(interaction._cache["rewards"]),
                    "original_rewards": torch.zeros_like(
                        interaction._cache["original_rewards"]
                    ),
                }

    if style == "individual" and reward_discount is not None:
        cache.apply_reward_discount(turn_discount=reward_discount)
    return interactions
