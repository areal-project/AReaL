# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

from .protocol import EnvBackend

RewardMode = Literal["episode", "dense"]
MissingRewardPolicy = Literal["reject", "ignore", "zero", "terminal_fallback"]
TruncationPolicy = Literal["drop", "retain_terminal", "retain_zero"]


class EnvRolloutRejected(RuntimeError):
    """A rollout policy declined to export the episode for training."""


@dataclass(slots=True)
class EnvRolloutResult:
    """Environment outcome and reward in the form consumed by the proxy."""

    reward: float | dict[str, float] | None
    turns: int
    terminated: bool
    truncated: bool
    stop_reason: str | None

    def proxy_reward(self) -> float | dict[str, float]:
        if self.reward is None:
            raise EnvRolloutRejected(
                f"Environment rollout was dropped (truncated={self.truncated}, "
                f"stop_reason={self.stop_reason!r})"
            )
        return self.reward


def _resolve_rewards(
    rewards: dict[str, float | None],
    reward_mode: RewardMode,
    missing_reward_policy: MissingRewardPolicy,
) -> float | dict[str, float] | None:
    terminal_reward = rewards[next(reversed(rewards))]
    if reward_mode == "episode":
        return terminal_reward

    dense = {}
    for completion_id, reward in rewards.items():
        if reward is not None:
            dense[completion_id] = reward
        elif missing_reward_policy == "zero":
            dense[completion_id] = 0.0
        elif (
            missing_reward_policy == "terminal_fallback" and terminal_reward is not None
        ):
            dense[completion_id] = terminal_reward
        elif missing_reward_policy != "ignore":
            return None
    return dense


async def run_environment(
    backend: EnvBackend,
    task_data: Mapping[str, Any],
    decide: Callable[[Any], Awaitable[tuple[str, Any]]],
    *,
    max_turns: int,
    reward_mode: RewardMode = "episode",
    missing_reward_policy: MissingRewardPolicy = "reject",
    truncation_policy: TruncationPolicy = "drop",
    seed: int | None = None,
) -> EnvRolloutResult:
    """Run one session and associate each reward with its generating completion.

    ``episode`` uses the final step's reward. Callers passing a ``dense`` reward
    map to the proxy must use individual export, which handles omitted assignments
    during reward discounting. ``ignore`` omits missing entries, returning an empty
    map if every entry is missing. ``terminal_fallback`` fills missing entries with
    the final step's reward when available.

    ``retain_terminal`` requires a reward on the last step of a truncated episode.
    ``retain_zero`` assigns zero to that step and still applies the missing-reward
    policy to earlier dense steps.
    """
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    if reward_mode not in ("episode", "dense"):
        raise ValueError(f"Unknown reward mode: {reward_mode!r}")
    if missing_reward_policy not in ("reject", "ignore", "zero", "terminal_fallback"):
        raise ValueError(f"Unknown missing reward policy: {missing_reward_policy!r}")
    if truncation_policy not in ("drop", "retain_terminal", "retain_zero"):
        raise ValueError(f"Unknown truncation policy: {truncation_policy!r}")

    rewards: dict[str, float | None] = {}
    async with await backend.acquire(task_data) as session:
        observation = await session.reset(seed=seed)
        for _ in range(max_turns):
            completion_id, action = await decide(observation)
            if completion_id in rewards:
                raise ValueError(f"Duplicate completion ID: {completion_id!r}")
            step = await session.step(action)
            if step.error is not None:
                raise RuntimeError(f"Environment step failed: {step.error}")
            if step.reward is not None and not isfinite(step.reward):
                raise ValueError("Environment reward must be finite")
            rewards[completion_id] = None if step.reward is None else float(step.reward)
            if step.terminated or step.truncated:
                break
            observation = step.observation

    terminated = step.terminated
    truncated = step.truncated or not terminated
    stop_reason = step.stop_reason if terminated or step.truncated else "max_turns"
    rewards_for_proxy = None
    if not truncated or truncation_policy != "drop":
        if truncated and truncation_policy == "retain_zero":
            rewards[completion_id] = 0.0
        if not truncated or rewards[completion_id] is not None:
            rewards_for_proxy = _resolve_rewards(
                rewards, reward_mode, missing_reward_policy
            )
    return EnvRolloutResult(
        rewards_for_proxy, len(rewards), terminated, truncated, stop_reason
    )
