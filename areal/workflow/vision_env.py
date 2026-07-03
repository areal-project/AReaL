# SPDX-License-Identifier: Apache-2.0

"""Environment interface for multi-turn tool-calling vision-language RL.

A `MultiTurnVisionEnv` owns all task-specific semantics (the initial prompt
incl. any system/tool instructions, how to parse the model's action, how to
grade it, the feedback to return, and when the episode terminates). The
`VisionMultiTurnWorkflow` is task-agnostic: it drives generation, calls
`env.step()` each turn, and accumulates the training trajectory.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class EnvResetResult:
    """Initial prompt for turn 0.

    messages_chat : list[dict]
        Full vLLM chat-completions message list for turn 0 (e.g. a system turn
        with tool instructions followed by the user turn with image_url + text
        content blocks). The workflow derives the HF-processor prompt string
        from this same list, so both the token path and the vLLM chat path stay
        consistent. The number of image content blocks must equal len(images).
    images : list
        Images for turn 0 (e.g. dataset ``images``); processed once.
    """

    messages_chat: list[dict[str, Any]]
    images: list[Any]


@dataclass
class EnvStepResult:
    """Result of one ``env.step()``.

    observation : str | None
        Text feedback to append as the next user turn. MUST be pure text (no
        image; image observations are out of scope). None when no further prompt
        is needed (typically when ``done``).
    reward : float
        Scalar reward for this turn (envs decide sparse/terminal semantics).
    done : bool
        True when the episode terminates after this step.
    """

    observation: str | None
    reward: float
    done: bool


class MultiTurnVisionEnv(ABC):
    """Per-episode environment for multi-turn tool-calling VLM RL.

    One instance is created per episode: the workflow resolves ``env_factory``,
    instantiates it with ``env_args``, calls ``reset(data)`` once, then
    ``step(assistant_text)`` per turn. Implementations hold per-episode state
    (ground truth, turn counter, ...). ``env_args`` must be JSON-serializable
    (no callables / live handles) since it travels as workflow kwargs to the
    rollout worker.
    """

    @abstractmethod
    def reset(self, data: dict[str, Any]) -> EnvResetResult:
        """Initialize episode state from a dataset row; return the turn-0 prompt."""
        raise NotImplementedError

    @abstractmethod
    def step(self, assistant_text: str) -> EnvStepResult:
        """Advance one turn given the model's decoded assistant output."""
        raise NotImplementedError

    def get_metrics(self) -> dict[str, float]:
        """Optional per-episode scalar metrics (e.g. {"turns", "acc"})."""
        return {}
