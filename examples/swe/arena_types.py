"""Lightweight Arena configuration types shared by preflight and training."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArenaRewardRefConfig:
    """Expected Arena reward implementation pinned for one Stream."""

    key: str = ""
    version: str = ""


@dataclass
class ArenaStreamConfig:
    """Per-Stream routing, mixture weighting, and reward configuration."""

    name: str = ""
    stream_id: str = ""
    sampling_weight: float = 1.0
    harness: str = ""
    llm_protocol: str = ""
    task_envs: dict[str, str] = field(default_factory=dict)
    expected_reward_ref: ArenaRewardRefConfig = field(
        default_factory=ArenaRewardRefConfig
    )
    reward_threshold: float | None = None
    reward_transform_fn: str = ""
    # Shared snapshot of per-data-id rewards and optional Harness bindings.
    # An explicit Stream harness overrides per-task pins; arena_harness is a fallback.
    # Created before training if absent; reused and checked on subsequent starts.
    # Mutually exclusive with expected_reward_ref (the Stream-default mode).
    reward_mapping_file: str = ""
    # Preserve inherited arena_harness provenance across controller/worker serialization.
    _harness_is_fallback: bool = False

    @property
    def harness_override(self) -> str:
        """Return the explicit Stream Harness, excluding the common fallback."""
        return "" if self._harness_is_fallback else self.harness
