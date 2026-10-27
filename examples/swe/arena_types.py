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
    # Per-task Harness pins take precedence over the Stream-level harness.
    # Created before training if absent; reused and checked on subsequent starts.
    # Mutually exclusive with expected_reward_ref (the Stream-default mode).
    reward_mapping_file: str = ""
    # Optional read-only JSON snapshot of ordered Env IDs, task types and hashes.
    # Exact published versions are validated before any workers are created.
    # Without this field, dataset selection continues to use the live Stream.
    dataset_snapshot_file: str = ""
