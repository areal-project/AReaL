# SPDX-License-Identifier: Apache-2.0

"""Example-local configuration for offline prefix replay."""

import math
from dataclasses import dataclass, field
from typing import Any

from areal.api.cli_args import GenerationHyperparameters, PPOConfig


def build_prefix_replay_workflow_kwargs(
    gconfig: GenerationHyperparameters,
) -> dict[str, Any]:
    """Let grouped rollout own n_samples, with one action per proxy call."""
    kwargs = gconfig.to_openai_completions_args_dict()
    kwargs["n"] = 1
    kwargs["extra_body"] = {
        **kwargs.get("extra_body", {}),
        "max_total_tokens": gconfig.max_tokens,
    }
    return kwargs


@dataclass
class PrefixReplayConfig:
    """Offline prefix-pool construction options from the ReOPD paper."""

    kappa: float = 0.6
    seed: int = 42
    input_mode: str = "auto"
    drop_system_messages: bool = False
    route_metadata_field: str | None = "task"
    default_route: str | None = None
    parse_tool_call_args: bool = False
    cache_processed_dataset: bool = True
    cleanup_processed_dataset: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.kappa, (int, float))
            or isinstance(self.kappa, bool)
            or not math.isfinite(self.kappa)
            or not 0.0 < self.kappa <= 1.0
        ):
            raise ValueError("prefix_replay.kappa must be in (0, 1]")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("prefix_replay.seed must be an integer")
        if self.input_mode not in ("auto", "trajectory", "prefix"):
            raise ValueError(
                "prefix_replay.input_mode must be 'auto', 'trajectory', or 'prefix'"
            )
        if self.route_metadata_field is not None and (
            not isinstance(self.route_metadata_field, str)
            or not self.route_metadata_field.strip()
        ):
            raise ValueError(
                "prefix_replay.route_metadata_field must be a non-empty string or null"
            )
        if self.default_route is not None and (
            not isinstance(self.default_route, str) or not self.default_route.strip()
        ):
            raise ValueError(
                "prefix_replay.default_route must be a non-empty string or null"
            )
        if not isinstance(self.parse_tool_call_args, bool):
            raise TypeError("prefix_replay.parse_tool_call_args must be a boolean")
        if not isinstance(self.cache_processed_dataset, bool):
            raise TypeError("prefix_replay.cache_processed_dataset must be a boolean")
        if not isinstance(self.cleanup_processed_dataset, bool):
            raise TypeError("prefix_replay.cleanup_processed_dataset must be a boolean")


@dataclass
class PrefixReplayOPDConfig(PPOConfig):
    """Use main's MOPD source routing with a replay-specific data loader."""

    prefix_replay: PrefixReplayConfig = field(default_factory=PrefixReplayConfig)
