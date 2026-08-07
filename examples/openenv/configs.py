# SPDX-License-Identifier: Apache-2.0
"""Training config for OpenEnv-driven RL runs.

Extends :class:`GRPOConfig` with a nested :class:`OpenEnvConfig` and a couple
of workflow-specific knobs. The trainer instantiates
:class:`areal.workflow.openenv.OpenEnvWorkflow` from these values directly, so
no external YAML plumbing is needed to add a new environment beyond editing
this file's ``openenv`` block.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig
from areal.api.openenv_api import OpenEnvConfig


@dataclass
class OpenEnvExperimentConfig(GRPOConfig):
    """GRPO config with an OpenEnv environment attached."""

    openenv: OpenEnvConfig = field(
        default_factory=lambda: OpenEnvConfig(
            env_client_class="echo_env.EchoEnv",
            base_url="https://openenv-echo-env.hf.space",
        ),
        metadata={"help": "OpenEnv environment/adapter configuration."},
    )
    initial_user_prompt: str = field(
        default="",
        metadata={
            "help": "Optional prompt to send to the model before the first observation."
        },
    )
    dataset_size: int = field(
        default=1024,
        metadata={
            "help": "Number of synthetic seed rows in the in-memory dataset. "
            "Each row spawns one episode."
        },
    )
    eval_dataset_size: int = field(
        default=64,
        metadata={"help": "Number of synthetic seed rows in the eval dataset."},
    )
