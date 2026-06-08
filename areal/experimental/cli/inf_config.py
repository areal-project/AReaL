# SPDX-License-Identifier: Apache-2.0

"""Config loading for ``areal inf`` — reuses the training yaml schema.

`areal inf run` deliberately does NOT introduce a separate inference yaml.
The training yaml already describes a complete v2 microservice deployment
(scheduler, rollout backend, parallelism, image, ...); for standalone
inference we just ignore the train/actor sections and act on `rollout`
and `scheduler`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from areal.api.cli_args import PPOConfig


def load_inference_config(
    config_path: Path, overrides: list[str]
) -> tuple["PPOConfig", str]:
    """Load the training yaml and resolve the service name.

    Returns the parsed PPOConfig (so the supervisor can reach .scheduler,
    .rollout, etc. exactly the way PPOTrainer does) plus the resolved
    service name in the form ``{experiment_name}/{trial_name}``.
    """
    from areal.api.cli_args import PPOConfig, load_expr_config

    argv = ["--config", str(config_path), *overrides]
    cfg, _ = load_expr_config(argv, PPOConfig)
    service_name = f"{cfg.experiment_name}/{cfg.trial_name}"
    return cfg, service_name
