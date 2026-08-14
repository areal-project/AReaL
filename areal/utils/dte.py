# SPDX-License-Identifier: Apache-2.0
"""Dependency-free configuration helpers for separation DTE updates."""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from enum import Enum
from typing import Any


def apply_dte_config_envvars(
    config: Any,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Validate ``actor.dte`` and propagate its switches to GPU workers."""
    dte_cfg = getattr(config.actor, "dte", None)
    if dte_cfg is None or not dte_cfg.enabled:
        return {}

    topology = config.rollout.scheduling_strategy.type
    topology = topology.value if isinstance(topology, Enum) else topology
    if topology != "separation":
        raise ValueError(
            "actor.dte is currently supported only with rollout scheduling "
            f"strategy 'separation', got {topology!r}"
        )
    if dte_cfg.transfer != "delta":
        raise ValueError("actor.dte.transfer must be 'delta' when DTE is enabled")
    if dte_cfg.delta_method != "adamw":
        raise ValueError("actor.dte.delta_method must be 'adamw'")
    if dte_cfg.anchor_interval < 0:
        raise ValueError("actor.dte.anchor_interval must be non-negative")

    exported_env = {
        "DTE_SEPARATION_WEIGHT_UPDATE": "1",
        "DTE_DELTA_TRANSFER": "1",
        "DTE_DELTA_ANCHOR_INTERVAL": str(dte_cfg.anchor_interval),
        "DTE_STREAMING_RECONSTRUCT": "1",
    }
    environ = os.environ if environ is None else environ
    environ.update(exported_env)

    for cfg_part in (config.actor, config.rollout):
        for spec in getattr(cfg_part, "scheduling_spec", ()) or ():
            if isinstance(spec, dict):
                spec.setdefault("env_vars", {}).update(exported_env)
                continue
            env_vars = getattr(spec, "env_vars", None)
            if env_vars is None:
                env_vars = {}
                setattr(spec, "env_vars", env_vars)
            env_vars.update(exported_env)

    return exported_env
