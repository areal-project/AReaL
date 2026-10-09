# SPDX-License-Identifier: Apache-2.0
"""Dependency-free configuration helpers for separation DTE updates."""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from enum import Enum
from typing import Any


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _backend_name(value: Any) -> str:
    return str(_enum_value(value)).split(":", maxsplit=1)[0].lower()


def apply_dte_config_envvars(
    config: Any,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Validate delta weight-transfer config and propagate runtime switches."""
    if not getattr(config.actor, "enable_delta_weight_update", False):
        return {}

    topology = _enum_value(config.rollout.scheduling_strategy.type)
    if topology != "separation":
        raise ValueError(
            "actor.enable_delta_weight_update=true is currently supported only "
            "with rollout scheduling strategy 'separation', "
            f"got {topology!r}"
        )
    delta_method = config.actor.weight_update_delta_method
    if delta_method != "adamw":
        raise ValueError("actor.weight_update_delta_method must be 'adamw'")
    anchor_interval = config.actor.weight_update_anchor_interval
    if anchor_interval < 0:
        raise ValueError("actor.weight_update_anchor_interval must be non-negative")

    actor_backend = _backend_name(config.actor.backend)
    if actor_backend != "megatron":
        raise ValueError(
            "delta weight transfer currently requires actor.backend='megatron:*'; "
            f"got {config.actor.backend!r}"
        )
    rollout_backend = _backend_name(config.rollout.backend)
    if rollout_backend != "sglang":
        raise ValueError(
            "delta weight transfer currently requires rollout.backend='sglang:*'; "
            f"got {config.rollout.backend!r}"
        )
    if config.actor._version != "v2":
        raise ValueError(
            "delta weight transfer currently requires actor._version='v2'; "
            f"got {config.actor._version!r}"
        )
    if config.rollout._version != "v2":
        raise ValueError(
            "delta weight transfer currently requires rollout._version='v2'; "
            f"got {config.rollout._version!r}"
        )
    if config.actor.weight_update_mode != "awex":
        raise ValueError(
            "delta weight transfer currently requires "
            "actor.weight_update_mode='awex'; "
            f"got {config.actor.weight_update_mode!r}"
        )
    if config.actor.use_lora:
        raise ValueError("delta weight transfer does not support actor.use_lora=True")

    exported_env = {
        # Select the separation-specific AWEX update path. Together with
        # DTE_DELTA_TRANSFER, this enables the DTE runtime protocol.
        "DTE_SEPARATION_WEIGHT_UPDATE": "1",
        # Send sparse deltas between full synchronization anchors.
        "DTE_DELTA_TRANSFER": "1",
        # Force a full sync after N committed deltas; zero disables this anchor.
        "DTE_DELTA_ANCHOR_INTERVAL": str(anchor_interval),
        # Reconstruct pre-step tensors lazily during Hugging Face conversion.
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
