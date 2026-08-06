# SPDX-License-Identifier: Apache-2.0
"""Dependency-free helpers for DTE runtime configuration."""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from enum import Enum
from typing import Any


def dte_weight_update_topology_gates(
    enabled: bool, topology: str | Enum
) -> tuple[bool, bool]:
    """Return mutually exclusive ``(colocation, separation)`` DTE gates."""
    topology_value = topology.value if isinstance(topology, Enum) else topology
    if topology_value not in {"colocation", "separation"}:
        raise ValueError(f"Unsupported DTE scheduling topology: {topology_value!r}")
    return (
        bool(enabled) and topology_value == "colocation",
        bool(enabled) and topology_value == "separation",
    )


def apply_dte_config_envvars(
    config: Any,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Export ``actor.dte`` config into process and scheduling environments."""
    dte_cfg = getattr(config.actor, "dte", None)
    if dte_cfg is None:
        return {}

    environ = os.environ if environ is None else environ
    exported_env: dict[str, str] = {}

    def set_env(name: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, bool):
            env_value = "1" if value else "0"
        else:
            env_value = str(value)
        environ[name] = env_value
        exported_env[name] = env_value

    transfer = dte_cfg.transfer
    delta_transfer: bool | None = None
    if transfer is not None:
        transfer = transfer.strip().lower()
        if transfer not in {"full", "delta"}:
            raise ValueError(
                "actor.dte.transfer must be 'full' or 'delta', "
                f"got {dte_cfg.transfer!r}"
            )
        delta_transfer = transfer == "delta"

    delta_method = dte_cfg.delta_method
    detector = None
    if delta_method is not None:
        delta_method = delta_method.strip().lower()
        if delta_method == "adamw":
            detector = "inversion"
        elif delta_method == "snapshot":
            detector = "snapshot"
        else:
            raise ValueError(
                "actor.dte.delta_method must be 'snapshot' or 'adamw', "
                f"got {dte_cfg.delta_method!r}"
            )
    elif delta_transfer:
        detector = "inversion"

    enabled = dte_cfg.enabled
    if enabled is False and delta_transfer:
        raise ValueError(
            "actor.dte.enabled=false conflicts with actor.dte.transfer='delta'. "
            "Use actor.dte.enabled=true for delta transfer, or set "
            "actor.dte.transfer=full."
        )
    if enabled is None and delta_transfer:
        enabled = True

    rollout_strategy = config.rollout.scheduling_strategy.type
    if enabled is not None:
        colocate_enabled, separation_enabled = dte_weight_update_topology_gates(
            enabled, rollout_strategy
        )
        set_env("DTE_COLOCATE_WEIGHT_UPDATE", colocate_enabled)
        set_env("DTE_SEPARATION_WEIGHT_UPDATE", separation_enabled)

    if enabled is not None:
        set_env("DTE_DELTA_TRANSFER", delta_transfer)
        set_env("DTE_DELTA_DETECTOR", detector)
        set_env("DTE_DELTA_ANCHOR_INTERVAL", dte_cfg.anchor_interval)
        set_env("DTE_DELTA_BYTES_RATIO", dte_cfg.bytes_ratio)
        set_env("DTE_DELTA_VERIFY_SNAPSHOT", dte_cfg.verify_snapshot)
    set_env(
        "DTE_RELEASE_TRAIN_WEIGHTS_AFTER_UPDATE",
        dte_cfg.release_train_weights_after_update,
    )
    set_env(
        "DTE_SYNC_MODEL_PARAMS_BEFORE_PAYLOAD",
        dte_cfg.sync_model_params_before_payload,
    )
    set_env("DTE_DELTA_INVERSION_DEBUG", dte_cfg.inversion_debug)
    set_env(
        "DTE_DELTA_INVERSION_BF16_MARGIN_REL",
        dte_cfg.inversion_bf16_margin_rel,
    )

    if exported_env:
        for cfg_part in (
            getattr(config, "actor", None),
            getattr(config, "rollout", None),
        ):
            specs = getattr(cfg_part, "scheduling_spec", None)
            if not specs:
                continue
            for spec in specs:
                if isinstance(spec, dict):
                    spec.setdefault("env_vars", {}).update(exported_env)
                    continue
                env_vars = getattr(spec, "env_vars", None)
                if env_vars is None:
                    env_vars = {}
                    setattr(spec, "env_vars", env_vars)
                env_vars.update(exported_env)

    return exported_env
