# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from areal.api.cli_args import BaseExperimentConfig
from areal.utils.environ import is_single_controller

if TYPE_CHECKING:
    from areal.api.cli_args import AwexConfig


@dataclass
class AwexRuntimeHandle:
    meta_server_addr: str | None = None
    owns_meta_server: bool = False
    _stop_fn: Callable[[], bool] | None = None
    _finalizer: weakref.finalize | None = None

    def close(self) -> None:
        if (
            self.owns_meta_server
            and self._finalizer is not None
            and self._finalizer.alive
        ):
            self._finalizer()


def _get_weight_update_engine_config(config: BaseExperimentConfig):
    if hasattr(config, "actor"):
        return config.actor
    if hasattr(config, "model"):
        return config.model
    raise ValueError(
        f"Config {type(config).__name__} must have either 'actor' or 'model' attribute"
    )


def _resolve_meta_server_override() -> str | None:
    for key in ("AREAL_AWEX_META_SERVER_ADDR", "AWEX_META_SERVER_ADDR"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return None


def _import_meta_server_fns():
    try:
        from awex.meta.meta_server import start_meta_server, stop_meta_server
    except ImportError as exc:
        raise RuntimeError(
            "Awex runtime bootstrap requires the awex package to be installed."
        ) from exc
    return start_meta_server, stop_meta_server


def _safe_stop_meta_server(stop_fn: Callable[[], bool]) -> None:
    stop_fn()


def prepare_awex_runtime(config: BaseExperimentConfig) -> AwexRuntimeHandle:
    engine_config = _get_weight_update_engine_config(config)
    if engine_config.weight_update_mode != "awex":
        return AwexRuntimeHandle()

    awex_cfg: AwexConfig | None = getattr(config, "awex", None)
    if awex_cfg is None:
        raise ValueError("Awex config is required when weight_update_mode is 'awex'.")

    meta_server_addr = _resolve_meta_server_override() or awex_cfg.meta_server_addr
    if meta_server_addr and meta_server_addr.lower() != "auto":
        awex_cfg.meta_server_addr = meta_server_addr
        return AwexRuntimeHandle(meta_server_addr=meta_server_addr)

    if not is_single_controller():
        raise ValueError(
            "AWEX auto-start is supported only in single-controller mode. "
            "Set awex.meta_server_addr explicitly when running in SPMD mode."
        )

    start_meta_server, stop_meta_server = _import_meta_server_fns()
    meta_ip, meta_port = start_meta_server()
    resolved_addr = f"{meta_ip}:{meta_port}"
    handle = AwexRuntimeHandle(
        meta_server_addr=resolved_addr,
        owns_meta_server=True,
        _stop_fn=stop_meta_server,
    )
    handle._finalizer = weakref.finalize(
        handle, _safe_stop_meta_server, stop_meta_server
    )
    awex_cfg.meta_server_addr = resolved_addr
    return handle
