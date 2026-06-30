# SPDX-License-Identifier: Apache-2.0

"""User config loader for ``areal inf``.

~/.areal/inf/config.toml overrides built-in CLI defaults. Sections map
to verbs:

  [default]              admin_api_key, log_level (applied to all verbs)
  [launch]               run-time gateway / router / strategy / timeouts
  [scheduler]            scheduler backend selection (applied to run only)
  [register.internal]    register / inline-register defaults

Precedence (highest first):
  1. explicit CLI flag
  2. config.toml
  3. hard-coded default in the click @option

Missing / malformed file is treated as empty — never crashes the CLI.
"""

from __future__ import annotations

from pathlib import Path

from areal.v2.cli.config import BindingMap, ConfigLoader
from areal.v2.cli.inference.state import INF_NAMESPACE

INF_BINDINGS: BindingMap = {
    # [default]
    (
        "default",
        "service",
    ): (
        (
            "run",
            "stop",
            "status",
            "register",
            "deregister",
            "models",
            "logs",
        ),
        "service",
    ),
    ("default", "admin_api_key"): ("run", "admin_api_key"),
    ("default", "log_level"): ("run", "log_level"),
    # [launch] — applied to `run`
    ("launch", "gateway_host"): ("run", "host"),
    ("launch", "gateway_port"): ("run", "port"),
    ("launch", "routing_strategy"): ("run", "routing_strategy"),
    ("launch", "launch_timeout"): ("run", "launch_timeout"),
    # [scheduler] — applied to `run` only; other verbs read from ServiceState
    ("scheduler", "type"): ("run", "scheduler"),
    # [register.internal] — applied to `register`
    ("register.internal", "backend"): ("register", "backend"),
    ("register.internal", "model_health_timeout"): ("register", "model_health_timeout"),
    ("register.internal", "engine_args"): ("register", "engine_args"),
    ("register.internal", "proxy_args"): ("register", "proxy_args"),
}


inf_config_loader = ConfigLoader(namespace=INF_NAMESPACE, bindings=INF_BINDINGS)


def load_click_default_map(extra: Path | None = None) -> dict:
    return inf_config_loader.load_click_default_map(extra=extra)
