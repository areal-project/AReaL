# SPDX-License-Identifier: Apache-2.0

"""User config loader for ``areal weight-update``.

``~/.areal/weight-update/config.toml`` overrides built-in defaults:

  [default]   service, gateway, admin_api_key

Precedence (highest first): explicit CLI flag, config.toml, click default.
A missing or malformed file is treated as empty.
"""

from __future__ import annotations

from pathlib import Path

from areal.v2.cli.config import BindingMap, ConfigLoader
from areal.v2.cli.weight_update.state import WU_NAMESPACE

WU_BINDINGS: BindingMap = {
    ("default", "service"): (("status", "pairs"), "service"),
    ("default", "gateway"): (("status", "pairs"), "gateway"),
    ("default", "admin_api_key"): (("status", "pairs"), "admin_api_key"),
}

wu_config_loader = ConfigLoader(namespace=WU_NAMESPACE, bindings=WU_BINDINGS)


def load_click_default_map(extra: Path | None = None) -> dict:
    return wu_config_loader.load_click_default_map(extra=extra)
