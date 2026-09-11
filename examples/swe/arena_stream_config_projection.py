# SPDX-License-Identifier: Apache-2.0

"""Project inline Arena Streams from an AReaL training config."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
from typing import Any

import yaml


def _reject_interpolations(value: Any, *, path: str) -> None:
    """Reject values whose preflight meaning could differ after Hydra resolution."""
    if isinstance(value, str) and "${" in value:
        raise ValueError(
            f"{path} contains a Hydra interpolation; use a literal value so "
            "preflight and training see the same Stream configuration"
        )
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_interpolations(item, path=f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_interpolations(item, path=f"{path}.{key}")


def project_arena_streams(config_path: Path) -> bytes:
    """Return canonical top-level ``streams`` YAML from a training config."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Multi-Stream config must be a mapping: {config_path}")
    econfig = config.get("econfig")
    if not isinstance(econfig, dict):
        raise ValueError(f"Multi-Stream config is missing econfig: {config_path}")
    streams = econfig.get("arena_streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError(
            "Multi-Stream config requires a non-empty "
            f"econfig.arena_streams: {config_path}"
        )
    _reject_interpolations(streams, path="econfig.arena_streams")
    return yaml.safe_dump({"streams": streams}, sort_keys=False).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    payload = project_arena_streams(args.config)
    print(base64.b64encode(payload).decode("ascii"), end="")


if __name__ == "__main__":
    main()
