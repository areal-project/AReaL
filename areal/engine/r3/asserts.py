# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib import metadata
from typing import Any

import areal


class R3Error(ValueError):
    """Raised when R3 routing replay data or setup is structurally invalid."""


def _package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-installed"


def r3_version_context() -> dict[str, str]:
    return {
        "areal": getattr(areal, "__version__", "unknown"),
        "megatron-core": _package_version("megatron-core"),
        "mbridge": _package_version("mbridge"),
    }


def format_context(context: dict[str, Any]) -> str:
    items = {**r3_version_context(), **context}
    return ", ".join(f"{key}={value!r}" for key, value in items.items())


def r3_error(message: str, **context: Any) -> R3Error:
    return R3Error(f"{message} ({format_context(context)})")


def r3_assert(condition: bool, message: str, **context: Any) -> None:
    if not condition:
        raise r3_error(message, **context)
