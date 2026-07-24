# SPDX-License-Identifier: Apache-2.0

"""Utilities for safely serializing application configuration."""

from __future__ import annotations

from typing import Any

REDACTED_VALUE = "<redacted>"


def _is_sensitive_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.lower()
    return (
        normalized in {"authorization", "password", "secret", "token"}
        or "api_key" in normalized
        or normalized.endswith(("_password", "_secret", "_token", "_credential"))
        or "private_key" in normalized
    )


def redact_sensitive_config(value: Any) -> Any:
    """Return a copy with credentials redacted while preserving ordinary fields."""
    if isinstance(value, dict):
        return {
            key: REDACTED_VALUE
            if _is_sensitive_key(key) and item not in (None, "")
            else redact_sensitive_config(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive_config(item) for item in value)
    return value
