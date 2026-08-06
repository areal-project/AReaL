# SPDX-License-Identifier: Apache-2.0
"""DTE weight-digest recovery helpers for launchers."""

from __future__ import annotations

import dataclasses
import json
import os
import time
from collections.abc import Mapping
from typing import Any

WEIGHT_CAPTURE_ROOT_ENV = "AREAL_DTE_WEIGHT_CAPTURE_ROOT"
WEIGHT_COMPARE_GROUP_ENV = "AREAL_DTE_WEIGHT_COMPARE_GROUP"
MISMATCH_FLAG_NAME = "mismatch_flag.json"
RECOVERABLE_MISMATCH_REASONS = frozenset({"expected_peer_timeout"})
UNREADABLE_MISMATCH_REASON = "mismatch_flag_unreadable"


@dataclasses.dataclass(frozen=True)
class DTERecoveryFlag:
    path: str
    reason: str | None
    data: dict[str, Any] | None = None
    error: str | None = None

    @property
    def recoverable(self) -> bool:
        return self.reason in RECOVERABLE_MISMATCH_REASONS


def _clean_env_value(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text or None


def _env_value(
    env_vars: Mapping[str, object] | None,
    environ: Mapping[str, str],
    name: str,
) -> str | None:
    if env_vars is not None and name in env_vars:
        return _clean_env_value(env_vars.get(name))
    return _clean_env_value(environ.get(name))


def mismatch_flag_path(
    env_vars: Mapping[str, object] | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    environ = os.environ if environ is None else environ
    capture_root = _env_value(env_vars, environ, WEIGHT_CAPTURE_ROOT_ENV)
    compare_group = _env_value(env_vars, environ, WEIGHT_COMPARE_GROUP_ENV)
    if capture_root is None or compare_group is None:
        return None
    return os.path.join(capture_root, compare_group, MISMATCH_FLAG_NAME)


def extract_mismatch_reason(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None

    mismatch = data.get("mismatch")
    if isinstance(mismatch, dict):
        reason = mismatch.get("reason")
        if reason == "mismatch_flag_observed":
            nested_reason = extract_mismatch_reason(mismatch.get("flag"))
            if nested_reason is not None:
                return nested_reason
        if isinstance(reason, str) and reason:
            return reason

    reason = data.get("reason")
    if isinstance(reason, str) and reason:
        return reason
    return None


def load_mismatch_flag(
    env_vars: Mapping[str, object] | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> DTERecoveryFlag | None:
    path = mismatch_flag_path(env_vars, environ=environ)
    if path is None or not os.path.exists(path):
        return None

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return DTERecoveryFlag(
            path=path,
            reason=UNREADABLE_MISMATCH_REASON,
            error=repr(exc),
        )
    return DTERecoveryFlag(
        path=path,
        reason=extract_mismatch_reason(data),
        data=data,
    )


def archive_mismatch_flag(path: str, *, run_id: int) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = f"{path}.archived_run{run_id}_{timestamp}"
    archive_path = base
    suffix = 1
    while os.path.exists(archive_path):
        suffix += 1
        archive_path = f"{base}.{suffix}"
    os.replace(path, archive_path)
    return archive_path
