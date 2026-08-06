# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_ENV_NAME = "AREAL_ROLLOUT_FINGERPRINT"
_CAPTURE_ROOT_ENV = "AREAL_DTE_WEIGHT_CAPTURE_ROOT"
_COMPARE_GROUP_ENV = "AREAL_DTE_WEIGHT_COMPARE_GROUP"
_CAPTURE_RUN_ENV = "AREAL_DTE_WEIGHT_CAPTURE_RUN"
_DEFAULT_CAPTURE_DIR = "dte_weight_capture"
_TRUTHY = {"1", "true", "yes", "on"}


def enabled() -> bool:
    return os.environ.get(_ENV_NAME, "0").strip().lower() in _TRUTHY


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_ints(values: Iterable[int]) -> str:
    return sha256_json([int(v) for v in values])


def _safe_slug(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_") or "unknown"


def _env_int(name: str, default: int = -1) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _capture_group() -> str:
    value = os.environ.get(_COMPARE_GROUP_ENV)
    if value:
        return _safe_slug(value)
    return _safe_slug(os.environ.get("EXP_NAME", "default"))


def _capture_run() -> str:
    value = os.environ.get(_CAPTURE_RUN_ENV)
    if value:
        return _safe_slug(value)
    trial = os.environ.get("TRIAL_NAME")
    if trial:
        transfer = os.environ.get("DTE_TRANSFER")
        delta_method = os.environ.get("DTE_DELTA_METHOD")
        suffix = "-".join(part for part in (transfer, delta_method) if part)
        return _safe_slug(f"{trial}-{suffix}" if suffix else trial)
    return _safe_slug(f"pid{os.getpid()}-{socket.gethostname()}")


def _capture_dir() -> Path:
    root = os.environ.get(_CAPTURE_ROOT_ENV)
    if not root:
        root = os.path.join(
            os.environ.get("AREAL_STORAGE", os.getcwd()),
            _DEFAULT_CAPTURE_DIR,
        )
    return Path(root) / _capture_group() / _capture_run()


def _runtime_context() -> dict[str, Any]:
    return {
        "ts": time.time(),
        "exp": os.environ.get("EXP_NAME", ""),
        "trial": os.environ.get("TRIAL_NAME", ""),
        "job": os.environ.get("SLURM_JOB_ID", ""),
        "capture_group": _capture_group(),
        "run": _capture_run(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "rank": _env_int("RANK"),
        "local_rank": _env_int("LOCAL_RANK"),
        "world_size": _env_int("WORLD_SIZE"),
        "transfer": os.environ.get("DTE_TRANSFER", ""),
        "delta_method": os.environ.get("DTE_DELTA_METHOD", ""),
    }


def _append_jsonl_atomic(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(record) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        delete=False,
    ) as f:
        tmp_name = f.name
        f.write(payload)
    try:
        with path.open("a", encoding="utf-8") as out:
            with open(tmp_name, encoding="utf-8") as src:
                out.write(src.read())
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _write_artifact(record: dict[str, Any]) -> None:
    try:
        out_dir = _capture_dir()
        _append_jsonl_atomic(out_dir / "rollout_fingerprint.jsonl", record)
        if record.get("event") == "seed_derived":
            seed_record = {
                key: record.get(key)
                for key in (
                    "ts",
                    "exp",
                    "trial",
                    "job",
                    "capture_group",
                    "run",
                    "hostname",
                    "pid",
                    "rank",
                    "local_rank",
                    "world_size",
                    "task_id",
                    "group_id",
                    "session_id",
                    "member",
                    "sample_sha256",
                    "seed",
                )
            }
            seed_record["event"] = "seed_derived"
            _append_jsonl_atomic(out_dir / "rollout_seed_manifest.jsonl", seed_record)
        if record.get("event") == "rollout_batch_selected":
            _append_jsonl_atomic(out_dir / "rollout_batch_manifest.jsonl", record)
        if record.get("event") == "train_step_manifest":
            _append_jsonl_atomic(out_dir / "train_step_manifest.jsonl", record)
    except Exception:
        # Fingerprint artifacts are diagnostic; logging must remain best-effort.
        return


def log_event(logger: Any, event: str, **fields: Any) -> None:
    if not enabled():
        return
    record = {"event": event, **_runtime_context(), **fields}
    _write_artifact(record)
    logger.info("AREAL_ROLLOUT_FINGERPRINT %s", canonical_json(record))
