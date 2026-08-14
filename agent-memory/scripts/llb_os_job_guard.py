#!/usr/bin/env python3
"""Cancel a tracked active LLB-OS job before replacing it, then track the replacement.

The registry contains record IDs only; credentials stay in the caller's AIS environment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGISTRY = Path(os.environ.get("LLB_OS_JOB_REGISTRY", "/storage/openpsi/users/yl/cfuse/llb_os_active_jobs.json"))
TERMINAL = {"succeeded", "success", "failed", "stopped", "cancelled", "canceled", "completed", "finished", "done"}


def load() -> dict[str, Any]:
    if not REGISTRY.exists():
        return {}
    try:
        data = json.loads(REGISTRY.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save(data: dict[str, Any]) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY.with_suffix(REGISTRY.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, REGISTRY)


def status(record_id: str) -> str:
    from aistudio_common.rest.job import query_job_status
    value = query_job_status(record_id)
    if isinstance(value, dict):
        value = value.get("status") or value.get("state") or value.get("phase") or "unknown"
    return str(value).strip().lower()


def preflight(task_key: str) -> None:
    data = load()
    item = data.get(task_key)
    if not isinstance(item, dict) or not item.get("record_id"):
        print(f"[LLB-OS guard] no tracked predecessor for {task_key}")
        return
    record_id = str(item["record_id"])
    current = status(record_id)
    if current in TERMINAL:
        print(f"[LLB-OS guard] predecessor {record_id} is terminal ({current}); no cancellation needed")
        return
    from pypai.job.execution_job.base_job import BaseJob
    print(f"[LLB-OS guard] cancelling predecessor {record_id} for {task_key} (status={current})")
    BaseJob.stop(record_id)
    data[task_key] = {**item, "status_before_replacement": current, "replaced_at": datetime.now(timezone.utc).isoformat()}
    save(data)


def register(task_key: str, record_id: str, job_name: str = "") -> None:
    data = load()
    data[task_key] = {
        "record_id": str(record_id),
        "job_name": job_name,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    save(data)
    print(f"[LLB-OS guard] tracking {task_key}: record_id={record_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("task_key")
    p = sub.add_parser("register")
    p.add_argument("task_key")
    p.add_argument("record_id")
    p.add_argument("--job-name", default="")
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args.task_key)
    else:
        register(args.task_key, args.record_id, args.job_name)


if __name__ == "__main__":
    main()
