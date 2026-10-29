"""Load explicitly pinned Arena datasets without changing a shared Stream."""

from __future__ import annotations

import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from examples.swe.arena_client import ArenaAPIError, ArenaOpenAPIClient
from examples.swe.arena_reward_mapping import parse_version_ref
from examples.swe.arena_types import ArenaRewardRefConfig


def load_arena_dataset_snapshot(
    arena: ArenaOpenAPIClient,
    filename: str,
    stream_id: str,
    llm_protocol: str,
) -> list[dict[str, str]]:
    """Validate a read-only, ordered snapshot against exact published Env versions.

    Format: ``{"format_version": 1, "stream_id": ..., "count": N, "rows": [...]}``.
    Each row requires ``data_id`` (``env:key@version``), ``arena_task_type`` and
    ``env_content_hash``. The snapshot is never created or rewritten here. Reward
    and Harness pins are still validated separately by the normal startup path.
    """
    payload = json.loads(Path(filename).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("stream_id") != stream_id:
        raise ArenaAPIError("Arena dataset snapshot Stream mismatch")
    if type(payload.get("format_version")) is not int or payload["format_version"] != 1:
        raise ArenaAPIError("Unsupported Arena dataset snapshot format_version")
    items = payload.get("rows")
    if not isinstance(items, list) or not items or payload.get("errors"):
        raise ArenaAPIError("Arena dataset snapshot requires complete non-empty rows")
    if type(payload.get("count")) is not int or payload["count"] != len(items):
        raise ArenaAPIError("Arena dataset snapshot count does not match its rows")

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    references: list[tuple[ArenaRewardRefConfig, str]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise ArenaAPIError("Invalid Arena dataset snapshot row")
        data_id = item.get("data_id")
        task_type = item.get("arena_task_type")
        content_hash = item.get("env_content_hash")
        if not isinstance(data_id, str) or not data_id.startswith("env:"):
            raise ArenaAPIError("Arena dataset snapshot requires env:key@version IDs")
        ref = parse_version_ref(data_id.removeprefix("env:"))
        if ref.version == "latest":
            raise ArenaAPIError("Arena dataset snapshot requires exact Env versions")
        if data_id in seen:
            raise ArenaAPIError("Duplicate Arena dataset snapshot data_id")
        if not isinstance(task_type, str):
            raise ArenaAPIError("Arena dataset snapshot requires string task types")
        if not isinstance(content_hash, str) or not content_hash:
            raise ArenaAPIError("Arena dataset snapshot requires Env content hashes")
        seen.add(data_id)
        references.append((ref, content_hash))
        rows.append(
            {
                "data_id": data_id,
                "stream_id": stream_id,
                "llm_protocol": llm_protocol,
                "arena_task_type": task_type,
            }
        )

    # Startup-only checks; bounded parallel reads avoid one network round trip per
    # Env becoming a serial delay for large datasets. map retains snapshot order.
    with httpx.Client(timeout=arena.timeout) as client:

        def validate(reference: tuple[ArenaRewardRefConfig, str]) -> None:
            ref, expected_hash = reference
            version = arena.get_published_version(
                "envs", ref.key, ref.version, client=client
            )
            if version.get("content_hash") != expected_hash:
                raise ArenaAPIError(
                    f"Arena dataset snapshot Env content hash mismatch: "
                    f"{ref.key}@{ref.version}"
                )

        with ThreadPoolExecutor(max_workers=min(8, len(references))) as pool:
            for _ in pool.map(validate, references):
                pass
    return rows
