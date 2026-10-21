"""Freeze per-task Arena reward and Harness references before starting workers."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from examples.swe.arena_client import ArenaAPIError, ArenaOpenAPIClient
from examples.swe.arena_types import ArenaRewardRefConfig


def parse_version_ref(value: Any) -> ArenaRewardRefConfig:
    """Parse an exact key@version reference without guessing a version."""
    if not isinstance(value, str) or value.count("@") != 1:
        raise ArenaAPIError("Arena source_reward_ref requires key@version")
    key, version = value.split("@")
    if not key or not version or any(char.isspace() for char in value):
        raise ArenaAPIError("Arena source_reward_ref requires non-empty key@version")
    return ArenaRewardRefConfig(key, version)


def load_reward_mapping(
    filename: str, stream_id: str
) -> dict[str, ArenaRewardRefConfig]:
    """Read a complete snapshot; malformed or foreign snapshots fail closed."""
    payload = json.loads(Path(filename).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("stream_id") != stream_id:
        raise ArenaAPIError("Arena reward mapping Stream mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items or payload.get("errors"):
        raise ArenaAPIError(
            "Arena reward mapping must contain complete non-empty items"
        )
    mapping = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise ArenaAPIError("Invalid Arena reward mapping item")
        data_id = item.get("data_id")
        ref = item.get("reward_ref")
        if not isinstance(data_id, str) or not data_id or not isinstance(ref, Mapping):
            raise ArenaAPIError("Arena reward mapping requires data_id and reward_ref")
        if data_id in mapping:
            raise ArenaAPIError(f"Duplicate Arena reward mapping data_id: {data_id}")
        key, version = ref.get("key"), ref.get("version")
        if not isinstance(key, str) or not isinstance(version, str):
            raise ArenaAPIError("Arena reward mapping requires string key and version")
        mapping[data_id] = parse_version_ref(f"{key}@{version}")
    if payload.get("count") != len(mapping):
        raise ArenaAPIError("Arena reward mapping count does not match its items")
    return mapping


def load_harness_mapping(filename: str, stream_id: str) -> dict[str, str]:
    """Read optional per-task Harness pins; legacy reward-only snapshots still work."""
    load_reward_mapping(filename, stream_id)
    payload = json.loads(Path(filename).expanduser().read_text(encoding="utf-8"))
    mapping = {}
    for item in payload["items"]:
        if "harness_ref" not in item:
            continue
        ref = _binding_ref(item["harness_ref"])
        mapping[item["data_id"]] = f"{ref.key}@{ref.version}"
    if mapping and len(mapping) != len(payload["items"]):
        raise ArenaAPIError("Arena Harness mapping must cover every data_id")
    return mapping


def _binding_ref(value: Any) -> ArenaRewardRefConfig:
    if isinstance(value, Mapping):
        key, version = value.get("key"), value.get("version")
        if not isinstance(key, str) or not isinstance(version, str):
            raise ArenaAPIError("Arena binding requires string key and version")
        value = f"{key}@{version}"
    return parse_version_ref(value)


def prepare_reward_mapping(
    arena: ArenaOpenAPIClient,
    stream_id: str,
    rows: list[dict[str, str]],
    filename: str,
    *,
    client: httpx.Client | None = None,
    stream: Mapping[str, Any] | None = None,
) -> dict[str, ArenaRewardRefConfig]:
    """Create once from published Env metadata, or validate the frozen snapshot.

    Call only during synchronous preflight/dataset setup. All rewards must still be
    published. Existing snapshots are never silently regenerated, including recovery.
    """
    if client is None:
        with httpx.Client(timeout=arena.timeout) as shared_client:
            return prepare_reward_mapping(
                arena, stream_id, rows, filename, client=shared_client, stream=stream
            )
    path = Path(filename).expanduser()
    data_ids = [row["data_id"] for row in rows]
    if not data_ids or len(data_ids) != len(set(data_ids)):
        raise ArenaAPIError("Arena reward mapping requires non-empty unique data_ids")
    items = []
    harnesses = {}
    bindings = (stream or {}).get("metadata", {}).get("per_task_bindings", {})
    if isinstance(bindings, str):
        bindings = json.loads(bindings)
    if not isinstance(bindings, Mapping):
        raise ArenaAPIError("Arena per_task_bindings must be an object")
    if any(not isinstance(value, Mapping) for value in bindings.values()):
        raise ArenaAPIError("Arena per_task_bindings values must be objects")
    if bindings and set(bindings) != set(data_ids):
        raise ArenaAPIError("Arena per_task_bindings dataset coverage mismatch")
    if path.exists():
        mapping = load_reward_mapping(str(path), stream_id)
        harnesses = load_harness_mapping(str(path), stream_id)
        if set(mapping) != set(data_ids):
            raise ArenaAPIError("Arena reward mapping dataset coverage mismatch")
    else:
        mapping = {}
        for data_id in sorted(data_ids):
            if not data_id.startswith("env:"):
                raise ArenaAPIError(
                    f"Expected env:key@version data_id, got {data_id!r}"
                )
            env_ref = parse_version_ref(data_id.removeprefix("env:"))
            env = arena.get_published_version(
                "envs", env_ref.key, env_ref.version, client=client
            )
            spec = env.get("spec")
            metadata = spec.get("metadata") if isinstance(spec, Mapping) else None
            metadata = metadata or {}
            binding = bindings.get(data_id, {})
            if not isinstance(binding, Mapping):
                raise ArenaAPIError(f"Invalid Arena binding for {data_id!r}")
            reward_sources = [
                value
                for value in (
                    binding.get("reward_ref"),
                    metadata.get("source_reward_ref"),
                    metadata.get("reward_ref"),
                )
                if value is not None
            ]
            if not reward_sources:
                raise ArenaAPIError(
                    f"Env {data_id!r} is missing source_reward_ref/reward_ref"
                )
            refs = [_binding_ref(value) for value in reward_sources]
            if any(ref != refs[0] for ref in refs):
                raise ArenaAPIError(
                    f"Conflicting Arena reward bindings for {data_id!r}"
                )
            ref = refs[0]
            mapping[data_id] = ref
            item = {
                "data_id": data_id,
                "reward_ref": {"key": ref.key, "version": ref.version},
                "env_content_hash": env.get("content_hash"),
            }
            harness_sources = [
                value
                for value in (binding.get("harness_ref"), metadata.get("harness_ref"))
                if value is not None
            ]
            if harness_sources:
                refs = [_binding_ref(value) for value in harness_sources]
                if any(ref != refs[0] for ref in refs):
                    raise ArenaAPIError(
                        f"Conflicting Arena Harness bindings for {data_id!r}"
                    )
                ref = refs[0]
                item["harness_ref"] = {"key": ref.key, "version": ref.version}
                harnesses[data_id] = f"{ref.key}@{ref.version}"
            items.append(item)
    if harnesses and set(harnesses) != set(data_ids):
        raise ArenaAPIError("Arena Harness mapping must cover every data_id")
    # A frozen snapshot must not silently drop or contradict declared Stream pins.
    for data_id, binding in bindings.items():
        if _binding_ref(binding.get("reward_ref")) != mapping[data_id]:
            raise ArenaAPIError(f"Frozen Arena reward binding mismatch for {data_id!r}")
        ref = _binding_ref(binding.get("harness_ref"))
        if harnesses.get(data_id) != f"{ref.key}@{ref.version}":
            raise ArenaAPIError(
                f"Frozen Arena Harness binding mismatch for {data_id!r}"
            )
    for harness in sorted(set(harnesses.values())):
        ref = parse_version_ref(harness)
        arena.get_published_version("harnesses", ref.key, ref.version, client=client)
    for key, version in sorted({(ref.key, ref.version) for ref in mapping.values()}):
        arena.get_published_version("rewards", key, version, client=client)
    if items:
        payload = {"stream_id": stream_id, "count": len(items), "items": items}
        path.parent.mkdir(parents=True, exist_ok=True)
        # Publish only a fully validated, durable file, without replacing another
        # controller's snapshot if concurrent starts race on the same run directory.
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if (
                    load_reward_mapping(str(path), stream_id) != mapping
                    or load_harness_mapping(str(path), stream_id) != harnesses
                ):
                    raise ArenaAPIError(
                        "Concurrent Arena reward mapping conflict"
                    ) from None
        finally:
            Path(temporary).unlink(missing_ok=True)
    return mapping
