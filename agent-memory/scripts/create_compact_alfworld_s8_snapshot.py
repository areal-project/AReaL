#!/usr/bin/env python3
"""Create a non-destructive compact AO copy of an ALFWorld checkpoint snapshot.

The source is opened read-only.  The copy deliberately omits Qdrant: the snapshot
loader rebuilds a fresh local index from the transformed cube dump, preventing
stale raw payloads from being retrieved.  Vectors and all memory IDs are retained.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

FORMAT = "compact_action_observation_v1"
MARKER = "COMPACT_TRAJECTORY_V1:"


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256(path: Path) -> str:
    return file_digest(path, "sha256")


def compact(value: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", (value or "").strip())
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def parse_trajectory(value: str) -> list[dict[str, Any]] | None:
    value = (value or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json|python|text)?\s*", "", value, count=1, flags=re.I)
        value = re.sub(r"\s*```$", "", value, count=1)
    candidates = [value]
    start = value.find("[")
    if start >= 0:
        quote = None
        escaped = False
        depth = 0
        for i, ch in enumerate(value[start:], start):
            if quote:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    quote = None
            elif ch in ("'", '"'):
                quote = ch
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    candidates.append(value[start : i + 1])
                    break
    for candidate in dict.fromkeys(candidates):
        for parser in (json.loads, ast.literal_eval):
            try:
                payload = parser(candidate)
                if isinstance(payload, dict):
                    payload = payload.get("trajectory", payload.get("messages", payload.get("steps")))
                if isinstance(payload, list) and all(isinstance(x, dict) for x in payload):
                    return payload
            except Exception:
                pass
    return None


def trajectory_summary(value: str) -> str | None:
    messages = parse_trajectory(value)
    if not messages:
        return None
    start = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == "user" and "Now, it's your turn" in str(msg.get("content", "")):
            start = i
    if start >= 0:
        messages = messages[start:]
    pairs: list[tuple[str, str]] = []
    observation = ""
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        if msg.get("role") == "user":
            if "Now, it's your turn" in content:
                content = content.split("Now, it's your turn", 1)[-1]
            observation = compact(content, 420)
        elif msg.get("role") == "assistant":
            match = re.search(r"(?im)^\s*Action\s*:\s*(.+)$", content)
            action = compact(match.group(1) if match else content, 240)
            if action:
                pairs.append((observation, action))
                observation = ""
    if not pairs:
        return None
    lines = [MARKER, "Archived Action/Observation Summary:"]
    for i, (obs, action) in enumerate(pairs[-24:], 1):
        if obs:
            lines.append(f"{i}. Observation: {obs}")
        lines.append(f"   Action: {action}")
    return "\n".join(lines)


def script_summary(content: str) -> str | None:
    """Convert existing procedural scripts into a concise, explicit action plan.

    S8 was saved after proceduralization, so it contains no raw TRAJECTORY
    payloads to recover observations from.  We retain the task goal as the
    first observation and each numbered script step as an action; this is an
    honest compact representation rather than inventing environment feedback.
    """
    if "# High-Level Script:" not in content and "## Steps" not in content:
        return None
    header = content.split("\n\n# High-Level Script:", 1)[0].strip()
    task = header[5:].strip() if header.startswith("Task:") else header
    steps_match = re.search(r"(?s)## Steps\s*(.*?)(?=\n## |\Z)", content)
    if not steps_match:
        return None
    steps = re.findall(r"(?m)^\s*\d+\.\s+(.*?)(?=\n\s*\d+\.\s+|\Z)", steps_match.group(1))
    actions = [compact(re.sub(r"\*+", "", step), 320) for step in steps]
    actions = [x for x in actions if x]
    if not actions:
        return None
    lines = [MARKER, "Archived Action/Observation Summary:"]
    if task:
        lines.append(f"1. Observation: Task goal: {compact(task, 500)}")
        lines.append(f"   Action: {actions[0]}")
        actions = actions[1:]
    for idx, action in enumerate(actions, 2):
        lines.append(f"{idx}. Action: {action}")
    return "\n".join(lines)


def transform_content(content: Any) -> tuple[Any, str]:
    if not isinstance(content, str):
        return content, "not_string"
    if MARKER in content:
        return content, "already_compact"
    marker = "\n\nTRAJECTORY:\n"
    if marker in content:
        header, raw = content.split(marker, 1)
        summary = trajectory_summary(raw)
        if summary:
            return header.rstrip() + "\n\n" + summary, "trajectory_transformed"
        return content, "trajectory_parse_failed"
    summary = script_summary(content)
    if summary:
        return summary, "script_transformed"
    return content, "not_compactable"


def transform_record(record: dict[str, Any], stats: dict[str, int]) -> None:
    # Cube dump wraps a TextualMemoryItem in ``payload``; local cache stores it
    # directly.  Update only metadata.full_content and preserve ID/vector/task
    # embeddings exactly.
    containers = [record]
    payload = record.get("payload")
    if isinstance(payload, dict):
        containers.append(payload)
    for container in containers:
        metadata = container.get("metadata")
        targets: list[dict[str, Any]] = []
        if isinstance(metadata, dict):
            targets.append(metadata)
            extra = metadata.get("model_extra")
            if isinstance(extra, dict):
                targets.append(extra)
        for target in targets:
            if "full_content" not in target:
                continue
            updated, status = transform_content(target["full_content"])
            stats[status] = stats.get(status, 0) + 1
            target["full_content"] = updated
            return
    stats["no_full_content"] = stats.get("no_full_content", 0) + 1


def stream_json_array(src: Path, dst: Path, stats: dict[str, int]) -> int:
    """Incrementally decode/write one top-level JSON array without loading 3GB."""
    decoder = json.JSONDecoder()
    count = 0
    buffer = ""
    started = False
    first = True
    with src.open("r", encoding="utf-8") as inf, dst.open("w", encoding="utf-8") as outf:
        outf.write("[\n")
        while True:
            chunk = inf.read(8 * 1024 * 1024)
            if chunk:
                buffer += chunk
            eof = not chunk
            pos = 0
            while True:
                while pos < len(buffer) and buffer[pos].isspace() or (pos < len(buffer) and buffer[pos] in "[,]"):
                    if buffer[pos] == "[":
                        started = True
                    pos += 1
                if pos >= len(buffer):
                    break
                if buffer[pos] == "]":
                    pos += 1
                    break
                if not started:
                    raise ValueError("cube textual_memory.json is not a top-level array")
                try:
                    obj, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    break
                if not isinstance(obj, dict):
                    raise ValueError("cube array contains non-object item")
                transform_record(obj, stats)
                if not first:
                    outf.write(",\n")
                json.dump(obj, outf, ensure_ascii=False, separators=(",", ":"))
                first = False
                count += 1
                pos = end
            buffer = buffer[pos:]
            if eof:
                if buffer.strip() not in ("", "]"):
                    raise ValueError("trailing/incomplete JSON while streaming cube dump")
                break
        outf.write("\n]\n")
    return count


def copy_tree_without_qdrant(source: Path, destination: Path) -> None:
    for name in ("local_cache",):
        shutil.copytree(source / name, destination / name, copy_function=shutil.copy2)
    shutil.copy2(source / "snapshot_meta.json", destination / "snapshot_meta.json")
    (destination / "cube").mkdir(parents=True)
    for item in (source / "cube").iterdir():
        if item.name != "textual_memory.json":
            target = destination / "cube" / item.name
            if item.is_dir():
                shutil.copytree(item, target, copy_function=shutil.copy2)
            else:
                shutil.copy2(item, target)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--destination", type=Path, required=True)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    required = [source / "cube" / "textual_memory.json", source / "local_cache" / "mem_cache.json"]
    if not source.is_dir() or any(not p.is_file() for p in required):
        raise SystemExit(f"invalid source snapshot: {source}")
    if destination.exists():
        raise SystemExit(f"destination must not already exist: {destination}")
    destination.mkdir(parents=True)
    source_hashes = {str(p.relative_to(source)): sha256(p) for p in required}
    try:
        copy_tree_without_qdrant(source, destination)
        stats: dict[str, int] = {}
        cache_path = destination / "local_cache" / "mem_cache.json"
        with cache_path.open(encoding="utf-8") as f:
            cache = json.load(f)
        if not isinstance(cache, dict):
            raise ValueError("mem_cache.json must be an object")
        for record in cache.values():
            if isinstance(record, dict):
                transform_record(record, stats)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, separators=(",", ":"))
        cube_count = stream_json_array(source / "cube" / "textual_memory.json", destination / "cube" / "textual_memory.json", stats)
        # Do not copy stale Qdrant records. Loader rebuilds it locally from transformed cube.
        (destination / "qdrant").mkdir()
        meta_path = destination / "snapshot_meta.json"
        meta = json.loads(meta_path.read_text())
        meta["cube_dir"] = str(destination / "cube")
        meta["qdrant_dir"] = str(destination / "qdrant")
        meta["textual_memory_md5"] = file_digest(destination / "cube" / "textual_memory.json", "md5")
        meta["compact_memory_format"] = FORMAT
        meta["compacted_from"] = str(source)
        meta["compacted_at"] = datetime.now(timezone.utc).isoformat()
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        after_hashes = {str(p.relative_to(source)): sha256(p) for p in required}
        if source_hashes != after_hashes:
            raise RuntimeError("SOURCE MUTATION DETECTED; refusing to report success")
        manifest = {
            "format": FORMAT, "source": str(source), "destination": str(destination),
            "source_sha256_before_after": {k: [source_hashes[k], after_hashes[k]] for k in source_hashes},
            "cube_records": cube_count, "mem_cache_records": len(cache), "transform_stats": stats,
            "qdrant": "intentionally empty; rebuild from transformed cube on load",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (destination / "compact_transform_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        print(f"ERROR: compact snapshot creation failed; incomplete destination preserved for diagnosis: {destination}", file=sys.stderr)
        raise

if __name__ == "__main__":
    raise SystemExit(main())
