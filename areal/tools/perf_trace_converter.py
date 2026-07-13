# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from glob import glob
from pathlib import Path


def _load_events(path: Path) -> list[dict]:
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as fin:
        for lineno, raw_line in enumerate(fin, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - invalid payload
                raise ValueError(
                    f"Failed to parse JSONL at line {lineno} in {path}: {exc}"
                ) from exc
    return events


def _extract_rank(event: dict) -> str | int | None:
    """Best-effort extraction of the rank identifier from a trace event."""

    args = event.get("args")
    if not isinstance(args, dict):
        return None
    rank = args.get("rank")
    if rank is None:
        return None
    if isinstance(rank, bool):  # guard against bool subclassing int
        return None
    if isinstance(rank, int | float):
        try:
            return int(rank)
        except (TypeError, ValueError, OverflowError):
            return None
    if isinstance(rank, str):
        text = rank.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return text
    return str(rank)


def _extract_role(event: dict) -> str | None:
    """Best-effort extraction of the role identifier from a trace event."""

    args = event.get("args")
    if not isinstance(args, dict):
        return None
    role = args.get("role")
    if role is None or not isinstance(role, str):
        return None
    return role.strip() or None


def _format_rank(rank: str | int) -> str:
    return str(rank)


def _rank_sort_key(rank: str | int | None) -> tuple[int, object]:
    if rank is None:
        return (2, 0)
    if isinstance(rank, int):
        return (0, rank)
    return (1, str(rank))


def _role_sort_key(role: str | None) -> tuple[int, str]:
    if role is None:
        return (1, "")
    return (0, role)


def _value_sort_key(value: object) -> tuple[int, object]:
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, int):
        return (1, value)
    if isinstance(value, float):
        return (2, value)
    if isinstance(value, str):
        return (3, value)
    return (4, repr(value))


def _metadata_name_sort_key(name: object) -> int:
    if name == "process_name":
        return 0
    if name == "process_sort_index":
        return 1
    if name == "thread_name":
        return 2
    return 3


def _tid_sort_key(value: object) -> tuple[int, object]:
    """Sort key for TIDs: positive ints < negative ints < others."""
    prio, val = _value_sort_key(value)
    if prio == 1:  # int
        if isinstance(val, int) and val < 0:
            return (2, -val)
        return (1, val)
    if prio >= 2:
        return (prio + 1, val)
    return (prio, val)


def _remap_process_and_thread_ids(
    events: list[dict],
    existing_process_names: dict[tuple[str | int, str | None, object], str]
    | None = None,
    existing_thread_names: dict[tuple[str | int, str | None, object, object], str]
    | None = None,
) -> list[dict]:
    """Remap pid/tid to be unique and return metadata events.

    This function modifies the `events` list in-place by replacing `pid` and
    `tid` values. It returns a new list of generated metadata events.
    """
    if existing_process_names is None:
        existing_process_names = {}
    if existing_thread_names is None:
        existing_thread_names = {}

    # pid_keys: (rank, role, original_pid)
    pid_keys: set[tuple[str | int, str | None, object]] = set()
    # tid_keys: (rank, role, original_pid, original_tid)
    tid_keys: set[tuple[str | int, str | None, object, object]] = set()

    for event in events:
        rank = _extract_rank(event)
        if rank is None:
            continue

        role = _extract_role(event)
        original_pid = event.get("pid")
        if original_pid is None:
            continue
        pid_keys.add((rank, role, original_pid))

        original_tid = event.get("tid")
        if original_tid is not None:
            tid_keys.add((rank, role, original_pid, original_tid))

    sorted_pid_keys = sorted(
        pid_keys,
        key=lambda item: (
            _rank_sort_key(item[0]),
            _role_sort_key(item[1]),
            _value_sort_key(item[2]),
        ),
    )

    pid_map: dict[tuple[str | int, str | None, object], int] = {}
    pid_labels: dict[int, tuple[str | int, str | None, object]] = {}
    for new_pid, key in enumerate(sorted_pid_keys):
        pid_map[key] = new_pid + 1
        pid_labels[new_pid + 1] = key

    tid_counters: dict[int, int] = {}
    tid_map: dict[tuple[str | int, str | None, object, object], int] = {}
    tid_labels: dict[tuple[int, int], tuple[str | int, str | None, object]] = {}

    sorted_tid_keys = sorted(
        tid_keys,
        key=lambda item: (
            _rank_sort_key(item[0]),
            _role_sort_key(item[1]),
            _value_sort_key(item[2]),
            _tid_sort_key(item[3]),
        ),
    )

    for key in sorted_tid_keys:
        rank, role, original_pid, original_tid = key
        new_pid = pid_map[(rank, role, original_pid)]
        next_tid = tid_counters.get(new_pid, new_pid)
        tid_counters[new_pid] = next_tid + 1
        tid_map[key] = next_tid
        tid_labels[(new_pid, next_tid)] = (rank, role, original_tid)

    for event in events:
        rank = _extract_rank(event)
        if rank is None:
            continue

        role = _extract_role(event)
        original_pid = event.get("pid")
        if original_pid is None:
            continue
        new_pid = pid_map[(rank, role, original_pid)]
        event["pid"] = new_pid

        original_tid = event.get("tid")
        if original_tid is not None:
            tid_key = (rank, role, original_pid, original_tid)
            if tid_key in tid_map:
                event["tid"] = tid_map[tid_key]
            else:
                # Defensive: leave event["tid"] as is, or set to None, or log warning
                event["tid"] = None

    metadata_events: list[dict] = []
    for pid, (rank, role, original_pid) in pid_labels.items():
        rank_text = _format_rank(rank)
        process_name = existing_process_names.get((rank, role, original_pid))
        if process_name is None:
            if role:
                process_name = f"[{role}] Rank {rank_text}, Process {original_pid}"
            else:
                process_name = f"[Rank {rank_text}, Process {original_pid}]"

        args: dict = {"name": process_name, "rank": rank}
        if role is not None:
            args["role"] = role

        metadata_events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "args": args,
            }
        )
        sort_args: dict = {"sort_index": pid, "rank": rank}
        if role is not None:
            sort_args["role"] = role
        metadata_events.append(
            {
                "name": "process_sort_index",
                "ph": "M",
                "pid": pid,
                "args": sort_args,
            }
        )

    for (pid, tid), (rank, role, original_tid) in tid_labels.items():
        # Retrieve the correct original_pid for this new_pid
        _, _, original_pid = pid_labels[pid]

        rank_text = _format_rank(rank)
        thread_name = existing_thread_names.get(
            (rank, role, original_pid, original_tid)
        )
        if thread_name is None:
            thread_name = f"[Thread {original_tid}]"

        thread_args: dict = {"name": thread_name, "rank": rank}
        if role is not None:
            thread_args["role"] = role

        metadata_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "args": thread_args,
            }
        )
        metadata_events.append(
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "args": {"sort_index": tid, "rank": rank},
            }
        )

    return metadata_events


def _resolve_trace_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.rglob("*.jsonl") if p.is_file())
    matches = [Path(p) for p in glob(str(source), recursive=True)]
    files = [p for p in matches if p.is_file()]
    return sorted(files)


def convert_jsonl_to_chrome_trace(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    *,
    display_time_unit: str = "ms",
) -> dict:
    """Convert newline-delimited trace events into Chrome Trace JSON.

    The ``input_path`` may point to a single JSONL file, a directory containing
    per-rank JSONL files, or a glob pattern. All matching files are concatenated
    in lexical order before emitting the Chrome trace payload.
    """

    sources = _resolve_trace_files(Path(input_path))
    if not sources:
        raise FileNotFoundError(f"No trace files matched input path: {input_path}")

    events: list[dict] = []
    for path in sources:
        events.extend(_load_events(path))

    existing_process_names: dict[tuple[str | int, str | None, object], str] = {}
    existing_thread_names: dict[tuple[str | int, str | None, object, object], str] = {}

    filtered_events: list[dict] = []
    ignored_metadata = {
        "process_name",
        "thread_name",
        "process_sort_index",
        "thread_sort_index",
    }
    for event in events:
        rank = _extract_rank(event)
        role = _extract_role(event)
        if event.get("ph") == "M":
            name = event.get("name")
            args = event.get("args", {})
            pid = event.get("pid")
            tid = event.get("tid")

            if rank is not None and pid is not None:
                if name == "process_name" and isinstance(args, dict):
                    pname = args.get("name")
                    if pname:
                        existing_process_names[(rank, role, pid)] = str(pname)
                elif (
                    name == "thread_name" and tid is not None and isinstance(args, dict)
                ):
                    tname = args.get("name")
                    if tname:
                        existing_thread_names[(rank, role, pid, tid)] = str(tname)

            if name in ignored_metadata:
                continue
        filtered_events.append(event)

    events = filtered_events

    # Collect all unique flow IDs / correlations to remap them sequentially
    # flow_id_keys: (rank, role, flow_id)
    flow_id_keys: set[tuple[str | int, str | None, object]] = set()
    for event in events:
        rank = _extract_rank(event)
        if rank is None:
            continue
        role = _extract_role(event)

        # Collect from flow events
        if event.get("ph") in ("s", "t", "f") and "id" in event:
            flow_id_keys.add((rank, role, event["id"]))

        # Collect from args.correlation
        args = event.get("args")
        if isinstance(args, dict) and "correlation" in args:
            flow_id_keys.add((rank, role, args["correlation"]))

    # Sort and create mapping
    sorted_flow_keys = sorted(
        flow_id_keys,
        key=lambda item: (
            _rank_sort_key(item[0]),
            _role_sort_key(item[1]),
            _value_sort_key(item[2]),
        ),
    )

    flow_id_map = {key: i for i, key in enumerate(sorted_flow_keys, start=1)}

    # Apply mapping
    for event in events:
        rank = _extract_rank(event)
        if rank is None:
            continue
        role = _extract_role(event)

        if event.get("ph") in ("s", "t", "f") and "id" in event:
            key = (rank, role, event["id"])
            if key in flow_id_map:
                event["id"] = flow_id_map[key]

        args = event.get("args")
        if isinstance(args, dict) and "correlation" in args:
            key = (rank, role, args["correlation"])
            if key in flow_id_map:
                args["correlation"] = flow_id_map[key]

    metadata_events = _remap_process_and_thread_ids(
        events,
        existing_process_names=existing_process_names,
        existing_thread_names=existing_thread_names,
    )

    metadata_events.sort(
        key=lambda event: (
            _rank_sort_key(event.get("args", {}).get("rank")),
            _role_sort_key(event.get("args", {}).get("role")),
            _metadata_name_sort_key(event.get("name")),
            _value_sort_key(event.get("pid")),
            _value_sort_key(event.get("tid")),
        )
    )

    events.sort(
        key=lambda event: (
            event.get("ts", 0),
            _value_sort_key(event.get("pid")),
            _value_sort_key(event.get("tid")),
        )
    )

    events = metadata_events + events

    chrome_trace = {
        "traceEvents": events,
        "displayTimeUnit": display_time_unit,
    }

    if output_path is not None:
        destination = Path(output_path)
        if destination.parent != Path(".") and not destination.parent.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as fout:
            json.dump(chrome_trace, fout, ensure_ascii=False)
    return chrome_trace


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PerfTracer JSONL output into Chrome Trace JSON format.",
    )
    parser.add_argument(
        "input",
        type=str,
        help=(
            "Path, directory, or glob pattern for PerfTracer JSONL files "
            "(per-rank outputs allowed)"
        ),
    )
    parser.add_argument(
        "output",
        type=str,
        nargs="?",
        help=(
            "Optional output path for the Chrome Trace JSON file. "
            "If not specified, the output location is inferred from input: "
            "for a directory, outputs to <dir>/traces.json; "
            "for a file, outputs to same dir with .json extension; "
            "for a glob, outputs to common parent dir/traces.json. "
            "Pass '-' to write to stdout."
        ),
    )
    parser.add_argument(
        "--display-time-unit",
        type=str,
        default="ms",
        help="Value for the displayTimeUnit field in the Chrome trace output",
    )
    return parser.parse_args(argv)


def _infer_output_path(input_path: str) -> Path:
    """Infer output path based on input path when output is not specified.

    Rules:
    - If input is a directory: output to <dir>/traces.json
    - If input is a file: output to same dir with .json extension
    - If input is a glob pattern: output to common parent dir/traces.json
    """
    input_as_path = Path(input_path)

    # Case 1: Input is an existing directory
    if input_as_path.is_dir():
        return input_as_path / "traces.json"

    # Case 2: Input is an existing file
    if input_as_path.is_file():
        # Replace .jsonl extension with .json, or just add .json
        if input_as_path.suffix.lower() == ".jsonl":
            return input_as_path.with_suffix(".json")
        else:
            return input_as_path.parent / f"{input_as_path.stem}.json"

    # Case 3: Input might be a glob pattern or non-existent path
    # Try to resolve it and find common parent
    resolved = _resolve_trace_files(input_as_path)
    if resolved:
        # Find common parent directory of all matched files
        if len(resolved) == 1:
            # Single file matched - same as Case 2
            matched_file = resolved[0]
            if matched_file.suffix.lower() == ".jsonl":
                return matched_file.with_suffix(".json")
            else:
                return matched_file.parent / f"{matched_file.stem}.json"
        else:
            # Multiple files - find common parent
            try:
                common_parent = Path(os.path.commonpath([p.parent for p in resolved]))
                return common_parent / "traces.json"
            except ValueError:
                # No common path (e.g., files on different drives on Windows)
                return Path.cwd() / "traces.json"

    # Fallback: treat as a potential directory or use parent
    if "*" in input_path or "?" in input_path:
        # It's a glob pattern - extract the base directory
        base = input_path.split("*")[0].split("?")[0]
        base_path = Path(base).parent if base else Path.cwd()
        return base_path / "traces.json"

    # Default fallback to current directory
    return Path.cwd() / "traces.json"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    emit_stdout = args.output == "-"
    if args.output is None:
        destination: str | os.PathLike[str] | None = _infer_output_path(args.input)
    elif emit_stdout:
        destination = None
    else:
        destination = args.output
    chrome_trace = convert_jsonl_to_chrome_trace(
        args.input,
        destination,
        display_time_unit=args.display_time_unit,
    )
    if emit_stdout:
        json.dump(chrome_trace, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
