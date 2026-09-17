# SPDX-License-Identifier: Apache-2.0

"""Post-process SFT profile artifacts into trace views and summaries."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from areal.tools.perf_trace_converter import write_kernel_profile_trace_views

_MEMORY_LOG_FIELDS = (
    "memory allocated",
    "memory reserved",
    "device memory used/total",
)


def _count_files(paths: Sequence[Path]) -> int:
    return sum(1 for path in paths if path.is_file())


def _max_nvidia_smi_mib(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None

    max_mib: int | None = None
    with path.open("r", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                used = int(row[2].strip())
            except ValueError:
                continue
            max_mib = used if max_mib is None else max(max_mib, used)
    return max_mib


def _max_trainer_memory_gb(log_path: Path | None) -> dict[str, float | None]:
    result: dict[str, float | None] = {field: None for field in _MEMORY_LOG_FIELDS}
    if log_path is None or not log_path.exists():
        return result

    patterns = {
        field: re.compile(rf"{re.escape(field)} \(GB\): ([0-9]+(?:\.[0-9]+)?)")
        for field in _MEMORY_LOG_FIELDS
    }
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        for field, pattern in patterns.items():
            match = pattern.search(line)
            if match is None:
                continue
            value = float(match.group(1))
            current = result[field]
            result[field] = value if current is None else max(current, value)
    return result


def _materialize_kernel_views(log_dir: Path, run_dir: Path) -> dict[str, list[str]]:
    trace_files = sorted(log_dir.glob("perf_tracer/*/traces-*.jsonl"))
    outputs: dict[str, list[str]] = {}
    for trace_file in trace_files:
        generated = write_kernel_profile_trace_views(trace_file)
        archive_dir = run_dir / "kernel_traces" / trace_file.parent.name
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived_paths = []
        for path in generated.values():
            archived_path = archive_dir / path.name
            shutil.copy2(path, archived_path)
            archived_paths.append(str(archived_path))
        outputs[str(trace_file)] = archived_paths
    return outputs


def _archive_memory_snapshots(
    snapshot_files: Sequence[Path],
    *,
    run_dir: Path,
    profile_step: int,
) -> list[str]:
    archive_dir = run_dir / "memory_snapshots" / f"step_{profile_step}"
    archived_paths: list[str] = []
    for snapshot_file in snapshot_files:
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived_path = archive_dir / snapshot_file.name
        shutil.copy2(snapshot_file, archived_path)
        archived_paths.append(str(archived_path))
    return archived_paths


def _write_markdown_summary(
    output_path: Path,
    *,
    profile_kind: str,
    profile_step: int,
    log_dir: Path,
    run_dir: Path,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# SFT Profile Summary",
        "",
        f"- profile_kind: `{profile_kind}`",
        f"- profile_step: `{profile_step}`",
        f"- log_dir: `{log_dir}`",
        f"- run_dir: `{run_dir}`",
        f"- trace_file_count: `{summary['trace_file_count']}`",
        f"- memory_snapshot_count: `{summary['memory_snapshot_count']}`",
        f"- peak_nvidia_smi_mib: `{summary['peak_nvidia_smi_mib']}`",
        "",
        "## Trainer Memory Peaks",
        "",
    ]
    for field, value in summary["trainer_memory_gb"].items():
        lines.append(f"- {field}: `{value}` GB")
    lines.extend(["", "## Kernel Trace Views", ""])
    if summary["kernel_trace_views"]:
        for trace_file, outputs in summary["kernel_trace_views"].items():
            lines.append(f"- `{trace_file}`")
            for output in outputs:
                lines.append(f"  - `{output}`")
    else:
        lines.append("- No kernel trace views generated.")
    lines.extend(["", "## Archived Memory Snapshots", ""])
    if summary["archived_memory_snapshots"]:
        for snapshot in summary["archived_memory_snapshots"]:
            lines.append(f"- `{snapshot}`")
    else:
        lines.append("- No memory snapshots archived.")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def postprocess_profile(
    *,
    log_dir: Path,
    run_dir: Path,
    profile_kind: str,
    profile_step: int,
    trainer_log: Path | None = None,
    nvidia_smi_csv: Path | None = None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)

    trace_files = sorted(log_dir.glob("perf_tracer/*/traces-*.jsonl"))
    snapshot_files = sorted(
        log_dir.glob(f"memory_snapshots/step_{profile_step}/snapshot_*.pickle")
    )
    kernel_views = (
        _materialize_kernel_views(log_dir, run_dir) if profile_kind == "kernel" else {}
    )
    archived_snapshots = _archive_memory_snapshots(
        snapshot_files,
        run_dir=run_dir,
        profile_step=profile_step,
    )
    summary: dict[str, Any] = {
        "profile_kind": profile_kind,
        "profile_step": profile_step,
        "log_dir": str(log_dir),
        "run_dir": str(run_dir),
        "trace_file_count": _count_files(trace_files),
        "memory_snapshot_count": _count_files(snapshot_files),
        "peak_nvidia_smi_mib": _max_nvidia_smi_mib(nvidia_smi_csv),
        "trainer_memory_gb": _max_trainer_memory_gb(trainer_log),
        "kernel_trace_views": kernel_views,
        "archived_memory_snapshots": archived_snapshots,
    }

    json_path = run_dir / "profile_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown_summary(
        run_dir / "profile_summary.md",
        profile_kind=profile_kind,
        profile_step=profile_step,
        log_dir=log_dir,
        run_dir=run_dir,
        summary=summary,
    )
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process AReaL SFT profile artifacts."
    )
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile-kind", choices=["kernel", "memory"], required=True)
    parser.add_argument("--profile-step", type=int, required=True)
    parser.add_argument("--trainer-log", type=Path)
    parser.add_argument("--nvidia-smi-csv", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = postprocess_profile(
        log_dir=args.log_dir,
        run_dir=args.run_dir,
        profile_kind=args.profile_kind,
        profile_step=args.profile_step,
        trainer_log=args.trainer_log,
        nvidia_smi_csv=args.nvidia_smi_csv,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
