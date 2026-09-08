# SPDX-License-Identifier: Apache-2.0

"""Attribute Kineto GPU events to their directly enclosing CPU operator.

Usage: python -m areal.tools.kernel_operator_stats TRACE... --output-dir OUTPUT
Input is the original Chrome JSON (optionally gzip), before track rewriting.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import re
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path
from typing import Any

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
API_CATEGORIES = {"cuda_runtime", "cuda_driver"}
COMM_FIELDS = {
    "collective": "Collective name",
    "process_group": "Process Group Description",
    "group_name": "Process Group Name",
    "group_size": "Group size",
    "group_ranks": "Process Group Ranks",
}
GROUP_FIELDS = ("category", "cpu_operator", "attribution", "component", *COMM_FIELDS)


class ComponentRules:
    """Explicit model/run-specific labels; unmatched events remain OTHER."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.scopes = [
            (re.compile(r["pattern"]), r["component"]) for r in spec["cpu_scopes"]
        ]
        self.kernels = [
            (re.compile(r["pattern"]), r["component"]) for r in spec["gpu_kernels"]
        ]

    @cache
    def scope(self, name: str) -> str:
        return next(
            (label for pattern, label in self.scopes if pattern.fullmatch(name)), ""
        )

    @cache
    def kernel(self, name: str) -> str:
        return next(
            (label for pattern, label in self.kernels if pattern.search(name)), ""
        )

    def communication(self, args: dict) -> str:
        for rule in self.spec["communications"]:
            if all(str(args.get(k)) == str(v) for k, v in rule["match"].items()):
                return rule["component"]
        return "COMMUNICATION_OTHER"


def _find_owners(events, operators, by_thread):
    owners = {}
    for thread, indices in by_thread.items():
        intervals = sorted(operators[thread], key=lambda i: events[i]["ts"])
        active = []
        cursor = 0
        for index in sorted(indices, key=lambda i: events[i]["ts"]):
            timestamp = events[index]["ts"]
            while (
                cursor < len(intervals) and events[intervals[cursor]]["ts"] <= timestamp
            ):
                op_index = intervals[cursor]
                op = events[op_index]
                heapq.heappush(active, (op["dur"], -op["ts"], op_index))
                cursor += 1
            while active:
                op = events[active[0][2]]
                if op["ts"] + op["dur"] > timestamp:
                    break
                heapq.heappop(active)
            if active:
                owners[index] = active[0][2]

    return owners


def attribute_gpu_events(
    events: list[dict[str, Any]], rules: ComponentRules | None = None
) -> tuple[list[dict], dict]:
    """Count each GPU event once, using launch-time CPU nesting on (pid, tid).

    Correlations are local to this input trace. Reused correlations are ambiguous
    unless their APIs form one nested runtime/driver call on the same thread.
    CPU intervals are half-open; GPU execution can outlive the enclosing CPU op.
    Only cpu_op ranges are operators; user annotations are not fallback owners.
    """
    operators = defaultdict(list)
    component_scopes = defaultdict(list)
    apis = defaultdict(list)
    gpu_indices = []
    for index, event in enumerate(events):
        if event.get("ph") != "X":
            continue
        category = event.get("cat")
        if (
            rules
            and category in ("cpu_op", "user_annotation")
            and rules.scope(event["name"])
        ):
            component_scopes[(event["pid"], event["tid"])].append(index)
        if category == "cpu_op":
            operators[(event["pid"], event["tid"])].append(index)
        elif category in API_CATEGORIES:
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                apis[str(correlation)].append(index)
        elif category in GPU_CATEGORIES:
            gpu_indices.append(index)

    # Resolve launch APIs first, then sweep CPU intervals once per thread.
    launches = {}
    ambiguous = set()
    by_thread = defaultdict(list)
    for correlation, indices in apis.items():
        index = min(indices, key=lambda i: events[i]["dur"])
        launch = events[index]
        if any(
            (events[i]["pid"], events[i]["tid"]) != (launch["pid"], launch["tid"])
            or events[i]["ts"] > launch["ts"]
            or events[i]["ts"] + events[i]["dur"] < launch["ts"] + launch["dur"]
            for i in indices
        ):
            ambiguous.add(correlation)
            continue
        launches[correlation] = index
        by_thread[(launch["pid"], launch["tid"])].append(index)

    owners = _find_owners(events, operators, by_thread)
    component_owners = (
        _find_owners(events, component_scopes, by_thread) if rules else {}
    )

    totals = defaultdict(lambda: {"gpu_events": 0, "gpu_time_us": 0.0})
    calls = defaultdict(set)
    reasons = Counter()
    input_time = Counter()
    for index in gpu_indices:
        event = events[index]
        correlation = str(event.get("args", {}).get("correlation"))
        launch_index = launches.get(correlation)
        owner_index = owners.get(launch_index)
        reason = (
            "ambiguous_launch"
            if correlation in ambiguous
            else "no_launch"
            if launch_index is None
            else "no_cpu_operator"
            if owner_index is None
            else "attributed"
        )
        name = (
            events[owner_index]["name"] if owner_index is not None else "[unattributed]"
        )
        owner_args = (
            events[owner_index].get("args", {}) if owner_index is not None else {}
        )
        comm = tuple(
            str(
                owner_args.get(field, "unknown" if name == "record_param_comms" else "")
            )
            for field in COMM_FIELDS.values()
        )
        component = ""
        if rules:
            scope_index = component_owners.get(launch_index)
            component = (
                rules.scope(events[scope_index]["name"])
                if scope_index is not None
                else "OTHER"
            )
            if event["cat"] == "kernel":
                if name == "record_param_comms":
                    component = rules.communication(owner_args)
                else:
                    component = rules.kernel(event["name"]) or component
            elif event["cat"] == "gpu_memcpy":
                component = "MEMCPY"
        key = (event["cat"], name, reason, component, *comm)
        totals[key]["gpu_events"] += 1
        totals[key]["gpu_time_us"] += event["dur"]
        if owner_index is not None:
            calls[key].add(owner_index)
        reasons[reason] += 1
        input_time[event["cat"]] += event["dur"]

    rows = []
    for key, values in totals.items():
        rows.append(
            dict(
                **dict(zip(GROUP_FIELDS, key)),
                **values,
                cpu_calls_with_gpu=len(calls[key]),
            )
        )
    quality = {
        "gpu_events": len(gpu_indices),
        "attribution_counts": dict(reasons),
        "input_gpu_time_us": dict(input_time),
        "aggregated_gpu_events": sum(r["gpu_events"] for r in rows),
        "aggregated_gpu_time_us": {
            cat: sum(r["gpu_time_us"] for r in rows if r["category"] == cat)
            for cat in input_time
        },
    }
    return rows, quality


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "trace",
        "category",
        "cpu_operator",
        "attribution",
        "component",
        *COMM_FIELDS,
        "gpu_events",
        "gpu_time_us",
        "cpu_calls_with_gpu",
        "gpu_time_share_pct",
        "mean_gpu_event_us",
    ]
    category_times = Counter()
    for row in rows:
        category_times[(row["trace"], row["category"])] += row["gpu_time_us"]
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for row in sorted(
            rows, key=lambda r: (r["trace"], r["category"], -r["gpu_time_us"])
        ):
            total = category_times[(row["trace"], row["category"])]
            writer.writerow(
                dict(
                    **row,
                    gpu_time_share_pct=100 * row["gpu_time_us"] / total if total else 0,
                    mean_gpu_event_us=row["gpu_time_us"] / row["gpu_events"],
                )
            )


def write_component_summary(rows: list[dict], path: Path) -> None:
    """Sum exclusive GPU-event components; CPU calls can span components."""
    totals = defaultdict(lambda: [0, 0.0])
    category_times = Counter()
    for row in rows:
        key = (row["component"], row["category"])
        totals[key][0] += int(row["gpu_events"])
        totals[key][1] += float(row["gpu_time_us"])
        category_times[row["category"]] += float(row["gpu_time_us"])
    total_time = sum(category_times.values())
    fields = [
        "component",
        "category",
        "gpu_events",
        "gpu_time_us",
        "all_gpu_time_share_pct",
        "category_time_share_pct",
    ]
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for (component, category), (count, duration) in sorted(
            totals.items(), key=lambda item: -item[1][1]
        ):
            writer.writerow(
                dict(
                    component=component,
                    category=category,
                    gpu_events=count,
                    gpu_time_us=duration,
                    all_gpu_time_share_pct=100 * duration / total_time
                    if total_time
                    else 0,
                    category_time_share_pct=100 * duration / category_times[category]
                    if category_times[category]
                    else 0,
                )
            )


def summarize_traces(
    paths: list[Path], output_dir: Path, rules: ComponentRules | None = None
) -> dict:
    """Read one rank/step trace at a time; merge only completed aggregates."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if rules:
        (output_dir / "component-rules.json").write_text(
            json.dumps(rules.spec, indent=2)
        )
    per_trace = []
    quality = {}
    for path in dict.fromkeys(p.resolve() for p in paths):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as fin:
            trace = json.load(fin)
        rows, checks = attribute_gpu_events(trace["traceEvents"], rules)
        del trace
        per_trace.extend(dict(trace=str(path), **row) for row in rows)
        quality[str(path)] = checks
        # Persist progress without retaining raw events from earlier traces.
        (output_dir / "validation.json").write_text(json.dumps(quality, indent=2))
    combined = {}
    for row in per_trace:
        key = tuple(row[field] for field in GROUP_FIELDS)
        if key not in combined:
            combined[key] = dict(row, trace="all")
        else:
            for field in ("gpu_events", "gpu_time_us", "cpu_calls_with_gpu"):
                combined[key][field] += row[field]
    _write_csv(output_dir / "operators-per-trace.csv", per_trace)
    _write_csv(output_dir / "operators-all-traces.csv", list(combined.values()))
    if rules:
        write_component_summary(
            list(combined.values()), output_dir / "component-summary.csv"
        )
    return quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--component-rules", type=Path)
    args = parser.parse_args()
    rules = (
        ComponentRules(json.loads(args.component_rules.read_text()))
        if args.component_rules
        else None
    )
    summarize_traces(args.traces, args.output_dir, rules)


if __name__ == "__main__":
    main()
