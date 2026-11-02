# SPDX-License-Identifier: Apache-2.0

"""Read existing summary rows, optional rank logs and Chrome complete spans.

No new training instrumentation. CPU wall spans are not CUDA kernel durations.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections import defaultdict
from typing import Any

STAGES = ("recompute_logp", "compute_advantage", "train_step")


def stage_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    result = {}
    for stage in STAGES:
        value = metrics.get(f"timeperf/{stage}")
        if value is not None:
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid timeperf/{stage}")
            result[stage] = float(value)
    return result


def parse_microbatch_log(line: str) -> dict | None:
    """One engine invocation on one rank; phase/step must come from caller context.

    Do not use the number of log lines as the global synchronized mb count.
    """
    if "microbatch #tokens:" in line:
        backend = "megatron"
        match = re.search(
            r"microbatch #tokens:\s*(\[[\d,\s]*\]).*padded to:\s*(\[[\d,\s]*\])", line
        )
    elif "Microbatch #tokens (rank " in line:
        backend = "archon"
        match = re.search(
            r"Microbatch #tokens \(rank \d+\):\s*(\[[\d,\s]*\]), padded to:\s*(\[[\d,\s]*\])",
            line,
        )
    else:
        return None
    if match is None:
        raise ValueError("Malformed engine microbatch log")
    tokens, padded = (ast.literal_eval(group) for group in match.groups())
    if (
        not tokens
        or len(tokens) != len(padded)
        or any(t <= 0 or p < t for t, p in zip(tokens, padded, strict=True))
    ):
        raise ValueError("Inconsistent engine token/padding lists")
    rank = re.search(r"(?:rank |Rank )(\d+)", line)
    return {
        "backend": backend,
        "rank": int(rank[1]) if rank else None,
        "microbatches": len(tokens),
        "real_tokens_per_microbatch": tokens,
        "padded_tokens_per_microbatch": padded,
        "phase": None,
        "global_step": None,
    }


def trace_events(text: str) -> list[dict]:
    """Chrome JSON object/list, or AReaL JSONL complete events."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        data = data.get("traceEvents", [data])
    if not isinstance(data, list) or any(not isinstance(event, dict) for event in data):
        raise ValueError("Expected a Chrome event list or JSONL events")
    return data


def _union_seconds(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, -math.inf
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total / 1e6


def actor_trace_times(
    events: list[dict], *, level: str = "auto", global_step: int | None = None
) -> dict:
    """Union nested spans per rank, max ranks, never sum trainer + worker scopes.

    A missing step can inherit from a containing same-thread span. Otherwise it
    remains unassigned unless the caller explicitly supplies a single-step id.
    """
    names = {
        "trainer": {
            "train.recompute_logp": "recompute_logp",
            "train.compute_advantage": "compute_advantage",
            "train.ppo_update": "train_step",
        },
        "worker": {
            "ppo_actor.compute_logp": "recompute_logp",
            "ppo_actor.compute_advantages": "compute_advantage",
            "ppo_actor.ppo_update": "train_step",
        },
    }
    if level not in {"auto", "trainer", "worker"}:
        raise ValueError("level must be auto, trainer or worker")
    complete = []
    for event in events:
        if event.get("ph") != "X":
            continue
        start, duration = event.get("ts"), event.get("dur")
        if (
            not isinstance(start, int | float)
            or not isinstance(duration, int | float)
            or not math.isfinite(start)
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ValueError("Invalid complete-event timestamp/duration")
        complete.append(event)
    if level == "auto":
        level = (
            "trainer"
            if any(event.get("name") in names["trainer"] for event in complete)
            else "worker"
        )
    intervals = defaultdict(list)
    unassigned = 0
    for event in complete:
        stage = names[level].get(event.get("name"))
        if stage is None:
            continue
        step = event.get("args", {}).get("global_step")
        if step is None:
            parents = [
                parent
                for parent in complete
                if parent is not event
                and parent.get("pid") == event.get("pid")
                and parent.get("tid") == event.get("tid")
                and parent.get("args", {}).get("rank")
                == event.get("args", {}).get("rank")
                and parent["ts"] <= event["ts"]
                and parent["ts"] + parent["dur"] >= event["ts"] + event["dur"]
                and "global_step" in parent.get("args", {})
            ]
            if parents:
                step = min(parents, key=lambda parent: parent["dur"])["args"][
                    "global_step"
                ]
        step = step if step is not None else global_step
        if step is None:
            unassigned += 1
            continue
        intervals[
            (
                str(step),
                stage,
                str(event.get("pid")),
                str(event.get("tid")),
                str(event.get("args", {}).get("rank")),
            )
        ].append((event["ts"], event["ts"] + event["dur"]))
    output = defaultdict(dict)
    for (step, stage, _, _, _), spans in intervals.items():
        output[step][stage] = max(output[step].get(stage, 0), _union_seconds(spans))
    return {
        "level": level,
        "steps_seconds": dict(output),
        "unassigned_events": unassigned,
        "scope": "CPU wall scopes; nested interval union then rank/thread maximum, not CUDA execution time",
    }
