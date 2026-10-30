# SPDX-License-Identifier: Apache-2.0

"""Advisory SWE progress and empirical remaining-time estimates for the demo."""

from __future__ import annotations

import json
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import aiofiles
import aiofiles.os

STAGES = ("initializing", "rollout", "grading", "cleanup")
TERMINAL = {"completed", "failed", "cancelled"}


class EpisodeProgress:
    """One writer per episode; observation failures never change its reward."""

    def __init__(self, directory: Path, *, task_id: str, group_id, sample_idx):
        self.path = directory / "progress.json"
        self.identity = dict(
            task_id=task_id,
            group_id=group_id,
            sample_idx=sample_idx,
            episode_id=directory.name,
        )
        self.stage = None
        self.entered = time.monotonic()
        self.durations: dict[str, float] = {}
        self.error = None

    async def mark(self, stage: str, *, phase: str | None = None, turn: int = 0):
        now = time.monotonic()
        if stage != self.stage:
            if self.stage in STAGES:
                self.durations[self.stage] = now - self.entered
            self.stage, self.entered = stage, now
        record = {
            **self.identity,
            "stage": stage,
            "phase": phase or stage,
            "turn": turn,
            "stage_started": self.entered,
            "stage_seconds": dict(self.durations),
            "wall_time": time.time(),
        }
        try:
            # Atomic visibility without checkpoint-style durability/fsync costs.
            async with aiofiles.open(self.path.with_suffix(".tmp"), "w") as stream:
                await stream.write(json.dumps(record))
            await aiofiles.os.replace(self.path.with_suffix(".tmp"), self.path)
        except OSError as error:
            self.error = f"{type(error).__name__}: {error}"


class SWEProgress:
    """Local file observations only; never controls native admission or readiness."""

    def __init__(
        self,
        output: Path,
        *,
        startup_seconds: float = 90,
        checkpoint_seconds: float = 40,
        release_seconds: float = 15,
        margin_seconds: float = 15,
    ):
        self.root = output / "swe-episodes"
        self.priors = {
            "startup_seconds": startup_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "release_seconds": release_seconds,
        }
        self.margin = margin_seconds
        self._cache = {}
        self._last_scan = float("-inf")
        self._read_errors = 0
        self._lock = threading.Lock()

    def _records(self, now):
        if now - self._last_scan >= 1:
            self._last_scan = now
            self._read_errors = 0
            for path in self.root.glob("*/progress.json"):
                try:
                    stamp = path.stat().st_mtime_ns
                    if path not in self._cache or self._cache[path][0] != stamp:
                        self._cache[path] = (stamp, json.loads(path.read_text()))
                except (OSError, ValueError):
                    self._read_errors += 1
                    self._cache.pop(path, None)
        return [record for _, record in self._cache.values()]

    @staticmethod
    def _remaining(record, history, now):
        if record["stage"] == "completed":
            return {"low": 0.0, "high": 0.0, "samples": 0, "source": "completed"}
        if record["stage"] not in STAGES:
            return None
        stage = record["stage"]
        elapsed = max(0.0, now - record["stage_started"])
        matched = [h for h in history if h["task_id"] == record["task_id"]]
        source = "same_task" if len(matched) >= 3 else "mixed_tasks"
        cohort = matched if len(matched) >= 3 else history
        # Conditional empirical durations; an overdue stage has unknown ETA,
        # rather than incorrectly becoming zero seconds remaining.
        remaining = sorted(
            sum(h["stage_seconds"].get(s, 0) for s in STAGES[STAGES.index(stage) :])
            - elapsed
            for h in cohort
            if h["stage_seconds"].get(stage, -1) >= elapsed
        )
        if len(remaining) < 3:
            return None
        return {
            "low": remaining[int((len(remaining) - 1) * 0.25)],
            "high": remaining[int((len(remaining) - 1) * 0.9)],
            "samples": len(remaining),
            "source": source,
        }

    def __call__(self, status: dict) -> dict:
        with self._lock:
            now = time.monotonic()
            records = self._records(now)
            history = [r for r in records if r["stage"] == "completed"]
            grouped = defaultdict(list)
            for record in records:
                if record["group_id"] is not None:
                    grouped[record["group_id"]].append(record)
            groups, signature, estimates = [], [], []
            group_size = status["group_size"]
            for group_id, members in sorted(grouped.items()):
                if all(m["stage"] in TERMINAL for m in members):
                    continue
                phases = dict(Counter(m["phase"] for m in members))
                stages = dict(Counter(m["stage"] for m in members))
                # Generation/tool turns are visible but don't wake the LLM.
                signature.append((group_id, sorted(stages.items())))
                remaining = [self._remaining(m, history, now) for m in members]
                indices = {m["sample_idx"] for m in members}
                # Native v2 does not populate sample_idx. Each run() has a
                # unique episode directory; use those identities for advisory
                # membership only. Partial/duplicate attempts remain unknown.
                complete_membership = len(members) == group_size and (
                    indices == set(range(group_size))
                    or indices == {None}
                    and len({m["episode_id"] for m in members}) == group_size
                )
                eta = None
                if complete_membership and all(r is not None for r in remaining):
                    eta = {
                        "low": max(r["low"] for r in remaining),
                        "high": max(r["high"] for r in remaining),
                    }
                    estimates.append(eta)
                groups.append(
                    {
                        "group_id": group_id,
                        "task_id": members[0]["task_id"],
                        "phases": phases,
                        "observed_samples": len(members),
                        "samples": [
                            {
                                "sample_idx": m["sample_idx"],
                                "episode_id": m["episode_id"],
                                "phase": m["phase"],
                                "turn": m["turn"],
                                "stage_elapsed_seconds": max(
                                    0, now - m["stage_started"]
                                ),
                                "remaining_seconds": estimate,
                            }
                            for m, estimate in zip(members, remaining)
                        ],
                        "remaining_seconds": eta,
                    }
                )
            needed = max(0, status["batch_size"] - status["counts"].get("ready", 0))
            eta = None
            if needed == 0:
                eta = {"low": 0.0, "high": 0.0}
            elif len(estimates) >= needed:
                eta = {
                    k: sorted(e[k] for e in estimates)[needed - 1]
                    for k in ("low", "high")
                }
            costs = {}
            measured = status.get("last_measured_durations", {})
            for key, prior in self.priors.items():
                costs[key] = {
                    "seconds": measured.get(key, prior),
                    "source": "current_run" if key in measured else "configured_prior",
                }
            resume_budget = costs["startup_seconds"]["seconds"] + self.margin
            suspend_cost = (
                costs["checkpoint_seconds"]["seconds"]
                + costs["release_seconds"]["seconds"]
                if status.get("actor_running", True)
                else 0
            )
            return {
                "event_signature": json.dumps(signature),
                "groups": groups,
                "completed_history_samples": len(history),
                "batch_remaining_seconds": eta,
                "eta_confidence": "unknown" if eta is None else "low",
                "eta_method": "Empirical stage durations, slowest sample per group; descriptive range, not a bound. Unknown for insufficient/overdue history or missing members. Native export/admission alone decides readiness.",
                "read_errors": self._read_errors,
                "lifecycle_costs": costs,
                "restore_margin_seconds": self.margin,
                "start_within_seconds": None
                if eta is None
                else max(0, eta["low"] - resume_budget),
                "estimated_free_window_seconds": None
                if eta is None
                else (eta["low"] - resume_budget - suspend_cost),
            }
