# SPDX-License-Identifier: Apache-2.0

"""Observe a frozen Expert event stream without changing its scheduling rules."""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from areal.elastic.policy.basic.serving import ServingMetrics, ServingSLO
from areal.elastic.policy.reference.scenario import Group, Request


class ServingInfeasible(Exception):
    """Too many requests have irreversibly violated the request-level SLO."""


@dataclass
class RequestLatency:
    request_id: str
    expected_tokens: int
    ready_at: float | None = None
    first_token_at: float | None = None
    last_token_at: float | None = None
    generated_tokens: int = 0
    maximum_tbt_seconds: float = 0.0
    completed: bool = False

    @property
    def ttft_seconds(self) -> float | None:
        if self.first_token_at is None or self.ready_at is None:
            return None
        return self.first_token_at - self.ready_at


class ServingObserver:
    """TTFT starts at logical readiness, before admission or backend assignment.

    Initial model nodes are ready at external episode arrival. Later nodes are
    ready when all dependencies finish. Tokens are simulator generation events,
    not measured client-visible streaming; unmeasured network tails stay explicit.
    """

    def __init__(self, groups: tuple[Group, ...], slo: ServingSLO) -> None:
        self.slo = slo
        self.requests: dict[str, RequestLatency] = {}
        self.dependents: dict[str, list[str]] = defaultdict(list)
        self.remaining: dict[str, int] = {}
        self.deadlines: list[tuple[float, str]] = []
        self.bad_requests: set[str] = set()
        self.behavior: dict[str, Counter[int]] = defaultdict(Counter)
        self.training_checks: list[dict[str, Any]] = []
        for group in groups:
            for node in group.nodes:
                key = f"{group.group_id}/{node.name}"
                self.remaining[key] = len(node.predecessors)
                for predecessor in node.predecessors:
                    self.dependents[f"{group.group_id}/{predecessor}"].append(key)
                if isinstance(node, Request):
                    self.requests[key] = RequestLatency(key, node.output_tokens)
                    if not node.predecessors:
                        self.ready(key, group.available_at)
        self.required_passes = math.ceil(
            len(self.requests) * slo.minimum_request_fraction - 1e-12
        )

    def ready(self, key: str, when: float) -> None:
        self.requests[key].ready_at = when
        heapq.heappush(self.deadlines, (when + self.slo.ttft_seconds, key))

    def observe(
        self,
        kind: str,
        subject: str,
        when: float,
        details: dict[str, Any],
        *,
        actor_version: int,
    ) -> None:
        if kind == "token_generated":
            request = self.requests[subject]
            if request.ready_at is None:
                raise ValueError("Token emitted before its request became ready")
            if details["output_index"] != request.generated_tokens + 1:
                raise ValueError("Token observation must preserve output order")
            if request.last_token_at is None:
                request.first_token_at = when
            else:
                request.maximum_tbt_seconds = max(
                    request.maximum_tbt_seconds, when - request.last_token_at
                )
            request.generated_tokens += 1
            request.last_token_at = when
            self.behavior[subject.rsplit("/", 1)[0]][details["version"]] += 1
            if (
                request.ttft_seconds > self.slo.ttft_seconds + 1e-12
                or request.maximum_tbt_seconds > self.slo.tbt_seconds + 1e-12
            ):
                self.bad_requests.add(subject)
        elif kind in ("request_complete", "external_complete"):
            if kind == "request_complete":
                self.requests[subject].completed = True
            for child in self.dependents[subject]:
                self.remaining[child] -= 1
                if self.remaining[child] == 0 and child in self.requests:
                    self.ready(child, when)
        elif kind == "actor_start":
            versions: Counter[int] = Counter()
            for group in details["groups"]:
                versions.update(self.behavior[group])
            self.training_checks.append(
                {
                    "step": subject,
                    "time": when,
                    "preupdate_actor_version": actor_version,
                    "generated_tokens": sum(versions.values()),
                    "behavior_version_counts": dict(versions),
                    "maximum_version_lag": actor_version - min(versions)
                    if versions
                    else None,
                }
            )

    def check_possible(self, when: float) -> None:
        """A failure budget is a safe early-stop bound, never a passing shortcut."""
        while self.deadlines and self.deadlines[0][0] < when - 1e-12:
            deadline, key = heapq.heappop(self.deadlines)
            first = self.requests[key].first_token_at
            if first is None or first > deadline + 1e-12:
                self.bad_requests.add(key)
        if len(self.requests) - len(self.bad_requests) < self.required_passes:
            raise ServingInfeasible(
                f"Serving SLO cannot reach {self.slo.minimum_request_fraction:.2%}: "
                f"{len(self.bad_requests)}/{len(self.requests)} requests already failed"
            )

    def metrics(self) -> ServingMetrics:
        complete = [
            r
            for r in self.requests.values()
            if r.completed and r.generated_tokens == r.expected_tokens
        ]
        ttft = {
            r.request_id
            for r in complete
            if r.ttft_seconds <= self.slo.ttft_seconds + 1e-12
        }
        tbt = {
            r.request_id
            for r in complete
            if r.maximum_tbt_seconds <= self.slo.tbt_seconds + 1e-12
        }
        observed = [r for r in self.requests.values() if r.first_token_at is not None]
        return ServingMetrics(
            self.slo,
            len(self.requests),
            len(complete),
            len(ttft),
            len(tbt),
            len(ttft & tbt),
            max((r.ttft_seconds for r in observed), default=None),
            max((r.maximum_tbt_seconds for r in observed), default=None),
            "logical_request_ready_to_simulated_token; admission included; client/network tails unmeasured",
        )

    def export(self) -> dict[str, Any]:
        return {
            "serving": asdict(self.metrics()),
            "maximum_possible_request_fraction": (
                len(self.requests) - len(self.bad_requests)
            )
            / len(self.requests),
            "irreversibly_failed_requests": sorted(self.bad_requests),
            "requests": [
                dict(asdict(r), ttft_seconds=r.ttft_seconds)
                for r in self.requests.values()
            ],
            "training_version_checks": self.training_checks,
            "actual_version_lag_is_observation_only": True,
        }
