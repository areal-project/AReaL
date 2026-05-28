# SPDX-License-Identifier: Apache-2.0

"""Pluggable routing strategies for worker selection."""

from __future__ import annotations

from typing import Protocol

from areal.experimental.inference_service.router.state import WorkerInfo


class RoutingStrategy(Protocol):
    """Protocol for routing strategy implementations."""

    def pick(self, workers: list[WorkerInfo]) -> WorkerInfo | None:
        """Select a worker from the list, or return None if empty."""
        ...


class RoundRobinStrategy:
    """Cycle through workers in order."""

    def __init__(self) -> None:
        self._idx = 0

    def pick(self, workers: list[WorkerInfo]) -> WorkerInfo | None:
        if not workers:
            return None
        w = workers[self._idx % len(workers)]
        self._idx += 1
        return w


def get_strategy(name: str) -> RoutingStrategy:
    """Instantiate a routing strategy by name."""
    if name == "round_robin":
        return RoundRobinStrategy()
    if name == "least_busy":
        raise NotImplementedError(
            "least_busy strategy is not yet implemented: "
            "active_requests tracking is required but not wired up. "
            "Use 'round_robin' instead."
        )
    raise ValueError(f"Unknown routing strategy: {name}")
