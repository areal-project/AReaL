# SPDX-License-Identifier: Apache-2.0

"""Cheap, causal workload summaries used to gate expensive inline searches."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import fmean

from .observations import Turn


@dataclass(frozen=True)
class InlineMoments:
    mean: float
    p90: float
    cv: float

    @classmethod
    def from_values(cls, values: tuple[int, ...]) -> InlineMoments:
        mean = fmean(values)
        ordered = sorted(values)
        position = (len(ordered) - 1) * 0.9
        index = int(position)
        fraction = position - index
        p90 = (
            ordered[index] * (1 - fraction)
            + ordered[min(index + 1, len(ordered) - 1)] * fraction
        )
        deviation = sqrt(fmean((value - mean) ** 2 for value in values))
        return cls(mean, p90, deviation / max(mean, 1e-12))

    def distance(self, other: InlineMoments) -> float:
        return max(
            abs(self.mean - other.mean) / max(self.mean, other.mean, 1),
            abs(self.p90 - other.p90) / max(self.p90, other.p90, 1),
            # Near-zero CV noise must not appear to be a huge relative change.
            abs(self.cv - other.cv) / max(self.cv, other.cv, 0.5),
        )


@dataclass(frozen=True)
class InlineWorkloadFeatures:
    episodes: int
    initial_input: InlineMoments
    generated: InlineMoments
    actor_tokens: InlineMoments
    maximum_context: InlineMoments
    mean_turns: float
    multiturn_fraction: float

    @classmethod
    def from_turns(
        cls, episodes: tuple[tuple[Turn, ...], ...]
    ) -> InlineWorkloadFeatures:
        """Only complete, already published training episodes enter this window."""
        return cls(
            len(episodes),
            InlineMoments.from_values(tuple(e[0].input_tokens for e in episodes)),
            InlineMoments.from_values(
                tuple(sum(t.output_tokens for t in e) for e in episodes)
            ),
            InlineMoments.from_values(
                tuple(
                    sum(t.input_tokens + t.output_tokens for t in e) for e in episodes
                )
            ),
            InlineMoments.from_values(
                tuple(
                    max(t.input_tokens + t.output_tokens for t in e) for e in episodes
                )
            ),
            fmean(len(e) for e in episodes),
            fmean(len(e) > 1 for e in episodes),
        )

    def distance(self, other: InlineWorkloadFeatures) -> float:
        return max(
            self.initial_input.distance(other.initial_input),
            self.generated.distance(other.generated),
            self.actor_tokens.distance(other.actor_tokens),
            self.maximum_context.distance(other.maximum_context),
            abs(self.mean_turns - other.mean_turns)
            / max(self.mean_turns, other.mean_turns, 1),
            abs(self.multiturn_fraction - other.multiturn_fraction),
        )


@dataclass(frozen=True)
class InlineTriggerCheck:
    update: int
    search: bool
    reason: str
    distance: float
    confirmations: int
    features: InlineWorkloadFeatures


class InlineSearchTrigger:
    def __init__(self, *, change_ratio: float, change_checks: int, maximum_gap: int):
        self.change_ratio = change_ratio
        self.change_checks = change_checks
        self.maximum_gap = maximum_gap
        self.last_probe_update = 0
        self.last_search_update = 0
        self.anchor: InlineWorkloadFeatures | None = None
        self.confirmations = 0

    def check(
        self, update: int, features: InlineWorkloadFeatures
    ) -> InlineTriggerCheck:
        self.last_probe_update = update
        distance = features.distance(self.anchor) if self.anchor is not None else 0.0
        self.confirmations = (
            self.confirmations + 1 if distance >= self.change_ratio else 0
        )
        if self.anchor is None:
            reason = "initial"
        elif self.confirmations >= self.change_checks:
            reason = "workload_change"
        elif update - self.last_search_update >= self.maximum_gap:
            reason = "refresh"
        else:
            reason = "confirming_change" if self.confirmations else "stable"
        search = reason in {"initial", "workload_change", "refresh"}
        result = InlineTriggerCheck(
            update, search, reason, distance, self.confirmations, features
        )
        if search:
            self.anchor, self.last_search_update = features, update
            self.confirmations = 0
        return result
