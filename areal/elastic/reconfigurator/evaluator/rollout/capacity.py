# SPDX-License-Identifier: Apache-2.0

"""Bounded serial probes of rollout SLO capacity."""

from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class RolloutCapacity:
    lower: float | None
    upper: float | None
    probes: tuple[tuple[float, float | None], ...]
    seconds: float

    @property
    def rate(self):
        # Use demonstrated capacity when available. With only a failing rate,
        # retain its upper bound as an optimistic search estimate.
        return self.lower if self.lower is not None else self.upper


def probe_capacity(check, observed_rate, target, *, max_probes=4):
    started = perf_counter()
    lower = upper = None
    probes = []
    rate = observed_rate
    for _ in range(max_probes):
        score = check(rate)
        probes.append((rate, score))
        if score is None:
            break
        if score + 1e-12 >= target:
            lower = rate
        else:
            upper = rate
        if lower is None:
            rate /= 2
        elif upper is None:
            rate *= 2
        else:
            rate = (lower + upper) / 2
    return RolloutCapacity(lower, upper, tuple(probes), perf_counter() - started)
