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


def rollout_slo(pattern, rate, config, model, slo, concurrency):
    """Replay a 60s workload sample in isolation at an offered rate."""
    from dataclasses import replace

    from .replay import offered_fraction
    from .sglang import scheduler_for

    period = len(pattern.arrivals) / rate
    scaled = replace(
        pattern,
        period_seconds=period,
        arrivals=tuple(
            (offset * period / pattern.period_seconds, family)
            for offset, family in pattern.arrivals
        ),
    )
    # Cover at least the first offer when probing a low arrival rate.
    forecast = scaled.forecast(0.0, max(60.0, scaled.arrivals[0][0] + 1.0))
    if forecast is None:
        raise ValueError("Capacity estimates require observed shapes for all families")
    # Allow final offers to produce the first call's TPOT measurement.
    forecast = replace(forecast, input_closes_at=forecast.input_closes_at + 75.0)
    return offered_fraction(
        forecast, config, model, scheduler_for("sglang"), slo, concurrency
    )
