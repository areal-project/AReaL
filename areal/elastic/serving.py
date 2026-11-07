# SPDX-License-Identifier: Apache-2.0

"""Request-level serving feasibility, independent of any simulator or runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass

from areal.elastic.contracts import nonnegative, positive_int

from .contracts import OnlineObservation, RequestView, RollingSLO


@dataclass(frozen=True)
class ServingSLO:
    ttft_seconds: float
    tpot_seconds: float
    minimum_request_fraction: float

    def __post_init__(self) -> None:
        nonnegative("ttft_seconds", self.ttft_seconds, positive=True)
        nonnegative("tpot_seconds", self.tpot_seconds, positive=True)
        nonnegative(
            "minimum_request_fraction", self.minimum_request_fraction, positive=True
        )
        if self.minimum_request_fraction > 1:
            raise ValueError("minimum_request_fraction must be at most one")

    def rejection(self, metrics: ServingMetrics | None, expected: int) -> str:
        """Missing coverage is not evidence that an SLO was met."""
        if metrics is None:
            return "Serving SLO requires request-level metrics"
        if metrics.slo != self:
            return "Serving metrics use different SLO thresholds"
        if (
            metrics.expected_requests != expected
            or metrics.completed_requests != expected
        ):
            return "Serving metrics did not cover the identical complete request window"
        if metrics.joint_fraction + 1e-12 < self.minimum_request_fraction:
            return (
                f"Serving SLO: {metrics.joint_met_requests}/{expected} requests "
                f"jointly meet TTFT/TPOT; required {self.minimum_request_fraction:.2%}"
            )
        return ""


@dataclass(frozen=True)
class ServingMetrics:
    """A request passes when TTFT and its mean time per output token pass.

    expected_requests includes requests delayed before admission. Tool execution
    between requests is excluded from TPOT. Single-token outputs have no
    decode interval and use zero TPOT.
    """

    slo: ServingSLO
    expected_requests: int
    completed_requests: int
    ttft_met_requests: int
    tpot_met_requests: int
    joint_met_requests: int
    maximum_ttft_seconds: float | None
    maximum_tpot_seconds: float | None
    timing_boundary: str

    def __post_init__(self) -> None:
        positive_int("expected_requests", self.expected_requests)
        for name in (
            "completed_requests",
            "ttft_met_requests",
            "tpot_met_requests",
            "joint_met_requests",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= self.expected_requests:
                raise ValueError(f"Invalid {name}")
        if (
            max(self.ttft_met_requests, self.tpot_met_requests)
            > self.completed_requests
        ):
            raise ValueError("Only completed requests contribute passing counts")
        if self.joint_met_requests > min(
            self.ttft_met_requests, self.tpot_met_requests
        ):
            raise ValueError("Joint passes cannot exceed either marginal pass count")
        for name in ("maximum_ttft_seconds", "maximum_tpot_seconds"):
            if getattr(self, name) is not None:
                nonnegative(name, getattr(self, name))
        if not self.timing_boundary:
            raise ValueError("An explicit timing boundary is required")

    @property
    def joint_fraction(self) -> float:
        return self.joint_met_requests / self.expected_requests


def request_tpot_seconds(request: RequestView) -> float | None:
    """Mean decode gap within one model request; missing timing stays unknown."""
    if request.first_token_at is None or request.generated < 1:
        return None
    if request.generated == 1:
        return 0.0
    if request.last_token_at is None:
        return None
    return (request.last_token_at - request.first_token_at) / (request.generated - 1)


def request_slo_status(
    request: RequestView, slo: ServingSLO, *, now: float
) -> bool | None:
    """Only finalized TPOT can fail: later tokens can improve a pending mean."""
    first = request.first_token_at
    if first is None:
        if (
            request.completed_at is None
            and now - request.ready_at > slo.ttft_seconds + 1e-12
        ):
            return False
        return None
    if first - request.ready_at > slo.ttft_seconds + 1e-12:
        return False
    if request.completed_at is None:
        return None
    tpot = request_tpot_seconds(request)
    return None if tpot is None else tpot <= slo.tpot_seconds + 1e-12


def rolling_slo(view: OnlineObservation, slo: ServingSLO, seconds: float) -> RollingSLO:
    rows = [
        r
        for r in view.requests
        if r.completed_at is None or r.completed_at >= view.now - seconds
    ]
    complete = passed = decided = unknown_completed = 0
    ttfts, tpots = [], []
    for request in rows:
        first = request.first_token_at
        if first is not None:
            ttfts.append(first - request.ready_at)
        elif request.completed_at is None:
            ttfts.append(view.now - request.ready_at)
        tpot = request_tpot_seconds(request)
        if tpot is not None:
            tpots.append(tpot)
        status = request_slo_status(request, slo, now=view.now)
        if request.completed_at is not None:
            complete += 1
            unknown_completed += status is None
        if status is not None:
            decided += 1
            passed += status

    def quantile(values):
        return (
            sorted(values)[
                max(0, math.ceil(len(values) * slo.minimum_request_fraction) - 1)
            ]
            if values
            else 0.0
        )

    return RollingSLO(
        complete,
        len(rows) - complete,
        decided,
        passed,
        passed / decided if decided and not unknown_completed else None,
        quantile(ttfts),
        quantile(tpots),
    )
