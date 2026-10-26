# SPDX-License-Identifier: Apache-2.0

"""Request-level serving feasibility, independent of any simulator or runtime."""

from __future__ import annotations

from dataclasses import dataclass

from .cost_model.rollout import nonnegative, positive_int


@dataclass(frozen=True)
class ServingSLO:
    ttft_seconds: float
    tbt_seconds: float
    minimum_request_fraction: float

    def __post_init__(self) -> None:
        nonnegative("ttft_seconds", self.ttft_seconds, positive=True)
        nonnegative("tbt_seconds", self.tbt_seconds, positive=True)
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
                f"jointly meet TTFT/TBT; required {self.minimum_request_fraction:.2%}"
            )
        return ""


@dataclass(frozen=True)
class ServingMetrics:
    """A request passes only when both TTFT and its largest token gap pass.

    expected_requests includes requests delayed before admission. Tool execution
    between requests is never an inter-token gap. Single-token outputs have no
    token gap and satisfy TBT with a value of zero.
    """

    slo: ServingSLO
    expected_requests: int
    completed_requests: int
    ttft_met_requests: int
    tbt_met_requests: int
    joint_met_requests: int
    maximum_ttft_seconds: float | None
    maximum_tbt_seconds: float | None
    timing_boundary: str

    def __post_init__(self) -> None:
        positive_int("expected_requests", self.expected_requests)
        for name in (
            "completed_requests",
            "ttft_met_requests",
            "tbt_met_requests",
            "joint_met_requests",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= self.expected_requests:
                raise ValueError(f"Invalid {name}")
        if max(self.ttft_met_requests, self.tbt_met_requests) > self.completed_requests:
            raise ValueError("Only completed requests contribute passing counts")
        if self.joint_met_requests > min(self.ttft_met_requests, self.tbt_met_requests):
            raise ValueError("Joint passes cannot exceed either marginal pass count")
        for name in ("maximum_ttft_seconds", "maximum_tbt_seconds"):
            if getattr(self, name) is not None:
                nonnegative(name, getattr(self, name))
        if not self.timing_boundary:
            raise ValueError("An explicit timing boundary is required")

    @property
    def joint_fraction(self) -> float:
        return self.joint_met_requests / self.expected_requests
