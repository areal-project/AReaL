# SPDX-License-Identifier: Apache-2.0

"""Conditional RPC timing, separate from measured GPU-iteration calibration."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..core import Estimate, code_revision, fingerprint, integer, nonnegative
from .rollout import ExpertRolloutModel, RolloutConfig


@dataclass(frozen=True)
class RPCTimingModel:
    """Effective service scale plus an asynchronous response tail.

    Fit these jointly against RPC labels using calibration-only queue simulations.
    RPC labels cannot identify GPU efficiency or a physical response component;
    both estimates remain assumed, even inside the request-length bounds. Batch,
    cache and KV coverage require separate observations. No fitting/search here.
    """

    version: str
    profile: str
    service_scale: float
    response_seconds: float
    sample_ids: tuple[str, ...]
    request_bounds: tuple[tuple[str, float, float], ...]
    allow_request_extrapolation: bool = False

    @staticmethod
    def profile_for(base: ExpertRolloutModel, config: RolloutConfig) -> str:
        """Bind the entire frozen service, including any GPU calibration."""
        return fingerprint(code_revision(), base, config)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids))
        object.__setattr__(
            self, "request_bounds", tuple(tuple(b) for b in self.request_bounds)
        )
        if not self.version or not self.profile or not self.sample_ids:
            raise ValueError("RPC timing needs version, base profile and sample IDs")
        if any(not s for s in self.sample_ids) or len(set(self.sample_ids)) != len(
            self.sample_ids
        ):
            raise ValueError("RPC sample IDs must be nonempty and unique")
        nonnegative("service_scale", self.service_scale, positive=True)
        nonnegative("response_seconds", self.response_seconds)
        if type(self.allow_request_extrapolation) is not bool:
            raise ValueError("allow_request_extrapolation must be boolean")
        if len(self.request_bounds) != 2 or {b[0] for b in self.request_bounds} != {
            "input_tokens",
            "output_tokens",
        }:
            raise ValueError("RPC bounds must contain input_tokens and output_tokens")
        for name, low, high in self.request_bounds:
            nonnegative(name, low, positive=True)
            nonnegative(name, high, positive=True)
            if high < low:
                raise ValueError("Reversed RPC request bounds")

    def _mismatch(self, profile: str) -> Estimate | None:
        if profile != self.profile:
            return Estimate(
                None, "unsupported", notes=("RPC base rollout profile mismatch",)
            )
        return None

    def service_duration(self, profile: str, duration: Estimate) -> Estimate:
        """Scale iterations conditionally; never label an RPC fit GPU-calibrated."""
        mismatch = self._mismatch(profile)
        if mismatch is not None:
            return mismatch
        if duration.seconds is None:
            return duration
        return replace(
            duration,
            seconds=duration.seconds * self.service_scale,
            evidence="assumed",
            notes=duration.notes
            + ("RPC-fitted effective scale; GPU/response decomposition unidentified",),
            source_ids=duration.source_ids + (self.version,),
        )

    def response_delay(
        self, profile: str, input_tokens: int, output_tokens: int
    ) -> Estimate:
        """Tail after GPU completion; occupies no GPU request/KV slot."""
        integer("input_tokens", input_tokens, minimum=1)
        integer("output_tokens", output_tokens, minimum=1)
        mismatch = self._mismatch(profile)
        if mismatch is not None:
            return mismatch
        values = dict(input_tokens=input_tokens, output_tokens=output_tokens)
        violations = tuple(
            f"Outside RPC request bound: {name}={values[name]} not in [{low}, {high}]"
            for name, low, high in self.request_bounds
            if not low <= values[name] <= high
        )
        if violations and not self.allow_request_extrapolation:
            return Estimate(None, "unsupported", notes=violations)
        return Estimate(
            self.response_seconds,
            "assumed",
            notes=violations
            + (
                "Conditional RPC tail; not an independently measured component",
                "Request bounds do not certify cache, batching or KV coverage",
            ),
            source_ids=(self.version,),
        )
