# SPDX-License-Identifier: Apache-2.0

"""Candidate-only headroom checks at the evaluator's existing Actor calls."""

import math
from dataclasses import replace

from areal.elastic.policy.basic.online import ActorMemorySafety


class EvaluationMemoryModels:
    def __init__(self, models, target, reserve, observed_at):
        self.models, self.target, self.reserve = models, target, reserve
        self.checked = set()
        self.report = ActorMemorySafety(
            True,
            reserve,
            0,
            None,
            None,
            None,
            (),
            observed_at,
            "Only batches used by candidate evaluation are checked",
            scope="evaluation",
        )

    def __getattr__(self, name):
        return getattr(self.models, name)

    def actor(self, layout, batch):
        # Old Actor work before cutover retains the physical model. Only the
        # proposed Actor layout must retain policy headroom in this evaluation.
        if (layout.actor_dp, layout.actor_tp) == (
            self.target.actor_dp,
            self.target.actor_tp,
        ) and batch not in self.checked:
            try:
                estimate = getattr(self.models, "actor_memory", None)
                if estimate is None:
                    raise ValueError("Actor memory estimate unavailable")
                used, limit, measured_oom = estimate(layout, batch)
                if (
                    not math.isfinite(used)
                    or used < 0
                    or not math.isfinite(limit)
                    or limit <= 0
                ):
                    raise ValueError("Invalid Actor memory estimate")
                self.checked.add(batch)
                report = replace(self.report, checked_batches=len(self.checked))
                if report.peak_gib is None or used > report.peak_gib:
                    report = replace(
                        report,
                        peak_gib=used,
                        capacity_gib=limit,
                        allowed_gib=limit * (1 - self.reserve),
                    )
                self.report = report
                if measured_oom or used > limit * (1 - self.reserve):
                    raise ValueError(
                        "Evaluated Actor batch exceeds the memory safety budget or has a measured OOM"
                    )
            except ValueError as exc:
                self.report = replace(self.report, admissible=False, reason=str(exc))
        if not self.report.admissible:
            # A failure caught inside capacity estimation must still reject the
            # candidate; it cannot turn into a merely unknown supply rate.
            raise ValueError(self.report.reason)
        return self.models.actor(layout, batch)
