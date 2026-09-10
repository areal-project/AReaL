# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

import torch

MOPD_CONTRIBUTIONS_KEY = "_mopd_teacher_contributions"


def aggregate_mopd_targets(
    data: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Aggregate raw weighted teacher log-probabilities on an actor DP head."""
    if data is None:
        return None
    for trajectory in data:
        route = trajectory.get("mopd_route")
        contributions = trajectory.pop(MOPD_CONTRIBUTIONS_KEY, None)
        if not isinstance(contributions, dict) or not contributions:
            raise ValueError(
                f"MOPD route {route!r} has no positive teacher contributions"
            )

        logp_sum: torch.Tensor | None = None
        weight_sum: torch.Tensor | None = None
        for teacher_id, contribution in contributions.items():
            if not isinstance(contribution, dict):
                raise TypeError(
                    f"MOPD contribution from {teacher_id!r} must be a mapping"
                )
            teacher_logp = contribution.get("logp")
            weight = contribution.get("weight")
            if not isinstance(teacher_logp, torch.Tensor):
                raise TypeError(
                    f"MOPD contribution from {teacher_id!r} has no tensor logp"
                )
            if (
                not isinstance(weight, (int, float))
                or isinstance(weight, bool)
                or not math.isfinite(weight)
                or weight <= 0
            ):
                raise ValueError(
                    f"MOPD contribution weight from {teacher_id!r} must be "
                    "finite and positive"
                )
            if logp_sum is not None and teacher_logp.shape != logp_sum.shape:
                raise ValueError(
                    f"MOPD teacher logp shape mismatch for route {route!r}: "
                    f"expected {tuple(logp_sum.shape)}, got {tuple(teacher_logp.shape)}"
                )
            weighted_logp = teacher_logp * weight
            token_weight = torch.full_like(teacher_logp, weight)
            logp_sum = weighted_logp if logp_sum is None else logp_sum + weighted_logp
            weight_sum = (
                token_weight if weight_sum is None else weight_sum + token_weight
            )

        assert logp_sum is not None and weight_sum is not None
        trajectory["mopd_teacher_logp_sum"] = logp_sum
        trajectory["mopd_teacher_weight_sum"] = weight_sum
        trajectory.pop("mopd_route", None)
    return data
