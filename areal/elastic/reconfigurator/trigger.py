# SPDX-License-Identifier: Apache-2.0

"""Decide whether the observed request SLO warrants a layout search."""

from typing import Literal


def trigger(
    *,
    joint_fraction: float | None,
    completed_requests: int,
    pressure_fraction: float,
    slack_fraction: float,
    minimum_completed_requests: int,
) -> Literal["none", "pressure", "slack"]:
    """Use the joint TTFT/TPOT pass fraction from one observation window."""
    if not 0 <= pressure_fraction <= slack_fraction <= 1:
        raise ValueError("Expected 0 <= pressure_fraction <= slack_fraction <= 1")
    if joint_fraction is None:
        return "none"
    if joint_fraction < pressure_fraction:
        return "pressure"
    if (
        joint_fraction > slack_fraction
        and completed_requests >= minimum_completed_requests
    ):
        return "slack"
    return "none"
