# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from areal.utils import logging

logger = logging.getLogger("AwexPairState")


@dataclass
class AwexPairState:
    """Per-pair state for a non-colocated AWEX process group."""

    weights_update_group: Any | None
    control_group: Any | None
    transfer_plan: Any
    transfer_rank: int
    runtime_state: dict[str, Any] | None = None


def teardown_pair_process_groups(
    state: AwexPairState,
    pair_name: str,
    distributed: Any,
) -> list[Exception]:
    """Destroy each pair group independently and retain failed handles."""
    if not distributed.is_initialized():
        state.weights_update_group = None
        state.control_group = None
        return []

    errors: list[Exception] = []
    for attr in ("weights_update_group", "control_group"):
        group = getattr(state, attr)
        if group is None:
            continue
        try:
            distributed.destroy_process_group(group)
        except Exception as exc:
            logger.warning(
                "Failed to teardown %s for AWEX pair '%s'",
                attr,
                pair_name,
                exc_info=True,
            )
            errors.append(exc)
        else:
            setattr(state, attr, None)
    return errors
