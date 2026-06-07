# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from areal.infra.rpc.rtensor import RTensor
from areal.utils.data import unpack_groups_to_sequences


def prepare_dta_rollout_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize rollout trajectories into sequence-level items for DTA."""
    return unpack_groups_to_sequences(RTensor.localize(batch))
