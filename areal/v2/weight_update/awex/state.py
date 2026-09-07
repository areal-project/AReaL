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


@dataclass
class MegatronColocatePairState:
    """Per-pair state for the training side of colocated AWEX."""

    kv_store_url: str
    transfer_rank: int
    infer_world_size: int
    admin_api_key: str
    timeout_s: float
    http_client: Any | None


@dataclass
class SGLangColocatePairState:
    """Per-pair state for the inference side of colocated AWEX."""

    weights_update_group: Any | None
    transfer_rank: int
    kv_store_url: str
    infer_world_size: int
    train_world_size: int
    admin_api_key: str
    timeout_s: float
    http_client: Any | None
    transport: Any | None
    train_to_infer_device_mapping: dict[int, int]
    infer_to_train_device_mapping: dict[int, int]
    send_transfer_plan: Any
    recv_transfer_plan: Any


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
