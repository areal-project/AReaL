# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AwexPairState:
    """Per-pair state for a non-colocated AWEX process group."""

    weights_update_group: Any
    transfer_plan: Any
    transfer_rank: int


@dataclass
class MegatronColocatePairState:
    """Per-pair state for the training side of colocated AWEX."""

    kv_store_url: str
    transfer_rank: int
    infer_world_size: int
    admin_api_key: str
    timeout_s: float
    http_client: Any


@dataclass
class SGLangColocatePairState:
    """Per-pair state for the inference side of colocated AWEX."""

    weights_update_group: Any
    transfer_rank: int
    kv_store_url: str
    infer_world_size: int
    train_world_size: int
    admin_api_key: str
    timeout_s: float
    http_client: Any
    transport: Any
    train_to_infer_device_mapping: dict[int, int]
    infer_to_train_device_mapping: dict[int, int]
    send_transfer_plan: Any
    recv_transfer_plan: Any
