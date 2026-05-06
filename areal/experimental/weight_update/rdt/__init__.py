# SPDX-License-Identifier: Apache-2.0
"""RDT weight update backend using one-sided RDMA (YR/NIXL).

See docs/rfc/rdt_weight_update_backend.md for details.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import aiohttp  # pyright: ignore[reportMissingImports]

from areal.infra.rpc.serialization import deserialize_value
from areal.infra.utils.concurrent import run_async_task
from areal.utils import logging

logger = logging.getLogger("RDTWeightUpdate")


async def _fetch_kv_metadata_async(
    kv_store_url: str,
    pair_name: str,
) -> tuple[Any, Any]:
    """Fetch infer and training parameter metadata from gateway KV store.

    Args:
        kv_store_url: Gateway URL (e.g., "http://10.0.0.1:7080")
        pair_name: Unique identifier for the TW-IW pair

    Returns:
        tuple[Any, Any]: (infer_params_meta, training_params_meta)
    """
    infer_url = f"{kv_store_url}/weight_meta/{pair_name}/infer_params_meta"
    train_url = f"{kv_store_url}/weight_meta/{pair_name}/training_params_meta"

    async with aiohttp.ClientSession() as session:

        async def _get(url: str) -> Any:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("value", data)

        infer_json, train_json = await asyncio.gather(_get(infer_url), _get(train_url))

    return deserialize_value(infer_json), deserialize_value(train_json)


def fetch_kv_metadata(kv_store_url: str, pair_name: str) -> tuple[Any, Any]:
    """Sync wrapper around :func:`_fetch_kv_metadata_async`.

    Args:
        kv_store_url: Gateway URL
        pair_name: TW-IW pair identifier

    Returns:
        tuple[Any, Any]: (infer_params_meta, training_params_meta)
    """
    return run_async_task(_fetch_kv_metadata_async, kv_store_url, pair_name)


def deserialize_actor_handle_bytes(actor_bytes_b64: str) -> Any:
    """Deserialize Base64-encoded actor handle bytes.

    Args:
        actor_bytes_b64: Base64-encoded cloudpickle bytes

    Returns:
        Any: Ray actor handle
    """
    import ray

    actor_bytes = base64.b64decode(actor_bytes_b64)
    return ray.cloudpickle.loads(actor_bytes)


def get_tensor_transport() -> str:
    """Get appropriate tensor transport based on device type.

    Returns:
        str: "YR" for NPU, "NIXL" for CUDA GPU
    """
    from areal.infra.platforms import current_platform

    device_type = current_platform.device_type
    if device_type == "npu":
        return "YR"
    elif device_type == "cuda":
        return "NIXL"
    else:
        raise RuntimeError(f"Unsupported device type for RDT: {device_type}")


__all__ = [
    "fetch_kv_metadata",
    "deserialize_actor_handle_bytes",
    "get_tensor_transport",
    "WeightTransportActor",
]


# WeightTransportActor - lazy import to avoid circular dependencies
def __getattr__(name: str):
    if name == "WeightTransportActor":
        from areal.experimental.weight_update.rdt.weight_transport_actor import (
            WeightTransportActor,
        )

        return WeightTransportActor
    raise AttributeError(f"module {__name__} has no attribute {name}")
