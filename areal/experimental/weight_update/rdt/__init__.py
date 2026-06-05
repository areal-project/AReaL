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


def serialize_actor_handle_bytes(actor_handle: Any) -> str:
    """Serialize actor handle to Base64-encoded cloudpickle bytes.

    Args:
        actor_handle: Ray actor handle

    Returns:
        str: Base64-encoded string
    """
    import ray

    handle_bytes = ray.cloudpickle.dumps(actor_handle)
    return base64.b64encode(handle_bytes).decode()


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


def try_register_nixl_memory(tensor: Any) -> bool:
    """Try to register tensor memory with NIXL for lifetime of the process.

    Pre-registering avoids per-step NIXL registration overhead when the same
    buffer is reused across multiple weight update steps. Returns True if
    registration succeeded, False if the API is unavailable or call fails.

    Args:
        tensor: A PyTorch tensor whose memory should be registered with NIXL

    Returns:
        bool: True if registration succeeded, False otherwise
    """
    try:
        from ray.experimental import (
            register_nixl_memory,  # pyright: ignore[reportMissingImports]
        )

        register_nixl_memory(tensor)
        return True
    except (ImportError, AttributeError, Exception) as e:
        logger.warning(
            "register_nixl_memory not available or failed: %s. "
            "NIXL buffer reuse will be disabled.",
            e,
        )
        return False


def try_set_target_for_ref(ref: Any, target: list) -> bool:
    """Try to set target buffers for an RDT ObjectRef to receive into.

    When set_target_for_ref succeeds, ray.get() writes RDMA data directly
    into the pre-allocated target buffers, avoiding per-step GPU allocation.
    Returns True if succeeded, False if the API is unavailable or call fails.

    Args:
        ref: A Ray ObjectRef for an RDT object
        target: A list of tensors to be used as target receive buffers

    Returns:
        bool: True if succeeded, False otherwise
    """
    try:
        from ray.experimental import (
            set_target_for_ref,  # pyright: ignore[reportMissingImports]
        )

        set_target_for_ref(ref, target)
        return True
    except (ImportError, AttributeError, Exception) as e:
        logger.warning(
            "set_target_for_ref not available or failed: %s. "
            "Falling back to standard ray.get().",
            e,
        )
        return False


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
    "serialize_actor_handle_bytes",
    "deserialize_actor_handle_bytes",
    "get_tensor_transport",
    "try_register_nixl_memory",
    "try_set_target_for_ref",
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
