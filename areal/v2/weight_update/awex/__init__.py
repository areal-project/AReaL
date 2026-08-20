# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import os
from typing import Any

import aiohttp  # pyright: ignore[reportMissingImports]

from areal.infra.rpc.serialization import deserialize_value
from areal.infra.utils.concurrent import run_async_task
from areal.utils import logging

logger = logging.getLogger("AwexHTTP")


def resolve_physical_gpu_id(
    device_index: int | None = None, *, strict: bool = False
) -> int:
    """Map a process-local CUDA device index to its physical GPU id.

    ``CUDA_VISIBLE_DEVICES`` remaps device indices: with ``CVD="2,3"``, the
    process-local device 1 is physical GPU 3. Colocate pairing keys on
    ``(ip, physical_gpu)``, and co-resident processes may see different CVD
    mappings, so CVD is the ground truth here (under slurm-style isolation
    ``torch.cuda.current_device()`` may simply return 0).

    Falls back to the process-local index when CVD is unset or unparsable
    (e.g. GPU UUID entries).
    """
    import torch

    idx = device_index if device_index is not None else torch.cuda.current_device()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd:
        return idx
    try:
        entries = [e.strip() for e in cvd.split(",") if e.strip()]
        return int(entries[idx])
    except (IndexError, ValueError) as exc:
        if strict:
            raise RuntimeError(
                "Cannot resolve a numeric physical GPU id from "
                f"device_index={idx}, CUDA_VISIBLE_DEVICES={cvd!r}"
            ) from exc
        logger.warning(
            "Cannot map device index %d through CUDA_VISIBLE_DEVICES=%r; "
            "falling back to the process-local index.",
            idx,
            cvd,
        )
        return idx


def awex_wu_use_group() -> bool:
    """Resolve whether ``batch_send_recv`` should use ``use_group=True``.

    Why: on some hardware/driver combinations, ``torch.distributed.batch_isend_irecv``
    can hang during weight update. ``AWEX_WU_USE_GROUP=0`` lets the caller fall back
    to per-op send/recv to bypass the hang. Defaults to ``1`` (True).
    """
    return bool(int(os.getenv("AWEX_WU_USE_GROUP", "1")))


async def _fetch_kv_metadata(
    kv_store_url: str,
    pair_name: str,
) -> tuple[Any, Any]:
    """Fetch infer and training parameter metadata from the gateway KV store.

    Uses a shared ``aiohttp.ClientSession`` with ``asyncio.gather`` so both
    requests share a TCP connection pool and execute concurrently.

    Returns
    -------
    tuple[Any, Any]
        (infer_params_meta, training_params_meta) — deserialized Python objects.
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
    """Sync wrapper around :func:`_fetch_kv_metadata`.

    Bridges async ``aiohttp`` into the synchronous adapter context using
    :func:`~areal.infra.utils.concurrent.run_async_task`.
    """
    return run_async_task(_fetch_kv_metadata, kv_store_url, pair_name)
