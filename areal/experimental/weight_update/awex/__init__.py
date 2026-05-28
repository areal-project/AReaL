# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from typing import Any

import aiohttp  # pyright: ignore[reportMissingImports]

from areal.infra.rpc.serialization import deserialize_value
from areal.infra.utils.concurrent import run_async_task
from areal.utils import logging

logger = logging.getLogger("AwexHTTP")


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
