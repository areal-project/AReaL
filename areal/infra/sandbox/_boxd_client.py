# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import atexit
from typing import Any

from areal.utils import logging

logger = logging.getLogger("BoxdClientManager")

AsyncBoxd = None


def _load_boxd_sdk() -> type[Any]:
    global AsyncBoxd

    if AsyncBoxd is not None:
        return AsyncBoxd

    try:
        from boxd import AsyncBoxd as ImportedAsyncBoxd
    except ImportError as exc:
        raise ImportError(
            "BoxdClientManager requires the optional 'boxd' dependency. Install it with `uv sync --extra sandbox`."
        ) from exc

    AsyncBoxd = ImportedAsyncBoxd
    return AsyncBoxd


class BoxdClientManager:
    """Process-wide access point for AsyncBoxd clients.

    The manager keeps boxd initialization and configuration in one place for both
    the synchronous runner and the async TIR tool. The SDK reads ``BOXD_API_KEY``
    and ``BOXD_REGION`` from the environment; ``configure()`` overrides them
    (``api_key``, ``region``, ``timeout``, ...). The underlying gRPC channel is tied
    to the running event loop, so the manager recreates the client when callers
    move across loops.
    """

    _client = None
    _lock = asyncio.Lock()
    _config_overrides: dict[str, Any] = {}
    _atexit_registered = False
    _loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def configure(cls, **config_kwargs: Any) -> None:
        if cls._client is not None:
            raise RuntimeError("BoxdClientManager is already initialized")

        cls._config_overrides = dict(config_kwargs)

    @classmethod
    def config_overrides(cls) -> dict[str, Any]:
        return dict(cls._config_overrides)

    @classmethod
    def active_loop(cls) -> asyncio.AbstractEventLoop | None:
        if cls._loop is not None and cls._loop.is_running():
            return cls._loop
        return None

    @classmethod
    async def get_client(cls):
        current_loop = asyncio.get_running_loop()

        async with cls._lock:
            needs_new_client = (
                cls._client is None
                or cls._loop is None
                or cls._loop.is_closed()
                or cls._loop is not current_loop
            )

            if needs_new_client:
                if cls._client is not None:
                    await cls._close_client()
                if cls._loop is not None and cls._loop is not current_loop:
                    logger.debug("Reinitializing AsyncBoxd client for a new event loop")

                async_boxd_cls = _load_boxd_sdk()
                cls._client = async_boxd_cls(**cls._config_overrides)
                cls._loop = current_loop
                if not cls._atexit_registered:
                    atexit.register(cls._close_sync)
                    cls._atexit_registered = True
                logger.debug("Initialized AsyncBoxd client")

            return cls._client

    @classmethod
    async def close(cls) -> None:
        async with cls._lock:
            await cls._close_client()

    @classmethod
    async def _close_client(cls) -> None:
        client = cls._client
        if client is None:
            return

        loop = cls._loop
        cls._client = None
        cls._loop = None

        try:
            if (
                loop is not None
                and loop.is_running()
                and loop is not asyncio.get_running_loop()
            ):
                future = asyncio.run_coroutine_threadsafe(client.close(), loop)
                await asyncio.wrap_future(future)
            else:
                await client.close()
        finally:
            logger.debug("Closed AsyncBoxd client")

    @classmethod
    def _close_sync(cls) -> None:
        try:
            asyncio.run(cls.close())
        except Exception:
            return
