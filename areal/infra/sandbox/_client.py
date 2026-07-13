# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import atexit
from typing import Any

from areal.utils import logging

logger = logging.getLogger("DaytonaClientManager")

AsyncDaytona = None
DaytonaConfig = None


def _load_daytona_sdk() -> tuple[type[Any], type[Any]]:
    global AsyncDaytona, DaytonaConfig

    if AsyncDaytona is not None and DaytonaConfig is not None:
        return AsyncDaytona, DaytonaConfig

    try:
        from daytona import AsyncDaytona as ImportedAsyncDaytona
        from daytona import DaytonaConfig as ImportedDaytonaConfig
    except ImportError as exc:
        raise ImportError(
            "DaytonaClientManager requires the optional 'daytona' dependency. Install it with `uv sync --extra sandbox`."
        ) from exc

    AsyncDaytona = ImportedAsyncDaytona
    DaytonaConfig = ImportedDaytonaConfig
    return AsyncDaytona, DaytonaConfig


class DaytonaClientManager:
    """Process-wide access point for AsyncDaytona clients.

    The manager keeps Daytona initialization and configuration in one place for both
    the synchronous runner and async TIR tool. The underlying SDK client is tied to
    the currently running event loop, so the manager recreates it when callers move
    across loops instead of trying to reuse a closed aiohttp session.

    Each client is created with ``connection_pool_maxsize=None`` to match AReaL's
    concurrent rollout workloads.
    """

    _client = None
    _lock = asyncio.Lock()
    _config_overrides: dict[str, Any] = {}
    _atexit_registered = False
    _loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def configure(cls, **config_kwargs: Any) -> None:
        if cls._client is not None:
            raise RuntimeError("DaytonaClientManager is already initialized")

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
                    logger.debug(
                        "Reinitializing AsyncDaytona client for a new event loop"
                    )

                async_daytona_cls, daytona_config_cls = _load_daytona_sdk()
                config = daytona_config_cls(
                    connection_pool_maxsize=None,
                    **cls._config_overrides,
                )
                cls._client = async_daytona_cls(config)
                cls._loop = current_loop
                if not cls._atexit_registered:
                    atexit.register(cls._close_sync)
                    cls._atexit_registered = True
                logger.debug("Initialized AsyncDaytona client")

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
            logger.debug("Closed AsyncDaytona client")

    @classmethod
    def _close_sync(cls) -> None:
        try:
            asyncio.run(cls.close())
        except Exception:
            return
