# SPDX-License-Identifier: Apache-2.0

"""Cancellation-safe streaming cleanup for Agent Service proxy hops."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from contextlib import suppress

from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send


class AsyncCleanupOnce:
    """Run one async cleanup task exactly once and shield it from its caller."""

    def __init__(
        self,
        callback: Callable[[], Awaitable[None]],
        *,
        task_name: str,
    ) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        if type(task_name) is not str or not task_name.strip():
            raise ValueError("task_name must be a non-blank string")
        self._callback = callback
        self._task_name = task_name
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        await self._callback()

    @staticmethod
    def _consume_result(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        with suppress(Exception):
            task.result()

    async def __call__(self) -> None:
        # There is no await between the check and assignment, so one event loop
        # cannot interleave two callers and create duplicate cleanup tasks.
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name=self._task_name)
        task = self._task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # The cleanup task keeps running.  Consume a later failure if this
            # cancelled caller is the only observer.
            task.add_done_callback(self._consume_result)
            raise


class _CleanupAsyncIterator(AsyncIterator[bytes]):
    """Trigger cleanup on EOF, failure, cancellation, or explicit early close."""

    def __init__(
        self,
        content: AsyncIterable[bytes],
        cleanup: AsyncCleanupOnce,
    ) -> None:
        self._iterator = content.__aiter__()
        self._cleanup = cleanup

    def __aiter__(self) -> _CleanupAsyncIterator:
        return self

    async def __anext__(self) -> bytes:
        try:
            return await self._iterator.__anext__()
        except BaseException:
            await self._cleanup()
            raise

    async def aclose(self) -> None:
        close = getattr(self._iterator, "aclose", None)
        try:
            if callable(close):
                await close()
        finally:
            # Unlike an unstarted async generator's ``aclose``, this always
            # executes, so a disconnect before the first byte cannot leak state.
            await self._cleanup()


class CleanupStreamingResponse(StreamingResponse):
    """A StreamingResponse whose upstream cleanup has no start-byte gap."""

    def __init__(
        self,
        content: AsyncIterable[bytes],
        *,
        cleanup: Callable[[], Awaitable[None]],
        cleanup_task_name: str,
        **kwargs,
    ) -> None:
        self._cleanup_once = AsyncCleanupOnce(
            cleanup,
            task_name=cleanup_task_name,
        )
        super().__init__(
            _CleanupAsyncIterator(content, self._cleanup_once),
            **kwargs,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Covers ASGI disconnect/cancellation before Starlette starts the
            # body iterator.  Iterator cleanup and this path share one task.
            await self._cleanup_once()


__all__ = ["AsyncCleanupOnce", "CleanupStreamingResponse"]
