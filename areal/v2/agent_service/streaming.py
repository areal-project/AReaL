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


class CleanupAsyncIterator(AsyncIterator[bytes]):
    """Close one async source before running one downstream cleanup.

    EOF, source failure, cancellation, and explicit early close all join the
    same cancellation-shielded finish task.  Once finishing starts, further
    reads fail closed even if source close or downstream cleanup later fails.
    """

    def __init__(
        self,
        content: AsyncIterable[bytes],
        *,
        cleanup: Callable[[], Awaitable[None]],
        cleanup_task_name: str,
    ) -> None:
        # Preserve CleanupStreamingResponse's eager constructor validation:
        # invalid cleanup configuration must fail before touching the source.
        if not callable(cleanup):
            raise TypeError("callback must be callable")
        self._cleanup = cleanup
        self._finished = False
        self._finish_once = AsyncCleanupOnce(
            self._finish,
            task_name=cleanup_task_name,
        )
        self._iterator = content.__aiter__()
        self._read_task: asyncio.Task[bytes] | None = None

    def __aiter__(self) -> CleanupAsyncIterator:
        return self

    async def __anext__(self) -> bytes:
        if self._finished:
            raise StopAsyncIteration
        if self._read_task is not None:
            raise RuntimeError("concurrent stream reads are not supported")

        # Own the source read in a distinct task.  Explicit close can then
        # cancel and join that task without waiting for (or cancelling) the
        # consumer that is itself waiting for finish, avoiding a close cycle.
        read_task = asyncio.create_task(
            self._read_next(),
            name="areal-cleanup-async-iterator-read",
        )
        self._read_task = read_task
        try:
            chunk = await read_task
        except BaseException as primary_error:
            # Publish the terminal state before the first cleanup await.  A
            # failed close must not make this source readable again.
            self._finished = True
            try:
                await self._finish_once()
            except BaseException as finish_error:
                if isinstance(primary_error, StopAsyncIteration):
                    # EOF has no primary failure to preserve.  A source-close
                    # or downstream-cleanup failure must remain observable.
                    raise
                primary_error.add_note(
                    f"stream finalization also failed: {finish_error!r}"
                )
                for note in getattr(finish_error, "__notes__", ()):
                    primary_error.add_note(f"stream finalization detail: {note}")
            raise
        else:
            # Close may win after the source produced a chunk but before this
            # consumer resumed.  Never publish bytes after authority teardown
            # has begun.
            if self._finished:
                await self._finish_once()
                raise StopAsyncIteration
            return chunk
        finally:
            if self._read_task is read_task and read_task.done():
                self._read_task = None

    async def _read_next(self) -> bytes:
        return await self._iterator.__anext__()

    async def _cancel_active_read(self) -> None:
        read_task = self._read_task
        if read_task is None:
            return
        if not read_task.done():
            read_task.cancel()
        # The read's exception is delivered to its consumer.  Finish only
        # needs to know that source code (including its finally block) has
        # stopped before it calls source.aclose and downstream cleanup.
        await asyncio.gather(read_task, return_exceptions=True)
        if self._read_task is read_task:
            self._read_task = None

    async def _finish(self) -> None:
        await self._cancel_active_read()
        close = getattr(self._iterator, "aclose", None)
        if callable(close):
            try:
                await close()
            except BaseException as source_close_error:
                try:
                    await self._cleanup()
                except BaseException as cleanup_error:
                    source_close_error.add_note(
                        f"downstream cleanup also failed: {cleanup_error!r}"
                    )
                raise
        else:
            await self._cleanup()
            return

        # Without a source-close failure, downstream cleanup remains the
        # primary finish failure and is propagated unchanged.
        await self._cleanup()

    async def aclose(self) -> None:
        # Unlike an unstarted async generator's own ``aclose``, this wrapper's
        # finish task always executes, so a pre-first-byte close cannot skip
        # downstream cleanup.
        self._finished = True
        await self._finish_once()


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
        self._cleanup_iterator = CleanupAsyncIterator(
            content,
            cleanup=cleanup,
            cleanup_task_name=cleanup_task_name,
        )
        super().__init__(
            self._cleanup_iterator,
            **kwargs,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        except BaseException as primary_error:
            # Covers ASGI disconnect/cancellation before iteration and while a
            # yielded chunk is being sent.  Closing the iterator first ensures
            # the source body cannot outlive its downstream authority.
            try:
                await self._cleanup_iterator.aclose()
            except BaseException as close_error:
                # The response failure explains why cleanup began.  Preserve
                # it as primary and retain close/cleanup failures as notes.  A
                # finish failure may already be the same object propagated by
                # body iteration; avoid adding a recursive self-note then.
                if close_error is not primary_error:
                    primary_error.add_note(
                        f"response stream finalization also failed: {close_error!r}"
                    )
                    for note in tuple(getattr(close_error, "__notes__", ())):
                        primary_error.add_note(
                            f"response stream finalization detail: {note}"
                        )
            raise
        else:
            # Without a response failure, close/cleanup failure remains the
            # primary error and must be propagated to the caller.
            await self._cleanup_iterator.aclose()


__all__ = [
    "AsyncCleanupOnce",
    "CleanupAsyncIterator",
    "CleanupStreamingResponse",
]
