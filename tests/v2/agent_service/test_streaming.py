# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest
from starlette.requests import ClientDisconnect

from areal.v2.agent_service.streaming import (
    CleanupAsyncIterator,
    CleanupStreamingResponse,
)


class _FiniteSource:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.next_calls = 0
        self.close_calls = 0

    def __aiter__(self) -> _FiniteSource:
        return self

    async def __anext__(self) -> bytes:
        self.next_calls += 1
        if self.next_calls == 1:
            return b"chunk"
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.close_calls += 1
        self.order.append("source-close")


def _asgi_24_scope() -> dict[str, object]:
    return {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "asgi": {"version": "3.0", "spec_version": "2.4"},
    }


async def _receive_must_not_run() -> dict[str, str]:
    raise AssertionError("ASGI 2.4 StreamingResponse must not poll receive")


@pytest.mark.asyncio
async def test_cleanup_iterator_eof_closes_source_then_cleanup_exactly_once() -> None:
    order: list[str] = []
    cleanup_calls = 0
    source = _FiniteSource(order)

    async def cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        order.append("cleanup")

    iterator = CleanupAsyncIterator(
        source,
        cleanup=cleanup,
        cleanup_task_name="test-stream-eof-cleanup",
    )

    assert await anext(iterator) == b"chunk"
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)
    await asyncio.gather(iterator.aclose(), iterator.aclose())
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)

    assert order == ["source-close", "cleanup"]
    assert source.close_calls == cleanup_calls == 1


@pytest.mark.asyncio
async def test_cleanup_iterator_source_error_closes_and_fails_closed() -> None:
    order: list[str] = []

    class FailingSource:
        def __init__(self) -> None:
            self.close_calls = 0

        def __aiter__(self) -> FailingSource:
            return self

        async def __anext__(self) -> bytes:
            raise RuntimeError("injected source failure")

        async def aclose(self) -> None:
            self.close_calls += 1
            order.append("source-close")

    source = FailingSource()

    async def cleanup() -> None:
        order.append("cleanup")

    iterator = CleanupAsyncIterator(
        source,
        cleanup=cleanup,
        cleanup_task_name="test-stream-source-error-cleanup",
    )

    with pytest.raises(RuntimeError, match="injected source failure"):
        await anext(iterator)
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)

    assert order == ["source-close", "cleanup"]
    assert source.close_calls == 1


@pytest.mark.asyncio
async def test_cleanup_iterator_cancelled_read_runs_source_finally_first() -> None:
    order: list[str] = []
    read_started = asyncio.Event()

    async def source():
        try:
            read_started.set()
            await asyncio.Event().wait()
            yield b"unreachable"
        finally:
            order.append("source-finally")

    async def cleanup() -> None:
        order.append("cleanup")

    iterator = CleanupAsyncIterator(
        source(),
        cleanup=cleanup,
        cleanup_task_name="test-stream-cancelled-read-cleanup",
    )
    reading = asyncio.create_task(anext(iterator))
    await asyncio.wait_for(read_started.wait(), timeout=1)
    reading.cancel()

    with pytest.raises(asyncio.CancelledError):
        await reading
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)

    assert order == ["source-finally", "cleanup"]


@pytest.mark.asyncio
async def test_explicit_close_cancels_and_joins_active_source_read() -> None:
    order: list[str] = []
    read_started = asyncio.Event()

    async def source():
        try:
            read_started.set()
            await asyncio.Event().wait()
            yield b"unreachable"
        finally:
            order.append("source-finally")
            await asyncio.sleep(0)
            order.append("source-finally-done")

    async def cleanup() -> None:
        order.append("cleanup")

    iterator = CleanupAsyncIterator(
        source(),
        cleanup=cleanup,
        cleanup_task_name="test-stream-active-read-close-cleanup",
    )
    reading = asyncio.create_task(anext(iterator))
    await asyncio.wait_for(read_started.wait(), timeout=1)

    await asyncio.wait_for(iterator.aclose(), timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await reading

    assert order == ["source-finally", "source-finally-done", "cleanup"]


@pytest.mark.asyncio
async def test_cleanup_iterator_explicit_never_started_close_is_exactly_once() -> None:
    order: list[str] = []

    class NeverStartedSource:
        def __init__(self) -> None:
            self.next_calls = 0
            self.close_calls = 0

        def __aiter__(self) -> NeverStartedSource:
            return self

        async def __anext__(self) -> bytes:
            self.next_calls += 1
            return b"unexpected"

        async def aclose(self) -> None:
            self.close_calls += 1
            order.append("source-close")

    source = NeverStartedSource()

    async def cleanup() -> None:
        order.append("cleanup")

    iterator = CleanupAsyncIterator(
        source,
        cleanup=cleanup,
        cleanup_task_name="test-stream-never-started-cleanup",
    )
    await asyncio.gather(*(iterator.aclose() for _ in range(8)))
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)

    assert source.next_calls == 0
    assert source.close_calls == 1
    assert order == ["source-close", "cleanup"]


@pytest.mark.asyncio
async def test_body_primary_error_preserves_secondary_finish_failures_as_notes() -> (
    None
):
    class FailingSource:
        def __init__(self) -> None:
            self.close_calls = 0

        def __aiter__(self) -> FailingSource:
            return self

        async def __anext__(self) -> bytes:
            raise ValueError("primary body failure")

        async def aclose(self) -> None:
            self.close_calls += 1
            raise RuntimeError("secondary source close failure")

    source = FailingSource()
    cleanup_calls = 0

    async def cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        raise LookupError("secondary cleanup failure")

    iterator = CleanupAsyncIterator(
        source,
        cleanup=cleanup,
        cleanup_task_name="test-stream-primary-error-cleanup",
    )

    with pytest.raises(ValueError, match="primary body failure") as exc_info:
        await anext(iterator)

    notes = getattr(exc_info.value, "__notes__", ())
    assert any("secondary source close failure" in note for note in notes)
    assert any("secondary cleanup failure" in note for note in notes)
    assert source.close_calls == cleanup_calls == 1


@pytest.mark.asyncio
async def test_eof_without_primary_error_surfaces_cleanup_failure() -> None:
    class EmptySource:
        def __aiter__(self) -> EmptySource:
            return self

        async def __anext__(self) -> bytes:
            raise StopAsyncIteration

        async def aclose(self) -> None:
            return None

    async def cleanup() -> None:
        raise RuntimeError("injected EOF cleanup failure")

    iterator = CleanupAsyncIterator(
        EmptySource(),
        cleanup=cleanup,
        cleanup_task_name="test-stream-eof-failed-cleanup",
    )

    with pytest.raises(RuntimeError, match="injected EOF cleanup failure"):
        await anext(iterator)


@pytest.mark.asyncio
async def test_source_close_failure_still_runs_cleanup_once() -> None:
    order: list[str] = []
    cleanup_calls = 0

    class FailingCloseSource:
        def __init__(self) -> None:
            self.close_calls = 0

        def __aiter__(self) -> FailingCloseSource:
            return self

        async def __anext__(self) -> bytes:
            return b"unused"

        async def aclose(self) -> None:
            self.close_calls += 1
            order.append("source-close")
            raise RuntimeError("injected source close failure")

    source = FailingCloseSource()

    async def cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        order.append("cleanup")

    iterator = CleanupAsyncIterator(
        source,
        cleanup=cleanup,
        cleanup_task_name="test-stream-failed-source-close-cleanup",
    )

    for _ in range(2):
        with pytest.raises(RuntimeError, match="injected source close failure"):
            await iterator.aclose()
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)

    assert source.close_calls == cleanup_calls == 1
    assert order == ["source-close", "cleanup"]


@pytest.mark.asyncio
async def test_cancelled_cleanup_waiter_does_not_cancel_finish_task() -> None:
    order: list[str] = []
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    source = _FiniteSource(order)

    async def cleanup() -> None:
        cleanup_started.set()
        await release_cleanup.wait()
        order.append("cleanup")

    iterator = CleanupAsyncIterator(
        source,
        cleanup=cleanup,
        cleanup_task_name="test-stream-shielded-cleanup",
    )
    closing = asyncio.create_task(iterator.aclose())
    await asyncio.wait_for(cleanup_started.wait(), timeout=1)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    release_cleanup.set()
    await asyncio.wait_for(iterator.aclose(), timeout=1)

    assert source.close_calls == 1
    assert order == ["source-close", "cleanup"]


def test_cleanup_iterator_rejects_bad_cleanup_before_touching_source() -> None:
    touched = False

    class UntouchedSource:
        def __aiter__(self):
            nonlocal touched
            touched = True
            return self

    with pytest.raises(TypeError, match="callback must be callable"):
        CleanupAsyncIterator(
            UntouchedSource(),
            cleanup=None,  # type: ignore[arg-type]
            cleanup_task_name="test-invalid-cleanup",
        )

    assert not touched


@pytest.mark.asyncio
async def test_asgi_24_start_send_failure_closes_never_started_source_first() -> None:
    order: list[str] = []

    class NeverStartedSource:
        def __init__(self) -> None:
            self.next_calls = 0
            self.close_calls = 0

        def __aiter__(self) -> NeverStartedSource:
            return self

        async def __anext__(self) -> bytes:
            self.next_calls += 1
            return b"unexpected"

        async def aclose(self) -> None:
            self.close_calls += 1
            order.append("source-close")

    source = NeverStartedSource()

    async def cleanup() -> None:
        order.append("cleanup")

    async def send(message: dict[str, object]) -> None:
        assert message["type"] == "http.response.start"
        raise OSError("injected disconnect before response start")

    response = CleanupStreamingResponse(
        source,
        cleanup=cleanup,
        cleanup_task_name="test-asgi-start-send-failure-cleanup",
    )
    with pytest.raises(ClientDisconnect):
        await response(_asgi_24_scope(), _receive_must_not_run, send)  # type: ignore[arg-type]

    assert source.next_calls == 0
    assert source.close_calls == 1
    assert order == ["source-close", "cleanup"]


@pytest.mark.asyncio
async def test_send_failure_stays_primary_when_source_close_and_cleanup_fail() -> None:
    class FailingCloseSource:
        def __init__(self) -> None:
            self.next_calls = 0
            self.close_calls = 0

        def __aiter__(self) -> FailingCloseSource:
            return self

        async def __anext__(self) -> bytes:
            self.next_calls += 1
            return b"unexpected"

        async def aclose(self) -> None:
            self.close_calls += 1
            raise RuntimeError("secondary response source close failure")

    source = FailingCloseSource()
    cleanup_calls = 0

    async def cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        raise LookupError("secondary response cleanup failure")

    async def send(message: dict[str, object]) -> None:
        assert message["type"] == "http.response.start"
        raise OSError("primary response send failure")

    response = CleanupStreamingResponse(
        source,
        cleanup=cleanup,
        cleanup_task_name="test-asgi-primary-send-failure-cleanup",
    )
    with pytest.raises(ClientDisconnect) as exc_info:
        await response(_asgi_24_scope(), _receive_must_not_run, send)  # type: ignore[arg-type]

    notes = getattr(exc_info.value, "__notes__", ())
    assert any("secondary response source close failure" in note for note in notes)
    assert any("secondary response cleanup failure" in note for note in notes)
    assert source.next_calls == 0
    assert source.close_calls == cleanup_calls == 1


@pytest.mark.asyncio
async def test_asgi_24_chunk_send_failure_runs_source_finally_first() -> None:
    order: list[str] = []
    cleanup_calls = 0

    async def source():
        try:
            yield b"chunk"
            await asyncio.Event().wait()
        finally:
            order.append("source-finally")

    async def cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        order.append("cleanup")

    async def send(message: dict[str, object]) -> None:
        if message["type"] == "http.response.body":
            assert message["body"] == b"chunk"
            raise OSError("injected disconnect during chunk send")

    response = CleanupStreamingResponse(
        source(),
        cleanup=cleanup,
        cleanup_task_name="test-asgi-chunk-send-failure-cleanup",
    )
    with pytest.raises(ClientDisconnect):
        await response(_asgi_24_scope(), _receive_must_not_run, send)  # type: ignore[arg-type]

    assert cleanup_calls == 1
    assert order == ["source-finally", "cleanup"]


@pytest.mark.asyncio
async def test_cancelled_asgi_send_runs_source_finally_before_cleanup() -> None:
    order: list[str] = []
    chunk_send_started = asyncio.Event()

    async def source():
        try:
            yield b"chunk"
            await asyncio.Event().wait()
        finally:
            order.append("source-finally")

    async def cleanup() -> None:
        order.append("cleanup")

    async def send(message: dict[str, object]) -> None:
        if message["type"] == "http.response.body":
            chunk_send_started.set()
            await asyncio.Event().wait()

    response = CleanupStreamingResponse(
        source(),
        cleanup=cleanup,
        cleanup_task_name="test-asgi-cancelled-send-cleanup",
    )
    serving = asyncio.create_task(
        response(_asgi_24_scope(), _receive_must_not_run, send)  # type: ignore[arg-type]
    )
    await asyncio.wait_for(chunk_send_started.wait(), timeout=1)
    serving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await serving

    assert order == ["source-finally", "cleanup"]


@pytest.mark.asyncio
async def test_asgi_cancellation_stays_primary_when_cleanup_fails() -> None:
    order: list[str] = []
    chunk_send_started = asyncio.Event()

    async def source():
        try:
            yield b"chunk"
            await asyncio.Event().wait()
        finally:
            order.append("source-finally")

    async def cleanup() -> None:
        order.append("cleanup")
        raise RuntimeError("secondary cancellation cleanup failure")

    async def send(message: dict[str, object]) -> None:
        if message["type"] == "http.response.body":
            chunk_send_started.set()
            await asyncio.Event().wait()

    response = CleanupStreamingResponse(
        source(),
        cleanup=cleanup,
        cleanup_task_name="test-asgi-primary-cancellation-cleanup",
    )
    serving = asyncio.create_task(
        response(_asgi_24_scope(), _receive_must_not_run, send)  # type: ignore[arg-type]
    )
    await asyncio.wait_for(chunk_send_started.wait(), timeout=1)
    serving.cancel()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await serving

    notes = getattr(exc_info.value, "__notes__", ())
    assert any("secondary cancellation cleanup failure" in note for note in notes)
    assert order == ["source-finally", "cleanup"]
