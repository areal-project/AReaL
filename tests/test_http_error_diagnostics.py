# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import aiohttp
import pytest
from aiohttp import web

from areal.infra.utils import http


@asynccontextmanager
async def http_server(
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_route("*", "/generate", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"127.0.0.1:{runner.addresses[0][1]}"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body", [b"input length exceeds context limit", b"bad \xff body"]
)
async def test_http_error_preserves_status_body_without_request_or_headers(
    body: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings = []
    monkeypatch.setattr(
        http.logger, "warning", lambda message, *args: warnings.append(message % args)
    )

    async def handler(request: web.Request) -> web.Response:
        payload = await request.json()
        assert payload["image_data"] == "private-image-content"
        return web.Response(
            status=400, body=body, headers={"Set-Cookie": "private-response-cookie"}
        )

    async with http_server(handler) as addr, aiohttp.ClientSession() as session:
        with pytest.raises(RuntimeError) as exc_info:
            await http.arequest_with_retry(
                addr,
                "/generate",
                payload={
                    "image_data": "private-image-content",
                    "input_ids": [1] * 10000,
                },
                session=session,
            )
        assert not session.closed

    cause = exc_info.value.__cause__
    assert isinstance(cause, aiohttp.ClientResponseError)
    assert cause.status == 400
    expected_body = body.decode("utf-8", errors="replace")
    for message in [str(exc_info.value), *warnings]:
        assert expected_body in message
        assert "private-image-content" not in message
        assert "private-response-cookie" not in message
        assert "input_ids" not in message


@pytest.mark.asyncio
async def test_http_error_collects_chunked_body_before_raising() -> None:
    async def handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(status=400)
        await response.prepare(request)
        await response.write(b"input length ")
        await asyncio.sleep(0.01)
        await response.write(b"exceeds context limit")
        await response.write_eof()
        return response

    async with http_server(handler) as addr:
        with pytest.raises(RuntimeError, match="input length exceeds context limit"):
            await http.arequest_with_retry(addr, "/generate")


@pytest.mark.asyncio
async def test_http_error_bounds_body_read_before_server_finishes() -> None:
    release = asyncio.Event()

    async def handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(status=413)
        await response.prepare(request)
        await response.write(b"x" * 5000)
        await release.wait()
        return response

    async with http_server(handler) as addr:
        try:
            with pytest.raises(RuntimeError) as exc_info:
                await asyncio.wait_for(
                    http.arequest_with_retry(addr, "/generate"), timeout=1.0
                )
            assert "[truncated]" in str(exc_info.value)
            assert len(str(exc_info.value)) < 5000
            assert exc_info.value.__cause__.status == 413
        finally:
            release.set()


@pytest.mark.asyncio
async def test_http_error_body_timeout_preserves_http_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(http, "_HTTP_ERROR_BODY_TIMEOUT", 0.02)
    release = asyncio.Event()

    async def handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(status=503)
        await response.prepare(request)
        await response.write(b"upstream unavailable")
        await release.wait()
        return response

    async with http_server(handler) as addr:
        try:
            with pytest.raises(RuntimeError) as exc_info:
                await http.arequest_with_retry(addr, "/generate", timeout=1.0)
            assert exc_info.value.__cause__.status == 503
            assert "upstream unavailable" in str(exc_info.value)
            assert "body read failed: TimeoutError" in str(exc_info.value)
        finally:
            release.set()


@pytest.mark.asyncio
async def test_http_retry_after_error_returns_successful_json() -> None:
    attempts = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return web.json_response({"error": "temporarily unavailable"}, status=503)
        return web.json_response({"output_ids": [42]})

    async with http_server(handler) as addr:
        result = await http.arequest_with_retry(
            addr, "/generate", max_retries=2, retry_delay=0
        )
    assert result == {"output_ids": [42]}
    assert attempts == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [False, True])
async def test_http_invalid_method_respects_session_ownership(
    borrowed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exception outside the retry handler closes only an owned session."""
    from unittest.mock import AsyncMock, Mock

    session = Mock(close=AsyncMock())
    factory = Mock(return_value=session)
    monkeypatch.setattr(http.aiohttp, "ClientSession", factory)
    monkeypatch.setattr(http, "get_default_connector", Mock())

    with pytest.raises(ValueError, match="Unsupported HTTP method: PATCH"):
        await http.arequest_with_retry(
            "localhost:1234",
            "/generate",
            session=session if borrowed else None,
            method="PATCH",
            max_retries=3,
        )

    assert session.close.await_count == (0 if borrowed else 1)
    assert factory.call_count == (0 if borrowed else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [False, True])
async def test_http_cancel_during_retry_delay_respects_session_ownership(
    borrowed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation in backoff propagates without another attempt or a leak."""
    from unittest.mock import AsyncMock, Mock

    entered_backoff = asyncio.Event()
    context = AsyncMock()
    context.__aenter__.side_effect = aiohttp.ClientConnectionError("disconnected")
    session = Mock(post=Mock(return_value=context), close=AsyncMock())
    monkeypatch.setattr(http.aiohttp, "ClientSession", Mock(return_value=session))
    monkeypatch.setattr(http, "get_default_connector", Mock())

    async def backoff(delay: float) -> None:
        assert delay == 1.0
        entered_backoff.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(http.asyncio, "sleep", backoff)
    task = asyncio.create_task(
        http.arequest_with_retry(
            "localhost:1234",
            "/generate",
            session=session if borrowed else None,
            max_retries=3,
        )
    )
    try:
        await asyncio.wait_for(entered_backoff.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert session.post.call_count == 1
        assert session.close.await_count == (0 if borrowed else 1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("stage", ["headers", "body"])
async def test_http_cancel_real_request_respects_session_ownership(
    borrowed: bool, stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation releases an owned real connector, not a caller's session."""
    entered = asyncio.Event()
    release = asyncio.Event()
    sessions = []
    connectors = []
    original_session = aiohttp.ClientSession
    original_text = aiohttp.ClientResponse.text

    async def track_text(response, *args, **kwargs):
        entered.set()
        return await original_text(response, *args, **kwargs)

    if stage == "body":
        monkeypatch.setattr(aiohttp.ClientResponse, "text", track_text)

    def track_session(**kwargs) -> aiohttp.ClientSession:
        session = original_session(**kwargs)
        sessions.append(session)
        connectors.append(session.connector)
        return session

    monkeypatch.setattr(http.aiohttp, "ClientSession", track_session)

    async def handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/plain"})
        if stage == "body":
            await response.prepare(request)
            await response.write(b"partial response")
        else:
            entered.set()
        await release.wait()
        return response

    async with http_server(handler) as addr:
        session = track_session() if borrowed else None
        task = asyncio.create_task(
            http.arequest_with_retry(
                addr,
                "/generate",
                session=session,
                max_retries=3,
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=1.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(sessions) == 1
            assert sessions[0].closed is (not borrowed)
            assert connectors[0].closed is (not borrowed)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            release.set()
            for created in sessions:
                await created.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [False, True])
async def test_http_invalid_json_respects_session_ownership(
    borrowed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed successful JSON is not retried and cannot leak a session."""
    import json

    sessions = []
    original_session = aiohttp.ClientSession
    attempts = 0

    def track_session(**kwargs) -> aiohttp.ClientSession:
        session = original_session(**kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(http.aiohttp, "ClientSession", track_session)

    async def handler(request: web.Request) -> web.Response:
        nonlocal attempts
        attempts += 1
        return web.Response(text="{invalid json", content_type="application/json")

    async with http_server(handler) as addr:
        session = track_session() if borrowed else None
        try:
            with pytest.raises(json.JSONDecodeError):
                await http.arequest_with_retry(
                    addr,
                    "/generate",
                    session=session,
                    max_retries=3,
                )
            assert attempts == 1
            assert len(sessions) == 1
            assert sessions[0].closed is (not borrowed)
        finally:
            for created in sessions:
                await created.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("succeeds", [False, True])
async def test_http_retry_completion_respects_session_ownership(
    borrowed: bool, succeeds: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normal success and exhausted retries retain ownership and diagnostics."""
    sessions = []
    original_session = aiohttp.ClientSession
    attempts = 0

    def track_session(**kwargs) -> aiohttp.ClientSession:
        session = original_session(**kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(http.aiohttp, "ClientSession", track_session)

    async def handler(request: web.Request) -> web.Response:
        nonlocal attempts
        attempts += 1
        if succeeds and attempts == 2:
            return web.json_response({"result": "ok"})
        return web.Response(status=503, text="temporarily unavailable")

    async with http_server(handler) as addr:
        session = track_session() if borrowed else None
        try:
            request = http.arequest_with_retry(
                addr,
                "/generate",
                session=session,
                max_retries=2,
                retry_delay=0,
            )
            if succeeds:
                assert await request == {"result": "ok"}
            else:
                with pytest.raises(RuntimeError, match="Failed after 2 retries") as exc:
                    await request
                assert isinstance(exc.value.__cause__, aiohttp.ClientResponseError)
                assert exc.value.__cause__.status == 503
                assert "temporarily unavailable" in str(exc.value)
            assert attempts == 2
            assert len(sessions) == 1
            assert sessions[0].closed is (not borrowed)
        finally:
            for created in sessions:
                await created.close()
