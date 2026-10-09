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
