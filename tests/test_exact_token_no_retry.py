# SPDX-License-Identifier: Apache-2.0

"""A refusal the server will repeat verbatim must not be retried.

Retrying an exact-token refusal re-sends the same bytes and gets the same
answer, so it only multiplies the request and the alarming log line. Every
other 4xx stays retryable: AReaL's own weight-update handlers answer
worker-side failures with 400, and those are what ``request_retries`` covers.
"""

from __future__ import annotations

import aiohttp
import pytest
from aiohttp import web

from areal.infra.utils.http import arequest_with_retry
from areal.utils.vision_canary import EXACT_TOKEN_REFUSAL


async def _serve(app: web.Application) -> tuple[str, web.AppRunner]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    return f"127.0.0.1:{port}", runner


async def _count_attempts(body: str, status: int, max_retries: int) -> int:
    """Serve ``body`` for every request and report how many arrived."""
    attempts = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal attempts
        attempts += 1
        return web.Response(status=status, text=body)

    app = web.Application()
    app.router.add_post("/generate", handler)
    addr, runner = await _serve(app)
    try:
        with pytest.raises((aiohttp.ClientResponseError, RuntimeError)):
            await arequest_with_retry(
                addr=addr,
                endpoint="/generate",
                payload={},
                method="POST",
                max_retries=max_retries,
                timeout=10,
                retry_delay=0.01,
            )
    finally:
        await runner.cleanup()
    return attempts


@pytest.mark.asyncio
async def test_an_exact_token_refusal_is_sent_once():
    """Test that the refusal short-circuits the retry loop."""
    body = f"{EXACT_TOKEN_REFUSAL} Rendered prompt does not match expected_token_ids."

    assert await _count_attempts(body, status=400, max_retries=3) == 1


@pytest.mark.asyncio
async def test_the_uninspectable_branch_is_also_sent_once():
    """Test that both of the patch's refusal branches skip retries."""
    body = f"{EXACT_TOKEN_REFUSAL} Cannot verify expected_token_ids: ..."

    assert await _count_attempts(body, status=400, max_retries=3) == 1


@pytest.mark.asyncio
async def test_an_ordinary_400_keeps_its_retries():
    """Test that the opt-out is scoped to the signature.

    Weight-update handlers answer worker-side failures with 400 and rely on
    those being retried, so a blanket no-retry-on-4xx rule would break them.
    """
    assert await _count_attempts("worker busy", status=400, max_retries=3) == 3


@pytest.mark.asyncio
async def test_a_503_keeps_its_retries():
    """Test that ordinary server errors are unaffected."""
    assert await _count_attempts("Service Unavailable", status=503, max_retries=3) == 3


@pytest.mark.asyncio
async def test_the_early_exit_does_not_leak_a_session():
    """Test that skipping the retry loop still releases the session.

    The helper opens its own session when the caller supplies none, and closes
    it on the paths that fall out of the retry loop. Returning early has to
    release it too, or every refusal leaks a connection pool.
    """
    import gc
    import warnings

    body = f"{EXACT_TOKEN_REFUSAL} Rendered prompt does not match expected_token_ids."

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await _count_attempts(body, status=400, max_retries=3)
        gc.collect()

    leaks = [w for w in caught if "Unclosed client session" in str(w.message)]
    assert not leaks


@pytest.mark.asyncio
async def test_a_caller_supplied_session_is_left_open():
    """Test that the early exit does not close a session it does not own.

    Callers share one session across many requests; closing it here would break
    every later call rather than just this one.
    """
    body = f"{EXACT_TOKEN_REFUSAL} Cannot verify expected_token_ids: ..."

    async def handler(request: web.Request) -> web.Response:
        return web.Response(status=400, text=body)

    app = web.Application()
    app.router.add_post("/generate", handler)
    addr, runner = await _serve(app)
    session = aiohttp.ClientSession()
    try:
        with pytest.raises(aiohttp.ClientResponseError):
            await arequest_with_retry(
                addr=addr,
                endpoint="/generate",
                payload={},
                method="POST",
                max_retries=3,
                timeout=10,
                retry_delay=0.01,
                session=session,
            )
        assert not session.closed
    finally:
        await session.close()
        await runner.cleanup()
