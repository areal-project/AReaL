from __future__ import annotations

import asyncio
import socket
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from areal.api.cli_args import InferenceEngineConfig
from areal.infra.utils.concurrent import register_loop_cleanup, run_async_task
from areal.infra.utils.http import (
    HttpxAsyncClientCleanup,
    close_httpx_client_from_sync,
    close_httpx_client_on_owner_loop,
    register_httpx_client_loop_cleanup,
)
from areal.v2.inference_service.controller.controller import RolloutControllerV2


def _controller() -> RolloutControllerV2:
    return RolloutControllerV2(
        config=InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-key",
            setup_timeout=1.0,
            workers_ready_timeout=1.0,
        ),
        scheduler=MagicMock(n_gpus_per_node=8),
    )


def test_destroy_rejects_open_async_client_after_owner_loop_closed() -> None:
    controller = _controller()
    owner_loop = MagicMock()
    owner_loop.is_closed.return_value = True
    client = MagicMock()
    client.is_closed = False
    client.aclose = AsyncMock()
    cleanup = HttpxAsyncClientCleanup(client, owner_loop)
    controller._async_client = client
    controller._async_client_loop = owner_loop
    controller._async_client_cleanup = cleanup

    with pytest.raises(RuntimeError, match="owner event loop closed"):
        controller.destroy()

    client.aclose.assert_not_awaited()
    assert controller._async_client is client
    assert controller._async_client_loop is owner_loop
    assert controller._async_client_cleanup is cleanup
    assert controller._destroyed is False


def test_get_async_client_rejects_access_after_destroy() -> None:
    controller = _controller()
    controller.destroy()
    client = MagicMock()
    client.aclose = AsyncMock()
    loop = asyncio.new_event_loop()
    try:
        with (
            patch(
                "areal.v2.inference_service.controller.controller.create_httpx_client",
                return_value=client,
            ) as create_client,
            pytest.raises(RuntimeError, match="has been destroyed"),
        ):
            loop.run_until_complete(controller._get_async_client())
    finally:
        loop.close()

    create_client.assert_not_called()
    client.aclose.assert_not_awaited()
    assert controller._async_client is None
    assert controller._async_client_loop is None
    assert controller._async_client_cleanup is None


def test_async_client_closes_on_owner_loop_shutdown_before_destroy() -> None:
    controller = _controller()
    client = MagicMock()
    client.is_closed = False
    close_loops: list[asyncio.AbstractEventLoop] = []

    async def close_client() -> None:
        close_loops.append(asyncio.get_running_loop())
        client.is_closed = True

    client.aclose = AsyncMock(side_effect=close_client)
    loop = asyncio.new_event_loop()
    try:
        with patch(
            "areal.v2.inference_service.controller.controller.create_httpx_client",
            return_value=client,
        ):
            assert loop.run_until_complete(controller._get_async_client()) is client
    finally:
        loop.close()

    client.aclose.assert_awaited_once_with()
    assert close_loops == [loop]
    assert client.is_closed is True
    assert controller._async_client is None
    assert controller._async_client_loop is None
    assert controller._async_client_cleanup is None

    controller.destroy()

    client.aclose.assert_awaited_once_with()
    assert controller._destroyed is True


def test_cancelled_loop_cleanup_preserves_failure_and_runs_later_callbacks() -> None:
    loop = asyncio.new_event_loop()
    cancelled = asyncio.CancelledError("transport close cancelled")
    client = MagicMock()
    client.aclose = AsyncMock(side_effect=cancelled)
    cleanup = HttpxAsyncClientCleanup(client, loop)
    later_callbacks: list[str] = []
    cleared: list[bool] = []

    # Callbacks run LIFO, so register the observer first and the failing HTTP
    # cleanup second.  The observer proves that cancellation did not abort the
    # rest of loop teardown.
    register_loop_cleanup(lambda: later_callbacks.append("ran"), loop=loop)
    register_httpx_client_loop_cleanup(cleanup, on_closed=lambda: cleared.append(True))
    try:
        loop.close()
    finally:
        # Keep the test from leaking an open loop if teardown raises.
        if not loop.is_closed():
            loop._cleanup_orig_close()

    assert loop.is_closed()
    assert later_callbacks == ["ran"]
    assert cleared == []
    assert cleanup.succeeded is False
    assert cleanup.error is cancelled
    client.aclose.assert_awaited_once_with()


def test_sequential_run_async_task_clients_close_on_their_own_loops() -> None:
    controller = _controller()
    clients: list[MagicMock] = []
    close_loops: list[asyncio.AbstractEventLoop] = []

    for _ in range(2):
        client = MagicMock()
        client.is_closed = False

        async def close_client(client=client) -> None:
            close_loops.append(asyncio.get_running_loop())
            client.is_closed = True

        client.aclose = AsyncMock(side_effect=close_client)
        clients.append(client)

    with patch(
        "areal.v2.inference_service.controller.controller.create_httpx_client",
        side_effect=clients,
    ):
        assert run_async_task(controller._get_async_client) is clients[0]
        assert controller._async_client is None
        assert run_async_task(controller._get_async_client) is clients[1]

    assert controller._async_client is None
    assert controller._async_client_loop is None
    assert controller._async_client_cleanup is None
    assert close_loops[0] is not close_loops[1]
    for client in clients:
        client.aclose.assert_awaited_once_with()


def test_run_async_task_closes_real_keepalive_transport_before_loop_exit() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(2.0)
    peer_saw_eof = threading.Event()
    server_done = threading.Event()
    server_errors: list[BaseException] = []

    def serve_one_keepalive_response() -> None:
        try:
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(2.0)
                request = bytearray()
                while b"\r\n\r\n" not in request:
                    chunk = conn.recv(4096)
                    if not chunk:
                        raise AssertionError("client closed before request headers")
                    request.extend(chunk)
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Length: 2\r\n"
                    b"Connection: keep-alive\r\n\r\nok"
                )
                if conn.recv(1) == b"":
                    peer_saw_eof.set()
        except BaseException as exc:
            server_errors.append(exc)
        finally:
            server_done.set()

    thread = threading.Thread(target=serve_one_keepalive_response, daemon=True)
    thread.start()
    controller = _controller()
    old_policy = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    try:

        async def make_request() -> None:
            client = await controller._get_async_client()
            response = await client.get(f"http://127.0.0.1:{listener.getsockname()[1]}")
            assert response.status_code == 200
            assert response.content == b"ok"

        run_async_task(make_request)
        assert peer_saw_eof.wait(timeout=2.0)
        assert server_done.wait(timeout=2.0)
        assert server_errors == []
        assert controller._async_client is None
        assert controller._async_client_loop is None
        assert controller._async_client_cleanup is None
    finally:
        asyncio.set_event_loop_policy(old_policy)
        listener.close()
        thread.join(timeout=2.0)
        controller.destroy()


def test_destroy_retains_async_client_when_live_loop_close_fails() -> None:
    controller = _controller()
    owner_loop = MagicMock()
    primary = RuntimeError("async transport close failed")
    client = MagicMock()
    client.is_closed = False
    client.aclose = AsyncMock(side_effect=primary)
    cleanup = HttpxAsyncClientCleanup(client, owner_loop)
    controller._async_client = client
    controller._async_client_loop = owner_loop
    controller._async_client_cleanup = cleanup

    with (
        patch(
            "areal.v2.inference_service.controller.controller."
            "close_httpx_client_from_sync",
            side_effect=primary,
        ),
        pytest.raises(RuntimeError) as exc_info,
    ):
        controller.destroy()

    assert exc_info.value is primary
    assert controller._async_client is client
    assert controller._async_client_loop is owner_loop
    assert controller._async_client_cleanup is cleanup
    assert controller._destroyed is False


def test_transport_close_failure_is_never_mistaken_for_httpx_closed() -> None:
    primary = RuntimeError("transport resource is still open")

    class FailingCloseTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.resource_open = True
            self.close_calls = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, request=request)

        async def aclose(self) -> None:
            self.close_calls += 1
            raise primary

    controller = _controller()
    transport = FailingCloseTransport()
    client = httpx.AsyncClient(transport=transport)
    loop = asyncio.new_event_loop()
    try:
        with patch(
            "areal.v2.inference_service.controller.controller.create_httpx_client",
            return_value=client,
        ):
            assert loop.run_until_complete(controller._get_async_client()) is client
            cleanup = controller._async_client_cleanup
            assert cleanup is not None
    finally:
        loop.close()

    assert client.is_closed is True
    assert transport.resource_open is True
    assert transport.close_calls == 1
    assert cleanup.succeeded is False
    assert cleanup.error is primary
    assert controller._async_client is client
    assert controller._async_client_cleanup is cleanup

    for _ in range(2):
        with pytest.raises(RuntimeError) as exc_info:
            controller.destroy()
        assert exc_info.value is primary
        assert controller._async_client is client
        assert controller._async_client_cleanup is cleanup
        assert transport.close_calls == 1


@pytest.mark.asyncio
async def test_async_client_close_propagates_failure_on_owner_loop() -> None:
    owner_loop = asyncio.get_running_loop()
    primary = RuntimeError("async transport close failed")
    client = MagicMock()
    client.is_closed = False
    client.aclose = AsyncMock(side_effect=primary)
    cleanup = HttpxAsyncClientCleanup(client, owner_loop)

    with pytest.raises(RuntimeError) as exc_info:
        await close_httpx_client_on_owner_loop(cleanup)

    assert exc_info.value is primary
    client.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_async_client_close_is_single_flight() -> None:
    owner_loop = asyncio.get_running_loop()
    primary = RuntimeError("single transport close failed")
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    client = MagicMock()

    async def close_client() -> None:
        close_started.set()
        await release_close.wait()
        raise primary

    client.aclose = AsyncMock(side_effect=close_client)
    cleanup = HttpxAsyncClientCleanup(client, owner_loop)
    first = asyncio.create_task(cleanup.close())
    second: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(close_started.wait(), timeout=1.0)
        second = asyncio.create_task(cleanup.close())
        await asyncio.sleep(0)
        release_close.set()
        results = await asyncio.wait_for(
            asyncio.gather(first, second, return_exceptions=True), timeout=1.0
        )
    finally:
        release_close.set()
        pending = [
            task for task in (first, second) if task is not None and not task.done()
        ]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    assert results == [primary, primary]
    client.aclose.assert_awaited_once_with()
    assert cleanup.succeeded is False
    assert cleanup.error is primary


@pytest.mark.asyncio
async def test_cancelled_close_follower_does_not_cancel_shared_completion() -> None:
    owner_loop = asyncio.get_running_loop()
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    client = MagicMock()

    async def close_client() -> None:
        close_started.set()
        await release_close.wait()

    client.aclose = AsyncMock(side_effect=close_client)
    cleanup = HttpxAsyncClientCleanup(client, owner_loop)
    leader = asyncio.create_task(cleanup.close())
    follower: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(close_started.wait(), timeout=1.0)
        follower = asyncio.create_task(cleanup.close())
        await asyncio.sleep(0)

        follower.cancel()
        with pytest.raises(asyncio.CancelledError):
            await follower
        release_close.set()
        await asyncio.wait_for(leader, timeout=1.0)
    finally:
        release_close.set()
        pending = [
            task for task in (leader, follower) if task is not None and not task.done()
        ]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    client.aclose.assert_awaited_once_with()
    assert cleanup.succeeded is True
    assert cleanup.error is None


def test_sync_async_client_close_runs_on_live_owner_loop() -> None:
    owner_loop = asyncio.new_event_loop()
    owner_started = threading.Event()
    close_loops: list[asyncio.AbstractEventLoop] = []
    thread_errors: list[BaseException] = []

    def run_owner_loop() -> None:
        try:
            asyncio.set_event_loop(owner_loop)
            owner_loop.call_soon(owner_started.set)
            owner_loop.run_forever()
        except BaseException as exc:
            thread_errors.append(exc)
        finally:
            owner_loop.close()

    owner_thread = threading.Thread(target=run_owner_loop, daemon=True)
    owner_thread.start()
    try:
        assert owner_started.wait(timeout=1.0)
        client = MagicMock()
        client.is_closed = False

        async def close_client() -> None:
            close_loops.append(asyncio.get_running_loop())
            client.is_closed = True

        client.aclose = AsyncMock(side_effect=close_client)
        cleanup = HttpxAsyncClientCleanup(client, owner_loop)
        close_httpx_client_from_sync(cleanup)
    finally:
        if not owner_loop.is_closed():
            owner_loop.call_soon_threadsafe(owner_loop.stop)
        owner_thread.join(timeout=1.0)

    assert not owner_thread.is_alive()
    assert thread_errors == []
    assert close_loops == [owner_loop]
    client.aclose.assert_awaited_once_with()


def test_get_async_client_rejects_concurrent_foreign_event_loop() -> None:
    controller = _controller()
    owner_loop = asyncio.new_event_loop()
    owner_started = threading.Event()
    owner_errors: list[BaseException] = []
    close_loops: list[asyncio.AbstractEventLoop] = []
    client = MagicMock()

    async def close_client() -> None:
        close_loops.append(asyncio.get_running_loop())

    client.aclose = AsyncMock(side_effect=close_client)

    def run_owner_loop() -> None:
        try:
            asyncio.set_event_loop(owner_loop)
            owner_loop.call_soon(owner_started.set)
            owner_loop.run_forever()
        except BaseException as exc:
            owner_errors.append(exc)
        finally:
            owner_loop.close()

    owner_thread = threading.Thread(target=run_owner_loop, daemon=True)
    owner_thread.start()
    try:
        assert owner_started.wait(timeout=1.0)
        with patch(
            "areal.v2.inference_service.controller.controller.create_httpx_client",
            return_value=client,
        ):
            owner_client = asyncio.run_coroutine_threadsafe(
                controller._get_async_client(), owner_loop
            ).result(timeout=1.0)
            assert owner_client is client

            with pytest.raises(RuntimeError, match="active on another event loop"):
                asyncio.run(controller._get_async_client())

            client.aclose.assert_not_awaited()
    finally:
        if not owner_loop.is_closed():
            owner_loop.call_soon_threadsafe(owner_loop.stop)
        owner_thread.join(timeout=1.0)

    assert not owner_thread.is_alive()
    assert owner_errors == []
    assert close_loops == [owner_loop]
    client.aclose.assert_awaited_once_with()
    assert controller._async_client is None
    assert controller._async_client_loop is None
    assert controller._async_client_cleanup is None


@pytest.mark.asyncio
async def test_sync_async_client_close_rejects_owner_loop_reentrancy() -> None:
    owner_loop = asyncio.get_running_loop()
    client = MagicMock()
    client.is_closed = False
    client.aclose = AsyncMock()
    cleanup = HttpxAsyncClientCleanup(client, owner_loop)

    with pytest.raises(RuntimeError, match="from its owner event loop"):
        close_httpx_client_from_sync(cleanup)

    client.aclose.assert_not_awaited()
