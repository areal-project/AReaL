# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections import Counter
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from aiohttp import ClientSession, web

from areal.v2.inference_service.router.app import create_app
from areal.v2.inference_service.router.backend import (
    Acquire,
    BackendRegistry,
    BackendSpec,
    CloseSession,
    RegisterBackends,
    Release,
    RoutingError,
    SetPoolState,
)
from areal.v2.inference_service.router.client import RouteClient
from areal.v2.inference_service.router.config import RouterConfig

IDENTITY = {"pool_id": "pool", "run_id": "run"}


async def registry(capacities=(2, 2)):
    state = BackendRegistry()
    await state.register(
        RegisterBackends(
            **IDENTITY,
            weight_version=0,
            backends=[
                BackendSpec(
                    backend_id=str(i), backend_addr=f"http://backend-{i}", capacity=c
                )
                for i, c in enumerate(capacities)
            ],
        )
    )
    return state


async def transition(state, status, seq, version=0):
    await state.set_state(
        "pool",
        SetPoolState(
            run_id="run",
            control_seq=seq,
            state=status,
            weight_version=version,
        ),
    )


async def acquire(state, request_id, session_id=None):
    return await state.acquire(
        Acquire(**IDENTITY, request_id=request_id, session_id=session_id)
    )


async def release(state, lease, outcome="completed"):
    await state.release(Release(**IDENTITY, lease_id=lease.lease_id, outcome=outcome))


@pytest.mark.asyncio
async def test_new_sessions_balance_normalized_load_atomically_and_stay_sticky():
    state = await registry((2, 4))
    await transition(state, "READY", 0)
    leases = await asyncio.gather(
        *[acquire(state, str(i), f"session-{i}") for i in range(30)]
    )
    assert Counter(x.backend_id for x in leases) == {"0": 10, "1": 20}
    sticky = await acquire(state, "next-turn", "session-0")
    assert sticky.backend_id == leases[0].backend_id
    await asyncio.gather(*[release(state, x) for x in [*leases, sticky]])
    snapshot = await state.snapshot(**IDENTITY)
    assert snapshot["outstanding"] == 0
    assert snapshot["sessions"] == 30  # Tool waiting / idle bindings cost no N.


@pytest.mark.asyncio
async def test_duplicate_acquire_release_and_finished_request_cannot_double_count():
    state = await registry()
    await transition(state, "READY", 0)
    leases = await asyncio.gather(*[acquire(state, "request", "s") for _ in range(10)])
    assert len({x.lease_id for x in leases}) == 1
    with pytest.raises(RoutingError, match="REQUEST_CONFLICT"):
        await acquire(state, "request", "another-session")
    await transition(state, "PAUSED", 1)
    assert await acquire(state, "request", "s") == leases[0]
    await asyncio.gather(*[release(state, x) for x in leases])
    assert (await state.snapshot(**IDENTITY))["outstanding"] == 0
    with pytest.raises(RoutingError, match="REQUEST_FINISHED"):
        await acquire(state, "request", "s")


@pytest.mark.asyncio
async def test_pause_close_and_old_run_rejections_do_not_create_leases():
    state = await registry()
    with pytest.raises(RoutingError, match="PAUSED"):
        await acquire(state, "paused", "s")
    await transition(state, "READY", 0)
    lease = await acquire(state, "active", "s")
    close = CloseSession(**IDENTITY, session_id="s")
    await state.close_session(close)
    await state.close_session(close)
    with pytest.raises(RoutingError, match="SESSION_CLOSED"):
        await acquire(state, "late", "s")
    assert (await state.snapshot(**IDENTITY))["outstanding"] == 1
    await release(state, lease)
    assert (await state.snapshot(**IDENTITY))["sessions"] == 0
    with pytest.raises(RoutingError, match="SESSION_CLOSED"):
        await acquire(state, "much-later", "s")
    with pytest.raises(RoutingError, match="STALE_RUN"):
        await state.acquire(Acquire(pool_id="pool", run_id="old", request_id="old"))


@pytest.mark.asyncio
async def test_weight_publication_requires_drain_and_late_resume_cannot_reopen_failure():
    state = await registry()
    await transition(state, "READY", 0)
    lease = await acquire(state, "old", "s")
    await transition(state, "PAUSED", 1)
    with pytest.raises(RoutingError, match="NOT_DRAINED"):
        await transition(state, "PAUSED", 2, version=1)
    await release(state, lease, "aborted")
    await transition(state, "PAUSED", 2, version=1)
    await transition(state, "PAUSED", 2, version=1)
    with pytest.raises(RoutingError, match="STALE_CONTROL"):
        await transition(state, "READY", 1, version=1)
    await transition(state, "READY", 3, version=1)
    new = await acquire(state, "continuation", "s")
    assert new.backend_id == lease.backend_id
    assert (lease.weight_version, new.weight_version) == (0, 1)
    await transition(state, "FAILED", 4, version=1)
    with pytest.raises(RoutingError, match="FAILED"):
        # A lost acquire reply must not admit a first send after another
        # worker failed the pool, even though this request owns a reservation.
        await acquire(state, "continuation", "s")
    with pytest.raises(RoutingError, match="POOL_TERMINAL"):
        await transition(state, "READY", 5, version=1)
    assert (await state.snapshot(**IDENTITY))["outstanding"] == 1


@pytest.mark.asyncio
async def test_unhealthy_bound_backend_is_not_migrated_and_pools_cannot_double_own():
    state = await registry()
    await transition(state, "READY", 0)
    lease = await acquire(state, "r1", "s")
    await state.update_health("pool", "run", lease.backend_id, False)
    with pytest.raises(RoutingError, match="UNAVAILABLE"):
        await acquire(state, "r2", "s")
    with pytest.raises(RoutingError, match="BACKEND_ALREADY_OWNED"):
        await state.register(
            RegisterBackends(
                pool_id="other",
                run_id="other",
                weight_version=0,
                backends=[
                    BackendSpec(
                        backend_id="b", backend_addr=lease.backend_addr, capacity=2
                    )
                ],
            )
        )


@pytest.mark.asyncio
async def test_route_validates_target_and_preserves_owner_response_and_counts():
    app = create_app(RouterConfig(admin_api_key="test"))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://router",
        headers={"Authorization": "Bearer test"},
    ) as client:
        assert (
            await client.post("/register", json={"worker_addr": "http://owner"})
        ).status_code == 200
        original = await client.post("/route", json={"api_key": "test"})
        assert original.json() == {
            "worker_addr": "http://owner",
            "url": None,
            "api_key": None,
        }
        for body in (
            {"target": "unknown"},
            {"target": "backend"},
            {"target": "backend", **IDENTITY, "request_id": ""},
        ):
            assert (await client.post("/route", json=body)).status_code == 422
        assert (
            await client.post(
                "/routing/backends/register",
                json={
                    **IDENTITY,
                    "weight_version": 0,
                    "backends": [
                        {
                            "backend_id": "b",
                            "backend_addr": "http://backend",
                            "capacity": 1,
                        }
                    ],
                },
            )
        ).status_code == 200
        assert (
            await client.post(
                "/route", json={"target": "backend", **IDENTITY, "request_id": "r"}
            )
        ).json()["detail"]["code"] == "PAUSED"
        assert (
            await client.post("/route", json={"api_key": "test"})
        ).json() == original.json()
        await client.post(
            "/routing/pools/pool/state",
            json={
                "run_id": "run",
                "control_seq": 0,
                "state": "READY",
                "weight_version": 0,
            },
        )
        routed = await client.post(
            "/route", json={"target": "backend", **IDENTITY, "request_id": "r"}
        )
        assert set(routed.json()) == {
            "lease_id",
            "backend_id",
            "backend_addr",
            "weight_version",
        }


@pytest_asyncio.fixture
async def live_client():
    state = await registry()
    await transition(state, "READY", 0)
    controls = {
        "lose_acquire": False,
        "lose_release": False,
        "delay_acquire": None,
        "acquired": asyncio.Event(),
        "requests": [],
    }

    async def handler(request):
        path = request.path
        if path == "/generate":
            return await controls["generate"](request)
        body = await request.json() if request.method == "POST" else {}
        try:
            result = {"status": "ok"}
            if path == "/route":
                controls["requests"].append(body["request_id"])
                result = (await state.acquire(Acquire(**body))).model_dump()
                controls["acquired"].set()
                if controls["delay_acquire"] is not None:
                    await controls["delay_acquire"].wait()
                if controls["lose_acquire"]:
                    controls["lose_acquire"] = False
                    return web.json_response({}, status=500)
            elif path == "/routing/release":
                await state.release(Release(**body))
                if controls["lose_release"]:
                    controls["lose_release"] = False
                    return web.json_response({}, status=500)
            elif path == "/routing/close_session":
                await state.close_session(CloseSession(**body))
            elif request.method == "GET":
                result = await state.snapshot(
                    pool_id=path.split("/")[3], run_id=request.query["run_id"]
                )
            else:
                await state.set_state(path.split("/")[3], SetPoolState(**body))
            return web.json_response(result)
        except RoutingError as exc:
            return web.json_response({"detail": {"code": exc.code}}, status=409)

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    controls["address"] = f"http://127.0.0.1:{port}"
    route = RouteClient(
        f"http://127.0.0.1:{port}", "pool", "run", "test", metadata_timeout=2
    )
    route.set_sender_state(False, 0)
    async with ClientSession() as session:
        yield state, route, session, controls
        tasks = list(route._tasks)
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 3)
    await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["recover", "close", "timeout"])
async def test_unavailable_bound_backend_waits_without_sending(live_client, action):
    state, route, session, controls = live_client
    sent = []

    async def generate(lease):
        sent.append(lease.backend_id)
        return "result", "completed"

    await route.execute(session, "sticky", generate)
    backend = sent.pop()
    await state.update_health("pool", "run", backend, False)
    initial_requests = len(controls["requests"])
    if action == "timeout":
        route.control_timeout = 0.2
    task = asyncio.create_task(route.execute(session, "sticky", generate))
    async with asyncio.timeout(2):
        while len(controls["requests"]) == initial_requests:
            await asyncio.sleep(0.01)
    snapshot = await state.snapshot(**IDENTITY)
    assert snapshot["state"] == "READY"
    assert snapshot["outstanding"] == 0
    assert sent == []

    if action == "recover":
        await state.update_health("pool", "run", backend, True)
        assert await asyncio.wait_for(task, 2) == ("result", 0)
        assert sent == [backend]
    elif action == "close":
        await route.close_session(session, "sticky")
        with pytest.raises(RoutingError, match="SESSION_CLOSED"):
            await asyncio.wait_for(task, 2)
        assert sent == []
        assert (await state.snapshot(**IDENTITY))["state"] == "READY"
    else:
        with pytest.raises(RoutingError, match="CONTROL_TIMEOUT"):
            await asyncio.wait_for(task, 2)
        assert sent == []
        assert (await state.snapshot(**IDENTITY))["state"] == "FAILED"
    assert (await state.snapshot(**IDENTITY))["outstanding"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["recover", "pause", "cancel"])
async def test_no_healthy_backend_waits_without_creating_session(live_client, action):
    state, route, session, controls = live_client
    sent = []
    for backend in ("0", "1"):
        await state.update_health("pool", "run", backend, False)

    async def generate(lease):
        sent.append(lease.backend_id)
        return "result", "completed"

    task = asyncio.create_task(route.execute(session, "new", generate))
    async with asyncio.timeout(2):
        while not controls["requests"]:
            await asyncio.sleep(0.01)
    snapshot = await state.snapshot(**IDENTITY)
    assert snapshot["state"] == "READY"
    assert snapshot["sessions"] == snapshot["outstanding"] == 0
    assert sent == []

    if action == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await state.update_health("pool", "run", "0", True)
        async with asyncio.timeout(2):
            while route._tasks:
                await asyncio.sleep(0.01)
        assert sent == []
        assert (await state.snapshot(**IDENTITY))["sessions"] == 0
    else:
        if action == "pause":
            route.set_sender_state(True, 1)
        await state.update_health("pool", "run", "0", True)
        if action == "pause":
            await asyncio.sleep(0.3)
            assert not task.done()
            assert sent == []
            assert (await state.snapshot(**IDENTITY))["sessions"] == 0
            route.set_sender_state(False, 2)
        assert await asyncio.wait_for(task, 2) == ("result", 0)
        assert sent == ["0"]
    assert (await state.snapshot(**IDENTITY))["outstanding"] == 0


@pytest.mark.asyncio
async def test_lost_metadata_replies_never_duplicate_generation(live_client):
    state, route, session, controls = live_client
    controls.update(lose_acquire=True, lose_release=True)
    sent = []

    async def generate(lease):
        sent.append(lease)
        return "result", "completed"

    assert await route.execute(session, "s", generate) == ("result", 0)
    assert len(sent) == 1
    assert len(controls["requests"]) == 2
    assert len(set(controls["requests"])) == 1
    assert (await state.snapshot(**IDENTITY))["outstanding"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["close", "pause_and_resume", "cancel"])
async def test_late_acquire_cannot_send_after_close_pause_or_cancel(
    live_client, action
):
    state, route, session, controls = live_client
    barrier = asyncio.Event()
    controls["delay_acquire"] = barrier
    sent = []

    async def generate(lease):
        sent.append(lease)
        return "ok", "completed"

    task = asyncio.create_task(route.execute(session, "s", generate))
    await asyncio.wait_for(controls["acquired"].wait(), 2)
    first_id = controls["requests"][0]
    if action == "close":
        await route.close_session(session, "s")
    elif action == "pause_and_resume":
        route.set_sender_state(True, 1)
        route.set_sender_state(False, 2)
    else:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    barrier.set()
    result = await asyncio.gather(task, return_exceptions=True)
    await asyncio.gather(*list(route._tasks), return_exceptions=True)
    if action == "pause_and_resume":
        assert result == [("ok", 0)]
        assert controls["requests"][-1] != first_id
        assert len(sent) == 1
    else:
        assert not sent
    assert (await state.snapshot(**IDENTITY))["outstanding"] == 0


@pytest.mark.asyncio
async def test_cancelled_caller_retains_terminal_response_and_release(live_client):
    state, route, session, _ = live_client
    started, finish = asyncio.Event(), asyncio.Event()

    async def generate(lease):
        started.set()
        await finish.wait()
        return "tokens", "completed"

    caller = asyncio.create_task(route.execute(session, "s", generate))
    await asyncio.wait_for(started.wait(), 2)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    await route.close_session(session, "s")
    assert (await state.snapshot(**IDENTITY))["outstanding"] == 1
    finish.set()
    await asyncio.gather(*list(route._tasks))
    snapshot = await state.snapshot(**IDENTITY)
    assert snapshot["outstanding"] == snapshot["sessions"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("broken_transport", [False, True])
async def test_uncertain_execution_fails_pool_without_decrement_or_retry(
    live_client, broken_transport
):
    state, route, session, controls = live_client

    async def generate(lease):
        if broken_transport:
            await session.close()
        raise TimeoutError("reply lost")

    with pytest.raises(RoutingError, match="EXECUTION_UNCERTAIN"):
        await route.execute(session, "s", generate)
    snapshot = await state.snapshot(**IDENTITY)
    assert snapshot["state"] == "FAILED"
    assert snapshot["outstanding"] == 1
    assert len(controls["requests"]) == 1
    with pytest.raises(RoutingError, match="FAILED"):
        await route.execute(session, "s", generate)


@pytest.mark.asyncio
async def test_engine_direct_http_abort_continuation_uses_lease_versions(
    live_client, monkeypatch
):
    from areal.api.cli_args import GenerationHyperparameters, InferenceEngineConfig
    from areal.api.io_struct import ModelRequest
    from areal.engine.sglang_remote import SGLangBackend
    from areal.infra import workflow_context
    from areal.infra.remote_inf_engine import RemoteInfEngine

    state, _, http, controls = live_client
    await state.register(
        RegisterBackends(
            pool_id="gen",
            run_id="run",
            weight_version=0,
            backends=[
                BackendSpec(
                    backend_id="b", backend_addr=controls["address"], capacity=2
                )
            ],
        )
    )
    await state.set_state(
        "gen",
        SetPoolState(run_id="run", control_seq=0, state="READY", weight_version=0),
    )
    route = RouteClient(controls["address"], "gen", "run", "test", metadata_timeout=2)
    route.set_sender_state(False, 0)
    engine = RemoteInfEngine(
        InferenceEngineConfig(backend="sglang:d1"), SGLangBackend()
    )
    engine.route_client = route
    engine.addresses = ["original-management-address"]
    engine._workflow_executor = SimpleNamespace(is_paused=lambda: False)
    calls, updates = [], []

    async def publish_after_drain():
        for _ in range(200):
            if (await state.snapshot("gen", "run"))["outstanding"] == 0:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("abort lease did not drain")
        await state.set_state(
            "gen",
            SetPoolState(run_id="run", control_seq=2, state="PAUSED", weight_version=1),
        )
        route.set_sender_state(False, 3)
        await state.set_state(
            "gen",
            SetPoolState(run_id="run", control_seq=3, state="READY", weight_version=1),
        )

    async def generate(request):
        calls.append(await request.json())
        # A response-time engine version must never relabel the old segment.
        engine.set_version(99)
        aborted = len(calls) == 1
        if aborted:
            await state.set_state(
                "gen",
                SetPoolState(
                    run_id="run", control_seq=1, state="PAUSED", weight_version=0
                ),
            )
            route.set_sender_state(True, 1)
            updates.append(asyncio.create_task(publish_after_drain()))
        return web.json_response(
            {
                "meta_info": {
                    "finish_reason": {"type": "abort" if aborted else "stop"},
                    "output_token_logprobs": [[-0.1, 4 if aborted else 5]],
                }
            }
        )

    async def get_http():
        return http

    controls["generate"] = generate
    monkeypatch.setattr(workflow_context, "get_aiohttp_session", get_http)
    response = await asyncio.wait_for(
        engine.agenerate(
            ModelRequest(
                input_ids=[1, 2, 3],
                gconfig=GenerationHyperparameters(
                    n_samples=1, max_new_tokens=5, max_tokens=16
                ),
            )
        ),
        5,
    )
    await asyncio.gather(*updates)
    assert response.output_tokens == [4, 5]
    assert response.output_versions == [0, 1]
    assert calls[1]["input_ids"] == [1, 2, 3, 4]
    assert engine.addresses == ["original-management-address"]
    snapshot = await state.snapshot("gen", "run")
    assert snapshot["outstanding"] == snapshot["sessions"] == 0


@pytest.mark.asyncio
async def test_retirement_requires_explicit_process_teardown_before_reusing_backends():
    state = await registry()
    await transition(state, "READY", 0)
    lease = await acquire(state, "uncertain", "s")
    await transition(state, "FAILED", 1)
    with pytest.raises(RoutingError, match="POOL_TERMINAL"):
        await transition(state, "CLOSED", 2)
    await state.set_state(
        "pool",
        SetPoolState(
            run_id="run",
            control_seq=2,
            state="CLOSED",
            weight_version=0,
            old_processes_stopped=True,
        ),
    )
    await state.register(
        RegisterBackends(
            pool_id="recovered",
            run_id="new",
            weight_version=0,
            backends=[
                BackendSpec(backend_id="b", backend_addr=lease.backend_addr, capacity=2)
            ],
        )
    )
    with pytest.raises(RoutingError, match="CLOSED"):
        await acquire(state, "old-sender", "new-session")
    assert (await state.snapshot("recovered", "new"))["outstanding"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("healthy_before_pause", [False, True])
async def test_health_probe_across_pause_cannot_change_resumed_health(
    healthy_before_pause,
):
    state = await registry((2,))
    assert await state.health_targets() == []
    await transition(state, "READY", 0)
    target = (await state.health_targets())[0]
    pool_id, run_id, backend_id, _, probe_seq = target
    await state.update_health(
        pool_id, run_id, backend_id, healthy_before_pause, control_seq=probe_seq
    )
    await transition(state, "PAUSED", 1)
    assert await state.health_targets() == []
    # The same old response is invalid during pause and after resume, including
    # a late success that could otherwise reopen a genuinely failed backend.
    for resumed in (False, True):
        if resumed:
            await transition(state, "READY", 2)
        await state.update_health(
            pool_id,
            run_id,
            backend_id,
            not healthy_before_pause,
            control_seq=probe_seq,
        )
        snapshot = await state.snapshot(**IDENTITY)
        assert snapshot["backends"][backend_id]["healthy"] == healthy_before_pause
    current_seq = (await state.health_targets())[0][-1]
    await state.update_health(
        pool_id, run_id, backend_id, True, control_seq=current_seq
    )
    lease = await acquire(state, "after-resume", "sticky")
    assert lease.backend_id == backend_id
    await release(state, lease)


@pytest.mark.asyncio
@pytest.mark.parametrize("sender_paused", [False, True])
async def test_intentional_pause_does_not_expire_health_budget(
    live_client, sender_paused
):
    state, route, session, _ = live_client
    route.control_timeout = 0.01
    await transition(state, "PAUSED", 1)
    route.set_sender_state(sender_paused, 1)
    sent = []

    async def generate(lease):
        sent.append(lease)
        return "tokens", "completed"

    task = asyncio.create_task(route.execute(session, "paused", generate))
    try:
        await asyncio.sleep(0.15)
        assert not task.done()
        assert sent == []
        await transition(state, "READY", 2)
        route.set_sender_state(False, 2)
        assert await asyncio.wait_for(task, 2) == ("tokens", 0)
        assert len(sent) == 1
        assert (await state.snapshot(**IDENTITY))["outstanding"] == 0
    finally:
        route.close_local_session("paused")
        await asyncio.gather(task, return_exceptions=True)
