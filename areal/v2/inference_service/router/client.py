# SPDX-License-Identifier: Apache-2.0

"""Reusable metadata client and sender fence for direct backend generation."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import TypeVar

import aiohttp
import requests

from areal.v2.inference_service.router.backend import LeaseResponse, RoutingError

T = TypeVar("T")
current_backend_session: ContextVar[str | None] = ContextVar(
    "backend_session", default=None
)


@asynccontextmanager
async def backend_session(engine, session_id: str | None = None, *, close: bool = True):
    """Trusted trajectory identity, never taken from agent-provided metadata."""
    session_id = session_id or uuid.uuid4().hex
    token = current_backend_session.set(session_id)
    try:
        yield session_id
    finally:
        current_backend_session.reset(token)
        if close and isinstance(getattr(engine, "route_client", None), RouteClient):
            await engine.close_backend_session(session_id)


@dataclass
class Execution:
    session_id: str | None
    cancelled: bool = False


class RouteClient:
    """One sender's fence; HTTP sessions are supplied by its execution loop.

    Control RPCs can arrive on other threads. The lock protects only local state,
    never a network call. Tasks retain response/release responsibility after their
    caller is cancelled. An ambiguous generation fails the pool, retaining N.
    """

    def __init__(
        self,
        router_addr: str,
        pool_id: str,
        run_id: str,
        admin_api_key: str,
        metadata_timeout: float = 30.0,
        control_timeout: float = 1800.0,
    ):
        self.router_addr = router_addr.rstrip("/")
        self.identity = {"pool_id": pool_id, "run_id": run_id}
        self.headers = {"Authorization": f"Bearer {admin_api_key}"}
        self.metadata_timeout = metadata_timeout
        self.control_timeout = control_timeout
        self._lock = Lock()
        self._paused = True
        self._control_seq = -1
        self._closed: set[str] = set()
        self._failed = False
        self._tasks: set[asyncio.Task] = set()

    def set_sender_state(self, paused: bool, control_seq: int) -> None:
        with self._lock:
            if control_seq < self._control_seq or (
                control_seq == self._control_seq and paused != self._paused
            ):
                raise RoutingError("STALE_CONTROL")
            if self._failed and not paused:
                raise RoutingError("FAILED")
            self._paused = paused
            self._control_seq = control_seq

    async def _post(
        self, session: aiohttp.ClientSession, path: str, body: dict
    ) -> dict:
        deadline = time.monotonic() + self.metadata_timeout
        while True:
            try:
                async with session.post(
                    self.router_addr + path,
                    json=body,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(
                        total=min(5.0, self.metadata_timeout)
                    ),
                ) as response:
                    data = await response.json()
                    if response.status == 200:
                        return data
                    detail = data.get("detail", {})
                    if isinstance(detail, dict) and "code" in detail:
                        raise RoutingError(detail["code"])
                    response.raise_for_status()
                    raise RoutingError("INVALID_RESPONSE")
            except (TimeoutError, aiohttp.ClientError, ValueError, RuntimeError):
                if time.monotonic() >= deadline:
                    raise RoutingError("METADATA_UNCERTAIN") from None
                await asyncio.sleep(0.1)

    async def release(
        self, session: aiohttp.ClientSession, lease: LeaseResponse, outcome: str
    ) -> None:
        await self._post(
            session,
            "/routing/release",
            {
                **self.identity,
                "lease_id": lease.lease_id,
                "outcome": outcome,
            },
        )

    def close_local_session(self, session_id: str) -> None:
        with self._lock:
            self._closed.add(session_id)

    async def close_session(
        self, session: aiohttp.ClientSession, session_id: str
    ) -> None:
        self.close_local_session(session_id)
        await self._post(
            session,
            "/routing/close_session",
            {
                **self.identity,
                "session_id": session_id,
            },
        )

    async def fail_pool(self, session: aiohttp.ClientSession) -> None:
        with self._lock:
            self._failed = True
            self._paused = True
        # The data loop/connector may be the reason we are failing. Publish on
        # an independent blocking transport in a thread so that a broken aiohttp
        # session cannot leave the Controller observing a healthy pool.
        # Read/compare/retry also handles races with Controller pause commands.
        deadline = time.monotonic() + self.metadata_timeout
        url = f"{self.router_addr}/routing/pools/{self.identity['pool_id']}/state"
        with requests.Session() as control:
            control.trust_env = False
            while time.monotonic() < deadline:
                try:
                    response = await asyncio.to_thread(
                        control.get,
                        url,
                        params={"run_id": self.identity["run_id"]},
                        headers=self.headers,
                        timeout=min(5.0, deadline - time.monotonic()),
                    )
                    response.raise_for_status()
                    state = response.json()
                    if state["state"] in ("FAILED", "CLOSED"):
                        return
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    response = await asyncio.to_thread(
                        control.post,
                        url,
                        json={
                            "run_id": self.identity["run_id"],
                            "control_seq": state["control_seq"] + 1,
                            "state": "FAILED",
                            "weight_version": state["weight_version"],
                        },
                        headers=self.headers,
                        timeout=min(5.0, remaining),
                    )
                    response.raise_for_status()
                    return
                except (requests.RequestException, ValueError):
                    await asyncio.sleep(0.1)
        raise RoutingError("POOL_FAILURE_UNCONFIRMED")

    async def execute(
        self,
        session: aiohttp.ClientSession,
        session_id: str | None,
        generate: Callable[[LeaseResponse], Awaitable[tuple[T, str]]],
    ) -> tuple[T, int]:
        execution = Execution(session_id)
        task = asyncio.create_task(self._execute(session, execution, generate))
        with self._lock:
            self._tasks.add(task)

        def finalized(done: asyncio.Task) -> None:
            with self._lock:
                self._tasks.discard(done)
            if not done.cancelled():
                done.exception()  # Retrieve even when the original caller left.

        task.add_done_callback(finalized)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            with self._lock:
                execution.cancelled = True
            raise

    async def _execute(
        self,
        session: aiohttp.ClientSession,
        execution: Execution,
        generate: Callable[[LeaseResponse], Awaitable[tuple[T, str]]],
    ) -> tuple[T, int]:
        unavailable: tuple[int, float] | None = None
        while True:
            with self._lock:
                closed = execution.session_id in self._closed
                cancelled = execution.cancelled
                paused, seq, failed = self._paused, self._control_seq, self._failed
            if closed or cancelled:
                raise RoutingError("SESSION_CLOSED" if closed else "CANCELLED")
            if failed:
                raise RoutingError("FAILED")
            if paused:
                # Intentional training pauses do not consume the health budget.
                unavailable = None
                await asyncio.sleep(0.05)
                continue
            # The ID is reused by _post's HTTP retries, never by a second generate.
            request = {
                **self.identity,
                "target": "backend",
                "request_id": uuid.uuid4().hex,
                "session_id": execution.session_id,
            }
            try:
                lease = LeaseResponse.model_validate(
                    await self._post(session, "/route", request)
                )
            except RoutingError as exc:
                if exc.code == "PAUSED":
                    unavailable = None
                    await asyncio.sleep(0.05)
                    continue
                if exc.code == "UNAVAILABLE":
                    with self._lock:
                        interrupted = (
                            self._paused
                            or self._failed
                            or execution.cancelled
                            or execution.session_id in self._closed
                            or seq != self._control_seq
                        )
                    if interrupted:
                        unavailable = None
                        continue
                    # No lease was allocated. Allow health to recover without
                    # migrating a bound session. Bound continuous unavailability
                    # per control epoch; pause/resume starts a fresh budget.
                    now = time.monotonic()
                    if unavailable is None or unavailable[0] != seq:
                        unavailable = (seq, now + self.control_timeout)
                    elif now >= unavailable[1]:
                        await self.fail_pool(session)
                        raise RoutingError("CONTROL_TIMEOUT") from exc
                    await asyncio.sleep(0.25)
                    continue
                if exc.code not in ("SESSION_CLOSED", "STALE_RUN", "CLOSED"):
                    await self.fail_pool(session)
                raise
            unavailable = None
            with self._lock:
                # Registration and pause/close are mutually exclusive. Once
                # registered, the task owns terminal response + release, even
                # during cancellation. Pool draining also waits for this lease.
                send = not (
                    self._paused
                    or self._failed
                    or execution.cancelled
                    or execution.session_id in self._closed
                    or seq != self._control_seq
                )
                generation = asyncio.create_task(generate(lease)) if send else None
            try:
                if generation is None:
                    await self.release(session, lease, "not_sent")
                    continue
                result, outcome = await generation
                await self.release(session, lease, outcome)
                return result, lease.weight_version
            except BaseException as exc:
                # A timeout/disconnect/5xx or an unparseable response cannot
                # establish that SGLang has stopped. Do not decrement its lease.
                await self.fail_pool(session)
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise RoutingError("EXECUTION_UNCERTAIN") from exc
