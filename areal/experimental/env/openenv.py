# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

from .protocol import StepResult

if TYPE_CHECKING:
    from openenv.core.env_client import EnvClient


class OpenEnvSession:
    """Own the async lifecycle of one OpenEnv client."""

    def __init__(self, client: EnvClient):
        self._client = client
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> Self:
        try:
            await self._stack.enter_async_context(self._client)
        except asyncio.CancelledError:
            # The client can start its provider before the handshake completes.
            await self._client.close()
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._stack.__aexit__(exc_type, exc, traceback)

    async def aclose(self) -> None:
        await self._stack.aclose()

    async def reset(self, seed: int | None = None) -> Any:
        result = await self._client.reset(**({} if seed is None else {"seed": seed}))
        if result.done:
            raise ValueError("OpenEnv reset returned a completed episode")
        return result.observation

    async def step(self, action: Any) -> StepResult:
        result = await self._client.step(action)
        return StepResult(
            observation=result.observation,
            reward=result.reward,
            terminated=bool(result.done),
            info=dict(result.metadata or {}),
        )


class OpenEnvBackend:
    """Create an OpenEnv client from each task using an application-supplied factory.

    The factory constructs a fresh, unconnected client. OpenEnv remains optional
    for callers using other backends.
    """

    def __init__(self, client_factory: Callable[[Mapping[str, Any]], EnvClient]):
        self._client_factory = client_factory

    async def acquire(self, task_data: Mapping[str, Any]) -> OpenEnvSession:
        return OpenEnvSession(self._client_factory(task_data))
