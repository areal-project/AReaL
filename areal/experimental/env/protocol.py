# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Protocol, Self


@dataclass(slots=True)
class StepResult:
    """Environment outcome for the action just taken."""

    observation: Any
    reward: float | None = None
    terminated: bool = False
    truncated: bool = False
    stop_reason: str | None = None
    error: str | None = None
    info: dict[str, Any] = field(default_factory=dict)


class EnvSession(Protocol):
    """One rollout's environment lease, managed with ``async with``."""

    async def reset(self, seed: int | None = None) -> Any: ...

    async def step(self, action: Any) -> StepResult: ...

    async def aclose(self) -> None: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


class EnvBackend(Protocol):
    """Allocate an independent session for each rollout."""

    async def acquire(self, task_data: Mapping[str, Any]) -> EnvSession: ...
