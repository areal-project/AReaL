# SPDX-License-Identifier: Apache-2.0

"""Model-call admission, independent of backend abort/pause semantics."""

import asyncio
from contextlib import asynccontextmanager


class GenerationGate:
    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self.closed = False
        self.active = 0

    async def acquire(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: not self.closed)
            self.active += 1

    async def release(self) -> None:
        async with self._condition:
            self.active -= 1
            self._condition.notify_all()

    async def drain(self, timeout: float) -> None:
        async with self._condition:
            self.closed = True
            await asyncio.wait_for(
                self._condition.wait_for(lambda: self.active == 0), timeout
            )

    async def open(self) -> None:
        async with self._condition:
            self.closed = False
            self._condition.notify_all()

    @asynccontextmanager
    async def rebind(self):
        # Hold admission across the entire bridge replacement, including awaits.
        async with self._condition:
            if self.active:
                raise RuntimeError("Drain model calls before rebinding the backend")
            yield
