# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Vendored from tensorrt_llm.scaffolding.result

import asyncio
import queue
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class ScaffoldingOutput:
    text: str
    token_ids: list[int]
    data: Any = None


class ScaffoldingResult:
    """Result object for scaffolding requests.

    Uses a thread-safe ``queue.Queue`` for cross-thread communication so
    that producers (ScaffoldingLlm loop thread) and consumers
    (caller's event loop) can safely exchange data.
    """

    def __init__(self):
        super().__init__()
        self._queue: queue.Queue = queue.Queue()
        self.outputs = []
        # only support one output for now, so use an empty obj to init
        self.outputs.append(ScaffoldingOutput("", []))
        self._done = False
        self.task_collections = None

    def set_output(self, output: ScaffoldingOutput | Any):
        if isinstance(output, ScaffoldingOutput):
            self.set_output_streaming(output)
        # terminate
        self.set_output_streaming(None)

    def set_output_streaming(self, output: ScaffoldingOutput | Any):
        self._queue.put_nowait(output)

    def set_task_collections(self, task_collections: Mapping[str, Any]):
        self.task_collections = task_collections

    async def _aresult_step(self):
        """Asynchronously wait for the next item from the thread-safe queue."""
        loop = asyncio.get_running_loop()
        obj = await loop.run_in_executor(None, self._queue.get)
        if obj is None:
            self._done = True
        else:  # obj is ScaffoldingOutput
            self.outputs[0] = obj

    def result(self, timeout: float | None = None) -> "ScaffoldingResult":
        while not self._done:
            try:
                obj = self._queue.get(timeout=timeout)
            except queue.Empty:
                break
            if obj is None:
                self._done = True
            else:
                self.outputs[0] = obj
        return self

    async def aresult(self) -> "ScaffoldingResult":
        while not self._done:
            await self._aresult_step()
        return self

    def __await__(self):
        return self.aresult().__await__()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._done:
            raise StopAsyncIteration

        await self._aresult_step()
        return self
