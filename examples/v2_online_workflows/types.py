# SPDX-License-Identifier: Apache-2.0

"""Benchmark inputs and results for the native Hermes Agent Service."""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class EnvironmentSpec:
    kind: str


@dataclass(frozen=True)
class TaskInput:
    task_id: str
    message: str
    environment: EnvironmentSpec


@dataclass(frozen=True)
class TaskResult:
    answer: str
    completed: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GradeResult:
    reward: float
    reason: str
    extracted_answer: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class Harness(Protocol):
    async def run(
        self,
        task: TaskInput,
        *,
        request_id: str,
        inference_url: str,
        model: str,
        session_api_key: str,
    ) -> TaskResult: ...

    async def close(self, request_id: str) -> None: ...


def get_benchmark(name: str):
    if name == "gsm8k":
        from examples.v2_online_workflows.benchmarks.gsm8k import GSM8KBenchmark

        return GSM8KBenchmark()
    if name in ("dapo", "dapo_math"):
        from examples.v2_online_workflows.benchmarks.dapo import DapoBenchmark

        return DapoBenchmark()
    raise ValueError(f"Unknown benchmark: {name}")
