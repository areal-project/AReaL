# SPDX-License-Identifier: Apache-2.0

"""Wire-safe task contracts. Private benchmark records never enter TaskInput."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class EnvironmentSpec:
    kind: str
    image: str = ""
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("python", "repository"):
            raise ValueError(f"Unsupported task environment: {self.kind}")


@dataclass(frozen=True)
class TaskInput:
    task_id: str
    message: str
    environment: EnvironmentSpec


@dataclass(frozen=True)
class TaskResult:
    answer: str
    completed: bool
    environment: dict[str, Any]
    inference_calls: list[dict[str, Any]] | None = None
    model_calls: int = 0

    @classmethod
    def from_response(cls, response: dict[str, Any]) -> TaskResult:
        if response.get("status") != "completed":
            raise RuntimeError("Agent Service did not complete the request")
        metadata = response.get("metadata", {})
        environment = metadata.get("environment")
        if not isinstance(environment, dict) or not isinstance(
            metadata.get("completed"), bool
        ):
            raise RuntimeError("Expected task environment and completion metadata")
        if environment.get("infrastructure_error") or any(
            call.get("status") == "infrastructure_error"
            for call in environment.get("calls", [])
        ):
            raise RuntimeError("Task environment infrastructure failure")
        return cls(
            answer="".join(
                block.get("text", "")
                for item in response.get("output", [])
                if item.get("type") == "message"
                for block in item.get("content", [])
                if block.get("type") == "output_text"
            ),
            completed=metadata["completed"],
            environment=environment,
            inference_calls=metadata.get("inference_calls"),
            model_calls=metadata.get("api_calls", 0),
        )


@dataclass(frozen=True)
class GradeResult:
    reward: float
    reason: str
    extracted_answer: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.reward):
            raise ValueError("Benchmark reward must be finite")


@dataclass(frozen=True)
class TaskExecution:
    result: TaskResult
    metadata: dict[str, Any]
    score: GradeResult | None = None


class TaskExecutor(Protocol):
    """Optional execution backend; the episode client owns the RL lifecycle."""

    async def load(self, request_id: str) -> dict[str, Any]: ...

    async def save(self, request_id: str, **values: Any) -> None: ...

    async def execute(
        self,
        task: TaskInput,
        *,
        request_id: str,
        base_url: str,
        api_key: str,
        model: str,
        episode: dict[str, Any] | None,
        session_id: str | None = None,
    ) -> TaskExecution: ...


class Benchmark(Protocol):
    name: str
    request_prefix: str

    def load_tasks(self, path: Path) -> tuple[list[Any], dict[str, Any]]: ...

    def prepare(self, task: Any) -> TaskInput: ...

    def grade(self, task: Any, result: TaskResult, output: Path) -> Any: ...

    def result_metadata(self, task: Any) -> dict[str, Any]: ...


def get_benchmark(name: str) -> Benchmark:
    if name == "dapo_math":
        from examples.task_workflows.benchmarks.dapo_math import DapoBenchmark

        return DapoBenchmark()
    if name == "swe_rebench":
        from examples.task_workflows.benchmarks.swe_rebench import SWERebenchBenchmark

        return SWERebenchBenchmark()
    raise ValueError(f"Unknown benchmark: {name}")
