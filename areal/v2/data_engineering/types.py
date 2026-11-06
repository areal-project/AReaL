# SPDX-License-Identifier: Apache-2.0

"""Execution contracts; domain payloads and policies belong to recipes."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

JsonObject = dict[str, Any]


def positive_int(value: int, name: str, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class PublicTask:
    """The only task representation passed to a calibration backend."""

    task_id: str
    input: JsonObject
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class DataSample:
    """A public task and its private reference; not a generated trajectory."""

    task: PublicTask
    reference: Any


@dataclass(frozen=True)
class Attempt:
    """One backend response; execution failures are distinct from wrong answers."""

    task_id: str
    repeat: int
    response: Any = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration factory, repetition count, and implementation-owned parameters."""

    implementation: str
    repeats: int = 2
    params: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("implementation",):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ValueError(f"calibration.{name} must be nonempty")
        positive_int(self.repeats, "calibration.repeats")
        if not isinstance(self.params, dict):
            raise ValueError("calibration.params must be an object")


@dataclass(frozen=True)
class ExecutionConfig:
    """Local worker limits and transient-error retries; no cluster dependency."""

    max_workers: int = 1
    batch_size: int = 256
    backend: Literal["thread", "process"] = "thread"
    max_retries: int = 0
    retry_delay: float = 1.0

    def __post_init__(self) -> None:
        positive_int(self.max_workers, "execution.max_workers")
        positive_int(self.batch_size, "execution.batch_size")
        positive_int(self.max_retries, "execution.max_retries", minimum=0)
        if self.backend not in {"thread", "process"}:
            raise ValueError("execution.backend must be thread or process")
        if (
            isinstance(self.retry_delay, bool)
            or not isinstance(self.retry_delay, (int, float))
            or not math.isfinite(self.retry_delay)
            or self.retry_delay < 0
        ):
            raise ValueError("execution.retry_delay must be finite and nonnegative")


@dataclass(frozen=True)
class DataJobConfig:
    """Bounded, foreground data job; recipe parameters define domain behavior.

    Execution is chunked, checkpointed and optionally parallel.
    Input and exclusion paths are files. Exclusions are public-task JSONL files
    with task_id fields, including a prior snapshot's public.jsonl.
    """

    recipe: str
    target_count: int
    purpose: Literal["harness", "model", "joint"] = "joint"
    seed: int = 0
    params: JsonObject = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    exclude: tuple[str, ...] = ()
    calibration: CalibrationConfig | None = None
    max_batches: int = 1
    max_candidates: int = 8192
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.recipe, str) or not self.recipe.strip():
            raise ValueError("recipe must be a nonempty module:callable")
        if self.purpose not in {"harness", "model", "joint"}:
            raise ValueError("purpose must be harness, model, or joint")
        positive_int(self.target_count, "target_count")
        positive_int(self.seed, "seed", minimum=0)
        positive_int(self.max_batches, "max_batches")
        positive_int(self.max_candidates, "max_candidates")
        if self.target_count > self.max_candidates:
            raise ValueError("target_count exceeds max_candidates")
        if not isinstance(self.params, dict) or not isinstance(self.inputs, dict):
            raise ValueError("params and inputs must be objects")
        if any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in self.inputs.items()
        ):
            raise ValueError("inputs must map names to file paths")
        if not isinstance(self.exclude, (list, tuple)) or any(
            not isinstance(p, str) for p in self.exclude
        ):
            raise ValueError("exclude must be a sequence of file paths")
        if not isinstance(self.execution, ExecutionConfig):
            raise ValueError("execution must be an ExecutionConfig")
        if self.calibration is not None and not isinstance(
            self.calibration, CalibrationConfig
        ):
            raise ValueError("calibration must be a CalibrationConfig")

    @classmethod
    def from_dict(
        cls, data: JsonObject, *, base_dir: Path | None = None
    ) -> DataJobConfig:
        values = dict(data)
        if "execution" in values:
            values["execution"] = ExecutionConfig(**values["execution"])
        if not isinstance(values.get("inputs", {}), dict):
            raise ValueError("inputs must be an object")
        if not isinstance(values.get("exclude", ()), (list, tuple)):
            raise ValueError("exclude must be a sequence of file paths")
        if values.get("calibration") is not None:
            values["calibration"] = CalibrationConfig(**values["calibration"])
        base = base_dir or Path.cwd()

        def resolve(path: str) -> str:
            value = Path(path).expanduser()
            return str(
                (base / value).resolve() if not value.is_absolute() else value.resolve()
            )

        values["inputs"] = {k: resolve(v) for k, v in values.get("inputs", {}).items()}
        values["exclude"] = tuple(resolve(p) for p in values.get("exclude", ()))
        return cls(**values)


@dataclass(frozen=True)
class RecipeContext:
    seed: int
    target_count: int
    purpose: str
    inputs: Mapping[str, Path]
    batch_index: int = 0


@dataclass(frozen=True)
class GenerationShard:
    """A deterministic range of at most stop-start raw candidates in one batch."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        positive_int(self.start, "generation.start", minimum=0)
        positive_int(self.stop, "generation.stop")
        if self.stop <= self.start:
            raise ValueError("Generation shard must be nonempty")


@dataclass(frozen=True)
class RankedCandidate:
    """Detached selection record; lower priority/tie_breaker/task_id sorts first."""

    task_id: str
    priority: float
    tie_breaker: str
    group: str = ""
    bucket: str = ""
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.priority, bool) or not isinstance(
            self.priority, (int, float)
        ):
            raise ValueError("Selection priority must be finite")
        try:
            priority = float(self.priority)
        except OverflowError:
            raise ValueError("Selection priority must be finite") from None
        if not math.isfinite(priority):
            raise ValueError("Selection priority must be finite")
        object.__setattr__(self, "priority", priority)
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("Ranked task ID must be nonempty")
        if any(
            not isinstance(value, str)
            for value in (self.tie_breaker, self.group, self.bucket)
        ):
            raise ValueError("Selection labels must be strings")
        if not isinstance(self.metadata, dict):
            raise ValueError("Ranked metadata must be an object")


@dataclass(frozen=True)
class SelectionPolicy:
    """Group minima in insertion order and bucket caps for a greedy global merge."""

    minimum_per_group: dict[str, int] = field(default_factory=dict)
    maximum_per_bucket: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for limits in (self.minimum_per_group, self.maximum_per_bucket):
            if not isinstance(limits, dict):
                raise ValueError("Selection limits must be mappings")
            for label, limit in limits.items():
                if not isinstance(label, str):
                    raise ValueError("Selection limit labels must be strings")
                positive_int(limit, "selection.limit", minimum=0)


@dataclass(frozen=True)
class CalibrationContext:
    repeats: int
    batch_index: int


@dataclass(frozen=True)
class CalibrationBatch:
    """Responses and implementation-specific provenance, without required agent bindings."""

    attempts: tuple[Attempt, ...]
    provenance: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class Selection:
    task_ids: tuple[str, ...]
    statistics: JsonObject = field(default_factory=dict)


class InsufficientDataError(ValueError):
    """Only this exception asks the executor to generate another batch."""


class DataRecipe(Protocol):
    """Domain policy executed by run_data_job; all payloads must serialize to JSON.

    Factories return an object implementing these methods. Imported helper code
    and additional files can be declared with an optional source_files() method.
    Generation must replay deterministically from context, seed and declared inputs.
    Methods must not depend on calls to other methods or mutable instance state.
    Parallel workers construct their own recipes; no instance state is checkpointed.
    Optional generation_shards(context, shard_size) and generate_shard(context, shard)
    enable parallel generation. Shards form a contiguous range starting at zero.
    Optional selection_policy(context, calibrated), rank(sample, scores, context),
    and finalize_selection(selected, eligible_count, context) enable parallel ranking
    with a deterministic global merge. Both hook sets must be complete.
    Selection receives a disk-backed Sequence, not necessarily a list.
    Selection and scoring must not mutate retained samples.
    """

    def generate(self, context: RecipeContext) -> Iterable[DataSample]: ...
    def clean(self, sample: DataSample) -> DataSample | None: ...
    def score(self, sample: DataSample, response: Any) -> float: ...
    def select(
        self,
        samples: Sequence[DataSample],
        scores: Mapping[str, tuple[float | None, ...]] | None,
        context: RecipeContext,
    ) -> Selection: ...
    def training_row(self, sample: DataSample) -> JsonObject: ...


class ShardedDataRecipe(DataRecipe, Protocol):
    """Optional generation capability used by TIR and other partitionable recipes."""

    def generation_shards(
        self, context: RecipeContext, shard_size: int
    ) -> Iterable[GenerationShard]: ...
    def generate_shard(
        self, context: RecipeContext, shard: GenerationShard
    ) -> Iterable[DataSample]: ...


class RankedDataRecipe(DataRecipe, Protocol):
    """Optional parallel eligibility/ranking with an explicit global merge policy."""

    def selection_policy(
        self, context: RecipeContext, calibrated: bool
    ) -> SelectionPolicy: ...
    def rank(
        self,
        sample: DataSample,
        scores: tuple[float | None, ...] | None,
        context: RecipeContext,
    ) -> RankedCandidate | None: ...
    def finalize_selection(
        self,
        selected: Sequence[RankedCandidate],
        eligible_count: int,
        context: RecipeContext,
    ) -> Selection: ...


class Calibrator(Protocol):
    """Process public tasks using an implementation-owned calibration method.

    References and scoring stay with the recipe executor. Return a response or
    explicit failure for every task/repeat pair, including failed executions.
    Calls are bounded to execution.batch_size tasks and may run concurrently.
    Implementations must be reentrant when execution.max_workers exceeds one.
    Transient retries and recovery have at-least-once execution semantics.
    Backend-specific dependencies and version checks belong to the implementation.
    """

    async def __call__(
        self, tasks: Sequence[PublicTask], context: CalibrationContext
    ) -> CalibrationBatch: ...
