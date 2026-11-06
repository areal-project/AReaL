# SPDX-License-Identifier: Apache-2.0

"""Bounded parallel recipe execution with transactional local recovery."""

from __future__ import annotations

import asyncio
import copy
import fcntl
import importlib
import inspect
import json
import math
import multiprocessing
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import aclosing
from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Any, TypeVar, cast

from areal.utils import logging
from areal.utils.fs import atomic_write_json

from .selection import select_ranked_candidates
from .snapshot import (
    DatasetSnapshot,
    digest,
    file_digest,
    iter_jsonl,
    json_bytes,
    load_snapshot,
    publish_snapshot,
    write_file,
    write_jsonl,
)
from .storage import ScoreMapping, WorkStore, sample_from_dict
from .types import (
    Attempt,
    CalibrationBatch,
    CalibrationContext,
    Calibrator,
    DataJobConfig,
    DataRecipe,
    DataSample,
    ExecutionConfig,
    GenerationShard,
    InsufficientDataError,
    RankedCandidate,
    RankedDataRecipe,
    RecipeContext,
    Selection,
    SelectionPolicy,
    ShardedDataRecipe,
)

logger = logging.getLogger("DataEngineeringRunner")
T = TypeVar("T")
R = TypeVar("R")


def _load_factory(spec: str, params: dict[str, Any]) -> tuple[Any, dict[str, str]]:
    module_name, separator, name = spec.partition(":")
    if not separator or not module_name or not name or ":" in name:
        raise ValueError(f"Expected module:callable, got {spec!r}")
    factory = getattr(importlib.import_module(module_name), name)
    source = inspect.getsourcefile(factory)
    if source is None:
        raise ValueError(f"Factory needs an auditable Python source file: {spec}")
    instance = factory(**copy.deepcopy(params))
    return instance, {
        "implementation": spec,
        "source_digest": file_digest(Path(source)),
    }


def _sources(config: DataJobConfig, recipe: Any, calibrator: Any) -> dict[str, str]:
    paths = list(config.inputs.values()) + list(config.exclude)
    specs = [config.recipe]
    if config.calibration is not None:
        specs.append(config.calibration.implementation)
    for spec in specs:
        source = inspect.getsourcefile(importlib.import_module(spec.partition(":")[0]))
        if source is not None:
            paths.append(source)
    for component in (recipe, calibrator):
        declared = getattr(component, "source_files", None)
        if declared is not None:
            paths.extend(str(path) for path in declared())
    return {str(Path(path).resolve()): file_digest(Path(path)) for path in paths}


def _validate_sample(sample: DataSample, *, serializable: bool = True) -> None:
    if not isinstance(sample, DataSample):
        raise TypeError("Recipe must yield DataSample instances")
    task = sample.task
    if not isinstance(task.task_id, str) or not task.task_id.strip():
        raise ValueError("Task ID must be nonempty")
    if not isinstance(task.input, dict) or not isinstance(task.metadata, dict):
        raise ValueError("Task input and metadata must be objects")
    if serializable:
        json_bytes(asdict(sample))


def _validated_attempts(
    samples: list[DataSample], attempts: list[Attempt], repeats: int
) -> dict[tuple[str, int], Attempt]:
    expected = {
        (sample.task.task_id, index) for sample in samples for index in range(repeats)
    }
    actual: dict[tuple[str, int], Attempt] = {}
    for attempt in attempts:
        if not isinstance(attempt, Attempt) or type(attempt.repeat) is not int:
            raise ValueError(
                "Calibration must return Attempt instances with integer repeat indices"
            )
        key = attempt.task_id, attempt.repeat
        if key not in expected or key in actual:
            raise ValueError(
                "Calibration returned unknown or duplicate task/repeat identities"
            )
        if attempt.failure_reason is not None and (
            not isinstance(attempt.failure_reason, str) or not attempt.failure_reason
        ):
            raise ValueError("failure_reason must be a nonempty string or None")
        json_bytes(asdict(attempt))
        actual[key] = attempt
    if set(actual) != expected:
        raise ValueError("Calibration omitted task/repeat identities")
    return actual


def _grade(
    samples: list[DataSample],
    attempts: list[Attempt],
    repeats: int,
    recipe: DataRecipe,
) -> dict[str, tuple[float | None, ...]]:
    actual = _validated_attempts(samples, attempts, repeats)
    scores = {}
    for sample in samples:
        values = []
        for index in range(repeats):
            attempt = actual[sample.task.task_id, index]
            if attempt.failure_reason is not None:
                values.append(None)
                continue
            value = recipe.score(copy.deepcopy(sample), copy.deepcopy(attempt.response))
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("Recipe scores must be finite numbers in [0, 1]")
            values.append(float(value))
        scores[sample.task.task_id] = tuple(values)
    return scores


def _chunks(values: Iterable[T], size: int) -> Iterator[list[T]]:
    iterator = iter(values)
    while chunk := list(islice(iterator, size)):
        yield chunk


async def _bounded(
    values: Iterable[T],
    call: Callable[[T], Awaitable[R]],
    workers: int,
    cancelled: threading.Event,
) -> AsyncIterator[tuple[T, R]]:
    """Bound both input prefetch and active tasks, yielding in input order."""
    pending: list[tuple[T, asyncio.Task[R]]] = []
    iterator = iter(values)
    try:
        while True:
            _check_cancelled(cancelled)
            while len(pending) < workers:
                try:
                    value = next(iterator)
                    _check_cancelled(cancelled)
                except StopIteration:
                    break
                pending.append((value, asyncio.create_task(call(value))))
            if not pending:
                return
            value, task = pending[0]
            result = await task
            _check_cancelled(cancelled)
            pending.pop(0)
            yield value, result
    finally:
        for _, task in pending:
            task.cancel()
        await asyncio.gather(*(task for _, task in pending), return_exceptions=True)


async def _retry(call: Callable[[], Awaitable[R]], config: ExecutionConfig) -> R:
    for attempt in range(config.max_retries + 1):
        try:
            return await call()
        except (OSError, TimeoutError):
            if attempt == config.max_retries:
                raise
            await asyncio.sleep(min(config.retry_delay * 2 ** min(attempt, 16), 60))
    raise AssertionError("Unreachable retry state")


def _recipe_chunk(
    spec: str, params: dict[str, Any], phase: str, samples: list[DataSample]
) -> list[Any]:
    recipe, _ = _load_factory(spec, params)
    values = []
    for sample in samples:
        sample = copy.deepcopy(sample)
        _validate_sample(sample, serializable=phase != "clean")
        if phase == "clean":
            value = recipe.clean(sample)
            if value is not None:
                _validate_sample(value)
        else:
            value = recipe.training_row(sample)
            if not isinstance(value, dict):
                raise TypeError("Training rows must be objects")
            json_bytes(value)
        values.append(value)
    return values


@dataclass(frozen=True)
class _GeneratedShard:
    path: str
    raw_count: int
    digest: str


def _generation_file(
    spec: str,
    params: dict[str, Any],
    context: RecipeContext,
    shard: GenerationShard,
    destination: str,
) -> _GeneratedShard:
    path = Path(destination)
    metadata = path.with_suffix(".json")
    if metadata.exists():
        value = json.loads(metadata.read_bytes())
        result = _GeneratedShard(**value["result"])
        if (
            value["shard"] != asdict(shard)
            or result.path != str(path)
            or type(result.raw_count) is not int
            or not 0 <= result.raw_count <= shard.stop - shard.start
            or file_digest(path) != result.digest
        ):
            raise ValueError("Generated shard checkpoint identity mismatch")
        return result
    recipe, _ = _load_factory(spec, params)
    count = 0

    def rows():
        nonlocal count
        for original in recipe.generate_shard(context, shard):
            count += 1
            if count > shard.stop - shard.start:
                raise ValueError("Generation shard exceeded its declared range")
            _validate_sample(original, serializable=False)
            cleaned = recipe.clean(original)
            if cleaned is not None:
                _validate_sample(cleaned)
            yield {"sample": asdict(cleaned) if cleaned is not None else None}

    write_jsonl(path, rows())
    result = _GeneratedShard(str(path), count, file_digest(path))
    atomic_write_json(
        metadata,
        {"shard": asdict(shard), "result": asdict(result)},
        default=None,
        allow_nan=False,
    )
    return result


def _grade_chunk(
    spec: str,
    params: dict[str, Any],
    samples: list[DataSample],
    attempts: tuple[Attempt, ...],
    repeats: int,
) -> dict[str, tuple[float | None, ...]]:
    recipe, _ = _load_factory(spec, params)
    return _grade(samples, list(attempts), repeats, recipe)


def _rank_chunk(
    spec: str,
    params: dict[str, Any],
    samples: list[DataSample],
    scores: dict[str, tuple[float | None, ...]] | None,
    context: RecipeContext,
) -> list[RankedCandidate]:
    recipe, _ = _load_factory(spec, params)
    ranked = []
    for sample in samples:
        task_id = sample.task.task_id
        value = recipe.rank(
            sample, scores[task_id] if scores is not None else None, context
        )
        if value is not None:
            if not isinstance(value, RankedCandidate) or value.task_id != task_id:
                raise ValueError("Recipe rank must return the same task ID or None")
            json_bytes(asdict(value))
            ranked.append(value)
    ranked.sort(key=lambda item: (item.priority, item.tie_breaker, item.task_id))
    return ranked


def _has_hooks(recipe: DataRecipe, names: tuple[str, ...]) -> bool:
    present = [callable(getattr(recipe, name, None)) for name in names]
    if any(present) and not all(present):
        raise TypeError("Recipe must implement all hooks: " + ", ".join(names))
    return all(present)


async def _execute(
    config: DataJobConfig,
    output: Path,
    store: WorkStore,
    recipe: DataRecipe,
    calibrator: Calibrator | None,
    pool: ThreadPoolExecutor | ProcessPoolExecutor,
    sources: dict[str, str],
    identities: dict[str, Any],
    write_state: Callable,
    cancelled: threading.Event,
) -> DatasetSnapshot:
    execution = config.execution
    loop = asyncio.get_running_loop()

    async def worker(phase: str, chunk: list[DataSample]) -> list[Any]:
        return await _retry(
            lambda: loop.run_in_executor(
                pool, _recipe_chunk, config.recipe, config.params, phase, chunk
            ),
            execution,
        )

    _check_cancelled(cancelled)
    excluded: set[str] = set()
    for path in config.exclude:
        for row in iter_jsonl(Path(path)):
            _check_cancelled(cancelled)
            if not isinstance(row.get("task_id"), str) or not row["task_id"]:
                raise ValueError("Exclusion rows need a nonempty task_id")
            excluded.add(row["task_id"])
    context = RecipeContext(
        config.seed,
        config.target_count,
        config.purpose,
        {key: Path(value) for key, value in config.inputs.items()},
    )
    selection_data = store.get("selection")
    last_shortage = "No eligible candidates"
    calibration_batches = []
    for batch in range(config.max_batches):
        _check_cancelled(cancelled)
        context = replace(context, batch_index=batch)
        if selection_data is not None:
            break
        generated_key = f"generated:{batch}"
        if store.get(generated_key) is None:
            if _has_hooks(recipe, ("generation_shards", "generate_shard")):
                sharded_recipe = cast(ShardedDataRecipe, recipe)
                previous = sum(
                    store.get(f"generated:{i}")["raw_count"] for i in range(batch)
                )
                expected_start = 0

                def shards():
                    nonlocal expected_start
                    for index, shard in enumerate(
                        sharded_recipe.generation_shards(context, execution.batch_size)
                    ):
                        _check_cancelled(cancelled)
                        if (
                            not isinstance(shard, GenerationShard)
                            or shard.start != expected_start
                        ):
                            raise ValueError(
                                "Generation shards must be contiguous and start at zero"
                            )
                        expected_start = shard.stop
                        if previous + shard.stop > config.max_candidates:
                            raise InsufficientDataError("Candidate budget exhausted")
                        yield index, shard

                async def generate(value):
                    index, shard = value
                    return await _retry(
                        lambda: loop.run_in_executor(
                            pool,
                            _generation_file,
                            config.recipe,
                            config.params,
                            context,
                            shard,
                            str(output / "work" / f"generation-{batch}-{index}.jsonl"),
                        ),
                        execution,
                    )

                raw_count = 0
                async with aclosing(
                    _bounded(shards(), generate, execution.max_workers, cancelled)
                ) as results:
                    async for (index, _), result in results:
                        raw_count += result.raw_count
                        seen = 0
                        for chunk_index, rows in enumerate(
                            _chunks(iter_jsonl(Path(result.path)), execution.batch_size)
                        ):
                            _check_cancelled(cancelled)
                            seen += len(rows)
                            key = f"clean:{batch}:{index}:{chunk_index}"
                            if store.get(key) is None:
                                cleaned = [
                                    sample_from_dict(row["sample"])
                                    if row["sample"] is not None
                                    else None
                                    for row in rows
                                ]
                                store.retain(key, batch, cleaned, excluded)
                                write_state(
                                    output / "progress.json", store.get("counts")
                                )
                        if seen != result.raw_count:
                            raise ValueError(
                                "Generated shard checkpoint row count mismatch"
                            )
                store.commit(generated_key, {"raw_count": raw_count})
            else:
                # A partial batch replays generation, but skips committed cleaning chunks.
                previous = sum(
                    store.get(f"generated:{i}")["raw_count"] for i in range(batch)
                )
                raw_count = 0
                chunk_count = 0

                def inputs():
                    nonlocal raw_count, chunk_count
                    for index, chunk in enumerate(
                        _chunks(recipe.generate(context), execution.batch_size)
                    ):
                        _check_cancelled(cancelled)
                        raw_count += len(chunk)
                        chunk_count += 1
                        if previous + raw_count > config.max_candidates:
                            raise InsufficientDataError("Candidate budget exhausted")
                        key = f"clean:{batch}:{index}"
                        if store.get(key) is None:
                            yield key, chunk

                async def clean(value):
                    return await worker("clean", value[1])

                async with aclosing(
                    _bounded(inputs(), clean, execution.max_workers, cancelled)
                ) as results:
                    async for (key, _), cleaned in results:
                        store.retain(key, batch, cleaned, excluded)
                        write_state(output / "progress.json", store.get("counts"))
                store.commit(
                    generated_key, {"raw_count": raw_count, "chunks": chunk_count}
                )
        _check_cancelled(cancelled)
        fresh = store.samples(batch)
        # Diagnostic exports are derived from durable state and safe to rebuild.
        write_jsonl(
            output / "work" / f"batch-{batch}-candidates.jsonl",
            (asdict(s) for s in fresh),
        )
        _check_cancelled(cancelled)
        write_jsonl(
            output / "work" / f"batch-{batch}-public.jsonl",
            (asdict(s.task) for s in fresh),
        )
        _check_cancelled(cancelled)
        if calibrator is not None and fresh:
            assert config.calibration is not None
            calibration_context = CalibrationContext(config.calibration.repeats, batch)

            def calibration_inputs():
                for index, chunk in enumerate(_chunks(fresh, execution.batch_size)):
                    _check_cancelled(cancelled)
                    key = f"calibration:{batch}:{index}"
                    if store.get(key) is None:
                        yield key, chunk

            async def probe(value):
                _check_cancelled(cancelled)
                chunk = value[1]
                response_key = value[0].replace("calibration:", "responses:", 1)
                cached = store.get(response_key)
                if cached is None:
                    result = await _retry(
                        lambda: calibrator(
                            [copy.deepcopy(s.task) for s in chunk], calibration_context
                        ),
                        execution,
                    )
                    _check_cancelled(cancelled)
                    if not isinstance(result, CalibrationBatch):
                        raise TypeError("Calibration must return CalibrationBatch")
                    if not isinstance(result.provenance, dict):
                        raise ValueError("Calibration provenance must be an object")
                    json_bytes(result.provenance)
                    _validated_attempts(
                        chunk, list(result.attempts), config.calibration.repeats
                    )
                    store.commit(response_key, asdict(result))
                else:
                    result = CalibrationBatch(
                        tuple(Attempt(**row) for row in cached["attempts"]),
                        cached["provenance"],
                    )
                scores = await _retry(
                    lambda: loop.run_in_executor(
                        pool,
                        _grade_chunk,
                        config.recipe,
                        config.params,
                        chunk,
                        result.attempts,
                        config.calibration.repeats,
                    ),
                    execution,
                )
                _check_cancelled(cancelled)
                # Canonical task/repeat ordering is independent of backend completion order.
                attempts = {(a.task_id, a.repeat): a for a in result.attempts}
                rows = [
                    asdict(attempts[s.task.task_id, r])
                    | {"score": scores[s.task.task_id][r]}
                    for s in chunk
                    for r in range(config.calibration.repeats)
                ]
                return rows, result.provenance

            async with aclosing(
                _bounded(calibration_inputs(), probe, execution.max_workers, cancelled)
            ) as results:
                async for (key, _), (rows, provenance) in results:
                    store.calibrate(key, rows, provenance)
                    write_state(output / "progress.json", store.get("counts"))
        counts = store.get("counts") or {"generated": 0, "clean_candidates": 0}
        counts["batches"] = batch + 1
        store.commit("counts", counts)
        write_state(output / "progress.json", counts)
        shortage = store.get(f"shortage:{batch}")
        if shortage is not None:
            last_shortage = shortage
            continue
        parallel_selection = _has_hooks(
            recipe, ("selection_policy", "rank", "finalize_selection")
        )
        if parallel_selection:
            ranked_recipe = cast(RankedDataRecipe, recipe)
            policy = ranked_recipe.selection_policy(context, calibrator is not None)
            if not isinstance(policy, SelectionPolicy):
                raise TypeError("Recipe must return SelectionPolicy")

            def rank_inputs():
                for index, chunk in enumerate(_chunks(fresh, execution.batch_size)):
                    key = f"ranking:{batch}:{index}"
                    if store.get(key) is None:
                        score_values = (
                            ScoreMapping(store) if calibrator is not None else None
                        )
                        scores = (
                            {
                                s.task.task_id: score_values[s.task.task_id]
                                for s in chunk
                            }
                            if score_values is not None
                            else None
                        )
                        yield key, chunk, scores

            async def rank(value):
                return await _retry(
                    lambda: loop.run_in_executor(
                        pool,
                        _rank_chunk,
                        config.recipe,
                        config.params,
                        value[1],
                        value[2],
                        context,
                    ),
                    execution,
                )

            async with aclosing(
                _bounded(rank_inputs(), rank, execution.max_workers, cancelled)
            ) as results:
                async for (key, _, _), ranked in results:
                    store.rank(key, ranked)
        try:
            if parallel_selection:
                selected = select_ranked_candidates(
                    policy, config.target_count, store.ranked
                )
                selection = ranked_recipe.finalize_selection(
                    selected, (store.get("counts") or {}).get("eligible", 0), context
                )
            else:
                selection = recipe.select(
                    store.samples(),
                    ScoreMapping(store) if calibrator else None,
                    context,
                )
        except InsufficientDataError as error:
            last_shortage = str(error)
            store.commit(f"shortage:{batch}", last_shortage)
            logger.info("Batch %d needs more data: %s", batch, error)
            continue
        _check_cancelled(cancelled)
        if not isinstance(selection, Selection):
            raise TypeError("Recipe must return Selection")
        if len(selection.task_ids) != config.target_count or len(
            set(selection.task_ids)
        ) != len(selection.task_ids):
            raise ValueError("Recipe must return target_count distinct candidate IDs")
        for task_id in selection.task_ids:
            store.sample(task_id)
        selection_data = asdict(selection)
        json_bytes(selection_data)
        store.commit("selection", selection_data)
        break
    if selection_data is None:
        raise InsufficientDataError(f"Data budget exhausted: {last_shortage}")
    ids = selection_data["task_ids"]

    def projections():
        for index, task_ids in enumerate(_chunks(ids, execution.batch_size)):
            key = f"training:{index}"
            if store.get(key) is None:
                yield key, [store.sample(task_id) for task_id in task_ids]

    async def project(value):
        return await worker("training", value[1])

    async with aclosing(
        _bounded(projections(), project, execution.max_workers, cancelled)
    ) as results:
        async for (key, _), rows in results:
            store.commit(key, rows)
    _check_cancelled(cancelled)
    if (
        _sources(config, recipe, calibrator) != sources
        or _executor_sources() != identities["executor_digests"]
    ):
        raise ValueError("Input sources changed while the data job was running")
    if calibrator is not None:
        # Read provenance for every completed chunk, including pre-restart work.
        for batch in range(config.max_batches):
            for index, _ in enumerate(
                _chunks(store.samples(batch), execution.batch_size)
            ):
                provenance = store.get(f"calibration:{batch}:{index}")
                if provenance is not None:
                    calibration_batches.append(
                        {
                            "batch_index": batch,
                            "chunk_index": index,
                            "provenance": provenance,
                        }
                    )

    def training_rows():
        for index, _ in enumerate(_chunks(ids, execution.batch_size)):
            yield from store.get(f"training:{index}")

    _check_cancelled(cancelled)
    return publish_snapshot(
        output / "dataset",
        public=(asdict(store.sample(task_id).task) for task_id in ids),
        references=(
            {"task_id": task_id, "reference": store.sample(task_id).reference}
            for task_id in ids
        ),
        training=training_rows(),
        probes=store.probe_rows(),
        report=(store.get("counts") or {})
        | {
            "selected": len(ids),
            "selection": selection_data["statistics"],
            "calibrated": calibrator is not None,
        },
        provenance={
            "config": asdict(config),
            **identities,
            "calibration_batches": calibration_batches,
            "input_digests": sources,
        },
    )


def _executor_sources() -> dict[str, str]:
    sources = {
        name: file_digest(Path(__file__).parent / name)
        for name in (
            "runner.py",
            "snapshot.py",
            "types.py",
            "storage.py",
            "selection.py",
        )
    }
    sources["areal/utils/fs.py"] = file_digest(Path(inspect.getfile(atomic_write_json)))
    return sources


async def _run_data_job(
    config: DataJobConfig,
    output: str | Path,
    *,
    resume: bool,
    cancelled: threading.Event,
) -> DatasetSnapshot:
    """Own the job loop, SQLite connection and synchronous work in one thread."""
    _check_cancelled(cancelled)
    write_state = partial(
        atomic_write_json,
        indent=None,
        default=None,
        allow_nan=False,
        sort_keys=True,
        ensure_ascii=False,
    )
    config = DataJobConfig.from_dict(asdict(config))
    config_bytes = json_bytes(asdict(config))
    output = Path(output).resolve()
    if resume:
        if not output.is_dir() or not (output / "work/checkpoints.sqlite3").is_file():
            raise ValueError("Resume requires an existing checkpointed job")
        if (output / "config.json").read_bytes() != config_bytes:
            raise ValueError("Cannot resume: data job config changed")
    else:
        output.mkdir(parents=True, mode=0o700, exist_ok=False)
        (output / "work").mkdir(mode=0o700)
        write_file(output / "config.json", config_bytes)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Data job is already running") from None
        store = WorkStore(output / "work/checkpoints.sqlite3", resume=resume)
        pool = None
        state = None
        try:
            recipe, recipe_identity = _load_factory(config.recipe, config.params)
            for method in ("generate", "clean", "score", "select", "training_row"):
                if not callable(getattr(recipe, method, None)):
                    raise TypeError(f"Recipe is missing {method}()")
            _has_hooks(recipe, ("generation_shards", "generate_shard"))
            _has_hooks(recipe, ("selection_policy", "rank", "finalize_selection"))
            calibrator = None
            calibration_identity = None
            if config.calibration is not None:
                calibrator, calibration_identity = _load_factory(
                    config.calibration.implementation, config.calibration.params
                )
                if not callable(calibrator):
                    raise TypeError("Calibration factory must return an async callable")
            _check_cancelled(cancelled)
            sources = _sources(config, recipe, calibrator)
            _check_cancelled(cancelled)
            identities = {
                "recipe": recipe_identity,
                "calibration": calibration_identity,
                "executor_digests": _executor_sources(),
            }
            binding = {
                "config_digest": digest(config_bytes),
                "sources": sources,
                "format": "areal.data.work.v2",
                **identities,
            }
            stored_binding = store.get("binding")
            if resume:
                if stored_binding != binding:
                    raise ValueError(
                        "Cannot resume: sources changed or checkpoint identity differs"
                    )
            else:
                store.commit("binding", binding)
            state = (
                json.loads((output / "job.json").read_bytes())
                if resume and (output / "job.json").exists()
                else {"started_at": time.time(), "config_digest": digest(config_bytes)}
            )
            state.update(status="running")
            state.pop("error", None)
            state.pop("finished_at", None)
            write_state(output / "job.json", state)
            if (output / "dataset").exists():
                snapshot = load_snapshot(output / "dataset")
                manifest = json.loads((snapshot.root / "manifest.json").read_bytes())
                expected = json.loads(
                    json_bytes(
                        {
                            "config": asdict(config),
                            "input_digests": sources,
                            **identities,
                        }
                    )
                )
                if any(
                    manifest["provenance"].get(key) != value
                    for key, value in expected.items()
                ):
                    raise ValueError(
                        "Published snapshot differs from checkpoint identity"
                    )
                if state.get("dataset_id") not in (None, snapshot.dataset_id):
                    raise ValueError(
                        "Published snapshot differs from completed job identity"
                    )
            else:
                if config.execution.backend == "process":
                    pool = ProcessPoolExecutor(
                        max_workers=config.execution.max_workers,
                        mp_context=multiprocessing.get_context("spawn"),
                    )
                else:
                    pool = ThreadPoolExecutor(max_workers=config.execution.max_workers)
                snapshot = await _execute(
                    config,
                    output,
                    store,
                    recipe,
                    calibrator,
                    pool,
                    sources,
                    identities,
                    write_state,
                    cancelled,
                )
            _check_cancelled(cancelled)
            state.update(
                status="succeeded",
                dataset_id=snapshot.dataset_id,
                dataset_path=str(snapshot.root),
            )
            logger.info("Published %s", snapshot.dataset_id)
            return snapshot
        except BaseException as error:
            if state is None and not resume:
                state = {
                    "started_at": time.time(),
                    "config_digest": digest(config_bytes),
                }
            if state is not None:
                state.update(
                    status="cancelled"
                    if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
                    else "failed",
                    error={"type": type(error).__name__, "message": str(error)},
                )
            raise
        finally:
            try:
                if pool is not None:
                    shutdown = asyncio.create_task(
                        asyncio.to_thread(pool.shutdown, wait=True, cancel_futures=True)
                    )
                    # A queued cross-thread cancel must not interrupt this cleanup.
                    while not shutdown.done():
                        try:
                            await asyncio.shield(shutdown)
                        except asyncio.CancelledError:
                            continue
                    shutdown.result()
            finally:
                store.close()
                if state is not None:
                    state["finished_at"] = time.time()
                    write_state(output / "job.json", state)


async def run_data_job(
    config: DataJobConfig, output: str | Path, *, resume: bool = False
) -> DatasetSnapshot:
    """Execute or resume without running blocking stages on the caller's loop.

    A dedicated thread owns the job's event loop, plugins and SQLite connection.
    Cancellation propagates to async calibration and waits for resource cleanup.
    Synchronous calls cannot be interrupted; cancellation is checked between stages
    and chunks. Only committed chunks are reused, requiring identical config,
    declared inputs, recipe and executor code. External calls are at-least-once.
    """
    cancelled = threading.Event()
    ready: Future[asyncio.Task[DatasetSnapshot]] = Future()
    config = copy.deepcopy(config)

    def run() -> DatasetSnapshot:
        async def execute() -> DatasetSnapshot:
            task = asyncio.current_task()
            assert task is not None
            ready.set_result(task)
            return await _run_data_job(
                config, output, resume=resume, cancelled=cancelled
            )

        return asyncio.run(execute())

    def cancel_job(future: Future[asyncio.Task[DatasetSnapshot]]) -> None:
        task = future.result()
        try:
            task.get_loop().call_soon_threadsafe(task.cancel)
        except RuntimeError:
            # The job may have completed and closed its loop during this race.
            if not task.get_loop().is_closed():
                raise

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AReaLDataJob")
    work = asyncio.get_running_loop().run_in_executor(pool, run)
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        cancelled.set()
        ready.add_done_callback(cancel_job)
        # shield() keeps repeated caller cancellation from abandoning job cleanup.
        while True:
            try:
                await asyncio.shield(work)
            except asyncio.CancelledError:
                if work.done():
                    break
            except BaseException:
                break
            else:
                break
        raise
    finally:
        pool.shutdown(wait=False)


def _check_cancelled(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise asyncio.CancelledError()
