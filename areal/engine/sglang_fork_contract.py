# SPDX-License-Identifier: Apache-2.0

"""Runtime contract checks for SGLang forks used by AWEX colocation.

The AWEX plugin depends on scheduler internals that move between SGLang
versions.  Validate those internals before receiving weights so an incompatible
fork fails at startup instead of silently routing every shard to rank zero or
hanging in a collective.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from areal.utils.logging import getLogger

logger = getLogger("SGLangForkContract")


def resolve_scheduler_memory_method(scheduler: Any, name: str) -> Callable:
    """Find the native memory API on modern managers or legacy schedulers."""
    for owner in (getattr(scheduler, "weight_updater", None), scheduler):
        method = getattr(owner, name, None)
        if callable(method):
            return method
    raise RuntimeError(
        f"Neither scheduler.weight_updater nor scheduler provides {name}"
    )


def resolve_scheduler_parallel_value(
    scheduler: Any,
    name: str,
    *,
    default: int | None = None,
) -> int:
    """Resolve a scheduler parallel value across supported SGLang layouts.

    Newer Theta-derived schedulers keep rank state under ``scheduler.ps``;
    older releases expose it on ``scheduler`` or ``scheduler.tp_worker``.
    Conflicting values are rejected because choosing either one can silently
    corrupt the AWEX transfer plan.
    """

    sources = (
        ("scheduler.ps", getattr(scheduler, "ps", None)),
        ("scheduler", scheduler),
        ("scheduler.tp_worker", getattr(scheduler, "tp_worker", None)),
    )
    resolved: list[tuple[str, int]] = []
    for source_name, source in sources:
        value = getattr(source, name, None) if source is not None else None
        if value is None:
            continue
        try:
            resolved.append((source_name, int(value)))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid {name} from {source_name}: {value!r}") from exc

    if not resolved:
        if default is not None:
            return default
        checked = ", ".join(source for source, _ in sources)
        raise RuntimeError(
            f"Cannot resolve {name} from {checked}; refusing to use an implicit "
            "rank-zero default"
        )

    values = {value for _, value in resolved}
    if len(values) != 1:
        details = ", ".join(f"{source}={value}" for source, value in resolved)
        raise RuntimeError(f"Conflicting scheduler {name} values: {details}")
    return resolved[0][1]


def _mode() -> str:
    return os.environ.get("AREAL_SGLANG_CONTRACT", "strict").strip().lower()


def _report(violations: list[str], stage: str) -> None:
    if not violations:
        logger.info("[fork-contract] %s: all checks passed", stage)
        return
    lines = "\n".join(f"  - {violation}" for violation in violations)
    message = (
        f"[fork-contract] {stage}: {len(violations)} SGLang fork contract "
        f"violation(s) detected:\n{lines}"
    )
    if _mode() == "warn":
        logger.warning("%s\nAREAL_SGLANG_CONTRACT=warn: continuing anyway.", message)
        return
    raise RuntimeError(message)


def check_static_contract() -> None:
    """Validate class-level APIs before constructing the scheduler."""

    if _mode() == "off":
        return
    violations: list[str] = []

    try:
        from sglang.srt.managers.io_struct import PauseGenerationReqInput

        try:
            PauseGenerationReqInput(mode="retract")
        except Exception as exc:
            violations.append(
                f"PauseGenerationReqInput(mode='retract') rejected: {exc!r}"
            )
    except ImportError as exc:
        violations.append(f"PauseGenerationReqInput import failed: {exc!r}")

    try:
        from sglang.srt.managers.scheduler import Scheduler

        for method in (
            "process_input_requests",
            "pause_generation",
            "continue_generation",
            "flush_cache",
        ):
            if not callable(getattr(Scheduler, method, None)):
                violations.append(f"Scheduler.{method} missing")
        if not any(
            callable(getattr(Scheduler, name, None))
            for name in ("is_fully_idle", "_is_no_request")
        ):
            violations.append("Scheduler has neither is_fully_idle nor _is_no_request")
    except ImportError as exc:
        violations.append(f"Scheduler import failed: {exc!r}")

    try:
        from sglang.srt.managers.scheduler_components.weight_updater import (
            SchedulerWeightUpdaterManager,
        )

        if not callable(
            getattr(SchedulerWeightUpdaterManager, "release_memory_occupation", None)
        ):
            violations.append(
                "SchedulerWeightUpdaterManager.release_memory_occupation missing"
            )
    except ImportError:
        try:
            from sglang.srt.managers.scheduler import Scheduler

            if not callable(getattr(Scheduler, "release_memory_occupation", None)):
                violations.append(
                    "Neither the weight-updater manager nor the legacy Scheduler "
                    "memory-release API is available"
                )
        except ImportError as exc:
            violations.append(f"Scheduler memory-release API import failed: {exc!r}")

    _report(violations, "static")


def check_scheduler_contract(scheduler: Any) -> None:
    """Validate instance-level APIs when the AWEX plugin binds."""

    if _mode() == "off":
        return
    violations: list[str] = []

    if not hasattr(scheduler, "_engine_paused"):
        violations.append("scheduler._engine_paused missing")

    for name in ("tp_rank", "tp_size", "gpu_id"):
        try:
            resolve_scheduler_parallel_value(scheduler, name)
        except RuntimeError as exc:
            violations.append(str(exc))

    if not hasattr(scheduler, "tp_cpu_group"):
        violations.append("scheduler.tp_cpu_group missing")
    if not callable(getattr(scheduler, "process_input_requests", None)):
        violations.append("scheduler.process_input_requests missing")

    running_batch = getattr(scheduler, "running_batch", None)
    if running_batch is None or not callable(getattr(running_batch, "is_empty", None)):
        violations.append("scheduler.running_batch.is_empty missing")
    if not hasattr(scheduler, "waiting_queue"):
        violations.append("scheduler.waiting_queue missing")

    for name in ("release_memory_occupation", "resume_memory_occupation"):
        try:
            resolve_scheduler_memory_method(scheduler, name)
        except RuntimeError as exc:
            violations.append(str(exc))

    model_runner = getattr(getattr(scheduler, "tp_worker", None), "model_runner", None)
    if model_runner is None or getattr(model_runner, "model", None) is None:
        violations.append("scheduler.tp_worker.model_runner.model unreachable")

    _report(violations, "scheduler-bind")
