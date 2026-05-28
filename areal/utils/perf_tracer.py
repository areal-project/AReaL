# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import atexit
import functools
import getpass
import json
import os
import tempfile
import threading
import time
import warnings
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, cast

import orjson
from torch import profiler

from areal.api.cli_args import PerfTracerConfig, SessionTracerConfig
from areal.utils import logging

logger = logging.getLogger("PerfTracer")

# Context variable for storing task_id in async context
_current_task_id: ContextVar[int | None] = ContextVar("task_id", default=None)

# Context variable for storing session_id in async context
_current_session_id: ContextVar[int | None] = ContextVar("session_id", default=None)

# Context variable for propagating global_step into nested scopes
_current_global_step: ContextVar[int | None] = ContextVar("global_step", default=None)

# Context variable tracking whether a profiler session is active
_profiler_active: ContextVar[bool] = ContextVar("profiler_active", default=False)

# Suppress Pydantic warnings for standard dataclasses
# Pydantic may inspect all dataclasses even when not using pydantic.dataclasses
# and emit false warnings about field() parameters or frozen dataclasses
warnings.filterwarnings(
    "ignore",
    message=".*repr.*should be.*",
    category=UserWarning,
    module="pydantic",
)
warnings.filterwarnings(
    "ignore",
    message=".*frozen.*attribute.*provided to.*Field.*function.*",
    category=UserWarning,
    module="pydantic",
)


_THREAD_LOCAL = threading.local()


class PerfTraceCategory(str, Enum):
    """Categories for classifying performance trace events.

    These categories help organize and filter trace events in visualization tools
    like Perfetto or Chrome Tracing. Categories are used in @trace_perf decorators,
    trace_scope context managers, and instant markers.

    Attributes
    ----------
    COMPUTE : str
        CPU/GPU computation (forward pass, backward pass, loss calculation).
    COMM : str
        Distributed communication (all-reduce, broadcast, point-to-point).
    IO : str
        Disk I/O operations (checkpoint save/load, data loading).
    SYNC : str
        Synchronization barriers and locks.
    SCHEDULER : str
        Task scheduling and queue management.
    INSTR : str
        Instrumentation and profiling overhead.
    MISC : str
        Miscellaneous events that don't fit other categories.
    """

    COMPUTE = "compute"
    COMM = "comm"
    IO = "io"
    SYNC = "sync"
    SCHEDULER = "scheduler"
    INSTR = "instr"
    MISC = "misc"


Category = PerfTraceCategory


CategoryLike = PerfTraceCategory | str | None


_CATEGORY_ALIASES: dict[str, PerfTraceCategory] = {
    "compute": PerfTraceCategory.COMPUTE,
    "communication": PerfTraceCategory.COMM,
    "comm": PerfTraceCategory.COMM,
    "io": PerfTraceCategory.IO,
    "synchronization": PerfTraceCategory.SYNC,
    "sync": PerfTraceCategory.SYNC,
    "scheduling": PerfTraceCategory.SCHEDULER,
    "scheduler": PerfTraceCategory.SCHEDULER,
    "instrumentation": PerfTraceCategory.INSTR,
    "instr": PerfTraceCategory.INSTR,
    "misc": PerfTraceCategory.MISC,
}


_PERF_TRACE_FILENAME = "traces.jsonl"
_SESSION_TRACE_FILENAME = "sessions.jsonl"


class _NullContext(AbstractContextManager, AbstractAsyncContextManager):
    """No-op context manager returned when tracing is disabled.

    Provides both sync and async context manager interfaces that do nothing,
    allowing trace_scope and atrace_scope to be called unconditionally without
    overhead when tracing is disabled or session_id is None.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, exc_tb):
        return False


def _rank_qualified_filename(filename: str, rank: int) -> str:
    root, ext = os.path.splitext(filename)
    return f"{root}-r{rank}{ext}"


def _maybe_duration(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return end - start


def _normalize_save_interval(config: PerfTracerConfig) -> int:
    return max(config.save_interval, 1)


def _normalize_flush_threshold(config: SessionTracerConfig) -> int:
    try:
        value = int(config.flush_threshold)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid flush_threshold=%r; defaulting to 1",
            getattr(config, "flush_threshold", None),
        )
        return 1
    return max(value, 1)


def _normalize_category(category: CategoryLike) -> str:
    """Normalize a category specification to a standard string value.

    Converts various category representations (enum, string, aliases) to their
    canonical string values. Returns "misc" for None or invalid inputs.

    Parameters
    ----------
    category : CategoryLike
        Category as PerfTraceCategory enum, string, or None.

    Returns
    -------
    str
        Normalized category string value.
    """

    if category is None:
        return PerfTraceCategory.MISC.value
    if isinstance(category, PerfTraceCategory):
        return category.value
    if isinstance(category, str) and category.strip():
        lowered = category.strip().lower()
        alias = _CATEGORY_ALIASES.get(lowered)
        if alias is not None:
            return alias.value
        return category
    return PerfTraceCategory.MISC.value


def _default_trace_path(
    config: PerfTracerConfig,
    *,
    rank: int,
    role: str | None = None,
    filename: str = _PERF_TRACE_FILENAME,
    subdir: str | None = None,
) -> str:
    """Generate the default output path for trace files.

    Constructs a standardized path under fileroot/logs/user/experiment/trial/
    with rank-qualified filename. Used for both performance traces and session traces.

    Parameters
    ----------
    config : PerfTracerConfig
        Configuration containing fileroot, experiment_name, and trial_name.
    rank : int
        Rank identifier to include in filename.
    role : str | None, default=None
        Optional role identifier (e.g., "actor", "learner"). When provided,
        creates a subdirectory under the tracer directory.
    filename : str, default="traces.jsonl"
        Base filename before rank qualification.
    subdir : str | None, default=None
        Optional subdirectory under trial_name (e.g., "perf_tracer", "session_tracer").

    Returns
    -------
    str
        Absolute path to the trace output file.
    """

    base_dir = os.path.join(
        os.path.expanduser(os.path.expandvars(config.fileroot)),
        "logs",
        getpass.getuser(),
        config.experiment_name,
        config.trial_name,
    )
    if subdir:
        base_dir = os.path.join(base_dir, subdir)
    if role:
        base_dir = os.path.join(base_dir, role)
    return os.path.join(base_dir, _rank_qualified_filename(filename, rank))


class SessionTraceEvent(str, Enum):
    """Enumeration of lifecycle events for session tracking.

    These events represent key points in a session's lifecycle and are used to
    trigger state updates, timestamp recording, and metric computation. Events
    are typically recorded via trace_session_event() or through context managers
    like atrace_session_phase().

    Event categories:
    - Phase boundaries: GENERATE_START, GENERATE_END, REWARD_START, REWARD_END,
                       TOOLCALL_START, TOOLCALL_END

    The events map to SessionRecord state transitions and are bound to specific
    actions through EventBinding configurations in SessionRecord.build_event_rules().

    Attributes
    ----------
    FINALIZED : str
        Session has been finalized.
    GENERATE_START : str
        Generation phase has started within the workflow.
    GENERATE_END : str
        Generation phase has completed.
    REWARD_START : str
        Reward computation phase has started.
    REWARD_END : str
        Reward computation phase has completed.
    TOOLCALL_START : str
        Tool calling phase has started.
    TOOLCALL_END : str
        Tool calling phase has completed.

    See Also
    --------
    trace_session_event : Function to record these events
    SessionRecord : Class that processes these events
    EventBinding : Configuration for event handling
    """

    # Lifecycle markers
    FINALIZED = "finalized"

    # Phase boundaries
    GENERATE_START = "generate_start"
    GENERATE_END = "generate_end"
    REWARD_START = "reward_start"
    REWARD_END = "reward_end"
    TOOLCALL_START = "toolcall_start"
    TOOLCALL_END = "toolcall_end"


@dataclass
class PhaseSpan:
    """Represents a single execution of a phase with start and end timestamps.

    A phase may execute multiple times within a session (e.g., multiple generate calls),
    and each execution is tracked as a separate PhaseSpan.

    Attributes
    ----------
    start_ts : float
        Timestamp when the phase started (from time.time(), wall-clock time).
    end_ts : float | None
        Timestamp when the phase ended, or None if still in progress.
    """

    start_ts: float
    end_ts: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
        }


# NOTE: frozen=True is valid despite Pydantic warnings
@dataclass(frozen=True)
class PhaseSpec:
    """Configuration specification for a tracked phase in session lifecycle.

    Defines how a phase should be tracked, including which events mark its boundaries,
    whether multiple executions are allowed, and optional callbacks for state updates.

    Attributes
    ----------
    name : str
        Phase name (e.g., "generate", "reward", "execution").
    start_event : SessionTraceEvent
        Event that marks the start of this phase.
    end_event : SessionTraceEvent
        Event that marks the end of this phase.
    allow_multiple : bool, default=False
        Whether this phase can execute multiple times within a session.
    ready_on_complete : bool, default=False
        Whether session becomes ready for flushing when this phase completes.
    on_end : Callable | None, default=None
        Optional callback invoked when the phase ends for custom state updates.
    """

    name: str
    start_event: SessionTraceEvent
    end_event: SessionTraceEvent
    allow_multiple: bool = False
    ready_on_complete: bool = False
    on_end: Callable[[SessionRecord, dict[str, Any]], None] | None = None


# NOTE: frozen=True is valid despite Pydantic warnings
@dataclass(frozen=True)
class EventBinding:
    """Binding configuration for how a SessionTraceEvent updates SessionRecord state.

    Defines the actions to take when a specific event is recorded, such as updating
    timestamps, starting/ending phases, invoking callbacks, or marking readiness.

    Attributes
    ----------
    timestamp_attr : str | None, default=None
        SessionRecord attribute to update with event timestamp (e.g., "finalized_ts").
    phase : str | None, default=None
        Phase name to update if this event represents a phase boundary.
    role : str | None, default=None
        Phase role: "start" or "end" for phase boundary events.
    allow_multiple : bool, default=False
        Whether multiple executions of this phase are allowed.
    payload_handler : Callable | None, default=None
        Optional callback to process event payload and update record state.
    mark_ready : bool, default=False
        Whether this event marks the session as ready for flushing.
    """

    timestamp_attr: str | None = None
    phase: str | None = None
    role: str | None = None
    allow_multiple: bool = False
    payload_handler: Callable[[SessionRecord, dict[str, Any]], None] | None = None
    mark_ready: bool = False


# NOTE: frozen=True is valid despite Pydantic warnings
@dataclass(frozen=True)
class FieldSpec:
    """Specification for serializing a field from SessionRecord to JSON output.

    Defines how to extract or compute a value from a SessionRecord for inclusion
    in the serialized output. Can either read a direct attribute or invoke a
    computation function for derived metrics.

    Attributes
    ----------
    attr : str | None, default=None
        SessionRecord attribute name to read directly (e.g., "session_id", "status").
    key : str | None, default=None
        JSON key name in output. Defaults to attr if not specified.
    compute : Callable | None, default=None
        Function to compute derived value from SessionRecord (e.g., duration calculation).
    omit_if_none : bool, default=True
        Whether to omit this field from JSON if its value is None.
    """

    attr: str | None = None
    key: str | None = None
    compute: Callable[[SessionRecord], Any] | None = None
    omit_if_none: bool = True

    def resolve(self, record: SessionRecord) -> Any:
        if self.compute is not None:
            return self.compute(record)
        if self.attr is None:
            raise ValueError("RecordField requires either attr or compute")
        return getattr(record, self.attr)

    def key_name(self) -> str:
        if self.key is not None:
            return self.key
        if self.attr is None:
            raise ValueError("RecordField without attr must define key")
        return self.attr


@dataclass
class SessionRecord:
    """Record of a single session's lifecycle with timestamps and computed metrics.

    This class represents the complete execution trace of a rollout session, including
    all lifecycle events, phase executions, and derived performance metrics. It serves
    as both an in-memory state tracker during execution and the serialization format
    for persisted session traces.

    The record tracks three main aspects:
    1. Lifecycle timestamps: submission (submit_ts) and finalization (finalized_ts)
    2. Phase executions: generate and reward phases with start/end times, supporting
       multiple executions per phase
    3. Derived metrics: computed from timestamps (total_s, generate_s, reward_s)

    State management:
    - Starts with status="pending" on registration
    - Updates to "accepted", "rejected", "failed", or "dropped" based on execution
    - Becomes ready for flushing when reaching a terminal state or explicit completion
    - Tracks multiple executions of phases (e.g., multiple generate calls in one session)

    The class uses ClassVar configurations (PHASE_CONFIGS, FIELD_SPECS) to define:
    - Which phases to track and their event bindings
    - Which fields to serialize and how to compute derived values
    - Lifecycle event handling and state transitions

    Parameters
    ----------
    task_id : int
        Identifier for the task associated with this session.
    session_id : int
        Unique identifier for this session.
    rank : int
        Rank identifier for the process that created this session.
    role : str | None, default=None
        Optional role identifier (e.g., "actor", "learner") for the process.
    submit_ts : float
        Timestamp when the session was submitted (from time.time(), wall-clock time).
    status : str, default="pending"
        Current status: "pending", "accepted", "rejected", "failed", or "dropped".
    reason : str | None, default=None
        Reason for rejection if status is "rejected" or "dropped".
    finalized_ts : float | None, default=None
        Timestamp when the session result was finalized by the training loop.

    See Also
    --------
    SessionTracer : Manager that creates and tracks SessionRecord instances
    PhaseSpan : Represents a single phase execution with start/end timestamps
    PhaseSpec : Configuration for tracked phases and their events
    """

    task_id: int
    session_id: int
    rank: int
    role: str | None
    submit_ts: float
    status: str = "pending"
    reason: str | None = None
    finalized_ts: float | None = None
    phases: dict[str, list[PhaseSpan]] = field(init=False)
    counters: dict[str, int] = field(init=False)
    # NOTE: repr=False is valid for dataclasses.field() despite Pydantic warnings
    _active_phases: dict[str, PhaseSpan | None] = field(init=False, repr=False)

    PHASE_CONFIGS: ClassVar[tuple[PhaseSpec, ...]] = ()
    COUNTERS: ClassVar[tuple[str, ...]] = ()
    FIELD_SPECS: ClassVar[tuple[FieldSpec, ...]] = ()

    def __post_init__(self) -> None:
        self.phases = {cfg.name: [] for cfg in self.PHASE_CONFIGS}
        self._active_phases = {cfg.name: None for cfg in self.PHASE_CONFIGS}
        self.counters = {name: 0 for name in self.COUNTERS}

    @classmethod
    def default_phase_configs(cls) -> tuple[PhaseSpec, ...]:
        return (
            PhaseSpec(
                name="generate",
                start_event=SessionTraceEvent.GENERATE_START,
                end_event=SessionTraceEvent.GENERATE_END,
                allow_multiple=True,
            ),
            PhaseSpec(
                name="reward",
                start_event=SessionTraceEvent.REWARD_START,
                end_event=SessionTraceEvent.REWARD_END,
                allow_multiple=True,
            ),
            PhaseSpec(
                name="toolcall",
                start_event=SessionTraceEvent.TOOLCALL_START,
                end_event=SessionTraceEvent.TOOLCALL_END,
                allow_multiple=True,
            ),
        )

    def is_ready_to_flush(self) -> bool:
        if self.status in {"rejected", "failed", "dropped"}:
            return True
        if self.status == "accepted" and self.finalized_ts is not None:
            return True
        return False

    def increment_counter(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def apply_phase_event(
        self,
        phase: str,
        role: str,
        timestamp: float,
        *,
        allow_multiple: bool,
    ) -> None:
        entries = self.phases.setdefault(phase, [])
        current = self._active_phases.get(phase)
        if role == "start":
            if current is not None and current.end_ts is None and not allow_multiple:
                current.end_ts = timestamp
            entry = PhaseSpan(start_ts=timestamp)
            entries.append(entry)
            self._active_phases[phase] = entry
        elif role == "end":
            if current is None or current.end_ts is not None:
                entry = PhaseSpan(start_ts=timestamp)
                entries.append(entry)
            else:
                entry = current
            entry.end_ts = timestamp
            self._active_phases[phase] = None

    @staticmethod
    def _on_finalized(record: SessionRecord, payload: dict[str, Any]) -> None:
        """Handle terminal event for a session."""
        status = payload.get("status")
        if status is not None:
            record.status = status
        if "reason" in payload:
            record.reason = payload.get("reason")

    @classmethod
    def build_event_rules(cls) -> dict[SessionTraceEvent, EventBinding]:
        rules: dict[SessionTraceEvent, EventBinding] = {
            # FINALIZED is the canonical terminal event for a session; it
            # records the wait/return timestamp and invokes the unified
            # termination handler to set final status.
            SessionTraceEvent.FINALIZED: EventBinding(
                timestamp_attr="finalized_ts",
                payload_handler=cls._on_finalized,
                mark_ready=True,
            ),
        }
        for cfg in cls.PHASE_CONFIGS:
            rules[cfg.start_event] = EventBinding(
                phase=cfg.name,
                role="start",
                allow_multiple=cfg.allow_multiple,
            )
            if cfg.end_event is not None:
                rules[cfg.end_event] = EventBinding(
                    phase=cfg.name,
                    role="end",
                    payload_handler=cfg.on_end,
                    mark_ready=cfg.ready_on_complete,
                )
        return rules

    def _phase_total_duration(self, phase: str) -> float | None:
        durations = [
            entry.end_ts - entry.start_ts
            for entry in self.phases.get(phase, [])
            if entry.end_ts is not None
        ]
        if not durations:
            return None
        return sum(durations)

    @staticmethod
    def _compute_total_time(record: SessionRecord) -> float | None:
        return _maybe_duration(record.submit_ts, record.finalized_ts)

    @staticmethod
    def _compute_generate_time(record: SessionRecord) -> float | None:
        return record._phase_total_duration("generate")

    @staticmethod
    def _compute_reward_time(record: SessionRecord) -> float | None:
        return record._phase_total_duration("reward")

    @staticmethod
    def _compute_toolcall_time(record: SessionRecord) -> float | None:
        return record._phase_total_duration("toolcall")

    @classmethod
    def default_field_specs(cls) -> tuple[FieldSpec, ...]:
        return (
            FieldSpec("task_id"),
            FieldSpec("session_id"),
            FieldSpec("rank"),
            FieldSpec("role", omit_if_none=True),
            FieldSpec("status"),
            FieldSpec("reason", omit_if_none=True),
            FieldSpec("submit_ts"),
            FieldSpec("finalized_ts", omit_if_none=True),
            FieldSpec(
                compute=cls._compute_total_time,
                key="total_s",
                omit_if_none=True,
            ),
            FieldSpec(
                compute=cls._compute_generate_time,
                key="generate_s",
                omit_if_none=True,
            ),
            FieldSpec(
                compute=cls._compute_reward_time,
                key="reward_s",
                omit_if_none=True,
            ),
            FieldSpec(
                compute=cls._compute_toolcall_time,
                key="toolcall_s",
                omit_if_none=True,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for field_spec in self.FIELD_SPECS:
            value = field_spec.resolve(self)
            if field_spec.omit_if_none and value is None:
                continue
            data[field_spec.key_name()] = value
        if any(self.phases.values()):
            data["phases"] = {
                name: [entry.to_dict() for entry in entries]
                for name, entries in self.phases.items()
                if entries
            }
        if any(self.counters.values()):
            data["counters"] = {k: v for k, v in self.counters.items() if v}
        return data


SessionRecord.PHASE_CONFIGS = SessionRecord.default_phase_configs()
SessionRecord.FIELD_SPECS = SessionRecord.default_field_specs()
_SESSION_EVENT_RULES = SessionRecord.build_event_rules()

_SESSION_TRACE_METHOD_TO_EVENT: dict[str, SessionTraceEvent] = {
    "mark_finalized": SessionTraceEvent.FINALIZED,
    "mark_generate_start": SessionTraceEvent.GENERATE_START,
    "mark_generate_end": SessionTraceEvent.GENERATE_END,
    "mark_reward_start": SessionTraceEvent.REWARD_START,
    "mark_reward_end": SessionTraceEvent.REWARD_END,
    "mark_toolcall_start": SessionTraceEvent.TOOLCALL_START,
    "mark_toolcall_end": SessionTraceEvent.TOOLCALL_END,
}


def trace_session_event(
    method: str,
    session_id: int | None = None,
    task_id: int | None = None,
    **payload: Any,
) -> None:
    """Record a session lifecycle event for tracking.

    This is the primary function for recording session events. It routes the event
    to the global SessionTracer (if configured) and applies appropriate state updates
    to the corresponding SessionRecord.

    The function handles two types of operations:
    1. Counter increments: method="increment_counter" with name and value in payload
    2. Lifecycle events: method mapped to SessionTraceEvent enum values

    Parameters
    ----------
    method : str
        Event method name. Standard methods: "mark_finalized", "mark_generate_start",
        "mark_generate_end", "mark_reward_start", "mark_reward_end", "increment_counter".
    session_id : int | None, Optional
        Optional session ID to record the event for.
    task_id : int | None, optional
        Optional task identifier associated with the session.
    **payload : Any
        Additional event data. Common keys include "status", "reason"
        for execution_end events, and "name", "value" for counters.

    See Also
    --------
    SessionTraceEvent : Enum defining available lifecycle events
    trace_session_phase : Context manager for automatic phase start/end pairing
    atrace_session_phase : Async version of trace_session_phase
    """

    tracer = get_session_tracer()
    if tracer is None:
        return
    if session_id is not None:
        session_ids = [session_id]
    elif task_id is not None:
        session_ids = tracer.get_session_ids(task_id)
    else:
        return
    if method == "increment_counter":
        name = payload.get("name")
        if not name:
            return
        for session_id in session_ids:
            tracer.increment_counter(session_id, name, payload.get("value", 1))
        return
    event = _SESSION_TRACE_METHOD_TO_EVENT.get(method)
    if event is None:
        return
    for session_id in session_ids:
        tracer.record_event(session_id, event, **payload)


class _SyncSessionPhaseScope(AbstractContextManager[Any]):
    """Sync context manager for tracing session phases.

    Automatically calls mark_{phase}_start on enter and mark_{phase}_end on exit,
    ensuring proper pairing even if exceptions occur.
    """

    def __init__(
        self,
        session_id: int | None,
        phase: str,
        *,
        start_payload: dict[str, Any] | None = None,
        end_payload: dict[str, Any] | None = None,
    ) -> None:
        self._session_id = session_id
        self._phase = phase
        self._start_method = f"mark_{phase}_start"
        self._end_method = f"mark_{phase}_end"
        self._start_payload = start_payload or {}
        self._end_payload = end_payload or {}

    def __enter__(self) -> _SyncSessionPhaseScope:
        trace_session_event(
            self._start_method, session_id=self._session_id, **self._start_payload
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Always call end event, even if exception occurred
        trace_session_event(
            self._end_method, session_id=self._session_id, **self._end_payload
        )
        return False  # Don't suppress exceptions


class _AsyncSessionPhaseScope(AbstractAsyncContextManager[Any]):
    """Async context manager for tracing session phases.

    Automatically calls mark_{phase}_start on enter and mark_{phase}_end on exit,
    ensuring proper pairing even if exceptions occur.
    """

    def __init__(
        self,
        session_id: int | None,
        phase: str,
        *,
        start_payload: dict[str, Any] | None = None,
        end_payload: dict[str, Any] | None = None,
    ) -> None:
        self._session_id = session_id
        self._phase = phase
        self._start_method = f"mark_{phase}_start"
        self._end_method = f"mark_{phase}_end"
        self._start_payload = start_payload or {}
        self._end_payload = end_payload or {}

    async def __aenter__(self) -> _AsyncSessionPhaseScope:
        trace_session_event(
            self._start_method, session_id=self._session_id, **self._start_payload
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Always call end event, even if exception occurred
        trace_session_event(
            self._end_method, session_id=self._session_id, **self._end_payload
        )
        return False  # Don't suppress exceptions


def trace_session_phase(
    phase: str,
    *,
    start_payload: dict[str, Any] | None = None,
    end_payload: dict[str, Any] | None = None,
) -> AbstractContextManager[Any]:
    """Sync context manager for tracing session phases.

    Automatically pairs mark_{phase}_start and mark_{phase}_end events,
    ensuring they are always called together even if exceptions occur.

    Parameters
    ----------
    phase : str
        Phase name (e.g., "generate", "reward", "execution").
        Will call ``mark_{phase}_start`` and ``mark_{phase}_end``.
    start_payload : dict[str, Any] | None
        Optional payload to pass to the start event.
    end_payload : dict[str, Any] | None
        Optional payload to pass to the end event.

    Returns
    -------
    AbstractContextManager
        A sync context manager for the phase tracing.

    Examples
    --------
    >>> with trace_session_phase("generate"):
    ...     result = engine.generate(req)

    >>> with trace_session_phase("reward"):
    ...     reward = reward_fn(prompt, completion)

    >>> with trace_session_phase(
    ...     "execution",
    ...     end_payload={"status": "accepted"}
    ... ):
    ...     result = process_request()
    """
    session_id = get_session_id()

    if session_id is None:
        return _NullContext()
    return _SyncSessionPhaseScope(
        session_id,
        phase,
        start_payload=start_payload,
        end_payload=end_payload,
    )


def atrace_session_phase(
    phase: str,
    *,
    start_payload: dict[str, Any] | None = None,
    end_payload: dict[str, Any] | None = None,
) -> AbstractAsyncContextManager[Any]:
    """Async context manager for tracing session phases.

    Automatically pairs mark_{phase}_start and mark_{phase}_end events,
    ensuring they are always called together even if exceptions occur.

    Parameters
    ----------
    phase : str
        Phase name (e.g., "generate", "reward").
        Will call ``mark_{phase}_start`` and ``mark_{phase}_end``.
    start_payload : dict[str, Any] | None
        Optional payload to pass to the start event.
    end_payload : dict[str, Any] | None
        Optional payload to pass to the end event.

    Returns
    -------
    AbstractAsyncContextManager
        An async context manager for the phase tracing.

    Examples
    --------
    >>> async with atrace_session_phase("generate"):
    ...     result = await engine.agenerate(req)

    >>> async with atrace_session_phase("reward"):
    ...     reward = await reward_fn(prompt, completion)

    >>> async with atrace_session_phase(
    ...     "execution",
    ...     end_payload={"status": "accepted"}
    ... ):
    ...     result = await process_request()
    """
    session_id = get_session_id()

    if session_id is None:
        return _NullContext()
    return _AsyncSessionPhaseScope(
        session_id,
        phase,
        start_payload=start_payload,
        end_payload=end_payload,
    )


class SessionTracer:
    """Tracer for tracking per-session lifecycle events and computing derived metrics.

    This class manages the complete lifecycle of individual rollout sessions, from
    submission through finalization. It records timestamped events, tracks phase
    executions (generate, reward, etc.), and computes derived metrics like total
    latency and per-phase durations.

    The tracer automatically flushes completed sessions to disk in JSONL format
    when the flush threshold is reached or when explicitly requested. Each session
    record includes lifecycle timestamps, status information, phase breakdowns,
    and computed performance metrics.

    Key features:
    - Automatic task ID and session ID assignment on registration
    - Event-driven state updates with timestamp tracking
    - Phase execution tracking with support for multiple executions per phase
    - Derived metric computation (total_s, generate_s, reward_s, etc.)
    - Configurable flush threshold for batched I/O
    - Thread-safe operation with internal locking
    - Task-session hierarchy tracking (one task can have multiple sessions)

    Parameters
    ----------
    config : SessionTracerConfig
        Configuration for session tracing including flush threshold settings.
    output_path : str
        Absolute path to the output JSONL file where session records will be written.
    rank : int
        Rank identifier for this process in distributed training.
    role : str | None, default=None
        Optional role identifier (e.g., "actor", "learner") for the process.

    See Also
    --------
    SessionRecord : Data structure representing a single session's lifecycle
    PerfTracer : Main tracer that optionally includes session tracking
    trace_session_event : Function to record session lifecycle events
    """

    def __init__(
        self,
        config: SessionTracerConfig,
        *,
        output_path: str,
        rank: int,
        role: str | None = None,
    ) -> None:
        self._config = config
        self._rank = rank
        self._role = role
        self._lock = threading.Lock()
        self._next_session_id = 0
        # task id sequence and mapping from task_id -> set(session_id)
        self._task_to_sessions: dict[int, set[int]] = {}
        self._records: dict[int, SessionRecord] = {}
        self._ready: set[int] = set()
        self._flush_threshold = _normalize_flush_threshold(config)
        self._output_path = output_path
        self._event_rules = _SESSION_EVENT_RULES

    def register_task(self, task_id: int) -> None:
        """Register a new logical task (dataset-level) given a task_id.

        Tasks group multiple sessions (one per generated sample). Use
        :meth:`register_session` to create sessions that belong to a task.
        """
        with self._lock:
            self._task_to_sessions.setdefault(task_id, set())

    def register_session(self, task_id: int) -> int:
        """Register a new session and optionally associate it with a task.

        Returns the newly-created session_id.
        """
        now = time.time()
        with self._lock:
            session_id = self._next_session_id
            self._next_session_id += 1
            self._records[session_id] = SessionRecord(
                task_id=task_id,
                session_id=session_id,
                rank=self._rank,
                role=self._role,
                submit_ts=now,
            )
            self._task_to_sessions.setdefault(task_id, set()).add(session_id)
        return session_id

    def get_session_ids(self, task_id: int) -> list[int]:
        """Get all session IDs associated with the given task_id."""
        with self._lock:
            return list(self._task_to_sessions.get(task_id, []))

    def _apply_event(
        self,
        session_id: int,
        event: SessionTraceEvent,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        rule = self._event_rules.get(event)
        if rule is None:
            return False
        data = dict(payload or {})
        with self._lock:
            record = self._records.get(session_id)
            if record is None:
                return False
            timestamp = time.time()
            if rule.timestamp_attr is not None:
                setattr(record, rule.timestamp_attr, timestamp)
            if rule.phase is not None and rule.role is not None:
                record.apply_phase_event(
                    rule.phase,
                    rule.role,
                    timestamp,
                    allow_multiple=rule.allow_multiple,
                )
            if rule.payload_handler is not None:
                rule.payload_handler(record, data)
            ready = rule.mark_ready or record.is_ready_to_flush()
            if ready:
                self._ready.add(session_id)
                if len(self._ready) >= self._flush_threshold:
                    return True
            return False

    def record_event(
        self,
        session_id: int,
        event: SessionTraceEvent,
        **payload: Any,
    ) -> None:
        should_flush = self._apply_event(session_id, event, payload)
        if should_flush:
            self.flush()

    def increment_counter(self, session_id: int, name: str, value: int = 1) -> None:
        with self._lock:
            record = self._records.get(session_id)
            if record is None:
                return
            record.increment_counter(name, value)
            if record.is_ready_to_flush():
                self._ready.add(session_id)

    def flush(self, force: bool = False) -> None:
        with self._lock:
            if force:
                candidate_ids = list(self._records.keys())
            else:
                candidate_ids = list(self._ready)
            if not candidate_ids:
                return

            to_flush: list[tuple[int, SessionRecord, bool]] = []
            for session_id in candidate_ids:
                record = self._records.get(session_id)
                if record is None:
                    self._ready.discard(session_id)
                    continue
                if not force and not record.is_ready_to_flush():
                    continue
                was_ready = session_id in self._ready
                to_flush.append((session_id, record, was_ready))

            if not to_flush:
                return

            for session_id, _, _ in to_flush:
                self._records.pop(session_id, None)
                self._ready.discard(session_id)

        try:
            payload = [record.to_dict() for (_, record, _) in to_flush]
            lines = [json.dumps(item, ensure_ascii=False) for item in payload]

            parent = os.path.dirname(self._output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self._output_path, "a", encoding="utf-8") as fout:
                for line in lines:
                    fout.write(f"{line}\n")
                fout.flush()
                os.fsync(fout.fileno())
        except (OSError, TypeError) as exc:  # pragma: no cover - depends on filesystem
            logger.error(
                "Failed to append session trace to %s: %s",
                self._output_path,
                exc,
            )
            with self._lock:
                for session_id, record, was_ready in to_flush:
                    self._records[session_id] = record
                    if was_ready:
                        self._ready.add(session_id)

    def reset(self) -> None:
        self.flush(force=True)
        with self._lock:
            self._records.clear()
            self._ready.clear()
            self._next_session_id = 0
            self._flush_threshold = _normalize_flush_threshold(self._config)


class _Scope(AbstractContextManager[Any]):
    """Internal sync context manager for PerfTracer.trace_scope().

    Automatically records complete events (duration spans) with proper timestamp
    tracking and exception handling. Captures exception types in the args if an
    exception occurs during execution.
    """

    def __init__(
        self,
        tracer: PerfTracer,
        name: str,
        *,
        category: str,
        enable_profiler: bool = False,
        profiler_args: dict[str, Any] | None = None,
        args: dict[str, Any] | None,
    ) -> None:
        self._tracer = tracer
        self._name = name
        self._category = category
        self._args = args
        self._global_step_token: Token | None = None
        self._profiler_token: Token | None = None
        self._start_ns: int | None = None

        self._profiler = None
        if enable_profiler:
            profile_kwargs = dict(profiler_args or {})
            profile_kwargs.setdefault("record_shapes", False)
            profile_kwargs.setdefault("with_stack", False)
            self._profiler = profiler.profile(
                activities=[
                    profiler.ProfilerActivity.CPU,
                    profiler.ProfilerActivity.CUDA,
                ],
                **profile_kwargs,
            )

    def __enter__(self) -> _Scope:
        step = None
        if self._args is not None:
            step = self._args.get("global_step")
        if step is not None:
            self._global_step_token = _current_global_step.set(step)

        if self._profiler:
            self._profiler_token = _profiler_active.set(True)
            self._profiler.__enter__()

        self._start_ns = time.perf_counter_ns()

        return self

    def __exit__(self, exc_type, exc, exc_tb):
        if self._start_ns is None:
            return False

        if self._profiler:
            self._profiler.__exit__(exc_type, exc, exc_tb)

        end_ns = time.perf_counter_ns()

        if self._profiler:
            if self._profiler_token is not None:
                _profiler_active.reset(self._profiler_token)
                self._profiler_token = None

            with tempfile.NamedTemporaryFile("w+b", suffix=".json") as fp:
                self._profiler.export_chrome_trace(fp.name)
                with open(fp.name, "rb") as fin:
                    profiler_traces = orjson.loads(fin.read())

            self._tracer.merge_profiler_trace(
                profiler_traces,
                base_time_ns=self._start_ns,
                args=self._args,
            )

        if self._global_step_token is not None:
            _current_global_step.reset(self._global_step_token)
            self._global_step_token = None

        duration_ns = end_ns - self._start_ns
        args = dict(self._args or {})
        if exc_type is not None:
            args.setdefault("exception", exc_type.__name__)
        self._tracer._record_complete(
            self._name,
            self._start_ns,
            duration_ns,
            category=self._category,
            args=args,
        )
        return False


class _AsyncScope(AbstractAsyncContextManager[Any]):
    """Internal async context manager for PerfTracer.atrace_scope().

    Wraps _Scope to provide async context manager interface while reusing the
    same timestamp tracking and event recording logic.
    """

    def __init__(
        self,
        tracer: PerfTracer,
        name: str,
        *,
        category: str,
        enable_profiler: bool = False,
        profiler_args: dict[str, Any] | None = None,
        args: dict[str, Any] | None,
    ) -> None:
        self._scope = _Scope(
            tracer,
            name,
            category=category,
            enable_profiler=enable_profiler,
            profiler_args=profiler_args,
            args=args,
        )

    async def __aenter__(self) -> _AsyncScope:
        self._scope.__enter__()
        return self

    async def __aexit__(self, exc_type, exc, exc_tb):
        return self._scope.__exit__(exc_type, exc, exc_tb)


def _thread_id() -> int:
    cached = getattr(_THREAD_LOCAL, "tid", None)
    if cached is not None:
        return cached
    try:
        tid = threading.get_native_id()
    except AttributeError:  # pragma: no cover - Python <3.8 fallback
        tid = threading.get_ident()
    _THREAD_LOCAL.tid = tid
    return tid


class PerfTracer:
    """A lightweight tracer that emits Chrome Trace compatible JSON.

    This is the main tracer class for recording performance events during training
    and inference. It outputs Chrome Trace format JSON files that can be visualized
    in chrome://tracing or Perfetto.

    The tracer supports two tracking modes:
    1. Performance events: Trace scopes, instant events, counters (Chrome Trace format)
    2. Session lifecycle: Optional per-session tracking via integrated SessionTracer

    When session tracking is enabled (via config.session_tracer.enabled=True), the
    tracer automatically creates a SessionTracer instance that writes session records
    to a separate JSONL file. This enables detailed analysis of rollout session
    lifecycles alongside performance traces.

    Parameters
    ----------
    config : PerfTracerConfig
        Configuration including enable flags, output paths, and session tracer settings.
    rank : int
        Rank identifier for this process in distributed training.
    role : str | None, default=None
        Optional role identifier (e.g., "actor", "learner") for the process.

    See Also
    --------
    SessionTracer : Integrated session lifecycle tracker
    trace_session_event : Function to record session events
    """

    def __init__(
        self, config: PerfTracerConfig, *, rank: int, role: str | None = None
    ) -> None:
        if rank < 0:
            raise ValueError("rank must be a non-negative integer")
        self._config = config
        self._enabled = config.enabled
        self._rank = rank
        self._role = role
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._pid = os.getpid()
        self._origin_ns = time.perf_counter_ns()
        self._thread_meta_emitted: set[int] = set()
        self._process_meta_emitted: set[int] = set()
        # Virtual TID management for merged traces
        self._virtual_tid_map: dict[tuple[int, int], int] = {}
        self._next_virtual_tid: int = -1
        self._output_path = _default_trace_path(
            config,
            rank=rank,
            role=role,
            subdir="perf_tracer",
        )
        self._save_interval = _normalize_save_interval(config)
        self._profile_steps = self._normalize_profile_steps(config.profile_steps)
        self._session_tracer: SessionTracer | None = None
        self._configure_session_tracer(config, rank=rank, role=role)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, flag: bool) -> None:
        self._enabled = flag

    def _configure_session_tracer(
        self, config: PerfTracerConfig, *, rank: int, role: str | None = None
    ) -> None:
        session_cfg = getattr(config, "session_tracer", None)
        enabled = bool(session_cfg and getattr(session_cfg, "enabled", False))
        if enabled:
            session_cfg = cast(SessionTracerConfig, session_cfg)
            output_path = _default_trace_path(
                config,
                filename=_SESSION_TRACE_FILENAME,
                rank=rank,
                role=role,
                subdir="session_tracer",
            )
            if self._session_tracer is None:
                self._session_tracer = SessionTracer(
                    session_cfg,
                    output_path=output_path,
                    rank=rank,
                    role=role,
                )
            else:
                raise RuntimeError("Session tracer is already configured")
        else:
            if self._session_tracer is not None:
                self._session_tracer.flush(force=True)
            self._session_tracer = None

    @property
    def session_tracer(self) -> SessionTracer | None:
        return self._session_tracer

    # ------------------------------------------------------------------
    # Core recording API
    # ------------------------------------------------------------------
    def trace_scope(
        self,
        name: str,
        *,
        category: CategoryLike = None,
        enable_profiler: bool = False,
        profiler_args: dict[str, Any] | None = None,
        args: dict[str, Any] | None = None,
    ) -> AbstractContextManager[Any]:
        if not self._enabled:
            return _NullContext()
        global_step = self._get_global_step(args)
        profiler_active = _profiler_active.get(False)
        profiler_flag = not profiler_active and self._should_enable_profiler(
            enable_profiler, global_step
        )
        return _Scope(
            self,
            name,
            category=_normalize_category(category),
            enable_profiler=profiler_flag,
            profiler_args=profiler_args,
            args=args,
        )

    def atrace_scope(
        self,
        name: str,
        *,
        category: CategoryLike = None,
        enable_profiler: bool = False,
        profiler_args: dict[str, Any] | None = None,
        args: dict[str, Any] | None = None,
    ) -> AbstractAsyncContextManager[Any]:
        if not self._enabled:
            return _NullContext()
        global_step = self._get_global_step(args)
        profiler_active = _profiler_active.get(False)
        profiler_flag = not profiler_active and self._should_enable_profiler(
            enable_profiler, global_step
        )
        return _AsyncScope(
            self,
            name,
            category=_normalize_category(category),
            enable_profiler=profiler_flag,
            profiler_args=profiler_args,
            args=args,
        )

    def instant(
        self,
        name: str,
        *,
        category: CategoryLike = None,
        args: dict[str, Any] | None = None,
    ) -> None:
        if not self._enabled:
            return
        self._record_event(
            {
                "name": name,
                "ph": "i",
                "ts": self._now_us(),
                "pid": self._pid,
                "tid": _thread_id(),
                "cat": _normalize_category(category),
                "args": args or {},
                "s": "t",
            }
        )

    # ------------------------------------------------------------------
    # Kernel trace helpers
    # ------------------------------------------------------------------
    def merge_profiler_trace(
        self,
        profiler_traces: Any,
        base_time_ns: int,
        *,
        args: dict[str, Any] | None = None,
    ) -> None:
        """Merge PyTorch Profiler traces into the current trace stream.

        This method ingests a raw PyTorch Profiler JSON trace (loaded as a dict),
        aligns timestamps, remaps PIDs/TIDs to avoid conflicts, and merges relevant
        events into the PerfTracer's event stream.

        Key transformations:
        1. Timestamps: Shifted by `base_time_ns` to align with PerfTracer's clock.
        2. PIDs: All events are remapped to `self._pid` (current process).
        3. TIDs: Mapped to unique negative integers to avoid collision with real threads.
           - CPU threads from Profiler -> "Profiler CPU {tid}"
           - GPU streams -> "GPU {device} Stream {stream}"
        4. Metadata: `thread_name` events are generated for the new virtual TIDs.

        Parameters
        ----------
        profiler_traces : Any
            The loaded JSON object from PyTorch Profiler export.
        base_time_ns : int
            The start timestamp of the profiling session in nanoseconds (perf_counter).
            Used to align Profiler's absolute timestamps with PerfTracer's relative ones.
        args : dict[str, Any] | None
            Optional arguments to attach to the merged events (currently unused but
            kept for API consistency).
        """
        if not self._enabled:
            return

        trace_events = profiler_traces.get("traceEvents", [])
        if not trace_events:
            return

        ts_values = [e["ts"] for e in trace_events if "ts" in e]
        if ts_values:
            min_ts = min(ts_values)
            # timestamps in Chrome trace json are in microseconds
            time_offset_us = (base_time_ns - self._origin_ns) / 1000.0 - min_ts
        else:
            raise RuntimeError(
                "Profiler trace events do not contain timestamps ('ts' field)."
            )

        # First pass: collect thread names from metadata events to identify GPU streams
        thread_names = {}
        for event in trace_events:
            if event.get("ph") == "M" and event.get("name") == "thread_name":
                pid = event.get("pid")
                tid = event.get("tid")
                event_args = event.get("args", {})
                name = event_args.get("name")
                if pid is not None and tid is not None and name:
                    try:
                        pid_key = int(pid)
                        tid_key = int(tid)
                        thread_names[(pid_key, tid_key)] = name
                    except (ValueError, TypeError):
                        pass

        new_events = []

        # Metadata events to ignore from source (we generate our own or they are useless)
        ignored_metadata = {
            "process_name",
            "process_sort_index",
            "process_labels",
            "thread_name",
            "thread_sort_index",
        }
        # We also ignore "Spans", "Traces" PIDs which are internal to PyTorch Profiler
        ignored_pids = {"Spans", "Traces", ""}

        with self._lock:
            for event in trace_events:
                pid = event.get("pid")
                if pid in ignored_pids:
                    continue

                # Filter out metadata we don't want
                if event.get("ph") == "M" and event.get("name") in ignored_metadata:
                    continue

                tid = event.get("tid")

                # Ensure pid/tid are hashable (int)
                try:
                    pid_key = int(pid) if pid is not None else -1
                    tid_key = int(tid) if tid is not None else -1
                except (ValueError, TypeError):
                    continue

                # Filter out events with negative PIDs (e.g. -1)
                if pid_key < 0:
                    continue

                # Allocate virtual TID
                if (pid_key, tid_key) not in self._virtual_tid_map:
                    v_tid = self._next_virtual_tid
                    self._next_virtual_tid -= 1
                    self._virtual_tid_map[(pid_key, tid_key)] = v_tid

                    # Default to generic thread name
                    t_name = f"[Profiler Thread {tid_key}]"
                    original_t_name = thread_names.get((pid_key, tid_key))
                    if original_t_name:
                        parts = original_t_name.strip().lower().split()
                        if (
                            len(parts) >= 2
                            and parts[0] == "stream"
                            and parts[1].isdigit()
                        ):
                            t_name = f"[Profiler Stream {parts[1]}]"

                    # Emit thread_name event immediately
                    new_events.append(
                        {
                            "name": "thread_name",
                            "ph": "M",
                            "pid": self._pid,
                            "tid": v_tid,
                            "args": {"name": t_name, "rank": self._rank},
                        }
                    )

                virtual_tid = self._virtual_tid_map[(pid_key, tid_key)]

                # Create new event
                new_event = event.copy()
                new_event["pid"] = self._pid
                new_event["tid"] = virtual_tid

                # Adjust timestamp
                if "ts" in new_event:
                    new_event["ts"] = new_event["ts"] + time_offset_us

                # Attach rank and role info
                event_args = new_event.get("args")
                if not isinstance(event_args, dict):
                    event_args = {}
                    new_event["args"] = event_args
                event_args["rank"] = self._rank
                if self._role is not None:
                    event_args["role"] = self._role

                new_events.append(new_event)

            self._events.extend(new_events)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, *, step: int | None = None, force: bool = False) -> None:
        if self._session_tracer is not None and force:
            self._session_tracer.flush(force=True)

        if not self._enabled:
            return

        # Save only on the last step of each interval (0-indexed).
        # For example, if save_interval=3, saves at steps 2, 5, 8, ...
        interval = self._save_interval
        if (
            not force
            and step is not None
            and interval > 1
            and ((step + 1) % interval) != 0
        ):
            return

        with self._lock:
            if not self._events:
                return
            events_to_write: list[dict[str, Any]] = self._events
            self._events = []

        try:
            serialized_events = [
                json.dumps(event, ensure_ascii=False) for event in events_to_write
            ]
            output_path = self._output_path

            parent = os.path.dirname(output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(output_path, "a", encoding="utf-8") as fout:
                for line in serialized_events:
                    fout.write(f"{line}\n")
                fout.flush()
                os.fsync(fout.fileno())
        except (OSError, TypeError) as exc:  # pragma: no cover - depends on filesystem
            logger.error("Failed to append perf trace to %s: %s", output_path, exc)
            with self._lock:
                self._events[0:0] = events_to_write

    def reset(self) -> None:
        if self._session_tracer is not None:
            self._session_tracer.reset()
        with self._lock:
            self._events = []
            self._thread_meta_emitted = set()
            self._process_meta_emitted = set()
            self._origin_ns = time.perf_counter_ns()
            self._enabled = self._config.enabled
            self._save_interval = _normalize_save_interval(self._config)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _record_complete(
        self,
        name: str,
        start_ns: int,
        duration_ns: int,
        *,
        category: str,
        args: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "name": name,
            "ph": "X",
            "ts": self._relative_us(start_ns),
            # Chrome trace viewers drop complete events whose duration rounds to 0 µs,
            # so clamp to 1 µs to keep sub-microsecond spans visible.
            "dur": max(duration_ns // 1000, 1),
            "pid": self._pid,
            "tid": _thread_id(),
            "cat": category,
            "args": args or {},
        }
        self._record_event(event)

    def _record_event(self, event: dict[str, Any]) -> None:
        if not self._enabled:
            return
        tid = event.get("tid")
        if isinstance(tid, int):
            self._ensure_thread_metadata(tid)
        event["pid"] = self._pid
        self._ensure_process_metadata(self._pid)
        if event.get("ph") != "M":
            args = event.setdefault("args", {})
            args.setdefault("rank", self._rank)
            if self._role is not None:
                args.setdefault("role", self._role)
        with self._lock:
            self._events.append(event)

    def _get_global_step(self, args: dict[str, Any] | None) -> int | None:
        if args and "global_step" in args:
            return args["global_step"]
        return _current_global_step.get()

    def _should_enable_profiler(
        self,
        requested: bool,
        global_step: int | None,
    ) -> bool:
        return bool(
            requested
            and self._profile_steps
            and global_step is not None
            and global_step in self._profile_steps
        )

    @staticmethod
    def _normalize_profile_steps(steps: list[int] | None) -> frozenset[int]:
        if not steps:
            return frozenset()
        normalized: set[int] = set()
        for raw in steps:
            try:
                step = int(raw)
            except (TypeError, ValueError):
                logger.warning("PerfTracer: invalid profile_steps entry %r", raw)
                continue
            normalized.add(step)
        return frozenset(normalized)

    def _ensure_thread_metadata(self, tid: int) -> None:
        if tid in self._thread_meta_emitted:
            return
        self._thread_meta_emitted.add(tid)
        thread_name = threading.current_thread().name
        meta_event = {
            "name": "thread_name",
            "ph": "M",
            "pid": self._pid,
            "tid": tid,
            "args": {"name": thread_name},
        }
        with self._lock:
            self._events.append(meta_event)

    def _ensure_process_metadata(self, pid: int) -> None:
        if pid in self._process_meta_emitted:
            return
        self._process_meta_emitted.add(pid)
        if self._role is not None:
            rank_label = f"[{self._role}] Rank {self._rank}, Process"
        else:
            rank_label = f"Rank {self._rank}, Process"
        process_name_event = {
            "name": "process_name",
            "ph": "M",
            "pid": pid,
            "args": {"name": rank_label},
        }
        sort_event = {
            "name": "process_sort_index",
            "ph": "M",
            "pid": pid,
            "args": {"sort_index": self._rank},
        }
        with self._lock:
            self._events.extend([process_name_event, sort_event])

    def _now_us(self) -> int:
        return self._relative_us(time.perf_counter_ns())

    def _relative_us(self, ts_ns: int) -> int:
        return max((ts_ns - self._origin_ns) // 1000, 0)


GLOBAL_TRACER: PerfTracer | None = None
_GLOBAL_TRACER_LOCK = threading.Lock()


def _save_at_exit() -> None:
    tracer = GLOBAL_TRACER
    if tracer is None or not tracer.enabled:
        return
    try:
        tracer.save(force=True)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to flush perf trace on exit: %s", exc, exc_info=True)


atexit.register(_save_at_exit)


# ----------------------------------------------------------------------
# Module-level convenience functions
# ----------------------------------------------------------------------
def _require_configured_tracer() -> PerfTracer:
    tracer = GLOBAL_TRACER
    if tracer is None:
        raise RuntimeError(
            "PerfTracer is not configured. Call perf_tracer.configure(...) first."
        )
    return tracer


def get_tracer() -> PerfTracer:
    return _require_configured_tracer()


def get_session_tracer() -> SessionTracer | None:
    tracer = GLOBAL_TRACER
    if tracer is None:
        return None
    return tracer.session_tracer


def configure(
    config: PerfTracerConfig,
    *,
    rank: int,
    role: str | None = None,
) -> PerfTracer:
    global GLOBAL_TRACER
    with _GLOBAL_TRACER_LOCK:
        if GLOBAL_TRACER is not None:
            raise RuntimeError(
                "PerfTracer has already been configured. Call perf_tracer.reset() "
                "before configuring again."
            )
        GLOBAL_TRACER = PerfTracer(config, rank=rank, role=role)
        logger.info(
            "Configured global PerfTracer: enabled=%s, session_tracing=%s, rank=%s, role=%s",
            GLOBAL_TRACER.enabled,
            GLOBAL_TRACER.session_tracer is not None,
            rank,
            role,
        )
        return GLOBAL_TRACER


def is_configured() -> bool:
    global GLOBAL_TRACER
    with _GLOBAL_TRACER_LOCK:
        return GLOBAL_TRACER is not None


def reset() -> None:
    """Clear the global tracer so the next configure() call reinitializes it."""
    global GLOBAL_TRACER
    with _GLOBAL_TRACER_LOCK:
        tracer = GLOBAL_TRACER
        GLOBAL_TRACER = None
    if tracer is not None:
        tracer.reset()


def trace_scope(
    name: str,
    *,
    category: CategoryLike = None,
    enable_profiler: bool = False,
    profiler_args: dict[str, Any] | None = None,
    args: dict[str, Any] | None = None,
):
    tracer = GLOBAL_TRACER
    if tracer is None:
        return _NullContext()
    return tracer.trace_scope(
        name,
        category=category,
        enable_profiler=enable_profiler,
        profiler_args=profiler_args,
        args=args,
    )


def atrace_scope(
    name: str,
    *,
    category: CategoryLike = None,
    enable_profiler: bool = False,
    profiler_args: dict[str, Any] | None = None,
    args: dict[str, Any] | None = None,
):
    tracer = GLOBAL_TRACER
    if tracer is None:
        return _NullContext()
    return tracer.atrace_scope(
        name,
        category=category,
        enable_profiler=enable_profiler,
        profiler_args=profiler_args,
        args=args,
    )


def instant(
    name: str,
    *,
    category: CategoryLike = None,
    args: dict[str, Any] | None = None,
) -> None:
    tracer = GLOBAL_TRACER
    if tracer is None:
        return
    tracer.instant(name, category=category, args=args)


def save(*, step: int | None = None, force: bool = False) -> None:
    tracer = GLOBAL_TRACER
    if tracer is None:
        return
    tracer.save(step=step, force=force)


def trace_perf(name: str, *, category: CategoryLike = None):
    """
    Decorator for tracing function performance with PerfTracer.

    Automatically creates a trace scope around the entire function execution.
    Works with both sync and async functions.

    Parameters
    ----------
    name : str
        Trace name to display in the trace viewer.
    category : CategoryLike, optional
        Trace category (compute, io, comm, sync, scheduler, etc.).

    Examples
    --------
    >>> @trace_perf("ppo_update", category="compute")
    ... async def update_model(self, batch):
    ...     loss = compute_loss(batch)
    ...     loss.backward()
    ...     return loss

    >>> @trace_perf("save_checkpoint", category="io")
    ... def save(self, path):
    ...     torch.save(self.state_dict(), path)
    """

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with atrace_scope(name, category=category):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with trace_scope(name, category=category):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def set_session_id(session_id: int | None) -> None:
    """Set the session_id for the current async context."""
    _current_session_id.set(session_id)


def get_session_id() -> int | None:
    """Get the session_id from the current async context."""
    return _current_session_id.get()


def set_task_id(task_id: int | None) -> None:
    """Set the task_id for the current async context."""
    _current_task_id.set(task_id)


def get_task_id() -> int | None:
    """Get the task_id from the current async context."""
    return _current_task_id.get()


def register_task(task_id: int) -> None:
    """Register a new task given a task_id in the current async context."""
    tracer = get_session_tracer()
    if tracer is not None:
        tracer.register_task(task_id)


def register_session(task_id: int) -> int | None:
    """Register a new session under the given task_id and return the session_id."""
    session_id = None
    tracer = get_session_tracer()
    if tracer is not None:
        session_id = tracer.register_session(task_id)
    return session_id


def session_context():
    """Decorator factory that populates ``session_id`` from the active task.

    Each wrapped invocation registers a session when a ``task_id`` exists and
    records it in the session context for downstream tracing. No cleanup or
    restoration is performed after execution; if no task is active the session
    context is set to ``None``.

    Returns
    -------
    Callable
        Decorator applicable to both sync and async callables.
    """

    def decorator(func):
        def _activate_session() -> None:
            task_id = get_task_id()
            session_id = register_session(task_id) if task_id is not None else None
            set_session_id(session_id)

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                _activate_session()
                return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                _activate_session()
                return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def trace_session(phase: str):
    """
    Decorator for tracing session phases using contextvars.

    Automatically reads the active session from context (populated via
    :func:`session_context`) and traces the phase execution.

    Parameters
    ----------
    phase : str
        Phase name (e.g., "generate", "reward", "execution").
        Will call mark_{phase}_start and mark_{phase}_end.

    Examples
    --------
    >>> # Context is set by WorkflowExecutor before calling arun_episode
    >>> async def arun_episode(self, engine, data):
    ...     # session information is automatically available from context
    ...     resps = await self._do_generate(engine, req, n_samples)
    ...     results = await self._compute_rewards(resps)
    ...     return results

    >>> # Use decorator on methods - no need to pass session_id!
    >>> @trace_session("generate")
    ... async def _do_generate(self, engine, req, n_samples):
    ...     return await engine.agenerate(...)

    >>> @trace_session("reward")
    ... async def _compute_rewards(self, resps):
    ...     for resp in resps:
    ...         reward = await self.async_reward_fn(...)
    ...     return results
    """

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with atrace_session_phase(phase):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with trace_session_phase(phase):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


__all__ = [
    "PerfTracer",
    "SessionTracer",
    "PerfTraceCategory",
    "Category",
    "SessionTraceEvent",
    "trace_session_event",
    "trace_session_phase",
    "atrace_session_phase",
    "trace_perf",
    "trace_session",
    "session_context",
    "set_session_id",
    "get_session_id",
    "set_task_id",
    "get_task_id",
    "register_task",
    "register_session",
    "GLOBAL_TRACER",
    "get_tracer",
    "get_session_tracer",
    "configure",
    "reset",
    "trace_scope",
    "atrace_scope",
    "instant",
    "save",
]
