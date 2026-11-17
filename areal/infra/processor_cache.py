# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Hashable
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock, get_ident
from typing import Any, TypeVar, cast
from uuid import uuid4

T = TypeVar("T")


class ProcessorCallCache:
    """Thread-safe single-flight cache for identical calls within one group."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._results: dict[Hashable, Any] = {}
        self._inflight: dict[Hashable, Future[Any]] = {}
        self._generation = 0
        self._closed = False
        self._cache_id = uuid4().hex
        self._execution_ids: dict[Hashable, str] = {}

    def _claim(self, key: Hashable) -> tuple[Future[Any] | None, bool, int, str, str]:
        with self._lock:
            if self._closed:
                return None, False, self._generation, "bypass", uuid4().hex
            if key in self._results:
                future: Future[Any] = Future()
                future.set_result(self._results[key])
                return future, False, self._generation, "hit", self._execution_ids[key]

            future = self._inflight.get(key)
            is_owner = future is None
            if future is None:
                future = Future()
                self._inflight[key] = future
                self._execution_ids[key] = uuid4().hex
            return (
                future,
                is_owner,
                self._generation,
                "miss" if is_owner else "wait",
                self._execution_ids[key],
            )

    @contextmanager
    def _lookup(self, key: Hashable):
        future, owner, generation, outcome, execution = self._claim(key)
        kinds = {
            "openai_multimodal",
            "openai_vision_tensors",
            "sglang_qwen_image",
            "pixel_video_processor",
        }
        fields = {
            "cache_id": self._cache_id,
            "execution_id": execution,
            "lookup_id": uuid4().hex,
            "generation": generation,
            "cache_kind": key[0]
            if isinstance(key, tuple) and key and key[0] in kinds
            else "processor",
            "cache_outcome": outcome,
        }
        self._record("processor_cache", "lookup", fields)
        try:
            yield future, owner, generation, fields
        except BaseException as error:
            reason = (
                "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
            )
            self._record(
                "processor_cache", reason, fields, exception=type(error).__name__
            )
            raise
        else:
            self._record("processor_cache", "returned", fields)

    @staticmethod
    def _record(stage: str, reason: str, fields: dict, **details: Any) -> None:
        from areal.utils.perf_tracer import record_disposition

        record_disposition(
            "outcome",
            stage,
            reason,
            unit="operation",
            thread_id=get_ident(),
            **fields,
            **details,
        )

    def _compute(self, factory: Callable[[], T], fields: dict) -> T:
        self._record("processor_compute", "started", fields)
        try:
            result = factory()
        except BaseException as error:
            self._record(
                "processor_compute", "failed", fields, exception=type(error).__name__
            )
            raise
        self._record("processor_compute", "completed", fields)
        return result

    def _publish_result(
        self,
        key: Hashable,
        future: Future[Any],
        generation: int,
        result: Any,
    ) -> None:
        with self._lock:
            if generation == self._generation:
                self._results[key] = result
            self._inflight.pop(key, None)
            future.set_result(result)

    def _publish_exception(
        self, key: Hashable, future: Future[Any], exc: BaseException
    ) -> None:
        with self._lock:
            self._inflight.pop(key, None)
            self._execution_ids.pop(key, None)
            future.set_exception(exc)

    def get_or_compute(self, key: Hashable, factory: Callable[[], T]) -> T:
        """Return a cached value, computing it once across concurrent callers."""
        with self._lookup(key) as (future, is_owner, generation, fields):
            if future is None:
                return self._compute(factory, fields)
            if not is_owner:
                return cast(T, future.result())
            try:
                result = self._compute(factory, fields)
            except BaseException as exc:
                self._publish_exception(key, future, exc)
                raise
            self._publish_result(key, future, generation, result)
            return result

    async def aget_or_compute(self, key: Hashable, factory: Callable[[], T]) -> T:
        """Asynchronously compute once without occupying threads for waiters."""
        with self._lookup(key) as (future, is_owner, generation, fields):
            if future is None:
                return await asyncio.to_thread(self._compute, factory, fields)
            if not is_owner:
                return cast(T, await asyncio.shield(asyncio.wrap_future(future)))
            try:
                result = await asyncio.to_thread(self._compute, factory, fields)
            except BaseException as exc:
                self._publish_exception(key, future, exc)
                raise
            self._publish_result(key, future, generation, result)
            return result

    def close(self) -> None:
        """Disable caching and drop results owned by a completed group."""
        with self._lock:
            self._closed = True
            self._generation += 1
            self._results.clear()
            self._execution_ids.clear()

    @classmethod
    def make_key(cls, *values: Any) -> Hashable:
        """Convert processor inputs into a stable, hashable cache key."""
        return tuple(cls._freeze(value) for value in values)

    @classmethod
    def _freeze(cls, value: Any) -> Hashable:
        if isinstance(value, dict):
            return tuple(
                sorted((str(key), cls._freeze(item)) for key, item in value.items())
            )
        if isinstance(value, (list, tuple)):
            return tuple(cls._freeze(item) for item in value)
        if isinstance(value, (set, frozenset)):
            return tuple(sorted(repr(cls._freeze(item)) for item in value))
        try:
            hash(value)
        except TypeError:
            return repr(value)
        return cast(Hashable, value)


@dataclass
class _RegistryEntry:
    cache: ProcessorCallCache
    expected_users: int
    acquired_users: int = 0
    active_users: int = 0
    last_access_time: float = 0.0


class ProcessorCacheRegistry:
    """Own group caches that must be shared across proxy sessions."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._entries: dict[str, _RegistryEntry] = {}

    def acquire(self, group_id: str, expected_users: int) -> ProcessorCallCache:
        if expected_users < 2:
            raise ValueError("A shared processor cache requires at least two users.")

        with self._lock:
            entry = self._entries.get(group_id)
            if entry is None:
                entry = _RegistryEntry(
                    cache=ProcessorCallCache(),
                    expected_users=expected_users,
                )
                self._entries[group_id] = entry
            else:
                entry.expected_users = max(entry.expected_users, expected_users)
            entry.acquired_users += 1
            entry.active_users += 1
            entry.last_access_time = time.monotonic()
            return entry.cache

    def release(self, group_id: str) -> None:
        with self._lock:
            entry = self._entries.get(group_id)
            if entry is None:
                return
            entry.active_users = max(0, entry.active_users - 1)
            entry.last_access_time = time.monotonic()
            if entry.acquired_users >= entry.expected_users and entry.active_users == 0:
                self._entries.pop(group_id).cache.close()

    def discard(self, group_id: str) -> None:
        """Explicitly clear a completed or aborted group's cached results."""
        with self._lock:
            entry = self._entries.pop(group_id, None)
            if entry is not None:
                entry.cache.close()

    def discard_stale(self, timeout_seconds: float) -> None:
        cutoff = time.monotonic() - timeout_seconds
        with self._lock:
            stale_ids = [
                group_id
                for group_id, entry in self._entries.items()
                if entry.last_access_time < cutoff
            ]
            for group_id in stale_ids:
                self._entries.pop(group_id).cache.close()
