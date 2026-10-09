# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Hashable
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Lock
from typing import Any, TypeVar, cast

T = TypeVar("T")


class ProcessorCallCache:
    """Thread-safe single-flight cache for identical calls within one group."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._results: dict[Hashable, Any] = {}
        self._inflight: dict[Hashable, Future[Any]] = {}
        self._generation = 0
        self._closed = False

    def _claim(self, key: Hashable) -> tuple[Future[Any] | None, bool, int]:
        with self._lock:
            if self._closed:
                return None, False, self._generation
            if key in self._results:
                future: Future[Any] = Future()
                future.set_result(self._results[key])
                return future, False, self._generation

            future = self._inflight.get(key)
            is_owner = future is None
            if future is None:
                future = Future()
                self._inflight[key] = future
            return future, is_owner, self._generation

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
            future.set_exception(exc)

    def get_or_compute(self, key: Hashable, factory: Callable[[], T]) -> T:
        """Return a cached value, computing it once across concurrent callers."""
        future, is_owner, generation = self._claim(key)
        if future is None:
            return factory()

        if not is_owner:
            return cast(T, future.result())

        try:
            result = factory()
        except BaseException as exc:
            self._publish_exception(key, future, exc)
            raise

        self._publish_result(key, future, generation, result)
        return result

    async def aget_or_compute(self, key: Hashable, factory: Callable[[], T]) -> T:
        """Asynchronously compute once without occupying threads for waiters."""
        future, is_owner, generation = self._claim(key)
        if future is None:
            return await asyncio.to_thread(factory)
        if not is_owner:
            return cast(T, await asyncio.shield(asyncio.wrap_future(future)))

        try:
            result = await asyncio.to_thread(factory)
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
