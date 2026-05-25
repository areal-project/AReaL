# SPDX-License-Identifier: Apache-2.0

"""Worker and session registries for the Router service.

All state is in-memory (lost on restart). Thread-safe via asyncio.Lock.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class WorkerInfo:
    """A registered data proxy worker."""

    worker_id: str
    worker_addr: str
    is_healthy: bool = True
    active_requests: int = 0
    registered_at: float = field(default_factory=time.time)


class WorkerRegistry:
    """Thread-safe worker registry with health tracking."""

    def __init__(self) -> None:
        self._workers: dict[str, WorkerInfo] = {}  # worker_addr -> WorkerInfo
        self._id_to_addr: dict[str, str] = {}  # worker_id -> worker_addr
        self._lock = asyncio.Lock()

    async def register(self, worker_addr: str) -> str:
        """Add a worker. Returns existing worker_id if already registered."""
        async with self._lock:
            if worker_addr in self._workers:
                return self._workers[worker_addr].worker_id
            worker_id = str(uuid.uuid4())
            self._workers[worker_addr] = WorkerInfo(
                worker_id=worker_id, worker_addr=worker_addr
            )
            self._id_to_addr[worker_id] = worker_addr
            return worker_id

    async def deregister(self, worker_addr: str) -> None:
        """Remove a worker by address. No-op if not found."""
        async with self._lock:
            w = self._workers.pop(worker_addr, None)
            if w is not None:
                self._id_to_addr.pop(w.worker_id, None)

    async def deregister_by_id(self, worker_id: str) -> str | None:
        """Remove a worker by ID. Returns the worker_addr or None if not found."""
        async with self._lock:
            worker_addr = self._id_to_addr.pop(worker_id, None)
            if worker_addr is not None:
                self._workers.pop(worker_addr, None)
            return worker_addr

    async def get_by_id(self, worker_id: str) -> WorkerInfo | None:
        """Look up a worker by its ID."""
        async with self._lock:
            addr = self._id_to_addr.get(worker_id)
            if addr is None:
                return None
            return self._workers.get(addr)

    async def update_health(self, worker_addr: str, healthy: bool) -> None:
        """Set the health flag for a worker."""
        async with self._lock:
            w = self._workers.get(worker_addr)
            if w:
                w.is_healthy = healthy

    async def get_healthy_workers(self) -> list[WorkerInfo]:
        """Return only workers with ``is_healthy == True``."""
        async with self._lock:
            return [w for w in self._workers.values() if w.is_healthy]

    async def get_all_workers(self) -> list[WorkerInfo]:
        """Return all workers regardless of health."""
        async with self._lock:
            return list(self._workers.values())

    async def list_worker_addrs(self) -> list[str]:
        """Return all registered worker addresses."""
        async with self._lock:
            return list(self._workers.keys())


class SessionRegistry:
    """Maps session API keys and session IDs to worker addresses.

    Pinning persists after reward is set (needed for
    ``/export_trajectories``). Cleaned up after export_trajectories or
    when a worker is deleted.
    """

    def __init__(self) -> None:
        self._key_to_worker: dict[str, str] = {}  # session_api_key -> worker_addr
        self._id_to_worker: dict[str, str] = {}  # session_id -> worker_addr
        self._id_to_key: dict[str, str] = {}  # session_id -> session_api_key
        self._lock = asyncio.Lock()

    async def register_session(
        self, session_key: str, session_id: str, worker_addr: str
    ) -> None:
        """Store both session_key→worker and session_id→worker. Upsert semantics."""
        async with self._lock:
            self._key_to_worker[session_key] = worker_addr
            self._id_to_worker[session_id] = worker_addr
            self._id_to_key[session_id] = session_key

    async def lookup_by_key(self, session_key: str) -> str | None:
        """Return the worker address pinned to a session API key, or None."""
        async with self._lock:
            return self._key_to_worker.get(session_key)

    async def lookup_by_id(self, session_id: str) -> str | None:
        """Return the worker address pinned to a session ID, or None."""
        async with self._lock:
            return self._id_to_worker.get(session_id)

    async def revoke_by_worker(self, worker_addr: str) -> int:
        """Remove all sessions pinned to a worker (cascade on deletion).

        Returns the number of session keys removed.
        """
        async with self._lock:
            keys_to_remove = [
                k for k, v in self._key_to_worker.items() if v == worker_addr
            ]
            ids_to_remove = [
                k for k, v in self._id_to_worker.items() if v == worker_addr
            ]
            for k in keys_to_remove:
                del self._key_to_worker[k]
            for k in ids_to_remove:
                self._id_to_key.pop(k, None)
                del self._id_to_worker[k]
            return len(keys_to_remove)

    async def revoke_session(self, session_id: str) -> bool:
        """Remove a single session by its ID.

        Removes both the session_id→worker and session_key→worker mappings.
        Called after ``/export_trajectories`` to prevent unbounded growth.

        Returns True if the session was found and removed, False otherwise.
        """
        async with self._lock:
            if session_id not in self._id_to_worker:
                return False
            del self._id_to_worker[session_id]
            session_key = self._id_to_key.pop(session_id, None)
            if session_key is not None:
                self._key_to_worker.pop(session_key, None)
            return True

    async def session_key_for_id(self, session_id: str) -> str | None:
        async with self._lock:
            return self._id_to_key.get(session_id)

    async def count(self) -> int:
        """Return the number of registered session keys."""
        async with self._lock:
            return len(self._key_to_worker)


@dataclass
class ModelInfo:
    """A registered model (internal or external)."""

    name: str
    url: str  # empty string for internal models
    api_key: str | None
    data_proxy_addrs: list[str] = field(default_factory=list)


@dataclass
class GroupInfo:
    """A registered group of sessions."""

    group_id: str
    worker_addr: str
    session_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class ModelRegistry:
    """Thread-safe registry for model routing."""

    def __init__(self) -> None:
        self._models: dict[str, ModelInfo] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        name: str,
        url: str,
        api_key: str | None,
        data_proxy_addrs: list[str],
    ) -> None:
        async with self._lock:
            self._models[name] = ModelInfo(
                name=name,
                url=url,
                api_key=api_key,
                data_proxy_addrs=data_proxy_addrs,
            )

    async def get(self, name: str) -> ModelInfo | None:
        async with self._lock:
            return self._models.get(name)

    async def first(self) -> ModelInfo | None:
        async with self._lock:
            if not self._models:
                return None
            return next(iter(self._models.values()))

    async def list_names(self) -> list[str]:
        async with self._lock:
            return list(self._models.keys())

    async def remove(self, name: str) -> bool:
        async with self._lock:
            return self._models.pop(name, None) is not None


class GroupRegistry:
    """Maps group IDs to worker addresses and member session IDs."""

    def __init__(self) -> None:
        self._groups: dict[str, GroupInfo] = {}
        self._lock = asyncio.Lock()

    async def register_group(
        self, group_id: str, worker_addr: str, session_ids: list[str]
    ) -> None:
        """Store a group mapping. Raises ValueError if group_id already exists."""
        async with self._lock:
            if group_id in self._groups:
                raise ValueError(f"Group {group_id} already registered")
            self._groups[group_id] = GroupInfo(
                group_id=group_id,
                worker_addr=worker_addr,
                session_ids=list(session_ids),
            )

    async def lookup(self, group_id: str) -> GroupInfo | None:
        """Return the GroupInfo for a group_id, or None."""
        async with self._lock:
            return self._groups.get(group_id)

    async def revoke(self, group_id: str) -> list[str]:
        """Remove a group. Returns the session_ids that were in the group."""
        async with self._lock:
            info = self._groups.pop(group_id, None)
            if info is None:
                return []
            return info.session_ids
