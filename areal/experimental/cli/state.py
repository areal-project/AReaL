# SPDX-License-Identifier: Apache-2.0

"""On-disk state primitives shared across service-style CLIs.

Three layers live here:

1. **Global helpers** — ``areal_home`` (the user's CLI root) and
   ``atomic_write_json`` (a generic write primitive). Both are
   stateless; no namespace involvement.

2. **NamespacedStateStore** — every subcommand CLI binds a namespace
   (``inf``, ``agent``, ``train``, …) and lives under
   ``$AREAL_HOME/<namespace>/``. The store class collects the path
   resolution / pointer file / orphan-recovery operations so callers
   construct one instance per namespace and use methods on it rather
   than threading the namespace string through every call site.

3. **Contract types** — ``SupportsComponentProbe`` (Protocol) and
   ``ServiceStateBase`` (ABC). Subcommand CLIs implement their own
   ServiceState dataclass that satisfies the protocol / base class, and
   in return get to plug into scaffold's ``ServiceLifecycle`` /
   ``StatusReporter`` / etc. without further glue.
"""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

DEFAULT_SERVICE = "default"


# ---------------------------------------------------------------------------
# Global helpers
# ---------------------------------------------------------------------------


def areal_home() -> Path:
    """Return the AReaL CLI home directory.

    Resolves ``$AREAL_HOME`` if set, otherwise ``~/.areal``. Created on
    first access so callers can mkdir-then-write subdirs without an
    explicit setup step.
    """

    env = os.environ.get("AREAL_HOME")
    root = Path(env).expanduser() if env else Path.home() / ".areal"
    root.mkdir(parents=True, exist_ok=True)
    return root


def atomic_write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    """Atomically write ``data`` as JSON to ``path``.

    Writes to a unique tempfile in ``path``'s directory, fsync()s it to
    disk, then renames into place. ``NamedTemporaryFile(delete=False)``
    gives us a fresh name per call so concurrent writers do not stomp on
    each other's tempfiles, and the tempfile is unlinked on serialization
    or rename failure so half-written state never lingers on disk.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, suffix=".tmp"
    ) as f:
        tmp_path = Path(f.name)
        try:
            json.dump(data, f, indent=indent, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Namespace-bound state store
# ---------------------------------------------------------------------------


class NamespacedStateStore:
    """All ``$AREAL_HOME/<namespace>/...`` path resolution + pointer
    file management for one subcommand CLI.

    Construct one per CLI (e.g. ``store = NamespacedStateStore("inf")``)
    and use the methods on it. Paths are re-resolved on every call so
    tests that override the home directory via ``AREAL_HOME`` work
    without rebuilding the store.
    """

    def __init__(self, namespace: str) -> None:
        if not namespace:
            raise ValueError("NamespacedStateStore requires a non-empty namespace")
        self.namespace = namespace

    # --- directory roots -------------------------------------------------

    def root(self) -> Path:
        d = areal_home() / self.namespace
        d.mkdir(parents=True, exist_ok=True)
        return d

    def services_dir(self) -> Path:
        d = self.root() / "services"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def logs_root(self) -> Path:
        d = self.root() / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def logs_dir(self, service: str) -> Path:
        d = self.logs_root() / service
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --- file paths ------------------------------------------------------

    def service_state_path(self, service: str) -> Path:
        return self.services_dir() / f"{service}.json"

    def service_lock_path(self, service: str) -> Path:
        return self.services_dir() / f"{service}.lock"

    def current_service_path(self) -> Path:
        return self.root() / "current-service"

    def config_path(self) -> Path:
        return self.root() / "config.toml"

    # --- pointer file ----------------------------------------------------

    def list_service_names(self) -> list[str]:
        return sorted(p.stem for p in self.services_dir().glob("*.json"))

    def set_current_service(self, service: str) -> None:
        self.current_service_path().write_text(service + "\n")

    def clear_current_service(self, service: str) -> None:
        path = self.current_service_path()
        if path.exists() and path.read_text().strip() == service:
            path.unlink()

    def resolve_service_name(
        self,
        explicit: str | None = None,
        *,
        fallback: str = DEFAULT_SERVICE,
    ) -> str:
        """Resolve the active service name for a CLI call.

        Order: ``--service`` flag > current-service pointer file > the
        single running service (if exactly one) > ``fallback``.
        """

        if explicit:
            return explicit
        pointer = self.current_service_path()
        if pointer.exists():
            value = pointer.read_text().strip()
            if value:
                return value
        running = self.list_service_names()
        if len(running) == 1:
            return running[0]
        return fallback

    # --- best-effort orphan PID recovery ---------------------------------

    def recover_pids_from_raw_state(self, service: str) -> list[int]:
        """Walk the service-state file for ``service`` and pull any
        ``pid`` / ``pids`` numbers.

        Used by ``run --force`` to clean up children when the dataclass
        parse fails (state file from an older / corrupted schema).
        Subclasses with extra state files (e.g. inf's
        ``models/<svc>.json``) override to walk the additional files.
        """

        pids: list[int] = []
        pid_keys = {"pid", "pids"}

        def add(value) -> None:
            if isinstance(value, int) and value > 0:
                pids.append(value)
            elif isinstance(value, list):
                for item in value:
                    add(item)

        def walk(value) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in pid_keys:
                        add(item)
                    else:
                        walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        path = self.service_state_path(service)
        if path.exists():
            with open(path) as f:
                walk(json.load(f))

        seen: set[int] = set()
        unique: list[int] = []
        for pid in pids:
            if pid not in seen:
                seen.add(pid)
                unique.append(pid)
        return unique


# ---------------------------------------------------------------------------
# Contract types
# ---------------------------------------------------------------------------


@runtime_checkable
class SupportsComponentProbe(Protocol):
    """Minimum surface a subcommand's component handle must expose so
    scaffold helpers can probe it and identify it.

    Each subcommand CLI's handle type — inf's ``TaskHandle``, agent's
    ``ProcessState``, etc. — already provides ``addr`` (HTTP base for
    ``/health`` probes) and ``pid`` (for local liveness / kill paths).
    No inheritance needed: structural duck typing.
    """

    @property
    def addr(self) -> str: ...

    @property
    def pid(self) -> int: ...


class ServiceStateBase(ABC):
    """Abstract base every subcommand CLI's ServiceState should satisfy.

    The base nails down the universal fields and the two methods
    (``gateway_alive`` / ``components``) that scaffold's lifecycle and
    status reporters rely on. Subclasses are free to add backend-specific
    fields (engine handles, model registries, agent pair configs, etc.).
    """

    service: str
    admin_api_key: str
    launch_mode: str
    started_at: float

    @abstractmethod
    def gateway_alive(self) -> bool:
        """Return True iff the service's central entry point is reachable.

        ``ServiceLifecycle`` uses this to decide "running" — every CLI
        must define what alive means for its own architecture (local PID
        alive / k8s pod ready / slurm job state).
        """

    @abstractmethod
    def components(self) -> Iterable[tuple[str, SupportsComponentProbe]]:
        """Yield ``(label, handle)`` for every component of this service.

        Used by ``StatusReporter`` to enumerate rows; the order is the
        order rows appear in the table. Labels are display-only strings
        (e.g. ``"gateway"``, ``"worker[qwen/0]"``).
        """
