# SPDX-License-Identifier: Apache-2.0

"""Shared Guard process: process management, port allocation, and child forking.

This module provides the base Guard functionality shared between:

- ``areal.infra.rpc.rpc_server`` (RPC server = guard + data + engine)
- ``areal.v2.inference_service.guard`` (inference service guard)

Key components:

- :class:`GuardState` — mutable shared state with hook system
- :func:`create_app` — Flask app factory with core guard routes
- :func:`make_base_parser` — CLI argument parser shared by entrypoints
- :func:`configure_state_from_args` — populate state from parsed CLI args
- :func:`run_server` — start werkzeug server with name_resolve registration
"""

from __future__ import annotations

import argparse
import errno
import getpass
import os
import signal
import socket
import subprocess
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from threading import Condition, Lock
from typing import Any

from flask import Flask, current_app, g, jsonify, request

from areal.infra.utils.proc import kill_process_tree, run_with_streaming_logs
from areal.utils import logging
from areal.utils.network import find_free_ports, format_hostport, is_port_free

logger = logging.getLogger("Guard")


class GuardState:
    """Mutable shared state for the Guard process.

    All guard-level state lives here so that both core routes and
    extension blueprints can access it via :func:`get_state`.

    The hook system allows blueprints to extend core endpoints:

    - **health hooks** — contribute extra fields to ``/health`` response
    - **configure hooks** — handle ``/configure`` payload
    - **cleanup hooks** — run during server shutdown
    """

    def __init__(self) -> None:
        # Server identity
        self.server_host: str = "0.0.0.0"
        self.server_port: int = 0

        # Experiment / trial config (used for log paths and name_resolve)
        self.experiment_name: str | None = None
        self.trial_name: str | None = None
        self.fileroot: str | None = None

        # Name-resolve config (used by run_server for service registration)
        self.name_resolve_type: str | None = None
        self.nfs_record_root: str | None = None
        self.etcd3_addr: str | None = None

        # Worker identity
        self.role: str | None = None
        self.worker_index: int = -1
        self.generation: str = uuid.uuid4().hex

        # Port tracking (thread-safe)
        self.allocated_ports: set[int] = set()
        self.port_reservations: dict[int, socket.socket] = {}
        self.fixed_worker_ports: dict[tuple[str, int], tuple[int, ...]] = {}
        self.allocated_ports_lock = Lock()

        # Forked child processes (thread-safe)
        self.forked_children: list[subprocess.Popen] = []
        self.forked_children_map: dict[tuple[str, int], subprocess.Popen] = {}
        self.forked_children_ports: dict[tuple[str, int], set[int]] = {}
        self.deleted_forked_children: set[tuple[str, int]] = set()
        self.forked_children_lock = Lock()
        self.fork_lifecycle_locks: dict[tuple[str, int], Lock] = {}
        self.fork_lifecycle_locks_lock = Lock()
        self.fork_requests_condition = Condition()
        self.accepting_fork_requests = True
        self.active_fork_requests = 0

        # Hook system — blueprints register hooks to extend core endpoints
        self._health_hooks: list[HealthHook] = []
        self._configure_hooks: list[ConfigureHook] = []
        self._cleanup_hooks: list[CleanupHook] = []

    def register_health_hook(self, hook: HealthHook) -> None:
        """Register a hook that contributes fields to ``/health`` response.

        The hook is called with no arguments and must return a dict of
        extra fields to merge into the health response.
        """
        self._health_hooks.append(hook)

    def register_configure_hook(self, hook: ConfigureHook) -> None:
        """Register a hook that handles ``/configure`` payload.

        The hook receives the full JSON dict and returns a result dict.
        Raise :class:`ValueError` for 400-worthy client errors.
        """
        self._configure_hooks.append(hook)

    def register_cleanup_hook(self, hook: CleanupHook) -> None:
        """Register a hook called during server shutdown."""
        self._cleanup_hooks.append(hook)

    @property
    def node_addr(self) -> str:
        """Return ``host:port`` string for this server (IPv6-safe)."""
        return format_hostport(self.server_host, self.server_port)


HealthHook = Callable[[], dict[str, Any]]
ConfigureHook = Callable[[dict], dict]
CleanupHook = Callable[[], None]


def get_state() -> GuardState:
    """Get the :class:`GuardState` from the current Flask app context."""
    return current_app.config["guard_state"]


def _fork_lifecycle_lock(state: GuardState, key: tuple[str, int]) -> Lock:
    """Return the stable per-worker lock that linearizes lifecycle requests."""
    with state.fork_lifecycle_locks_lock:
        return state.fork_lifecycle_locks.setdefault(key, Lock())


_FORK_LIFECYCLE_PATHS = {
    "/alloc_ports",
    "/reserve_worker_ports",
    "/fork",
    "/forked_worker_status",
    "/kill_forked_worker",
}


def begin_fork_request_drain(state: GuardState) -> None:
    """Reject new fork lifecycle requests before final Guard cleanup."""
    with state.fork_requests_condition:
        state.accepting_fork_requests = False


def wait_for_fork_request_drain(state: GuardState) -> None:
    """Wait until every admitted fork lifecycle request has completed."""
    with state.fork_requests_condition:
        while state.active_fork_requests:
            state.fork_requests_condition.wait()


def _shutdown_guard(state: GuardState, server: Any) -> None:
    """Drain lifecycle requests before final process and port cleanup."""
    begin_fork_request_drain(state)
    server.shutdown()
    server.server_close()
    wait_for_fork_request_drain(state)

    for hook in state._cleanup_hooks:
        try:
            hook()
        except Exception as e:
            logger.error(f"Error in cleanup hook: {e}")
    preserved_ports = cleanup_forked_children(state)
    cleanup_port_reservations(state, preserve_ports=preserved_ports)


_PORT_RESERVATION_PREFIX = "\0areal-port-reservation-v1-"


def _reserve_free_ports(state: GuardState, count: int) -> list[int]:
    """Atomically reserve node-wide ports for this Guard process.

    ``find_free_ports`` alone has a check-to-bind race when several Guard
    processes on one node allocate concurrently.  An abstract UNIX socket is
    used as a node-local, crash-safe lease for each candidate TCP port.  The
    socket remains open until the corresponding worker is removed.

    The caller must hold ``state.allocated_ports_lock``.
    """
    reservations: dict[int, socket.socket] = {}
    excluded_ports = set(state.allocated_ports)
    max_rounds = max(10, count * 10)

    try:
        for _ in range(max_rounds):
            remaining = count - len(reservations)
            if remaining == 0:
                break

            candidates = find_free_ports(
                remaining,
                exclude_ports=excluded_ports,
            )
            for port in candidates:
                excluded_ports.add(port)
                reservation = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    reservation.bind(f"{_PORT_RESERVATION_PREFIX}{port}")
                except OSError as exc:
                    reservation.close()
                    if exc.errno == errno.EADDRINUSE:
                        continue
                    raise

                # A non-AReaL process may have bound the TCP/UDP port between
                # candidate discovery and acquiring our cooperative lease.
                if not is_port_free(port):
                    reservation.close()
                    continue
                reservations[port] = reservation

        if len(reservations) != count:
            raise ValueError(
                f"Could only reserve {len(reservations)} node-wide ports "
                f"out of {count} requested after {max_rounds} rounds"
            )

        state.allocated_ports.update(reservations)
        state.port_reservations.update(reservations)
        return sorted(reservations)
    except Exception:
        for reservation in reservations.values():
            reservation.close()
        raise


def _release_reserved_ports_unlocked(
    state: GuardState,
    ports: set[int] | list[int],
) -> None:
    """Release port bookkeeping and node-wide leases with the lock held."""
    for port in ports:
        reservation = state.port_reservations.pop(port, None)
        if reservation is not None:
            reservation.close()
    state.allocated_ports.difference_update(ports)


def cleanup_port_reservations(
    state: GuardState,
    preserve_ports: set[int] | None = None,
) -> None:
    """Release node-wide leases except ports owned by children still alive."""
    with state.allocated_ports_lock:
        ports_to_preserve = set(preserve_ports or ())
        # Fixed worker ports are allocated as one group. Preserve the whole
        # group if any member still belongs to a child that failed to exit.
        for fixed_ports in state.fixed_worker_ports.values():
            if ports_to_preserve.intersection(fixed_ports):
                ports_to_preserve.update(fixed_ports)
        _release_reserved_ports_unlocked(
            state,
            set(state.port_reservations) - ports_to_preserve,
        )
        for key, fixed_ports in list(state.fixed_worker_ports.items()):
            if not ports_to_preserve.intersection(fixed_ports):
                state.fixed_worker_ports.pop(key, None)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def cleanup_forked_children(state: GuardState) -> set[int]:
    """Clean up all forked child processes.

    Each child's node-wide port lease remains held until that child is
    confirmed stopped. Failed terminations retain both tracking and leases so
    callers can retry without exposing the ports to another Guard.

    Returns
    -------
    set[int]
        Ports that must remain reserved because their child did not exit.
    """
    with state.forked_children_lock:
        children_to_kill = list(state.forked_children)
        for child in state.forked_children_map.values():
            if not any(existing is child for existing in children_to_kill):
                children_to_kill.append(child)

    if not children_to_kill:
        return set()

    logger.info(f"Cleaning up {len(children_to_kill)} forked child processes")
    for child in children_to_kill:
        try:
            if child.poll() is None:  # Still running
                kill_process_tree(child.pid, timeout=3, graceful=True)
            if child.poll() is None:
                raise RuntimeError("process tree is still alive after termination")
        except Exception as e:
            logger.error(f"Error killing forked child {child.pid}: {e}")
            continue

        with state.forked_children_lock:
            owned_keys = [
                key
                for key, current in state.forked_children_map.items()
                if current is child
            ]
            child_ports: set[int] = set()
            for key in owned_keys:
                state.forked_children_map.pop(key, None)
                child_ports.update(state.forked_children_ports.pop(key, set()))
                state.deleted_forked_children.add(key)
            state.forked_children = [
                current for current in state.forked_children if current is not child
            ]

        with state.allocated_ports_lock:
            for key in owned_keys:
                child_ports.update(state.fixed_worker_ports.pop(key, ()))
            _release_reserved_ports_unlocked(state, child_ports)
        logger.info(f"Killed forked child process {child.pid}")

    with state.forked_children_lock:
        remaining_keys = set(state.forked_children_map) | set(
            state.forked_children_ports
        )
        preserved_ports: set[int] = set()
        for ports in state.forked_children_ports.values():
            preserved_ports.update(ports)
    with state.allocated_ports_lock:
        for key in remaining_keys:
            preserved_ports.update(state.fixed_worker_ports.get(key, ()))
    return preserved_ports


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------


def create_app(state: GuardState) -> Flask:
    """Create a Flask app with core guard routes.

    Routes provided:

    - ``GET  /health`` — health check (extensible via health hooks)
    - ``POST /alloc_ports`` — allocate free ports
    - ``POST /fork`` — fork a child worker from a raw command
    - ``POST /forked_worker_status`` — reconcile a fork transaction key
    - ``POST /kill_forked_worker`` — kill a specific forked child
    - ``POST /configure`` — configure worker (extensible via configure hooks)

    Parameters
    ----------
    state : GuardState
        Shared mutable state for the guard process.

    Returns
    -------
    Flask
        Configured Flask application.
    """
    app = Flask(__name__)
    app.config["guard_state"] = state

    @app.before_request
    def _admit_fork_lifecycle_request():
        if request.path not in _FORK_LIFECYCLE_PATHS:
            return None
        with state.fork_requests_condition:
            if not state.accepting_fork_requests:
                return jsonify({"error": "Guard is shutting down"}), 503
            state.active_fork_requests += 1
            g.fork_lifecycle_admitted = True
        return None

    @app.teardown_request
    def _release_fork_lifecycle_request(_error):
        if not getattr(g, "fork_lifecycle_admitted", False):
            return
        g.fork_lifecycle_admitted = False
        with state.fork_requests_condition:
            assert state.active_fork_requests > 0
            state.active_fork_requests -= 1
            if state.active_fork_requests == 0:
                state.fork_requests_condition.notify_all()

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        s = get_state()
        result: dict[str, Any] = {
            "status": "healthy",
            "role": s.role,
            "worker_index": s.worker_index,
            "pid": os.getpid(),
            "generation": s.generation,
            "forked_children": len(s.forked_children),
        }
        # Collect additional fields from health hooks
        for hook in s._health_hooks:
            result.update(hook())
        return jsonify(result)

    @app.route("/alloc_ports", methods=["POST"])
    def alloc_ports():
        """Allocate multiple free ports.

        Expected JSON payload::

            {"count": 5}
        """
        try:
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({"error": "Invalid JSON in request body"}), 400

            count = data.get("count")
            if count is None:
                return jsonify({"error": "Missing 'count' field in request"}), 400

            if not isinstance(count, int) or count <= 0:
                return (
                    jsonify({"error": "'count' must be a positive integer"}),
                    400,
                )

            s = get_state()
            with s.allocated_ports_lock:
                ports = _reserve_free_ports(s, count)

            return jsonify({"status": "success", "ports": ports, "host": s.server_host})

        except Exception as e:
            logger.error(f"Error in alloc_ports: {e}\n{traceback.format_exc()}")
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    @app.route("/reserve_worker_ports", methods=["POST"])
    def reserve_worker_ports():
        """Reserve one fixed port group for a repeatedly forked worker."""
        try:
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({"error": "Invalid JSON in request body"}), 400

            role = data.get("role")
            worker_index = data.get("worker_index")
            count = data.get("count")
            if not isinstance(role, str) or not role:
                return jsonify({"error": "'role' must be a non-empty string"}), 400
            if not isinstance(worker_index, int) or worker_index < 0:
                return jsonify(
                    {"error": "'worker_index' must be a non-negative integer"}
                ), 400
            if not isinstance(count, int) or count <= 0:
                return jsonify({"error": "'count' must be a positive integer"}), 400

            s = get_state()
            key = (role, worker_index)
            with _fork_lifecycle_lock(s, key):
                with s.allocated_ports_lock:
                    ports = s.fixed_worker_ports.get(key)
                    if ports is None:
                        ports = tuple(_reserve_free_ports(s, count))
                        s.fixed_worker_ports[key] = ports
                    elif len(ports) != count:
                        return jsonify(
                            {
                                "error": (
                                    f"Fixed worker {role}/{worker_index} already has "
                                    f"{len(ports)} ports, requested {count}"
                                )
                            }
                        ), 409

            return jsonify(
                {
                    "status": "success",
                    "ports": list(ports),
                    "host": s.server_host,
                }
            )
        except Exception as e:
            logger.error(
                f"Error in reserve_worker_ports: {e}\n{traceback.format_exc()}"
            )
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    @app.route("/fork", methods=["POST"])
    def fork_worker():
        """Fork a new worker process on the same node.

        Launches the provided command list (``raw_cmd``) as-is.  The caller
        is responsible for allocating ports (via ``/alloc_ports``), building
        the full command, and polling for readiness after the response.

        Expected JSON payload::

            {
                "role": "actor",
                "worker_index": 0,
                "raw_cmd": ["python", "-m", "some.module", "--port", "8001"],
                "env": {"KEY": "value"}       // optional
            }

        Returns::

            {"status": "success", "host": "10.0.0.1", "pid": 42}
        """
        s = get_state()

        allocated_ports: list[int] = []
        key: tuple[str, int] | None = None
        try:
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({"error": "Invalid JSON in request body"}), 400

            role = data.get("role")
            worker_index = data.get("worker_index")
            raw_cmd = data.get("raw_cmd")
            allocated_ports = data.get("allocated_ports", [])

            if role is None:
                return (
                    jsonify({"error": "Missing 'role' field in request"}),
                    400,
                )
            if worker_index is None:
                return (
                    jsonify({"error": "Missing 'worker_index' field in request"}),
                    400,
                )
            if raw_cmd is None:
                return (
                    jsonify({"error": "Missing 'raw_cmd' field in request"}),
                    400,
                )
            if not isinstance(allocated_ports, list) or any(
                not isinstance(port, int) for port in allocated_ports
            ):
                return jsonify(
                    {"error": "'allocated_ports' must be a list of integers"}
                ), 400

            key = (role, worker_index)

            cmd = list(raw_cmd)

            # Optional per-process environment overrides
            env_overrides: dict[str, str] = data.get("env", {})

            logger.info(
                f"Forking new worker process for role '{role}' index {worker_index}"
            )

            # Build log paths
            log_dir = (
                Path(s.fileroot or "/tmp")
                / "logs"
                / getpass.getuser()
                / (s.experiment_name or "default")
                / (s.trial_name or "default")
            )
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{role}.log"
            merged_log = log_dir / "merged.log"

            logger.info(f"Forked worker logs will be written to: {log_file}")

            child_env = os.environ.copy()
            child_env.update(env_overrides)

            # Serialize the full lifecycle for this key without blocking other
            # workers while process termination or spawning takes place.
            with _fork_lifecycle_lock(s, key):
                with s.allocated_ports_lock:
                    fixed_ports = s.fixed_worker_ports.get(key)
                    if fixed_ports is not None and set(fixed_ports) != set(
                        allocated_ports
                    ):
                        return jsonify(
                            {
                                "error": (
                                    f"Forked worker {role}/{worker_index} does not "
                                    "match its fixed port reservation"
                                )
                            }
                        ), 409
                    if not set(allocated_ports).issubset(s.allocated_ports):
                        return jsonify(
                            {
                                "error": (
                                    f"Forked worker {role}/{worker_index} has stale "
                                    "or unreserved ports"
                                )
                            }
                        ), 409

                with s.forked_children_lock:
                    existing = s.forked_children_map.get(key)
                    if existing is not None and existing.poll() is None:
                        existing_ports = s.forked_children_ports.get(key, set())
                        if existing_ports != set(allocated_ports):
                            return jsonify(
                                {
                                    "error": (
                                        f"Forked worker {role}/{worker_index} already "
                                        "exists with different ports"
                                    )
                                }
                            ), 409
                        return jsonify(
                            {
                                "status": "success",
                                "host": s.server_host,
                                "pid": existing.pid,
                                "reused": True,
                            }
                        )
                    if existing is not None:
                        s.forked_children_map.pop(key, None)
                        s.forked_children_ports.pop(key, None)
                        try:
                            s.forked_children.remove(existing)
                        except ValueError:
                            pass

                    child_process = run_with_streaming_logs(
                        cmd,
                        log_file,
                        merged_log,
                        role,
                        env=child_env,
                    )
                    s.forked_children.append(child_process)
                    s.forked_children_map[key] = child_process
                    s.forked_children_ports[key] = set(allocated_ports)
                    s.deleted_forked_children.discard(key)

            logger.info(
                f"Forked worker for role '{role}' index "
                f"{worker_index} spawned (pid={child_process.pid})"
            )

            return jsonify(
                {
                    "status": "success",
                    "host": s.server_host,
                    "pid": child_process.pid,
                }
            )

        except Exception as e:
            if allocated_ports:
                assert key is not None
                with _fork_lifecycle_lock(s, key):
                    with s.forked_children_lock:
                        child_ports = s.forked_children_ports.get(key, set())
                    with s.allocated_ports_lock:
                        fixed_ports = set(s.fixed_worker_ports.get(key, ()))
                        transient_ports = (
                            set(allocated_ports) - fixed_ports - child_ports
                        )
                        _release_reserved_ports_unlocked(s, transient_ports)
            logger.error(f"Error in fork: {e}\n{traceback.format_exc()}")
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    @app.route("/forked_worker_status", methods=["POST"])
    def forked_worker_status():
        """Return the owner Guard's authoritative state for one fork key."""
        try:
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({"error": "Invalid JSON in request body"}), 400
            role = data.get("role")
            worker_index = data.get("worker_index")
            if not isinstance(role, str) or not role:
                return jsonify({"error": "'role' must be a non-empty string"}), 400
            if not isinstance(worker_index, int) or worker_index < 0:
                return jsonify(
                    {"error": "'worker_index' must be a non-negative integer"}
                ), 400

            s = get_state()
            key = (role, worker_index)
            with _fork_lifecycle_lock(s, key):
                with s.forked_children_lock:
                    process = s.forked_children_map.get(key)
                    exists = process is not None
                    alive = exists and process.poll() is None
                    pid = process.pid if process is not None else None
                    ports = sorted(s.forked_children_ports.get(key, set()))
            return jsonify(
                {
                    "status": "success",
                    "exists": exists,
                    "alive": alive,
                    "pid": pid,
                    "ports": ports,
                }
            )
        except Exception as e:
            logger.error(
                f"Error in forked_worker_status: {e}\n{traceback.format_exc()}"
            )
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    @app.route("/kill_forked_worker", methods=["POST"])
    def kill_forked_worker():
        """Kill a specific forked worker process.

        Expected JSON payload::

            {"role": "ref", "worker_index": 0}
        """
        s = get_state()

        try:
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({"error": "Invalid JSON in request body"}), 400

            role = data.get("role")
            worker_index = data.get("worker_index")
            release_ports = data.get("release_ports", False)

            if role is None:
                return (
                    jsonify({"error": "Missing 'role' field in request"}),
                    400,
                )
            if worker_index is None:
                return (
                    jsonify({"error": "Missing 'worker_index' field in request"}),
                    400,
                )
            if not isinstance(release_ports, bool):
                return jsonify({"error": "'release_ports' must be a boolean"}), 400

            key = (role, worker_index)

            # Keep this key linearizable across the blocking process kill. The
            # per-key lock allows unrelated workers to continue concurrently.
            with _fork_lifecycle_lock(s, key):
                # Read tracking state without removing it. Failed kills must
                # remain retryable and keep their ports reserved.
                with s.forked_children_lock:
                    child_process = s.forked_children_map.get(key)
                    already_deleted = key in s.deleted_forked_children

                if child_process is None:
                    if release_ports:
                        with s.allocated_ports_lock:
                            fixed_ports = set(s.fixed_worker_ports.pop(key, ()))
                            _release_reserved_ports_unlocked(s, fixed_ports)
                    message = (
                        f"Forked worker {role}/{worker_index} already removed"
                        if already_deleted
                        else f"Forked worker {role}/{worker_index} was not running"
                    )
                    return jsonify({"status": "success", "message": message})

                pid = child_process.pid

                try:
                    if child_process.poll() is None:  # Still running
                        kill_process_tree(pid, timeout=3, graceful=True)
                        if child_process.poll() is None:
                            raise RuntimeError(
                                "process tree is still alive after termination"
                            )
                        logger.info(
                            f"Killed forked worker {role}/{worker_index} (pid={pid})"
                        )
                except Exception as e:
                    logger.error(
                        f"Error killing forked worker "
                        f"{role}/{worker_index} (pid={pid}): {e}"
                    )
                    return (
                        jsonify(
                            {
                                "error": f"Failed to kill forked worker: {str(e)}",
                                "pid": pid,
                            }
                        ),
                        500,
                    )

                with s.forked_children_lock:
                    current_process = s.forked_children_map.get(key)
                    if current_process is not child_process:
                        return (
                            jsonify(
                                {
                                    "error": (
                                        f"Forked worker {role}/{worker_index} changed "
                                        "during termination"
                                    )
                                }
                            ),
                            409,
                        )
                    s.forked_children_map.pop(key, None)
                    child_ports = s.forked_children_ports.pop(key, set())
                    s.deleted_forked_children.add(key)
                    try:
                        s.forked_children.remove(child_process)
                    except ValueError:
                        logger.warning(
                            f"Process for {role}/{worker_index} was in map but not in list"
                        )
                with s.allocated_ports_lock:
                    fixed_ports = set(s.fixed_worker_ports.get(key, ()))
                    if release_ports:
                        fixed_ports.update(s.fixed_worker_ports.pop(key, ()))
                    ports_to_release = child_ports - fixed_ports
                    if release_ports:
                        ports_to_release.update(fixed_ports)
                    _release_reserved_ports_unlocked(s, ports_to_release)

                return jsonify(
                    {
                        "status": "success",
                        "message": (
                            f"Killed forked worker {role}/{worker_index} (pid={pid})"
                        ),
                    }
                )

        except Exception as e:
            logger.error(f"Error in kill_forked_worker: {e}\n{traceback.format_exc()}")
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    @app.route("/set_env", methods=["POST"])
    def set_env():
        """Set environment variables on the guard process.

        Forked child processes will inherit these via ``os.environ``.

        Expected JSON payload::

            {"env": {"KEY": "value", "KEY2": "value2"}}
        """
        try:
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({"error": "Invalid JSON in request body"}), 400

            env_payload = data.get("env")
            if env_payload is None:
                return jsonify({"error": "Missing 'env' field in request"}), 400
            if not isinstance(env_payload, dict):
                return jsonify({"error": "'env' must be a dictionary"}), 400

            for key, value in env_payload.items():
                os.environ[key] = str(value)

            logger.info("Updated %d environment variables", len(env_payload))
            return jsonify({"status": "success"})

        except Exception as e:
            logger.error(f"Error in set_env: {e}\n{traceback.format_exc()}")
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    @app.route("/configure", methods=["POST"])
    def configure():
        """Configure the worker process.

        Base implementation is a no-op. Blueprints register configure hooks
        to handle the payload (e.g., engine blueprint sets random seeds).

        Hooks may raise :class:`ValueError` for 400-worthy client errors.
        """
        s = get_state()

        try:
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({"error": "Invalid JSON in request body"}), 400

            if not s._configure_hooks:
                # No hooks registered — no-op (guard-only mode)
                logger.debug("Received /configure request (no-op)")
                return jsonify({"status": "ok"})

            # Dispatch to all registered configure hooks
            result: dict[str, Any] = {}
            for hook in s._configure_hooks:
                hook_result = hook(data)
                result.update(hook_result)

            result.setdefault("status", "success")
            return jsonify(result)

        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(
                f"Unexpected error in configure: {e}\n{traceback.format_exc()}"
            )
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    return app


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def make_base_parser(
    description: str = "AReaL Guard Service",
) -> argparse.ArgumentParser:
    """Create the base argument parser shared across guard-based CLIs.

    Includes: ``--host``, ``--port``, ``--experiment-name``, ``--trial-name``,
    ``--role``, ``--worker-index``, ``--name-resolve-type``,
    ``--nfs-record-root``, ``--etcd3-addr``, ``--fileroot``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to serve on (default: 0 = auto-assign)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    # Name-resolve / scheduler config
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--trial-name", type=str, required=True)
    parser.add_argument("--role", type=str, required=True)
    parser.add_argument("--worker-index", type=int, default=-1)
    parser.add_argument("--name-resolve-type", type=str, default="nfs")
    parser.add_argument(
        "--nfs-record-root", type=str, default="/tmp/areal/name_resolve"
    )
    parser.add_argument("--etcd3-addr", type=str, default="localhost:2379")
    parser.add_argument(
        "--fileroot",
        type=str,
        default=None,
        help="Root directory for log files.",
    )
    return parser


def configure_state_from_args(state: GuardState, args: argparse.Namespace) -> str:
    """Populate :class:`GuardState` from parsed CLI args.

    Returns the ``bind_host`` address for werkzeug (may differ from
    ``state.server_host`` when binding to ``0.0.0.0`` / ``::``).
    """
    from areal.utils.network import gethostip

    bind_host = args.host
    if bind_host == "0.0.0.0":
        host_ip = gethostip()
        if ":" in host_ip:
            bind_host = "::"
        state.server_host = host_ip
    elif bind_host == "::":
        state.server_host = gethostip()
    else:
        state.server_host = bind_host

    state.experiment_name = args.experiment_name
    state.trial_name = args.trial_name
    state.role = args.role
    state.fileroot = args.fileroot

    # Name-resolve config
    state.name_resolve_type = getattr(args, "name_resolve_type", "nfs")
    state.nfs_record_root = getattr(args, "nfs_record_root", "/tmp/areal/name_resolve")
    state.etcd3_addr = getattr(args, "etcd3_addr", "localhost:2379")

    # An explicit scheduler argument is authoritative.  Login shells on a
    # Slurm-allocated node can retain ``SLURM_PROCID`` even when AReaL uses the
    # local scheduler; unconditionally preferring that variable collapses all
    # local workers to index 0 and makes fork readiness identity checks fail.
    # Fall back to the Slurm task id only for launchers that omit the argument.
    worker_index = args.worker_index
    if worker_index == -1 and "SLURM_PROCID" in os.environ:
        worker_index = int(os.environ["SLURM_PROCID"])
    if worker_index == -1:
        raise ValueError("Invalid worker index. Not found from SLURM environ or args.")
    state.worker_index = worker_index

    return bind_host


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def run_server(
    state: GuardState,
    app: Flask,
    bind_host: str,
    port: int,
) -> None:
    """Start the werkzeug server and register with name_resolve.

    This is the shared server loop used by both the rpc_server and
    standalone guard entrypoints.  Handles SIGTERM, cleanup hooks,
    and forked-child cleanup on shutdown.
    """
    import logging as _logging

    from werkzeug.serving import make_server

    from areal.api.cli_args import NameResolveConfig
    from areal.utils import name_resolve, names

    _logging.getLogger("werkzeug").setLevel(_logging.WARNING)

    server = make_server(bind_host, port, app, threaded=True)
    state.server_port = server.socket.getsockname()[1]

    with state.allocated_ports_lock:
        state.allocated_ports.add(state.server_port)

    # Register with name_resolve
    if state.name_resolve_type is not None:
        name_resolve.reconfigure(
            NameResolveConfig(
                type=state.name_resolve_type,
                nfs_record_root=(state.nfs_record_root or "/tmp/areal/name_resolve"),
                etcd3_addr=state.etcd3_addr or "localhost:2379",
            )
        )

    worker_id = f"{state.role}/{state.worker_index}"
    key = names.worker_discovery(
        state.experiment_name,
        state.trial_name,
        state.role,
        state.worker_index,
    )
    name_resolve.add(key, state.node_addr, replace=True)

    logger.info(f"Starting Guard on {state.node_addr} for worker {worker_id}")

    def _sigterm_handler(signum, frame):
        """Convert SIGTERM to SystemExit so the finally block runs."""
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down (SIGINT)")
    except SystemExit:
        logger.info("Shutting down (SIGTERM)")
    finally:
        _shutdown_guard(state, server)
