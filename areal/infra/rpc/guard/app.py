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
import getpass
import os
import signal
import subprocess
import traceback
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import Any

from flask import Flask, current_app, jsonify, request

from areal.infra.utils.proc import kill_process_tree, run_with_streaming_logs
from areal.utils import logging
from areal.utils.network import find_free_ports, format_hostport

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

        # Port tracking (thread-safe)
        self.allocated_ports: set[int] = set()
        self.allocated_ports_lock = Lock()

        # Forked child processes (thread-safe)
        self.forked_children: list[subprocess.Popen] = []
        self.forked_children_map: dict[tuple[str, int], subprocess.Popen] = {}
        self.forked_children_lock = Lock()

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


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def cleanup_forked_children(state: GuardState) -> None:
    """Clean up all forked child processes.

    Copies the child list under the lock, then releases before blocking
    kills (avoids holding the lock for up to 4s × N children).
    """
    with state.forked_children_lock:
        if not state.forked_children:
            return
        children_to_kill = list(state.forked_children)
        state.forked_children.clear()
        state.forked_children_map.clear()

    logger.info(f"Cleaning up {len(children_to_kill)} forked child processes")
    for child in children_to_kill:
        try:
            if child.poll() is None:  # Still running
                kill_process_tree(child.pid, timeout=3, graceful=True)
                logger.info(f"Killed forked child process {child.pid}")
        except Exception as e:
            logger.error(f"Error killing forked child {child.pid}: {e}")


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------


def create_app(state: GuardState) -> Flask:
    """Create a Flask app with core guard routes.

    Routes provided:

    - ``GET  /health`` — health check (extensible via health hooks)
    - ``POST /alloc_ports`` — allocate free ports
    - ``POST /fork`` — fork a child worker from a raw command
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

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        s = get_state()
        result: dict[str, Any] = {
            "status": "healthy",
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
                ports = find_free_ports(count, exclude_ports=s.allocated_ports)
                s.allocated_ports.update(ports)

            return jsonify({"status": "success", "ports": ports, "host": s.server_host})

        except Exception as e:
            logger.error(f"Error in alloc_ports: {e}\n{traceback.format_exc()}")
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

        try:
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({"error": "Invalid JSON in request body"}), 400

            role = data.get("role")
            worker_index = data.get("worker_index")
            raw_cmd = data.get("raw_cmd")

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

            child_process = run_with_streaming_logs(
                cmd,
                log_file,
                merged_log,
                role,
                env=child_env,
            )

            with s.forked_children_lock:
                s.forked_children.append(child_process)
                s.forked_children_map[(role, worker_index)] = child_process

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
            logger.error(f"Error in fork: {e}\n{traceback.format_exc()}")
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

            key = (role, worker_index)

            # Remove from tracking structures (hold lock only for dict/list ops)
            with s.forked_children_lock:
                child_process = s.forked_children_map.pop(key, None)
                if child_process:
                    try:
                        s.forked_children.remove(child_process)
                    except ValueError:
                        logger.warning(
                            f"Process for {role}/{worker_index} was in map "
                            "but not in list"
                        )

            if child_process is None:
                return (
                    jsonify(
                        {"error": (f"Forked worker {role}/{worker_index} not found")}
                    ),
                    404,
                )

            pid = child_process.pid

            # Kill process tree (outside lock to avoid blocking)
            try:
                if child_process.poll() is None:  # Still running
                    kill_process_tree(pid, timeout=3, graceful=True)
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

    # Worker index (SLURM override)
    worker_index = args.worker_index
    if "SLURM_PROCID" in os.environ:
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
        # Run registered cleanup hooks (engine cleanup, perf_tracer, etc.)
        for hook in state._cleanup_hooks:
            try:
                hook()
            except Exception as e:
                logger.error(f"Error in cleanup hook: {e}")
        cleanup_forked_children(state)
        server.shutdown()
