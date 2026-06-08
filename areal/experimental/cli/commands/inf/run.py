# SPDX-License-Identifier: Apache-2.0

"""``areal inf run`` — launch the v2 inference service (detached).

Spawns an ``inf_supervisor`` subprocess that owns a ``RolloutControllerV2``,
which in turn manages the sglang workers + router + gateway + data-proxies.
The CLI process exits as soon as the supervisor reports ``ready``; the
supervisor stays up until ``areal inf stop`` (SIGTERM) tears it down via the
same ``controller.destroy()`` path that ``PPOTrainer.close()`` uses on
normal training shutdown.

While waiting for ready, the CLI streams the supervisor's ``main.log`` to
stderr so the user sees real-time progress (scheduler, worker spawn, sglang
load).  If the supervisor reports ``failed`` (init exception), the CLI exits
with the recorded reason instead of waiting for the launch timeout.

Logs follow the v2 training layout:
    {fileroot}/logs/{user}/{experiment_name}/{trial_name}/
        main.log               <- supervisor / driver
        inf-server.log         <- v2 worker (sglang)
        router.log / gateway.log / data-proxy*.log
        merged.log

Run name comes from ``experiment_name``/``trial_name`` in the yaml (or
their Hydra overrides) — same convention as ``areal run``.

Examples:
  areal inf run --config experiments/grpo.yaml
  areal inf run --config experiments/grpo.yaml --force
  areal inf run --config experiments/grpo.yaml trial_name=infer-only
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


_DESCRIPTION = __doc__


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run",
        help="Launch the v2 inference service (detached).",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, type=Path)
    p.add_argument(
        "--launch-timeout", type=float, default=600.0,
        help="Seconds to wait for the supervisor to become ready (sglang load can be slow).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Stop an existing service with the same name first.",
    )
    p.add_argument(
        "overrides", nargs=argparse.REMAINDER,
        help="Hydra-style overrides forwarded to the supervisor.",
    )
    p.set_defaults(func=_handle)


def _resolve_name(config_path: Path, overrides: list[str]) -> str:
    from areal.experimental.cli.runner import resolve_name

    return resolve_name(config_path, overrides=overrides)


def _refuse_or_replace(name: str, force: bool) -> None:
    from areal.experimental.cli.inf_state import (
        ServiceState,
        service_state_path,
        supervisor_alive,
    )

    p = service_state_path(name)
    if not p.exists():
        return
    try:
        existing = ServiceState.load(name)
    except (FileNotFoundError, ValueError, TypeError):
        return
    if supervisor_alive(existing):
        if not force:
            raise SystemExit(
                f"Service {name!r} is already running (supervisor pid={existing.supervisor_pid}). "
                f"Use --force to replace it, or `areal inf stop {name}` first."
            )
        os.kill(existing.supervisor_pid, signal.SIGTERM)
        deadline = time.time() + 30.0
        while time.time() < deadline and supervisor_alive(existing):
            time.sleep(0.5)
        if supervisor_alive(existing):
            try:
                os.kill(existing.supervisor_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    existing.remove()


def _spawn_supervisor(
    name: str, config_path: Path, overrides: list[str]
) -> int:
    """Spawn the supervisor detached.

    Supervisor itself redirects stdout/stderr to the v2-aligned ``main.log``
    inside the run, so we just discard whatever it emits to fd 1/2 BEFORE
    that redirect happens.
    """
    cmd = [
        sys.executable, "-m", "areal.experimental.cli.inf_supervisor",
        "--name", name,
        "--config", str(config_path),
        "--",
        *overrides,
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )
    return proc.pid


class _LogTailer:
    """Stream a (possibly not-yet-existing) log file to stderr line-by-line."""

    def __init__(self, path: Path):
        self.path = path
        self._fp = None
        self._buf = b""

    def _try_open(self) -> bool:
        if self._fp is not None:
            return True
        if not self.path.exists():
            return False
        try:
            self._fp = open(self.path, "rb")
        except OSError:
            return False
        return True

    def drain(self) -> None:
        if not self._try_open():
            return
        try:
            chunk = self._fp.read()
        except OSError:
            return
        if not chunk:
            return
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            try:
                sys.stderr.write(line.decode("utf-8", errors="replace") + "\n")
            except Exception:
                pass
        sys.stderr.flush()

    def close(self) -> None:
        if self._fp is not None:
            self.drain()
            try:
                self._fp.close()
            except Exception:
                pass


def _wait_ready(
    name: str, supervisor_pid: int, timeout_s: float, log_path: Path
) -> None:
    from areal.experimental.cli.inf_state import (
        service_failed_marker,
        service_ready_marker,
    )
    from areal.experimental.cli.state import pid_alive

    ready = service_ready_marker(name)
    failed = service_failed_marker(name)
    tailer = _LogTailer(log_path)

    deadline = time.time() + timeout_s
    try:
        while time.time() < deadline:
            tailer.drain()

            if failed.exists():
                reason = failed.read_text().strip() or "(unknown)"
                raise SystemExit(
                    f"Service {name!r} failed during init: {reason}\n"
                    f"See {log_path} for the full traceback."
                )
            if ready.exists():
                return
            if not pid_alive(supervisor_pid):
                # Process died WITHOUT writing the failed marker (segfault, OOM,
                # SIGKILL ...).  Let the user see the tail of the log either way.
                tailer.drain()
                raise SystemExit(
                    f"Supervisor for {name!r} (pid {supervisor_pid}) died "
                    f"before becoming ready. See {log_path}."
                )
            time.sleep(0.5)
        raise SystemExit(
            f"Service {name!r} did not become ready within {timeout_s:.0f}s. "
            f"See {log_path}."
        )
    finally:
        tailer.close()


def _peek_log_dir(config_path: Path, overrides: list[str]) -> Path:
    """Resolve the supervisor's log dir BEFORE spawning the supervisor.

    Done so the parent CLI can tail ``main.log`` from the moment the
    supervisor starts redirecting to it.
    """
    from areal.experimental.cli.inf_config import load_inference_config
    from areal.experimental.cli.inf_state import service_log_dir_for_config

    config, _ = load_inference_config(config_path, overrides)
    return service_log_dir_for_config(config)


def _handle(args: argparse.Namespace) -> int:
    from areal.experimental.cli.inf_state import (
        ServiceState,
        get_current_service,
        set_current_service,
    )

    config_path = args.config.expanduser().resolve()
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    overrides = args.overrides or []
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]

    name = _resolve_name(config_path, overrides)
    _refuse_or_replace(name, force=args.force)

    log_dir = _peek_log_dir(config_path, overrides)
    main_log = log_dir / "main.log"

    print(f"Starting service {name!r} ...", file=sys.stderr)
    print(f"  log dir: {log_dir}", file=sys.stderr)
    pid = _spawn_supervisor(name, config_path, overrides)
    print(f"  supervisor pid: {pid}", file=sys.stderr)
    print(
        f"  waiting up to {args.launch_timeout:.0f}s for /ready (streaming main.log) ...",
        file=sys.stderr,
    )
    print("  --------- supervisor output ---------", file=sys.stderr)

    try:
        _wait_ready(name, pid, args.launch_timeout, main_log)
    except SystemExit:
        if get_current_service() == name:
            set_current_service(None)
        raise

    state = ServiceState.load(name)
    if get_current_service() is None:
        set_current_service(state.name)
    print("  -------- supervisor ready --------", file=sys.stderr)
    print(f"\nService {name!r} ready.")
    print(f"  gateway: {state.gateway_addr or '(n/a)'}")
    print(f"  router:  {state.router_addr or '(n/a)'}")
    if state.server_addrs:
        print(f"  servers: {len(state.server_addrs)}")
        for addr in state.server_addrs[:4]:
            print(f"    - {addr}")
        if len(state.server_addrs) > 4:
            print(f"    ... (+{len(state.server_addrs) - 4} more)")
    print(f"  supervisor: pid={state.supervisor_pid}")
    print(f"  log:        {main_log}")
    return 0
