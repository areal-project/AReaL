# SPDX-License-Identifier: Apache-2.0

"""``areal inf run`` — launch the v2 inference service (detached).

Spawns an ``inf_supervisor`` subprocess that owns a ``RolloutControllerV2``,
which in turn manages the sglang workers + router + gateway + data-proxies.
The CLI process exits as soon as the supervisor reports ``ready``; the
supervisor stays up until ``areal inf stop`` (SIGTERM) tears it down via the
same ``controller.destroy()`` path that ``PPOTrainer.close()`` uses on
normal training shutdown.

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
    name: str, config_path: Path, overrides: list[str], log_file: Path
) -> int:
    cmd = [
        sys.executable, "-m", "areal.experimental.cli.inf_supervisor",
        "--name", name,
        "--config", str(config_path),
        "--",
        *overrides,
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lf = open(log_file, "ab", buffering=0)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=lf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    return proc.pid


def _wait_ready(name: str, supervisor_pid: int, timeout_s: float) -> None:
    from areal.experimental.cli.inf_state import service_ready_marker
    from areal.experimental.cli.state import pid_alive

    marker = service_ready_marker(name)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if marker.exists():
            return
        if not pid_alive(supervisor_pid):
            raise SystemExit(
                f"Supervisor for {name!r} (pid {supervisor_pid}) died before becoming ready. "
                f"See log."
            )
        time.sleep(0.5)
    raise SystemExit(
        f"Service {name!r} did not become ready within {timeout_s:.0f}s. "
        f"See supervisor log."
    )


def _handle(args: argparse.Namespace) -> int:
    from areal.experimental.cli.inf_state import (
        ServiceState,
        service_logs_dir,
        set_current_service,
        get_current_service,
    )

    config_path = args.config.expanduser().resolve()
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    overrides = args.overrides or []
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]

    name = _resolve_name(config_path, overrides)
    _refuse_or_replace(name, force=args.force)

    logs = service_logs_dir(name)
    log_file = logs / "supervisor.log"

    print(f"Starting service {name!r} ...", file=sys.stderr)
    pid = _spawn_supervisor(name, config_path, overrides, log_file)
    print(f"  supervisor pid: {pid}", file=sys.stderr)
    print(f"  log:            {log_file}", file=sys.stderr)
    print(
        f"  waiting up to {args.launch_timeout:.0f}s for /ready ...",
        file=sys.stderr,
    )

    try:
        _wait_ready(name, pid, args.launch_timeout)
    except SystemExit:
        if get_current_service() == name:
            set_current_service(None)
        raise

    state = ServiceState.load(name)
    if get_current_service() is None:
        set_current_service(state.name)
    print(f"\nService {name!r} ready.")
    print(f"  gateway: {state.gateway_addr or '(n/a)'}")
    print(f"  router:  {state.router_addr or '(n/a)'}")
    if state.server_addrs:
        print(f"  servers: {len(state.server_addrs)}")
        for addr in state.server_addrs[:4]:
            print(f"    - {addr}")
        if len(state.server_addrs) > 4:
            print(f"    ... (+{len(state.server_addrs) - 4} more)")
    print(f"  supervisor: pid={state.supervisor_pid}, log={log_file}")
    return 0
