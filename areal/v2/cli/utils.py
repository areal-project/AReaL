# SPDX-License-Identifier: Apache-2.0

"""Stateless utility helpers shared across subcommand CLIs.

Four groups live here:

- :func:`file_lock` — per-service mutual exclusion via POSIX flock.
- :func:`wait_http_health` / :func:`wait_client_health` — block until a
  newly-spawned component reports ready.
- :func:`json_or_table` — ``--json`` vs table dual-output dispatch.
- :func:`register_cli_logger` — register a CLI logger name + color in
  the global color table and return the configured Logger.
"""

from __future__ import annotations

import fcntl
import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

import click

from areal.utils.logging import LOGGER_COLORS_EXACT, getLogger
from areal.v2.cli.process import pid_alive

# ---------------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------------


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive flock on ``path`` for the duration of the ``with`` block.

    The lock file is created if missing; concurrent waiters block until
    the current holder exits the ``with`` block. Used by subcommand CLIs
    to serialize concurrent ``register`` / ``deregister`` / ``stop``
    calls against the same service's state file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Health polling
# ---------------------------------------------------------------------------


class _HealthCheckable(Protocol):
    def health(self, *, timeout: float) -> object: ...


def wait_http_health(
    url: str,
    *,
    pid: int,
    timeout: float,
    label: str,
    poll_interval: float = 0.5,
    request_timeout: float = 2.0,
) -> None:
    """Poll ``GET <url>/health`` until it returns < 5xx.

    Raises ``ClickException`` if the deadline elapses, or if the spawned
    PID dies during startup (cheap early-fail signal for the local
    case). Pass ``pid=0`` for non-local backends to skip the PID check.
    """

    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        if pid > 0 and not pid_alive(pid):
            raise click.ClickException(f"{label} subprocess died during startup")
        try:
            with urllib.request.urlopen(
                f"{url.rstrip('/')}/health", timeout=request_timeout
            ) as resp:
                if resp.status < 500:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            last_err = exc
            time.sleep(poll_interval)
    raise click.ClickException(f"{label} did not become healthy: {last_err}")


def wait_client_health(
    client: _HealthCheckable,
    *,
    timeout: float,
    label: str,
    poll_interval: float = 0.3,
    request_timeout: float = 1.5,
) -> None:
    """Poll ``client.health(timeout=...)`` until it returns without raising.

    Any exception from ``client.health`` is treated as "not ready yet"
    and the loop sleeps until the deadline.
    """

    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            client.health(timeout=request_timeout)
            return
        except Exception as exc:
            last_err = exc
            time.sleep(poll_interval)
    raise click.ClickException(
        f"{label} did not become healthy within {timeout:.0f}s (last error: {last_err})"
    )


# ---------------------------------------------------------------------------
# Output dispatch
# ---------------------------------------------------------------------------


def json_or_table(
    payload: Any,
    *,
    as_json: bool,
    table_renderer: Callable[[Any], None],
    indent: int = 2,
) -> None:
    """Emit ``payload`` as JSON or hand it to a table renderer.

    Most subcommand ``ps`` / ``status`` / ``models`` verbs follow the
    same pattern: ``--json`` dumps the raw payload, otherwise a table
    is rendered. Funneling both through this helper keeps the branching
    out of the command body.
    """

    if as_json:
        click.echo(json.dumps(payload, indent=indent, default=str))
        return
    table_renderer(payload)


# ---------------------------------------------------------------------------
# Logger registration
# ---------------------------------------------------------------------------


def register_cli_logger(name: str, color: str = "blue") -> logging.Logger:
    """Register ``name`` with ``color`` in ``areal.utils.logging``'s
    color table and return the configured Logger.

    Service-style CLIs conventionally use ``"blue"`` (infrastructure /
    scheduler category). See ``areal/utils/logging.py`` for the palette
    if a different color fits.
    """

    LOGGER_COLORS_EXACT[name] = color
    return getLogger(name)
