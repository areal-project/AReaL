# SPDX-License-Identifier: Apache-2.0

"""Cross-cutting state helpers shared by every sub-CLI.

The AReaL CLI persists a small amount of local state under ``~/.areal/``
(overridable with ``$AREAL_HOME``). Each sub-CLI owns a subdirectory with
the same internal shape, so commands can find services and jobs across
invocations without a background daemon:

    ~/.areal/
    ├── inf/
    │   ├── current-service
    │   ├── services/<name>.json
    │   └── logs/<name>/
    ├── agent/
    │   ├── current-service
    │   ├── services/<name>.json
    │   └── logs/<name>/
    ├── train/
    │   ├── current-run
    │   ├── runs/<name>.json
    │   └── logs/<name>/
    └── weight-update/
        ├── current-service
        ├── services/<name>.json
        └── logs/<name>/

The helpers in this module are deliberately tiny — atomic file write,
PID liveness, and the ``~/.areal/`` resolver. Per-namespace dataclasses
(``ServiceState``, ``RunState``, etc.) live alongside the verb files
that introduce them, not here.

This module must stay import-light. Do NOT add dependencies on torch,
aiohttp, fastapi, or any other heavy package — the lightness guard test
in ``tests/experimental/test_cli_lightness.py`` will reject the change.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def areal_home() -> Path:
    """Return the root state directory (``$AREAL_HOME`` or ``~/.areal``).

    Creates the directory if it doesn't exist. Sub-CLIs derive their own
    subdirectories from here; they should never bypass this function and
    hardcode ``~/.areal`` themselves.
    """
    env = os.environ.get("AREAL_HOME")
    root = Path(env).expanduser() if env else Path.home() / ".areal"
    root.mkdir(parents=True, exist_ok=True)
    return root


def namespace_dir(namespace: str) -> Path:
    """Return ``~/.areal/<namespace>/`` (created on demand).

    ``namespace`` is the on-disk directory name and follows the CLI
    verb prefix (``inf``, ``agent``, ``train``, ``weight-update``). The
    hyphenated form is used on disk to match the CLI surface exactly.
    """
    d = areal_home() / namespace
    d.mkdir(parents=True, exist_ok=True)
    return d


def pid_alive(pid: int) -> bool:
    """Cheap liveness probe: does *pid* still exist on this host?

    Uses ``kill(pid, 0)`` which signals nothing but raises ``ProcessLookupError``
    if the process is gone. Returns ``False`` for ``pid <= 0``. Note: a live
    PID does not mean the service is healthy — pair this with an HTTP probe
    when callers need real health status.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists but is owned by another user.
        return True
    return True


def atomic_write_text(path: Path, content: str) -> None:
    """Write *content* to *path* atomically (write-temp + rename).

    Prevents readers from observing a partially written file when two CLI
    invocations race on the same state file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    """Convenience wrapper: serialize *data* to JSON and write atomically."""
    atomic_write_text(path, json.dumps(data, indent=indent) + "\n")
