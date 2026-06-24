# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def areal_home() -> Path:
    """Return the AReaL CLI home directory.

    Resolves ``$AREAL_HOME`` if set, otherwise ``~/.areal``. The directory
    is created on first access so callers can mkdir-then-write subdirs
    without an explicit setup step.
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
            json.dump(data, f, indent=indent)
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
