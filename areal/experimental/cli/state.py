# SPDX-License-Identifier: Apache-2.0

"""Cross-cutting state helpers shared by every sub-CLI.

Layout under ``$AREAL_HOME`` (default ``~/.areal``):

    runs/<name>.json              top-level training runs
    runs/<name>.log               their captured stdout/stderr
    inf/services/<name>.json      inference service instances
    inf/logs/<name>/              per-service logs
    agent/services/<name>.json    agent service instances
    weight-update/...             diagnostics state
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def areal_home() -> Path:
    env = os.environ.get("AREAL_HOME")
    root = Path(env).expanduser() if env else Path.home() / ".areal"
    root.mkdir(parents=True, exist_ok=True)
    return root


def namespace_dir(namespace: str) -> Path:
    d = areal_home() / namespace
    d.mkdir(parents=True, exist_ok=True)
    return d


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(data, indent=indent) + "\n")


# ---- top-level training run state ----------------------------------------

def runs_dir() -> Path:
    d = areal_home() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sanitize_name(name: str) -> str:
    return name.replace("/", "__").replace(" ", "_")


def run_state_path(name: str) -> Path:
    return runs_dir() / f"{sanitize_name(name)}.json"


def run_log_path(name: str) -> Path:
    return runs_dir() / f"{sanitize_name(name)}.log"


@dataclass
class RunState:
    name: str
    driver: str
    config_path: str
    pid: int
    started_at: float
    status: str = "running"  # running | stopped | completed | failed
    log_path: str = ""
    scheduler_type: str | None = None
    overrides: list[str] = field(default_factory=list)
    argv: list[str] = field(default_factory=list)

    def save(self) -> None:
        atomic_write_json(run_state_path(self.name), asdict(self))

    @classmethod
    def load(cls, name: str) -> RunState:
        p = run_state_path(name)
        if not p.exists():
            raise FileNotFoundError(f"No run state for {name!r} at {p}")
        with open(p) as f:
            return cls(**json.load(f))

    def remove(self) -> None:
        p = run_state_path(self.name)
        if p.exists():
            p.unlink()
