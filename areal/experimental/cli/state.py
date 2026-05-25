# SPDX-License-Identifier: Apache-2.0

"""State files for runs launched via the AReaL CLI.

Layout under ``$AREAL_HOME`` (default ``~/.areal``)::

    runs/<sanitized-name>.json    state file
    logs/<sanitized-name>.log     captured stdout/stderr for background runs

The state file records the launching command, driver entry, config path,
PID, and last observed status. Concurrent re-launches under the same name
are refused if the recorded PID is still alive.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def areal_home() -> Path:
    home = os.environ.get("AREAL_HOME")
    if home:
        return Path(home).expanduser()
    return Path.home() / ".areal"


def runs_dir() -> Path:
    d = areal_home() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    d = areal_home() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sanitize_name(name: str) -> str:
    return name.replace("/", "__").replace(" ", "_")


def state_path(name: str) -> Path:
    return runs_dir() / f"{sanitize_name(name)}.json"


def log_path(name: str) -> Path:
    return logs_dir() / f"{sanitize_name(name)}.log"


@dataclass
class RunState:
    name: str
    command: str  # "run" | "train" | "inf" | "agent"
    driver: str
    config_path: str
    pid: int
    started_at: float
    status: str = "running"  # running | stopped | completed | failed
    log_path: str = ""
    scheduler_type: str | None = None
    argv: list[str] = field(default_factory=list)

    def save(self) -> None:
        p = state_path(self.name)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(asdict(self), f, indent=2)
        os.replace(tmp, p)

    @classmethod
    def load(cls, name: str) -> RunState:
        p = state_path(name)
        if not p.exists():
            raise FileNotFoundError(f"No state file for {name!r} at {p}")
        with open(p) as f:
            data = json.load(f)
        return cls(**data)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we cannot signal it; treat as alive.
        return True
    return True


def list_run_names() -> list[str]:
    return sorted(p.stem for p in runs_dir().glob("*.json"))


def load_all_runs() -> list[RunState]:
    out: list[RunState] = []
    for name in list_run_names():
        try:
            out.append(RunState.load(name))
        except (ValueError, FileNotFoundError, TypeError):
            continue
    return out
