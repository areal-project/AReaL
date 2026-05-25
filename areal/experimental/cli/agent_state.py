# SPDX-License-Identifier: Apache-2.0

"""State files for ``areal agent`` services.

Layout under ``$AREAL_HOME`` (default ``~/.areal``)::

    agent/
      current-service              name of the default agent service
      services/<name>.json         per-service info (PIDs, ports, agent class, ...)
      sessions/<name>.json         per-service session registry (see agent_sessions.py)
      logs/<name>/                 gateway.log, router.log, worker-<i>.log, proxy-<i>.log
      chats/<name>/<session>.jsonl chat transcripts (when --save-history)

Mirrors the inf-side state layer (``inf_state.py``) but tracks one extra
dimension: each service has N worker+data_proxy pairs, each with their own
host/port/PID. We bundle a pair into ``PairProcess`` to keep state ordered.

The CLI never holds the inference admin/session API key in plaintext: only an
``inf_api_key_present`` boolean is persisted.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from areal.experimental.cli.state import areal_home, pid_alive


def agent_dir() -> Path:
    d = areal_home() / "agent"
    d.mkdir(parents=True, exist_ok=True)
    return d


def agent_services_dir() -> Path:
    d = agent_dir() / "services"
    d.mkdir(parents=True, exist_ok=True)
    return d


def agent_logs_dir(name: str) -> Path:
    d = agent_dir() / "logs" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def agent_chats_dir(name: str) -> Path:
    d = agent_dir() / "chats" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def current_agent_service_file() -> Path:
    return agent_dir() / "current-service"


def agent_service_state_path(name: str) -> Path:
    return agent_services_dir() / f"{name}.json"


@dataclass
class PairProcess:
    """One worker + data-proxy pair launched together."""

    index: int
    worker_host: str
    worker_port: int
    worker_pid: int
    proxy_host: str
    proxy_port: int
    proxy_pid: int


@dataclass
class AgentServiceState:
    name: str
    agent_class: str
    num_pairs: int
    gateway_host: str
    gateway_port: int
    router_host: str
    router_port: int
    gateway_pid: int
    router_pid: int
    admin_api_key: str
    pairs: list[PairProcess] = field(default_factory=list)
    inf_addr: str = ""
    inf_model: str = ""
    inf_api_key_present: bool = False
    mode: str = "detached"
    log_level: str = "info"
    created_at: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def gateway_url(self) -> str:
        host = "127.0.0.1" if self.gateway_host in ("0.0.0.0", "::") else self.gateway_host
        return f"http://{host}:{self.gateway_port}"

    @property
    def router_url(self) -> str:
        host = "127.0.0.1" if self.router_host in ("0.0.0.0", "::") else self.router_host
        return f"http://{host}:{self.router_port}"

    def save(self) -> None:
        p = agent_service_state_path(self.name)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        doc = asdict(self)
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=2)
        os.replace(tmp, p)

    @classmethod
    def load(cls, name: str) -> AgentServiceState:
        p = agent_service_state_path(name)
        if not p.exists():
            raise FileNotFoundError(f"No agent service state for {name!r} at {p}")
        with open(p) as f:
            data = json.load(f)
        pairs = [PairProcess(**pp) for pp in data.pop("pairs", [])]
        return cls(pairs=pairs, **data)

    def remove(self) -> None:
        p = agent_service_state_path(self.name)
        if p.exists():
            p.unlink()


def list_agent_service_names() -> list[str]:
    return sorted(p.stem for p in agent_services_dir().glob("*.json"))


def load_all_agent_services() -> list[AgentServiceState]:
    out: list[AgentServiceState] = []
    for name in list_agent_service_names():
        try:
            out.append(AgentServiceState.load(name))
        except (ValueError, FileNotFoundError, TypeError):
            continue
    return out


def get_current_agent_service() -> str | None:
    p = current_agent_service_file()
    if not p.exists():
        return None
    name = p.read_text().strip()
    return name or None


def set_current_agent_service(name: str | None) -> None:
    p = current_agent_service_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    if name is None:
        if p.exists():
            p.unlink()
        return
    p.write_text(name + "\n")


def resolve_agent_service_name(explicit: str | None) -> str:
    """Apply design §11.1 resolution: explicit > current > sole > error."""
    if explicit:
        return explicit
    current = get_current_agent_service()
    if current:
        return current
    names = list_agent_service_names()
    if len(names) == 1:
        return names[0]
    if not names:
        raise SystemExit(
            "No agent service found. Start one with `areal agent run`."
        )
    raise SystemExit(
        "Multiple agent services exist; specify one with --service. "
        f"Known: {', '.join(names)}."
    )


def liveness_summary(state: AgentServiceState) -> dict[str, bool | list[bool]]:
    return {
        "gateway_pid_alive": pid_alive(state.gateway_pid),
        "router_pid_alive": pid_alive(state.router_pid),
        "worker_pids_alive": [pid_alive(p.worker_pid) for p in state.pairs],
        "proxy_pids_alive": [pid_alive(p.proxy_pid) for p in state.pairs],
    }
