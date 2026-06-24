# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from areal.experimental.cli.agent.state import agent_root

# (section, key) -> (verb_or_verbs, click_option_name).
# Anything not listed is silently ignored.
_BINDINGS: dict[tuple[str, str], tuple[str | tuple[str, ...], str]] = {
    (
        "default",
        "service",
    ): (
        (
            "run",
            "stop",
            "status",
            "ps",
            "logs",
        ),
        "service",
    ),
    ("default", "admin_api_key"): ("run", "admin_api_key"),
    ("default", "log_level"): ("run", "log_level"),
    ("run", "agent"): ("run", "agent"),
    ("run", "num_pairs"): ("run", "num_pairs"),
    ("run", "setup_timeout"): ("run", "setup_timeout"),
    ("run", "health_poll_interval"): ("run", "health_poll_interval"),
    ("run", "drain_timeout"): ("run", "drain_timeout"),
    ("run", "session_timeout"): ("run", "session_timeout"),
    ("inference", "addr"): ("run", "inf_addr"),
    ("inference", "api_key"): ("run", "inf_api_key"),
    ("inference", "model"): ("run", "inf_model"),
}


def config_path() -> Path:
    return agent_root() / "config.toml"


def _read_toml(path: Path) -> dict:
    try:
        import tomllib
    except ImportError:
        return {}
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def _flatten(toml: dict, parent: str = "") -> dict[tuple[str, str], object]:
    out: dict[tuple[str, str], object] = {}
    for k, v in toml.items():
        if isinstance(v, dict):
            sub_parent = f"{parent}.{k}" if parent else k
            for (sec, key), val in _flatten(v, sub_parent).items():
                out[(sec, key)] = val
        else:
            out[(parent, k)] = v
    return out


def load_click_default_map(extra: Path | None = None) -> dict:
    merged: dict[tuple[str, str], object] = {}
    for path in (config_path(), extra):
        if path is None:
            continue
        merged.update(_flatten(_read_toml(path)))

    default_map: dict[str, dict] = {}
    for (section, key), value in merged.items():
        binding = _BINDINGS.get((section, key))
        if binding is None:
            continue
        verbs, opt = binding
        if isinstance(verbs, str):
            verbs = (verbs,)
        for verb in verbs:
            default_map.setdefault(verb, {})[opt] = value
    return default_map
