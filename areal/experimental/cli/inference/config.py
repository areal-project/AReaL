# SPDX-License-Identifier: Apache-2.0

"""User config loader for `areal inf`.

~/.areal/inf/config.toml is an optional file that overrides built-in
CLI defaults.  Per design 12, sections map to verbs:

  [default]              admin_api_key, log_level (applied to all verbs)
  [launch]               run-time gateway / router / strategy / timeouts
  [scheduler]            scheduler backend selection (applied to run only)
  [register.internal]    register / inline-register defaults
  [collect]              collect verb defaults

Precedence (highest first):
  1. explicit CLI flag
  2. config.toml
  3. hard-coded default in the click @option

Missing / malformed file is treated as empty -- never crashes the CLI.
"""

from __future__ import annotations

from pathlib import Path

from areal.experimental.cli.inference.state import inf_root

# Maps each section + key in config.toml to (verb_name, click_option_name).
# This is the *whole* surface; anything not listed is silently ignored.
_BINDINGS: dict[tuple[str, str], tuple[str | tuple[str, ...], str]] = {
    # [default]
    (
        "default",
        "service",
    ): (
        (
            "run",
            "stop",
            "status",
            "register",
            "deregister",
            "models",
            "collect",
            "reward",
            "logs",
        ),
        "service",
    ),
    ("default", "admin_api_key"): ("run", "admin_api_key"),
    ("default", "log_level"): ("run", "log_level"),
    # [launch] — applied to `run`
    ("launch", "gateway_host"): ("run", "host"),
    ("launch", "gateway_port"): ("run", "port"),
    ("launch", "routing_strategy"): ("run", "routing_strategy"),
    ("launch", "launch_timeout"): ("run", "launch_timeout"),
    # [scheduler] — applied to `run` only; other verbs read from ServiceState
    ("scheduler", "type"): ("run", "scheduler"),
    # [register.internal] — applied to `register`
    ("register.internal", "backend"): ("register", "backend"),
    ("register.internal", "model_health_timeout"): ("register", "model_health_timeout"),
    ("register.internal", "engine_args"): ("register", "engine_args"),
    ("register.internal", "proxy_args"): ("register", "proxy_args"),
    # [collect]
    ("collect", "batch_size"): ("collect", "batch_size"),
    ("collect", "timeout"): ("collect", "timeout"),
    ("collect", "poll_interval"): ("collect", "poll_interval"),
    ("collect", "turn_discount"): ("collect", "turn_discount"),
    ("collect", "export_style"): ("collect", "export_style"),
    ("collect", "format"): ("collect", "format"),
}


def config_path() -> Path:
    return inf_root() / "config.toml"


def _read_toml(path: Path) -> dict:
    try:
        import tomllib  # py 3.11+
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
    """Flatten nested TOML to (section, key) -> value, joining keys with '.'."""
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
    """Return a click default_map dict from ~/.areal/inf/config.toml.

    `extra` is an optional override path (`--config FILE` from `inf run`).
    Its values take precedence over the user config but are still
    overridden by explicit CLI flags.
    """
    merged: dict[tuple[str, str], object] = {}
    for path in (config_path(), extra):
        if path is None:
            continue
        flat = _flatten(_read_toml(path))
        merged.update(flat)

    # Re-bucket into click's {verb: {option: value}} shape.
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
