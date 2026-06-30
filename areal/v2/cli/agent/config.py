# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from areal.v2.cli.agent.state import AGENT_NAMESPACE
from areal.v2.cli.config import BindingMap, ConfigLoader

# (section, key) -> (verb_or_verbs, click_option_name).
# Anything not listed in the agent's TOML is silently ignored.
AGENT_BINDINGS: BindingMap = {
    ("default", "service"): (("run", "stop", "status", "ps", "logs"), "service"),
    ("default", "admin_api_key"): ("run", "admin_api_key"),
    ("default", "log_level"): ("run", "log_level"),
    ("run", "agent"): ("run", "agent"),
    ("run", "num_pairs"): ("run", "num_pairs"),
    ("run", "setup_timeout"): ("run", "setup_timeout"),
    ("run", "health_poll_interval"): ("run", "health_poll_interval"),
    ("run", "drain_timeout"): ("run", "drain_timeout"),
    ("run", "session_timeout"): ("run", "session_timeout"),
}


agent_config_loader = ConfigLoader(namespace=AGENT_NAMESPACE, bindings=AGENT_BINDINGS)


def load_click_default_map(extra: Path | None = None) -> dict:
    return agent_config_loader.load_click_default_map(extra=extra)
