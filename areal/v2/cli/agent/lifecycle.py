# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from areal.v2.cli.agent.state import AGENT_NAMESPACE, ServiceState
from areal.v2.cli.lifecycle import ServiceLifecycle

agent_lifecycle = ServiceLifecycle(
    namespace=AGENT_NAMESPACE,
    state_class=ServiceState,
    stop_command="areal agent stop",
)
