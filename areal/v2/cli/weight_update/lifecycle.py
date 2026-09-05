# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from areal.v2.cli.lifecycle import ServiceLifecycle
from areal.v2.cli.weight_update.state import WU_NAMESPACE, ServiceState

# ``stop_command`` names the verb a user would run to clear a stale slot. This
# CLI has no stop: the gateway is the training controller's process, and the
# controller removes the state file when it shuts down. Point at the training
# side so the message is actionable rather than naming a verb that does not
# exist.
wu_lifecycle = ServiceLifecycle(
    namespace=WU_NAMESPACE,
    state_class=ServiceState,
    stop_command="stop the training job that owns it",
)
