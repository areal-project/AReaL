# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from areal.experimental.cli.inference.state import (
    INF_NAMESPACE,
    ModelState,
    RuntimeState,
    ServiceState,
    recover_pids_from_raw_state,
)
from areal.experimental.cli.lifecycle import ServiceLifecycle
from areal.experimental.cli.process import kill_pids


class InferenceLifecycle(ServiceLifecycle):
    """Adds the inf-specific behavior on top of scaffold's lifecycle:

    - State spans two files (service + model registry). ``force_replace_slot``
      walks the inf raw-JSON helper which knows about both files, and
      removes both on cleanup.
    """

    def force_replace_slot(self, service: str, *, grace_s: float = 5.0) -> None:
        path = self.state_path(service)
        if not path.exists():
            return
        pids: list[int] = []
        try:
            state = self.load_state(service)
            pids = self._collect_pids(state)
        except Exception:
            try:
                pids = recover_pids_from_raw_state(service)
            except Exception:
                pids = []
        if pids:
            kill_pids(pids, grace_s=grace_s)
        ServiceState.remove(service)
        ModelState.remove(service)


inf_lifecycle = InferenceLifecycle(
    namespace=INF_NAMESPACE,
    state_class=RuntimeState,
    stop_command="areal inf stop",
)
