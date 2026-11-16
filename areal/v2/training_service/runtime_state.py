# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports for Elastic checkpoint runtime state."""

from areal.elastic.infra.state.worker_checkpoint import (
    capture_runtime_state as capture_runtime_state,
)
from areal.elastic.infra.state.worker_checkpoint import (
    restore_random_state as restore_random_state,
)
