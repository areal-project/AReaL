# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports for Elastic switch checkpoints."""

from areal.elastic.infra.state.worker_checkpoint import (
    Backend as Backend,
)
from areal.elastic.infra.state.worker_checkpoint import (
    capture_runtime_state as capture_runtime_state,
)
from areal.elastic.infra.state.worker_checkpoint import (
    inspect_full_checkpoint as inspect_full_checkpoint,
)
from areal.elastic.infra.state.worker_checkpoint import (
    load_full_checkpoint as load_full_checkpoint,
)
from areal.elastic.infra.state.worker_checkpoint import (
    restore_random_state as restore_random_state,
)
from areal.elastic.infra.state.worker_checkpoint import (
    save_full_checkpoint as save_full_checkpoint,
)
