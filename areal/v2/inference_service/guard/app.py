# SPDX-License-Identifier: Apache-2.0

"""RPCGuard — inference service guard backed by the shared guard.

The inference guard extends the shared Guard's configuration hook with
node-local SGLang cache staging.
"""

from __future__ import annotations

import re
from pathlib import Path

from areal.infra.rpc.guard.app import (
    GuardState,
    create_app,
)
from areal.infra.rpc.guard.app import (
    cleanup_forked_children as _cleanup_impl,
)
from areal.utils import logging

logger = logging.getLogger("RPCGuard")

_state = GuardState()

app = create_app(_state)


def _configure_jit_cache(data: dict) -> dict:
    if "sglang_jit_cache" not in data:
        return {}
    from areal.v2.inference_service.sglang.jit_cache import stage_cache

    options = data["sglang_jit_cache"]
    directory = options["directory"]
    if options["action"] not in ("export", "install") or not re.fullmatch(
        r"\.sglang-jit-[a-z0-9_]+", directory
    ):
        raise ValueError("Invalid SGLang JIT cache operation")
    root = logging.get_experiment_log_path(
        _state.experiment_name, _state.trial_name, _state.fileroot
    )
    with _state.forked_children_lock:
        pids = [
            p.pid
            for (role, _), p in _state.forked_children_map.items()
            if role == "inf-server" and p.poll() is None
        ]
    return stage_cache(options["action"], Path(root) / directory, pids)


_state.register_configure_hook(_configure_jit_cache)


def cleanup_forked_children() -> None:
    _cleanup_impl(_state)
