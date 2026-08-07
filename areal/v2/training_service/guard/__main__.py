# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint: ``python -m areal.v2.training_service.guard``"""

from __future__ import annotations

import os

# Consume DTE_ACTOR_ALLOC_CONF before any torch/CUDA import. The submit
# scripts cannot put PYTORCH_CUDA_ALLOC_CONF into actor.scheduling_spec
# env_vars directly: rollout.scheduling_spec interpolates ${actor.scheduling_spec},
# so the value would leak into the SGLang server env, where torch_memory_saver
# rejects expandable_segments at startup. Only the train guard (this entrypoint)
# applies it; forked engine workers inherit it.
_alloc_conf = os.environ.get("DTE_ACTOR_ALLOC_CONF", "").strip()
if _alloc_conf:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _alloc_conf

# Same per-role trick for NCCL_CUMEM_ENABLE. The shared yaml env sets it to 0
# (SGLang side requires 0 with torch_memory_saver), but the actor needs cuMem
# support when expandable_segments is on: NCCL collectives over VMM-backed
# buffers with NCCL_CUMEM_ENABLE=0 hit cudaErrorIllegalAddress (observed at
# awex colocate connect, first TP collective, run 0704_..._expseg2 937706).
_nccl_cumem = os.environ.get("DTE_ACTOR_NCCL_CUMEM_ENABLE", "").strip()
if _nccl_cumem:
    os.environ["NCCL_CUMEM_ENABLE"] = _nccl_cumem

from areal.infra.rpc.guard.app import (  # noqa: E402
    configure_state_from_args,
    make_base_parser,
    run_server,
)
from areal.v2.training_service.guard.app import _state, app  # noqa: E402


def main() -> None:
    parser = make_base_parser(
        description="AReaL Train RPCGuard — HTTP gateway for coordinating forked workers"
    )
    args, _ = parser.parse_known_args()
    bind_host = configure_state_from_args(_state, args)
    run_server(_state, app, bind_host, args.port)


if __name__ == "__main__":
    main()
