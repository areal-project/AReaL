# SPDX-License-Identifier: Apache-2.0

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def patch_dist_group_timeout(timeout: timedelta):
    """
    Patch the default timeout for process groups in torch.distributed.

    Args:
        timeout (timedelta): Default timeout to set for all process group backends.
    """
    from torch.distributed import distributed_c10d

    if hasattr(distributed_c10d, "default_pg_timeout"):
        distributed_c10d.default_pg_timeout = timeout

    if hasattr(distributed_c10d, "default_pg_nccl_timeout"):
        distributed_c10d.default_pg_nccl_timeout = timeout


def warmup_process_groups(*groups: dist.ProcessGroup | None) -> None:
    """Force eager initialization of the collective communicator for each group.

    NCCL/HCCL communicators are created lazily on the first collective call.
    On Ascend NPU (HCCL), deferring init until a collective runs during
    training is prone to fail with HCCP process initialization errors
    (e.g. ``hcclCommInitRootInfoConfig`` error code 7) when multiple
    colocated engines (for example actor + reference) independently mint
    overlapping subgroups and trigger their first collective in the middle
    of training work. Running a small dummy all-reduce at setup time forces
    the communicator to be initialized while all ranks are aligned and the
    device is idle, which avoids the race.

    ``None`` groups and duplicates are skipped. No-op on CPU-only platforms
    or before ``dist.init_process_group``. Safe to call repeatedly;
    subsequent calls on already-initialized groups are cheap.
    """
    # Deferred import to keep this module importable without a platform
    # (e.g. during lightweight tooling or unit tests).
    from areal.infra.platforms import current_platform

    if not dist.is_initialized() or current_platform.device_type == "cpu":
        return

    # Preserve argument order while dropping None and duplicates.
    unique_groups = list(dict.fromkeys(g for g in groups if g is not None))
    if not unique_groups:
        return

    # Prefer LOCAL_RANK (set by torchrun/most launchers); fall back to the
    # device the caller has already configured via ``set_device``. This keeps
    # the helper usable under custom launchers that don't export LOCAL_RANK.
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is not None:
        local_rank = int(local_rank_env)
        current_platform.set_device(local_rank)
    else:
        local_rank = current_platform.current_device()

    device = torch.device(current_platform.device_type, local_rank)
    tensor = torch.zeros(1, device=device)
    for group in unique_groups:
        dist.all_reduce(tensor, group=group)


# Copy from pytorch and OpenRLHF to allow creating multiple main groups.
# This is needed because torch.distributed.init_process_group() only creates
# the default global group, and torch.distributed.new_group() only creates
# subgroups of the default group. AReaL needs independent process groups
# for weight synchronization between training and inference engines that
# run in separate launcher contexts (separate init_process_group calls).
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
# https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/distributed_util.py
def init_custom_process_group(
    backend=None,
    init_method=None,
    timeout=None,
    world_size=-1,
    rank=-1,
    store=None,
    group_name=None,
    backend_options=None,
):
    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        default_pg_timeout,
        rendezvous,
    )

    if store is not None and init_method is not None:
        raise RuntimeError("Cannot specify both init_method and store.")

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        backend_options=backend_options,
        timeout=timeout,
        group_desc=group_name,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg
