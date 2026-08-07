# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Sequence
from datetime import timedelta

import torch
import torch.distributed as dist

# The two probe sizes do not model training tensor shapes. They sit on either
# side of NCCL's message-size protocol thresholds (LL/LL128 vs Simple; 2KB vs
# 32MB payloads in bf16), and NCCL allocates transport buffers per communicator
# and protocol class on first use. Protocol selection also depends on
# collective type, topology and NCCL settings, so this targets the message
# classes seen in training rather than guaranteeing coverage of every size.
_PROBE_NUMELS: tuple[int, ...] = (1024, 16 * 1024 * 1024)


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


def _probe_device() -> torch.device:
    """The device warmup probes are allocated on.

    Prefer LOCAL_RANK (set by torchrun and most launchers); fall back to the
    device the caller has already configured via ``set_device``. This keeps the
    helper usable under custom launchers that don't export LOCAL_RANK. Note
    that the LOCAL_RANK branch also *sets* the device, matching what engines do
    during setup.
    """
    from areal.infra.platforms import current_platform

    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is not None:
        local_rank = int(local_rank_env)
        current_platform.set_device(local_rank)
    else:
        local_rank = current_platform.current_device()
    return torch.device(current_platform.device_type, local_rank)


def _warmup_device() -> torch.device | None:
    """The probe device, or None when there is nothing to warm up."""
    from areal.infra.platforms import current_platform

    if not dist.is_initialized() or current_platform.device_type == "cpu":
        return None
    return _probe_device()


def _unique(groups: tuple[dist.ProcessGroup | None, ...]) -> list[dist.ProcessGroup]:
    return list(dict.fromkeys(g for g in groups if g is not None))


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
    from areal.infra.platforms import current_platform

    if not dist.is_initialized() or current_platform.device_type == "cpu":
        return

    unique_groups = _unique(groups)
    if not unique_groups:
        return

    device = _probe_device()
    tensor = torch.zeros(1, device=device)
    for group in unique_groups:
        dist.all_reduce(tensor, group=group)


def nccl_process_groups() -> list[dist.ProcessGroup]:
    """Every collective-capable group this rank belongs to, in creation order.

    Sweeping the registry rather than naming groups keeps subgroups minted by
    other components (for example the MoE expert grad-bucket reduction) from
    being missed; a missed group still connects lazily at peak memory.

    Insertion order is the creation order, and non-member ranks get no entry,
    so any two ranks agree on the relative order of the groups they share.
    Only metadata is read here, so a raise cannot leave a peer waiting.
    """
    from torch.distributed.distributed_c10d import _world

    if not dist.is_initialized():
        return []

    groups: list[dist.ProcessGroup] = []
    for group in list(_world.pg_map):
        if "nccl" not in str(dist.get_backend(group)).lower():
            continue
        if dist.get_world_size(group=group) <= 1:
            continue
        groups.append(group)
    return groups


def warmup_collective_transports(
    *groups: dist.ProcessGroup | None,
    numels: Sequence[int] = _PROBE_NUMELS,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Pre-connect the all-reduce transport buffers of each group.

    Complements :func:`warmup_process_groups`: that one forces the communicator
    to be *created* with a single tiny all-reduce, which leaves the large Simple
    protocol buffer to be allocated during the first train step, at peak memory.

    Exceptions are deliberately not caught. Every statement below is a
    collective, so a rank that swallowed a failure and moved on would leave its
    peers blocked on an operation that never arrives.
    """
    device = _warmup_device()
    if device is None:
        return
    for group in _unique(groups):
        for numel in numels:
            dist.all_reduce(torch.zeros(numel, dtype=dtype, device=device), group=group)


def warmup_all_to_all_transports(
    *groups: dist.ProcessGroup | None,
    numels: Sequence[int] = _PROBE_NUMELS,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Pre-connect the all-to-all transport buffers of each group.

    Kept separate from :func:`warmup_collective_transports` because all-to-all
    allocates its own buffers and they live for the lifetime of the process.
    Warming a group that never dispatches tokens would hold that memory for
    nothing, which is the opposite of the point.
    """
    device = _warmup_device()
    if device is None:
        return
    for group in _unique(groups):
        world_size = dist.get_world_size(group=group)
        for numel in numels:
            aligned = (numel // world_size) * world_size
            if not aligned:
                continue
            src = torch.zeros(aligned, dtype=dtype, device=device)
            dist.all_to_all_single(torch.empty_like(src), src, group=group)


def warmup_sharded_transports(
    group: dist.ProcessGroup | None,
    *,
    numel: int = _PROBE_NUMELS[-1],
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Pre-connect the distributed optimizer's reduce_scatter / all_gather."""
    if group is None:
        return
    device = _warmup_device()
    if device is None:
        return
    world_size = dist.get_world_size(group=group)
    if world_size <= 1:
        return
    shard_numel = numel // world_size
    if shard_numel == 0:
        return

    flat = torch.zeros(shard_numel * world_size, dtype=dtype, device=device)
    shard = torch.empty(shard_numel, dtype=dtype, device=device)
    dist.reduce_scatter_tensor(shard, flat, group=group)
    dist.all_gather_into_tensor(flat, shard, group=group)


def warmup_p2p_transports(
    *groups: dist.ProcessGroup | None,
    prev_rank: int,
    next_rank: int,
    has_prev: bool,
    has_next: bool,
    numels: Sequence[int] = _PROBE_NUMELS,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Pre-connect the 2-rank pair communicators pipeline send/recv creates.

    ``prev_rank`` and ``next_rank`` are *global* ranks, matching Megatron's own
    convention: it resolves them with ``dist.get_global_rank(pp_group, ...)``
    and passes them alongside an explicit ``group=``.

    Both the unbatched and the batched form are exercised, because they can end
    up on different communicators. Non-blocking posts are issued before any
    wait, so no even/odd rotation is needed to stay deadlock-free.
    """
    device = _warmup_device()
    if device is None:
        return
    unique = _unique(groups)
    if not unique or not (has_prev or has_next):
        return

    for group in unique:
        for numel in numels:
            send_buf = torch.zeros(numel, dtype=dtype, device=device)
            recv_prev = torch.empty(numel, dtype=dtype, device=device)
            recv_next = torch.empty(numel, dtype=dtype, device=device)

            reqs = []
            if has_next:
                reqs.append(dist.isend(send_buf, next_rank, group=group))
            if has_prev:
                reqs.append(dist.irecv(recv_prev, prev_rank, group=group))
            if has_prev:
                reqs.append(dist.isend(send_buf, prev_rank, group=group))
            if has_next:
                reqs.append(dist.irecv(recv_next, next_rank, group=group))
            for req in reqs:
                req.wait()

            ops = []
            if has_prev:
                ops.append(dist.P2POp(dist.isend, send_buf, prev_rank, group))
                ops.append(dist.P2POp(dist.irecv, recv_prev, prev_rank, group))
            if has_next:
                ops.append(dist.P2POp(dist.isend, send_buf, next_rank, group))
                ops.append(dist.P2POp(dist.irecv, recv_next, next_rank, group))
            for work in dist.batch_isend_irecv(ops):
                work.wait()


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
