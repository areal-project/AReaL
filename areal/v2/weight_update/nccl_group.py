# SPDX-License-Identifier: Apache-2.0
"""NCCL process group initialization utilities for weight updates."""

import os

import torch
import torch.distributed as dist

from areal.infra.platforms import current_platform
from areal.utils import logging

logger = logging.getLogger("NCCLGroup")


# Copy from pytorch and OpenRLHF to allow creating multiple main groups.
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
    pg_options=None,
):
    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        default_pg_timeout,
        rendezvous,
    )

    assert (store is None) or (init_method is None), (
        "Cannot specify both init_method and store."
    )

    # NOTE: Processes launched with torchrun will set the following env var to True,
    # which blocks creating another TCP store for weight update.
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)

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

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = (
        "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    )
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg


def init_weights_update_group(
    master_address,
    master_port,
    rank,
    world_size,
    group_name,
    backend="nccl",
    role="",
):
    """Initialize the Torch process group for model parameter updates."""
    assert torch.distributed.is_initialized(), (
        "Default torch process group must be initialized"
    )
    assert group_name != "", "Group name cannot be empty"

    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    logger.info(
        f"init custom process group for {role}: master_address={master_address}, master_port={master_port}, "
        f"rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}, "
        f"current device id {current_platform.current_device()} "
        f"{current_platform.device_control_env_var} {visible_env or '(unset)'} "
        f"Local rank env {os.environ.get('LOCAL_RANK')} DEVICE env {os.environ.get('DEVICE')} "
        f"Global rank env {os.environ.get('RANK')}"
    )

    try:
        options = None
        if backend == "hccl":
            import torch_npu

            options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
            # first,using specified buffer size instead of global buffer size while large size has higher throughput,
            # because the memory used is 2 * buffer_size MB.
            # second,rollout and actor must have a same buffer size for group init.
            options.hccl_config = {
                "hccl_buffer_size": int(os.getenv("AWEX_P2P_HCCL_BUFFER_SIZE", "200"))
            }
        group = init_custom_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
            pg_options=options,
        )
        logger.info(f"Initialized custom process group: {group}")
        return group
    except Exception as e:
        raise RuntimeError(f"Failed to initialize custom process group: {e}.") from e


def setup_batch_isend_irecv(
    process_group, rank, world_size, tensor_size=10 * 10, dtype=torch.float32
):
    """
    Perform a simple communication using batch_isend_irecv to avoid the hang for later sub-ranks.

    Args:
    process_group (ProcessGroup): The process group to work on.
    tensor_size (int): Size of the tensor to send/receive.
    dtype (torch.dtype): Data type of the tensor.
    """
    assert process_group is not None, "Process group cannot be None"
    device = current_platform.current_device()
    logger.info(
        f"Setup batch isend irecv for rank {rank} world size {world_size} device {device}"
    )

    # Create tensors for sending and receiving
    torch_device = torch.device(f"{current_platform.device_type}:{device}")
    send_tensor = torch.full(
        (tensor_size,), rank, dtype=dtype, device=torch_device, requires_grad=False
    )
    recv_tensor = torch.zeros(
        (tensor_size,), dtype=dtype, device=torch_device, requires_grad=False
    )

    # Prepare the ops for batch_isend_irecv
    ops = []

    mid_point = world_size // 2
    if world_size <= 1:
        logger.info(f"Skip batch isend/irecv setup because world size={world_size}.")
    elif world_size % 2 == 0:
        # Even world_size: pair the first half with the second half.
        if rank < mid_point:
            target_rank = rank + mid_point
            if target_rank < world_size:
                ops.append(
                    dist.P2POp(
                        dist.irecv, recv_tensor, target_rank, group=process_group
                    )
                )
        else:
            target_rank = rank - mid_point
            if target_rank >= 0:
                ops.append(
                    dist.P2POp(
                        dist.isend, send_tensor, target_rank, group=process_group
                    )
                )
    else:
        # Odd world_size: use a simple ring so every rank has a partner.
        recv_from = (rank - 1 + world_size) % world_size
        send_to = (rank + 1) % world_size
        ops.append(dist.P2POp(dist.irecv, recv_tensor, recv_from, group=process_group))
        ops.append(dist.P2POp(dist.isend, send_tensor, send_to, group=process_group))

    # Execute batch_isend_irecv
    if ops:
        reqs = dist.batch_isend_irecv(ops)
        # Wait for all communications to complete
        for req in reqs:
            req.wait()

    # Synchronize
    current_platform.synchronize()
    dist.barrier(group=process_group, device_ids=[current_platform.current_device()])

    logger.info(
        f"Simple communication completed for process group of size {world_size}"
    )

    # Verify the results
    if world_size <= 1:
        return
    if world_size % 2 == 0:
        if rank < mid_point and rank + mid_point < world_size:
            expected_value = rank + mid_point
            assert torch.all(recv_tensor == expected_value), (
                f"Rank {rank} received incorrect data from rank {rank + mid_point}"
            )
    else:
        expected_value = (rank - 1 + world_size) % world_size
        assert torch.all(recv_tensor == expected_value), (
            f"Rank {rank} received incorrect data from rank {expected_value}"
        )

    logger.info("Simple communication verification successful")
