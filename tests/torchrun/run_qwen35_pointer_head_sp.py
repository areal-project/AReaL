# SPDX-License-Identifier: Apache-2.0
"""Two-NPU check that PointerHead gathers the sequence and splits its gradient."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu

from areal.infra.platforms import current_platform
from areal.models.mcore.qwen35_decision import PointerHead


def _broadcast_parameters(module: torch.nn.Module, src: int = 0) -> None:
    for param in module.parameters():
        dist.broadcast(param.data, src=src)


def run() -> None:
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 2:
        raise RuntimeError(f"expected 2 ranks, got {world}")
    device = torch.device(current_platform.device_type, rank)
    torch.manual_seed(0)

    head = PointerHead(8, dp=4, config=SimpleNamespace(sequence_parallel=True)).to(
        device
    )
    _broadcast_parameters(head)

    full = torch.randn(4, 2, 8, dtype=torch.bfloat16, device=device)
    dist.broadcast(full, src=0)
    local = full.chunk(world, dim=0)[rank]

    projected, extra = head(local)
    if extra is not None or projected.shape != (4, 2, 8):
        raise RuntimeError(
            f"rank {rank}: gathered projection shape {tuple(projected.shape)}"
        )
    if projected.dtype != torch.float32 or projected.device.type != "npu":
        raise RuntimeError(
            f"rank {rank}: projection is {projected.dtype} on {projected.device}"
        )
    with torch.no_grad():
        reference = torch.cat([head.q(full.float()), head.k(full.float())], dim=-1)
    torch.testing.assert_close(projected, reference)

    projected.sum().backward()
    local_grad = head.q.weight.grad.detach().clone()

    reference_head = PointerHead(8, dp=4).to(device)
    reference_head.load_state_dict(
        {key: value.detach().clone() for key, value in head.state_dict().items()}
    )
    reference_out, _ = reference_head(full.detach())
    reference_out.sum().backward()

    for param in head.parameters():
        dist.all_reduce(param.grad)
    for name, param in head.named_parameters():
        torch.testing.assert_close(param.grad, reference_head.get_parameter(name).grad)

    # Each rank owns a different sequence shard, so the pre-allreduce gradient
    # is only that shard's contribution.
    if torch.equal(local_grad, head.q.weight.grad):
        raise RuntimeError(f"rank {rank}: sequence-parallel gradient was not split")

    dist.barrier()
    if rank == 0:
        print("PASS pointer-head sequence-parallel gather", flush=True)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    current_platform.set_device(local_rank)
    dist.init_process_group(backend=current_platform.communication_backend)
    mpu.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        run()
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
