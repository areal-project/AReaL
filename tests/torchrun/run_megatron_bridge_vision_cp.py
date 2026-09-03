# SPDX-License-Identifier: Apache-2.0

import os

from areal.utils.torch_npu_compat import import_mindspeed_adaptor

import_mindspeed_adaptor()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (  # noqa: E402
    AllGatherVisionEmbeddings,
    ensure_requires_grad_for_cp_collective,
)

from areal.infra.platforms import current_platform  # noqa: E402


def run_case(
    name: str,
    lengths: list[int],
    dtype: torch.dtype,
    requires_grad: bool,
) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == len(lengths) == 2

    hidden_size = 8
    local_length = lengths[rank]
    device = current_platform.current_device()
    seqlens = [
        torch.tensor([length], dtype=torch.long, device=device) for length in lengths
    ]

    vision_output = torch.randn(local_length, hidden_size, dtype=dtype, device=device)
    if requires_grad:
        vision_output.requires_grad_(True)
    else:
        ensure_requires_grad_for_cp_collective((vision_output,))
    assert vision_output.requires_grad

    gathered = AllGatherVisionEmbeddings.apply(
        vision_output,
        seqlens,
        dist.group.WORLD,
    )
    total_length = sum(lengths)
    assert gathered.shape == (total_length, hidden_size)
    assert gathered.dtype == dtype

    row_weight = torch.arange(
        1,
        total_length + 1,
        dtype=dtype,
        device=device,
    ).unsqueeze(1)
    rank_weight = float(rank + 1)
    (gathered * row_weight * rank_weight).sum().backward()

    assert vision_output.grad is not None
    start = sum(lengths[:rank])
    expected = (
        row_weight[start : start + local_length] * sum(range(1, world_size + 1))
    ).expand(local_length, hidden_size)
    torch.testing.assert_close(vision_output.grad, expected, rtol=0, atol=0)
    dist.barrier()

    if rank == 0:
        print(f"PASS {name}: lengths={lengths}, dtype={dtype}")


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    current_platform.set_device(local_rank)
    dist.init_process_group(backend=current_platform.communication_backend)
    try:
        run_case("balanced", [3, 3], torch.float32, requires_grad=True)
        run_case("empty_rank", [4, 0], torch.bfloat16, requires_grad=False)
        run_case("frozen_outputs", [3, 3], torch.bfloat16, requires_grad=False)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
