# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from areal.engine.megatron_utils.packed_context_parallel import (
    reassemble_cp_packed_logprobs,
    split_packed_seqs_for_context_parallel,
)


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    group = dist.new_group([0, 1])

    from unittest.mock import patch

    device = torch.device("cuda", local_rank)
    full = torch.arange(32, dtype=torch.float32, device=device)
    cu_seqlens = torch.tensor([0, 8, 32], dtype=torch.long, device=device)
    with patch("areal.engine.megatron_utils.packed_context_parallel.mpu") as mocked_mpu:
        mocked_mpu.get_context_parallel_world_size.return_value = 2
        mocked_mpu.get_context_parallel_rank.return_value = rank
        mocked_mpu.get_context_parallel_group.return_value = group
        local = split_packed_seqs_for_context_parallel(full, cu_seqlens)
        assert local.ndim == 1 and local.numel() == full.numel() // 2
        result = reassemble_cp_packed_logprobs(local, cu_seqlens)
    torch.testing.assert_close(result, full, rtol=0.0, atol=0.0)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
