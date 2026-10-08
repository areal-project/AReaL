# SPDX-License-Identifier: Apache-2.0

from functools import wraps
from typing import Any

import torch
from torch.distributed.nn.functional import all_gather


def install_qwen4_exp_ple_cp_autograd() -> None:
    """Fix the pinned bridge's PLE gather without changing ID/vision gathers.

    PLE's causal convolution reads hidden states across CP shard boundaries.
    Retaining only the local shard's graph drops those remote input gradients.
    Replace only the PLE module's reconstruction binding; keep the bridge's
    sequence-parallel handling and packed zigzag permutation unchanged.
    """
    from mcore_bridge.model.modules import ple
    from mcore_bridge.utils import megatron_utils

    original = ple.reconstruct_tensor_cp
    if getattr(original, "_areal_cp_autograd", False):
        return

    @wraps(original)
    def reconstruct(
        tensor: torch.Tensor,
        packed_seq_params: Any,
        dim: int,
        cp_partition_mode: str = "zigzag",
    ) -> torch.Tensor:
        cp_size = megatron_utils.mpu.get_context_parallel_world_size()
        if cp_size <= 1 or not (torch.is_grad_enabled() and tensor.requires_grad):
            return original(tensor, packed_seq_params, dim, cp_partition_mode)
        group = megatron_utils.mpu.get_context_parallel_group()
        gathered = torch.cat(all_gather(tensor.contiguous(), group=group), dim=dim)
        if cp_partition_mode == "contiguous":
            return gathered
        if cp_partition_mode != "zigzag":
            raise ValueError(f"Unsupported PLE CP partition mode: {cp_partition_mode}")
        if dim != 0:
            gathered = gathered.transpose(0, dim).contiguous()
        result = megatron_utils._undo_attention_load_balancing(
            gathered, cp_size, packed_seq_params
        )
        return result if dim == 0 else result.transpose(0, dim).contiguous()

    reconstruct._areal_cp_autograd = True
    ple.reconstruct_tensor_cp = reconstruct
