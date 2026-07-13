# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_F
from torch import Tensor
from torch.distributed import ProcessGroup

from areal.models.fsdp.ulysses import (
    _gather_heads_scatter_seq as gather_heads_scatter_seq,
)
from areal.models.fsdp.ulysses import (
    _gather_seq_scatter_heads as gather_seq_scatter_heads,
)

__all__ = [
    "ulysses_slice_inputs",
    "gather_seq_scatter_heads",
    "gather_heads_scatter_seq",
    "ulysses_gather_output",
]


def _ulysses_slice_tensor(
    tensor: Tensor,
    cp_rank: int,
    cp_size: int,
) -> Tensor:
    """Slice a tensor along the last dimension for Ulysses SP."""
    total_len = tensor.shape[-1]

    if total_len % cp_size != 0:
        raise ValueError(
            f"Tensor length {total_len} not aligned to cp_size {cp_size}. "
            "Ensure pad_mb_list is called with batch_align_to=seq_len_divisor first."
        )

    chunk_size = total_len // cp_size
    start = cp_rank * chunk_size
    end = start + chunk_size

    return tensor[..., start:end].contiguous()


def ulysses_slice_inputs(
    inputs: dict[str, Any],
    labels: Tensor,
    cp_rank: int,
    cp_size: int,
) -> tuple[dict[str, Any], Tensor]:
    """Slice inputs and labels for Ulysses SP."""
    if cp_size <= 1:
        return inputs, labels

    inputs = dict(inputs)
    inputs["input_ids"] = _ulysses_slice_tensor(inputs["input_ids"], cp_rank, cp_size)
    labels = _ulysses_slice_tensor(labels, cp_rank, cp_size)

    inputs["position_ids"] = _ulysses_slice_tensor(
        inputs["position_ids"], cp_rank, cp_size
    )

    return inputs, labels


def ulysses_gather_output(
    output: Tensor,
    cp_group: ProcessGroup | None,
    seq_dim: int = 0,
) -> Tensor:
    """Gather output tensor from all CP ranks after forward pass."""
    if cp_group is None:
        return output

    cp_size = dist.get_world_size(cp_group)
    if cp_size <= 1:
        return output

    gathered = dist_F.all_gather(output, group=cp_group)
    return torch.cat(gathered, dim=seq_dim)
