# SPDX-License-Identifier: Apache-2.0

# Adapted from verl

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed._functional_collectives import all_to_all_single_autograd

_ULYSSES_SEQUENCE_PARALLEL_GROUP = None


def set_ulysses_sequence_parallel_group(group: dist.ProcessGroup | None):
    """
    Set ulysses sequence parallel process group.
    """
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    _ULYSSES_SEQUENCE_PARALLEL_GROUP = group


def get_ulysses_sequence_parallel_group() -> dist.ProcessGroup | None:
    """
    Get ulysses sequence parallel process group.
    """
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    return _ULYSSES_SEQUENCE_PARALLEL_GROUP


def get_ulysses_sequence_parallel_world_size(group: ProcessGroup | None = None) -> int:
    """
    Get ulysses sequence parallel world size.
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_world_size(group) if group else 1


def get_ulysses_sequence_parallel_rank(group: ProcessGroup | None = None) -> int:
    """
    Get ulysses sequence parallel rank.
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_rank(group) if group else 0


def _gather_seq_scatter_heads(
    x: Tensor,
    seq_dim: int,
    head_dim: int,
    unpadded_dim_size: int = 0,
    group: ProcessGroup | None = None,
) -> Tensor:
    """All-to-All: [bsz, seq/n, h, ...] -> [bsz, seq, h/n, ...]"""
    if group is None:
        return x

    sp_world = dist.get_world_size(group)
    if sp_world <= 1:
        return x

    x = all_to_all_tensor(x, scatter_dim=head_dim, gather_dim=seq_dim, group=group)

    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        padding_size = x.size(seq_dim) - unpadded_dim_size
        if padding_size > 0:
            x = _unpad_tensor(x, seq_dim, padding_size)

    return x


def _gather_heads_scatter_seq(
    x: Tensor,
    head_dim: int,
    seq_dim: int,
    group: ProcessGroup | None = None,
) -> Tensor:
    """All-to-All: [bsz, seq, h/n, ...] -> [bsz, seq/n, h, ...]"""
    if group is None:
        return x

    sp_world = dist.get_world_size(group)
    if sp_world <= 1:
        return x

    dim_size = x.size(seq_dim)
    if dim_size % sp_world != 0:
        padding_size = sp_world - (dim_size % sp_world)
        x = _pad_tensor(x, seq_dim, padding_size)

    return all_to_all_tensor(x, scatter_dim=seq_dim, gather_dim=head_dim, group=group)


def gather_seq_scatter_heads(
    x: Tensor,
    seq_dim: int,
    head_dim: int,
    unpadded_dim_size: int = 0,
    group: ProcessGroup | None = None,
) -> Tensor:
    """_gather_seq_scatter_heads with global group."""
    if group is None:
        group = get_ulysses_sequence_parallel_group()
    return _gather_seq_scatter_heads(x, seq_dim, head_dim, unpadded_dim_size, group)


def gather_heads_scatter_seq(
    x: Tensor,
    head_dim: int,
    seq_dim: int,
    group: ProcessGroup | None = None,
) -> Tensor:
    """_gather_heads_scatter_seq with global group."""
    if group is None:
        group = get_ulysses_sequence_parallel_group()
    return _gather_heads_scatter_seq(x, head_dim, seq_dim, group)


def _pad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    if padding_size == 0:
        return x
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def _unpad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    if padding_size == 0:
        return x
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -padding_size)
    return x[tuple(slc)]


def slice_input_tensor(
    x: Tensor, dim: int, padding: bool = True, group: dist.ProcessGroup | None = None
) -> Tensor:
    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_world_size = dist.get_world_size(group)
    sp_rank = get_ulysses_sequence_parallel_rank()
    dim_size = x.size(dim)
    # pad before slice
    if padding and dim_size % sp_world_size:
        padding_size = sp_world_size - (dim_size % sp_world_size)
        x = _pad_tensor(x, dim, padding_size)
    # slice the input tensor
    parts = x.size(dim) // sp_world_size
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(sp_rank * parts, (sp_rank + 1) * parts)
    return x[tuple(slc)].contiguous()


def all_to_all_tensor(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup | None = None,
) -> Tensor:
    """
    All-to-all communication for multi-dimensional tensors.

    Uses all_to_all_single_autograd for torch.compile compatibility.
    Autograd is handled internally by all_to_all_single_autograd.

    Args:
        local_input: Input tensor
        scatter_dim: Dimension to scatter across ranks
        gather_dim: Dimension where gathered data will be concatenated
        group: Process group for communication

    Returns:
        Tensor with scatter_dim reduced by world_size and gather_dim
        multiplied by world_size
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_world_size = dist.get_world_size(group)

    # Split input along scatter_dim
    # Each chunk has size: [..., scatter_dim_size // world_size, ...]
    chunks = list(torch.chunk(local_input, sp_world_size, dim=scatter_dim))

    # Stack chunks: [world_size, ...original_shape_with_reduced_scatter_dim...]
    stacked = torch.stack(chunks, dim=0)
    stacked_shape = stacked.shape

    # Flatten to 1D for all_to_all_single_autograd
    stacked_flat = stacked.reshape(-1).contiguous()

    # Perform all-to-all with built-in autograd support (equal split)
    received_flat = all_to_all_single_autograd(
        stacked_flat,
        output_split_sizes=None,
        input_split_sizes=None,
        group=group,
    )

    # Reshape back to [world_size, ...chunk_shape...]
    received = received_flat.reshape(stacked_shape)

    # Unbind world_size dimension and concatenate along gather_dim
    chunks_received = torch.unbind(received, dim=0)
    output = torch.cat(chunks_received, dim=gather_dim)

    return output.contiguous()


def ulysses_pad(
    input_ids_rmpad: torch.Tensor,
    position_ids_rmpad: torch.Tensor | None = None,
    sp_size: int = 1,
):
    if position_ids_rmpad is not None:
        assert position_ids_rmpad.size(-2) == 1
        assert input_ids_rmpad.size(-1) == position_ids_rmpad.size(-1)
    if sp_size <= 1:
        return input_ids_rmpad, position_ids_rmpad, 0
    _, total_seq_len = input_ids_rmpad.shape
    pad_size = (sp_size - total_seq_len % sp_size) % sp_size
    if pad_size > 0:
        input_ids_rmpad = torch.nn.functional.pad(
            input_ids_rmpad, (0, pad_size), value=0
        )
        if position_ids_rmpad is not None:
            pad_pos_ids = torch.arange(
                pad_size, device=position_ids_rmpad.device
            ).unsqueeze(0)
            if position_ids_rmpad.dim() == 3:
                pad_pos_ids = pad_pos_ids.unsqueeze(0).repeat(3, 1, 1)
            position_ids_rmpad = torch.cat((position_ids_rmpad, pad_pos_ids), dim=-1)
    return input_ids_rmpad, position_ids_rmpad, pad_size


def ulysses_pad_and_slice_inputs(
    input_ids_rmpad: torch.Tensor,
    position_ids_rmpad: torch.Tensor | None = None,
    sp_size: int = 1,
):
    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
        input_ids_rmpad, position_ids_rmpad, sp_size
    )
    input_ids_rmpad = slice_input_tensor(input_ids_rmpad, dim=1, padding=False)
    if position_ids_rmpad is not None:
        position_ids_rmpad = slice_input_tensor(
            position_ids_rmpad, dim=1, padding=False
        )
    return input_ids_rmpad, position_ids_rmpad, pad_size


def ulysses_prepare_inputs(
    padded_mb_input,
    ulysses_input_ids,
    ulysses_position_ids,
    sp_world_size,
):
    # init inputs with padded_mb_input and ulysses_inputs
    inputs = padded_mb_input.copy()
    inputs["input_ids"] = ulysses_input_ids
    if ulysses_position_ids is not None:
        inputs["position_ids"] = ulysses_position_ids

    # Pad and slice the loss inputs
    padded_input_ids = padded_mb_input["input_ids"]

    for key, value in list(inputs.items()):
        if key in {"input_ids", "position_ids"}:
            continue
        if not torch.is_tensor(value):
            continue

        if value.dim() >= 2 and value.shape[:2] == padded_input_ids.shape[:2]:
            # Please refer to ppo_loss_fn() in areal/trainer/ppo/critic.py
            if key in {"values", "returns", "loss_mask"}:
                sliced_value = slice_input_tensor(value, dim=1, padding=True)
                inputs[key] = sliced_value.squeeze(0)
            else:
                inputs[key] = value.squeeze(0)

    # Roll and slice the full input_ids as the labels in Ulysses SP.
    rolled_input_ids = torch.roll(padded_input_ids, shifts=-1, dims=-1)
    rolled_input_ids, _, _ = ulysses_pad_and_slice_inputs(
        rolled_input_ids, sp_size=sp_world_size
    )
    inputs["rolled_input_ids"] = rolled_input_ids.squeeze(0)
    return inputs
