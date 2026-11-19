# SPDX-License-Identifier: Apache-2.0

"""Core operations for training engines.

This module provides utilities and lightweight accumulation state shared across
different training engine implementations (FSDP, Megatron, etc.).
"""

from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from areal.infra.platforms import current_platform
from areal.utils.data import (
    MicroBatchList,
    pad_and_stack_tensors_along_first_dim,
    reorder_list,
    unpack_sequence,
)

__all__ = [
    "compute_total_loss_weight",
    "aggregate_eval_losses",
    "reorder_and_pad_outputs",
]


@dataclass
class _GradientAccumulationState:
    """Window metadata only; gradients remain in the backend's own buffers."""

    steps: int
    next_index: int = 0
    reference_loss_weight: torch.Tensor | None = None
    total_loss_weight: torch.Tensor | None = None
    num_microbatches: int = 0
    reference_microbatches: int = 0
    failed: bool = False
    # Megatron's native per-token finalizer and isolated MTP use distinct units.
    reference_num_tokens: torch.Tensor | None = None
    total_num_tokens: torch.Tensor | None = None
    reference_mtp_tokens: torch.Tensor | None = None
    total_mtp_tokens: torch.Tensor | None = None
    local_num_tokens: torch.Tensor | None = None
    mtp_metrics: torch.Tensor | None = None

    @property
    def is_first(self) -> bool:
        return self.next_index == 0

    @property
    def is_last(self) -> bool:
        return self.next_index == self.steps - 1

    def add_chunk(self, chunk_loss_weight: torch.Tensor, num_microbatches: int) -> None:
        """Record W_chunk, already summed over MBs and reduced by the backend.

        Each w_i is loss_weight_fn(microbatch_i), the configured loss normalizer.
        The caller computes W_chunk, so this class does not receive loss masks.
        """
        weight = chunk_loss_weight.detach()
        if self.reference_loss_weight is None:
            self.reference_loss_weight = weight * self.steps
            self.reference_microbatches = num_microbatches * self.steps
            self.total_loss_weight = weight.clone()
        else:
            self.total_loss_weight = self.total_loss_weight + weight
        self.num_microbatches += num_microbatches

    @property
    def gradient_scale(self) -> torch.Tensor:
        assert self.reference_loss_weight is not None
        assert self.total_loss_weight is not None
        return self.reference_loss_weight / self.total_loss_weight


@contextmanager
def _gradient_accumulation(
    engine: Any, steps: int, index: int
) -> Generator[_GradientAccumulationState | None, None, None]:
    """Manage consecutive chunk calls sharing one optimizer update.

    Engines use is_first/is_last to choose zero/step boundaries. Successful
    calls advance the index and the last call clears the metadata. A failed
    call poisons the window so an RPC retry cannot accumulate its gradients
    twice. This context does not pack inputs, run backward or store gradients.
    """
    if steps < 1 or not 0 <= index < steps:
        raise ValueError(
            f"Invalid accumulation_steps={steps}, accumulation_index={index}"
        )
    state = getattr(engine, "_gradient_accumulation_state", None)
    if state is not None and state.failed:
        raise RuntimeError(
            "Gradient accumulation failed; recreate the engine before retrying."
        )
    if steps == 1:
        if state is not None:
            raise ValueError(
                "Cannot interrupt an unfinished gradient accumulation window."
            )
        yield None
        return
    if state is None:
        if index != 0:
            raise ValueError(
                "Gradient accumulation must start at accumulation_index=0."
            )
        state = _GradientAccumulationState(steps)
        engine._gradient_accumulation_state = state
    if state.steps != steps or state.next_index != index:
        raise ValueError(
            f"Expected accumulation_steps={state.steps}, "
            f"accumulation_index={state.next_index}; got {steps}, {index}."
        )
    try:
        yield state
    except BaseException:
        # RPC retries must not add a chunk twice after a partial backward/step.
        state.failed = True
        raise
    else:
        state.next_index += 1
        if state.next_index == steps:
            engine._gradient_accumulation_state = None


@torch.no_grad()
def _scale_local_gradients(
    parameters: Iterable[torch.nn.Parameter], factor: torch.Tensor
) -> None:
    """Scale existing local shards, including gradients offloaded to CPU."""
    from torch.distributed.tensor import DTensor

    seen: set[int] = set()
    factors: dict[torch.device, torch.Tensor] = {}
    for parameter in parameters:
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        grad = parameter.grad
        if grad is None:
            continue
        if isinstance(grad, DTensor):
            grad = grad.to_local()
        if grad.device not in factors:
            factors[grad.device] = factor.to(device=grad.device)
        grad.mul_(factors[grad.device])


def _iter_persistent_buffers(
    module: torch.nn.Module,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield stable policy buffers, excluding non-persistent forward caches."""
    seen: set[int] = set()
    for prefix, child in module.named_modules():
        for name, buffer in child.named_buffers(recurse=False, remove_duplicate=False):
            if name in child._non_persistent_buffers_set:
                continue
            if id(buffer) in seen:
                continue
            seen.add(id(buffer))
            yield f"{prefix}.{name}" if prefix else name, buffer


def compute_total_loss_weight(
    mb_list: MicroBatchList,
    loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    dp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Compute total loss weight and all_reduce across data parallel group.

    This aggregates the loss weights from all micro-batches and reduces
    them across the data parallel group to get a global normalization factor.

    Parameters
    ----------
    mb_list : MicroBatchList
        The list of micro-batches.
    loss_weight_fn : Callable[[dict[str, Any]], torch.Tensor]
        Function to compute loss weight for each micro-batch.
    dp_group : dist.ProcessGroup
        The data parallel process group for all_reduce.

    Returns
    -------
    torch.Tensor
        The total loss weight (scalar tensor) after all_reduce.
    """
    total_weight = (
        torch.stack([loss_weight_fn(mb) for mb in mb_list.mbs])
        .sum()
        .detach()
        .clone()
        .to(dtype=torch.float32)
    )
    dist.all_reduce(total_weight, group=dp_group)
    assert total_weight > 0, (
        "Global total loss weight must be positive after all_reduce"
    )
    return total_weight


def aggregate_eval_losses(
    losses: list[torch.Tensor] | None,
    dp_group: dist.ProcessGroup,
    is_pp_last_stage: bool = True,
    pp_group: dist.ProcessGroup | None = None,
    pp_src_rank: int | None = None,
) -> torch.Tensor:
    """Aggregate evaluation losses from micro-batches.

    Parameters
    ----------
    losses : list[torch.Tensor] | None
        List of loss tensors from each micro-batch. None on non-last PP stages.
    dp_group : dist.ProcessGroup
        The data parallel process group for all_reduce.
    is_pp_last_stage : bool
        Whether this rank is the last PP stage. True by default.
    pp_group : dist.ProcessGroup | None
        Pipeline parallel group for broadcast. None if PP broadcast is not required.
    pp_src_rank : int | None
        Global rank of last PP stage (required if pp_group is set).

    Returns
    -------
    torch.Tensor
        The aggregated loss after summing and all_reduce.
    """
    if is_pp_last_stage:
        assert losses is not None, "losses required on last PP stage"
        loss = torch.stack(losses).sum(dtype=torch.float32)
        dist.all_reduce(loss, group=dp_group)
    else:
        device = current_platform.current_device()
        loss = torch.empty(1, device=device, dtype=torch.float32)

    if pp_group is not None:
        assert pp_src_rank is not None, "pp_src_rank required when pp_group is set"
        dist.broadcast(loss, src=pp_src_rank, group=pp_group)

    return loss


def reorder_and_pad_outputs(
    outputs: list[torch.Tensor],
    output_seqlens: list[int],
    mb_list: MicroBatchList,
    aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
) -> torch.Tensor:
    """Aggregate, reorder, and pad forward outputs from micro-batches.

    This handles the output post-processing for forward_batch:
    1. Aggregate outputs from all micro-batches
    2. Unpack by sequence lengths
    3. Reorder to match original input order
    4. Pad and stack along batch dimension

    Parameters
    ----------
    outputs : list[torch.Tensor]
        List of output tensors from each micro-batch.
    output_seqlens : list[int]
        Sequence lengths for unpacking.
    mb_list : MicroBatchList
        The micro-batch list containing reordering indices.
    aggregate_fn : Callable[[list[Any]], Any], optional
        Function to aggregate outputs, by default torch.cat.

    Returns
    -------
    torch.Tensor
        The processed outputs, padded and stacked along batch dimension.
    """
    res = aggregate_fn(outputs)
    seqlens = [output_seqlens[i] for i in mb_list.forward_indices]
    unpacked = unpack_sequence(res, lens=seqlens, dim=0)
    reordered = reorder_list(unpacked, mb_list.backward_indices)
    return pad_and_stack_tensors_along_first_dim(reordered)
