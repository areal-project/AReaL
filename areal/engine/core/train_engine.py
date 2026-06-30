# SPDX-License-Identifier: Apache-2.0

"""Core operations for training engines.

This module provides stateless utility functions that are shared across
different training engine implementations (FSDP, Megatron, etc.).
"""

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import torch
import torch.distributed as dist

from areal.api.engine_api import (
    LOSS_TERM_REDUCTION_MEAN,
    LossFnOutput,
    LossReduction,
    LossTerm,
)
from areal.infra.platforms import current_platform
from areal.utils.data import (
    MicroBatchList,
    pad_and_stack_tensors_along_first_dim,
    reorder_list,
    unpack_sequence,
)

__all__ = [
    "compute_global_normalizers",
    "compute_local_normalizers",
    "scale_loss_for_reduction",
    "aggregate_eval_losses",
    "reorder_and_pad_outputs",
]


def compute_local_normalizers(
    input_: dict[str, Any], loss_reduction: LossReduction
) -> dict[str, torch.Tensor]:
    return {term.name: term.normalizer_fn(input_) for term in loss_reduction.terms}


def _sum_term_normalizer(mb_list: MicroBatchList, term: LossTerm) -> torch.Tensor:
    return (
        torch.stack([term.normalizer_fn(mb) for mb in mb_list.mbs])
        .sum()
        .detach()
        .clone()
        .to(dtype=torch.float32)
    )


def compute_global_normalizers(
    mb_list: MicroBatchList,
    loss_reduction: LossReduction,
    dp_group: dist.ProcessGroup,
) -> dict[str, torch.Tensor]:
    normalizers = torch.stack(
        [_sum_term_normalizer(mb_list, term) for term in loss_reduction.terms]
    )
    dist.all_reduce(normalizers, group=dp_group)
    return {
        term.name: normalizer
        for term, normalizer in zip(
            loss_reduction.terms, normalizers.unbind(), strict=True
        )
    }


def _get_loss_term_value(
    loss: LossFnOutput, loss_reduction: LossReduction, term: LossTerm
) -> torch.Tensor:
    if isinstance(loss, Mapping):
        try:
            return loss[term.name]
        except KeyError as e:
            raise KeyError(
                f"loss output is missing term {term.name!r}; "
                f"available terms are {tuple(loss)}."
            ) from e
    if len(loss_reduction.terms) == 1:
        return loss
    raise TypeError("loss_fn must return a mapping for multi-term LossReduction.")


def _zero_like_loss(loss: LossFnOutput) -> torch.Tensor:
    if isinstance(loss, Mapping):
        first = next(iter(loss.values()))
        return first * 0.0
    return loss * 0.0


def _apply_local_reduction(
    term_loss: torch.Tensor,
    term: LossTerm,
    local_normalizer: torch.Tensor,
) -> torch.Tensor:
    if term.reduction == LOSS_TERM_REDUCTION_MEAN:
        return term_loss * local_normalizer
    return term_loss


def _safe_global_normalizer(global_normalizer: torch.Tensor) -> torch.Tensor:
    return torch.where(
        global_normalizer == 0,
        torch.ones_like(global_normalizer),
        global_normalizer,
    )


def _scale_loss_term(
    loss: LossFnOutput,
    loss_reduction: LossReduction,
    term: LossTerm,
    local_normalizers: dict[str, torch.Tensor],
    global_normalizers: dict[str, torch.Tensor],
    loss_multiplier: float,
) -> torch.Tensor:
    global_normalizer = global_normalizers[term.name]
    term_loss = _get_loss_term_value(loss, loss_reduction, term)
    term_loss = _apply_local_reduction(term_loss, term, local_normalizers[term.name])
    active = (global_normalizer != 0).to(dtype=term_loss.dtype)
    return (
        term_loss
        / _safe_global_normalizer(global_normalizer)
        * active
        * loss_multiplier
    )


def _iter_scaled_terms(
    loss: LossFnOutput,
    loss_reduction: LossReduction,
    local_normalizers: dict[str, torch.Tensor],
    global_normalizers: dict[str, torch.Tensor],
    loss_multiplier: float,
) -> Iterator[torch.Tensor]:
    for term in loss_reduction.terms:
        yield _scale_loss_term(
            loss,
            loss_reduction,
            term,
            local_normalizers,
            global_normalizers,
            loss_multiplier,
        )


def _sum_scaled_terms(
    terms: Iterable[torch.Tensor],
    loss: LossFnOutput,
) -> torch.Tensor:
    iterator = iter(terms)
    try:
        total = next(iterator)
    except StopIteration:
        return _zero_like_loss(loss)

    for term in iterator:
        total = total + term
    return total


def scale_loss_for_reduction(
    loss: LossFnOutput,
    loss_reduction: LossReduction,
    local_normalizers: dict[str, torch.Tensor],
    global_normalizers: dict[str, torch.Tensor],
    loss_multiplier: float,
) -> torch.Tensor:
    """Scale local loss terms into a globally normalized engine loss."""
    return _sum_scaled_terms(
        _iter_scaled_terms(
            loss,
            loss_reduction,
            local_normalizers,
            global_normalizers,
            loss_multiplier,
        ),
        loss,
    )


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
