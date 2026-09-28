# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api import InferenceEngine, TrainEngine, WorkflowLike
from areal.infra.platforms import current_platform
from areal.utils import stats_tracker
from areal.utils.data import (
    all_gather_tensor_container,
    broadcast_tensor_container,
    extract_single_valid_token_sequence,
    get_total_valid_tokens,
    split_and_unpad_tensor,
    tensor_container_to,
)
from areal.utils.seqpack import get_allocate_fn


class _TreeTokenOnlyTimeModel:
    def pred(self, stats: dict[str, Any]) -> float:
        return float(stats["n_tree_tokens"])


def _validate_group_indices(
    group_indices: list[list[int]], n_groups: int, n_items: int
) -> None:
    if len(group_indices) != n_groups:
        raise ValueError(
            f"group_indices must contain exactly {n_groups} groups, got {len(group_indices)}."
        )
    flat_indices = [idx for group in group_indices for idx in group]
    if len(flat_indices) != n_items:
        raise ValueError(
            f"group_indices must assign exactly {n_items} items, got {len(flat_indices)}."
        )
    if sorted(flat_indices) != list(range(n_items)):
        raise ValueError(
            "group_indices must be a permutation of [0, ..., n_items-1] "
            "(no duplicates, no missing/out-of-range indices)."
        )


@dataclass
class RedistributedData:
    all_data: list[dict[str, Any]]
    data: list[dict[str, Any]]
    rank: int
    group_indices: list[list[int]]
    dta_metrics: "DTAMetrics | None" = None


@dataclass(slots=True)
class DTAMetrics:
    n_tokens: float
    n_tree_tokens_before_allocation: float
    n_tree_tokens_after_allocation: float
    compression_ratio_before_allocation: float
    compression_ratio_after_allocation: float

    def to_stats(self) -> dict[str, float]:
        return {
            "dta/n_tokens": self.n_tokens,
            "dta/n_tree_tokens_before_allocation": self.n_tree_tokens_before_allocation,
            "dta/n_tree_tokens_after_allocation": self.n_tree_tokens_after_allocation,
            "dta/compression_ratio_before_allocation": self.compression_ratio_before_allocation,
            "dta/compression_ratio_after_allocation": self.compression_ratio_after_allocation,
        }


@dataclass(slots=True)
class DTAAllocationResult:
    group_indices: list[list[int]]
    metrics: DTAMetrics


def _dta_allocate(
    trajectories: list[dict[str, Any]],
    n_groups: int,
) -> DTAAllocationResult:
    from areal.experimental.dta.dp import LB_by_DFS_and_TM
    from areal.experimental.dta.token_trie import TokenTrie

    token_seqs: list[torch.Tensor] = []
    for idx, trajectory in enumerate(trajectories):
        try:
            seq = extract_single_valid_token_sequence(trajectory)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"Invalid trajectory format at index {idx} for DTA partitioning."
            ) from err
        token_seqs.append(seq)

    all_stats = TokenTrie(token_seqs).get_stats(mode="backward")
    n_total_tokens = float(all_stats["n_tokens"])
    n_tree_tokens_before = float(all_stats["n_tree_tokens"])

    config = SimpleNamespace(K=n_groups, mode="backward", block_size=None)
    group_indices = LB_by_DFS_and_TM(token_seqs, _TreeTokenOnlyTimeModel(), config)

    n_tree_tokens_after = 0.0
    for group in group_indices:
        if not group:
            continue
        group_token_seqs = [token_seqs[idx] for idx in group]
        group_stats = TokenTrie(group_token_seqs).get_stats(mode="backward")
        n_tree_tokens_after += float(group_stats["n_tree_tokens"])

    compression_ratio_before = (
        n_total_tokens / n_tree_tokens_before
        if n_tree_tokens_before > 0
        else float("nan")
    )
    compression_ratio_after = (
        n_total_tokens / n_tree_tokens_after
        if n_tree_tokens_after > 0
        else float("nan")
    )
    metrics = DTAMetrics(
        n_tokens=n_total_tokens,
        n_tree_tokens_before_allocation=n_tree_tokens_before,
        n_tree_tokens_after_allocation=n_tree_tokens_after,
        compression_ratio_before_allocation=compression_ratio_before,
        compression_ratio_after_allocation=compression_ratio_after,
    )
    return DTAAllocationResult(group_indices=group_indices, metrics=metrics)


def redistribute_trajectories(
    trajectories: list[dict[str, Any]],
    group=None,
    packing_algorithm: str = "ffd",
) -> RedistributedData:
    """Redistribute a list of trajectory dicts across a process group.

    Each trajectory dict should contain tensors with shape [batch_size, seqlen, *],
    where batch_size can vary per trajectory. This function gathers trajectories
    from all ranks and redistributes them for load balancing based on sequence lengths.

    Parameters
    ----------
    trajectories : list[dict[str, Any]]
        List of trajectory dictionaries from the local rank. Each trajectory
        contains tensors with shape [batch_size, seqlen, ...].
    group : dist.ProcessGroup, optional
        The process group for communication. If None, uses the default group.
    packing_algorithm : str, optional
        How to pack trajectories across data-parallel ranks: ``"ffd"`` or ``"kk"``
        balance by total sequence length; ``"dta"`` uses DTA DFS-order partitioning
        with ``n_tree_tokens`` as cost. Default ``"ffd"``.

    Returns
    -------
    RedistributedData
        Contains:
        - all_data: All trajectories gathered from all ranks (with padding removed)
        - data: List of trajectories assigned to the local rank
        - rank: Local rank in the group
        - group_indices: Assignment of trajectory indices to each rank
    """
    # All-gather trajectories from all ranks
    all_gathered = all_gather_tensor_container(trajectories, group=group)

    # Flatten the list of lists into a single list of trajectories
    all_data = []
    for traj_list in all_gathered:
        all_data.extend(traj_list)

    # Compute sequence lengths for load balancing
    seqlens = [get_total_valid_tokens(d) for d in all_data]

    # Remove pad positions from each trajectory (split_and_unpad_tensor
    # auto-derives trim lengths from attention_mask when traj_seqlens=None)
    all_data = [
        split_and_unpad_tensor(
            d, n_trajs=1, traj_group_sizes=[d["attention_mask"].shape[0]]
        )[0]
        for d in all_data
    ]

    n_groups = dist.get_world_size(group)
    if packing_algorithm == "dta":
        # Unpack group-level trajectories into sequence-level for DTA
        from areal.utils.data import unpack_groups_to_sequences

        all_data = unpack_groups_to_sequences(all_data)

        dta_result = _dta_allocate(all_data, n_groups)
        group_indices = dta_result.group_indices
        dta_metrics = dta_result.metrics
    elif packing_algorithm in ("ffd", "kk"):
        allocate_fn = get_allocate_fn(packing_algorithm)
        # Allocate trajectories to ranks using the configured packing algorithm
        # No capacity limit leads to balanced partition across this group
        group_indices = allocate_fn(
            seqlens, capacity=int(1e12), min_groups=dist.get_world_size(group)
        )
        dta_metrics = None
    else:
        raise ValueError(
            f"Unsupported packing_algorithm: {packing_algorithm!r}. "
            "Expected one of {'ffd', 'kk', 'dta'}."
        )
    _validate_group_indices(group_indices, n_groups=n_groups, n_items=len(all_data))

    # Select assigned trajectories for this rank (no concatenation — deferred to train side)
    local_indices = group_indices[dist.get_rank(group=group)]
    data = [all_data[i] for i in local_indices]
    return RedistributedData(
        all_data=all_data,
        data=data,
        rank=dist.get_rank(group=group),
        group_indices=group_indices,
        dta_metrics=dta_metrics,
    )


class DistRolloutCoordinator:
    def __init__(self, rollout_engine: InferenceEngine, train_engine: TrainEngine):
        self.rollout_engine = rollout_engine
        self.train_engine = train_engine

    def _broadcast_and_redistribute_trajectories(
        self,
        trajectories: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Broadcast and redistribute trajectories across distributed workers.

        This helper encapsulates:
        1. Redistribution within data parallel group (for load balancing)
        2. Broadcasting to context and model parallel group
        3. Synchronization barriers

        Parameters
        ----------
        trajectories : list[dict[str, Any]] | None
            List of trajectory dicts from data parallel head, None for other ranks.
            Each trajectory is a dict of tensors with shape [batch_size, seqlen, ...],
            where batch_size can vary per trajectory.

        Returns
        -------
        list[dict[str, Any]]
            Redistributed and broadcast batch available on all ranks (list of trajs)
        """
        rollout_packing = self.train_engine.config.packing_algorithm

        if trajectories is not None:
            redist = redistribute_trajectories(
                trajectories,
                group=self.train_engine.data_parallel_group,
                packing_algorithm=rollout_packing,
            )
            batch = redist.data
            dta_metrics_payload = [redist.dta_metrics]
        else:
            batch = None
            dta_metrics_payload = [None]

        current_platform.synchronize()
        dist.barrier(group=self.train_engine.cpu_group)

        dist.broadcast_object_list(
            dta_metrics_payload,
            src=self.train_engine.current_data_parallel_head(),
            group=self.train_engine.context_and_model_parallel_group,
        )
        dta_metrics = dta_metrics_payload[0]
        if dta_metrics is not None:
            stats_tracker.scalar(**dta_metrics.to_stats())

        batch = broadcast_tensor_container(
            batch,
            src_rank=self.train_engine.current_data_parallel_head(),
            group=self.train_engine.context_and_model_parallel_group,
        )

        current_platform.synchronize()
        dist.barrier(group=self.train_engine.cpu_group)

        return batch

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
    ) -> list[dict[str, Any]]:
        """Generate rollout batch with distributed coordination (synchronous).

        This method orchestrates distributed rollout generation:
        - Only data parallel heads generate rollouts (avoid redundancy)
        - Results are transferred to device and redistributed
        - Batch is broadcast to all workers
        - Synchronization barriers ensure consistency

        Must call connect_engine() before using this method.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            Input data batch for rollout generation
        workflow : WorkflowLike
            Workflow defining rollout logic
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        group_size : int, optional
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).

        Returns
        -------
        list[dict[str, Any]]
            Redistributed rollout trajectories on all ranks

        Raises
        ------
        RuntimeError
            If rollout engine not connected via connect_engine()
        """

        trajectories = None
        if self.train_engine.is_data_parallel_head():
            trajectories = self.rollout_engine.rollout_batch(
                data,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                group_size=group_size,
            )
            trajectories = tensor_container_to(
                trajectories, current_platform.current_device()
            )

        return self._broadcast_and_redistribute_trajectories(trajectories)

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ) -> list[dict[str, Any]]:
        """Prepare async rollout batch with distributed coordination.

        Similar to rollout_batch but uses prepare_batch for async training,
        where rollout generation happens concurrently with training.

        Must call connect_engine() before using this method.

        Parameters
        ----------
        dataloader : StatefulDataLoader
            Dataloader to pull samples from
        workflow : WorkflowLike
            Workflow defining rollout logic
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        should_accept_fn : Callable[[Dict[str, Any]], bool] | str, optional
            Filter function for accepting samples based on staleness
        group_size : int, optional
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).
        dynamic_bs : bool, optional
            If True, enables dynamic batch sizing. Default is False.

        Returns
        -------
        list[dict[str, Any]]
            Prepared rollout trajectories on all ranks

        Raises
        ------
        RuntimeError
            If rollout engine not connected via connect_engine()
        """

        trajectories = None
        if self.train_engine.is_data_parallel_head():
            trajectories = self.rollout_engine.prepare_batch(
                dataloader,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                should_accept_fn=should_accept_fn,
                group_size=group_size,
                dynamic_bs=dynamic_bs,
            )
            trajectories = tensor_container_to(
                trajectories, current_platform.current_device()
            )

        return self._broadcast_and_redistribute_trajectories(trajectories)
