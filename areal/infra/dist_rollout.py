# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api import InferenceEngine, TrainEngine, WorkflowLike
from areal.infra.platforms import current_platform
from areal.utils.data import (
    all_gather_tensor_container,
    broadcast_tensor_container,
    split_and_unpad_tensor,
    tensor_container_to,
)
from areal.utils.seqpack import get_allocate_fn


@dataclass
class RedistributedData:
    all_data: list[dict[str, Any]]
    data: list[dict[str, Any]]
    rank: int
    group_indices: list[list[int]]


def _all_gather_ragged_trajectory_lists(
    trajectories: list[dict[str, Any]], group=None
) -> list[list[dict[str, Any]]]:
    """Gather variable-length trajectory lists without semantic placeholders."""
    world_size = dist.get_world_size(group)
    local_group_rank = dist.get_rank(group=group)
    lengths: list[int | None] = [None] * world_size
    dist.all_gather_object(lengths, len(trajectories), group=group)
    if all(length == len(trajectories) for length in lengths):
        return all_gather_tensor_container(trajectories, group=group)

    source_ranks = (
        list(range(world_size))
        if group is None
        else dist.get_process_group_ranks(group)
    )
    gathered: list[list[dict[str, Any]]] = []
    for source_group_rank, (source_rank, length) in enumerate(
        zip(source_ranks, lengths, strict=True)
    ):
        assert length is not None
        source_items = trajectories if local_group_rank == source_group_rank else None
        rank_items = []
        for index in range(length):
            item = broadcast_tensor_container(
                source_items[index] if source_items is not None else None,
                src_rank=source_rank,
                group=group,
            )
            assert isinstance(item, dict)
            rank_items.append(item)
        gathered.append(rank_items)
    return gathered


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
        Packing algorithm to use ("ffd" or "kk"). Default is "ffd".

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
    all_gathered = _all_gather_ragged_trajectory_lists(trajectories, group=group)

    # Flatten the list of lists into a single list of trajectories
    all_data = []
    for traj_list in all_gathered:
        all_data.extend(traj_list)

    world_size = dist.get_world_size(group)
    if len(all_data) < world_size:
        raise RuntimeError(
            f"Cannot redistribute {len(all_data)} trainable trajectory groups "
            f"across {world_size} data-parallel ranks"
        )

    # Compute sequence lengths for load balancing
    seqlens = [d["attention_mask"].sum().item() for d in all_data]

    # Remove pad positions from each trajectory (split_and_unpad_tensor
    # auto-derives trim lengths from attention_mask when traj_seqlens=None)
    all_data = [
        split_and_unpad_tensor(
            d, n_trajs=1, traj_group_sizes=[d["attention_mask"].shape[0]]
        )[0]
        for d in all_data
    ]

    allocate_fn = get_allocate_fn(packing_algorithm)
    # Allocate trajectories to ranks using the configured packing algorithm
    # No capacity limit leads to balanced partition across this group
    group_indices = allocate_fn(seqlens, capacity=int(1e12), min_groups=world_size)
    local_indices = group_indices[dist.get_rank(group=group)]

    # Select assigned trajectories for this rank (no concatenation — deferred to train side)
    data = [all_data[i] for i in local_indices]
    return RedistributedData(
        all_data=all_data,
        data=data,
        rank=dist.get_rank(group=group),
        group_indices=group_indices,
    )


class DistRolloutCoordinator:
    def __init__(self, rollout_engine: InferenceEngine, train_engine: TrainEngine):
        self.rollout_engine = rollout_engine
        self.train_engine = train_engine

    def _synchronize_head_error(self, error: str | None) -> str | None:
        if not self.train_engine.is_data_parallel_head() or not dist.is_initialized():
            return error
        head_errors: list[str | None] = [None] * dist.get_world_size(
            self.train_engine.data_parallel_group
        )
        dist.all_gather_object(
            head_errors, error, group=self.train_engine.data_parallel_group
        )
        return next((message for message in head_errors if message), None)

    def _broadcast_and_redistribute_trajectories(
        self,
        trajectories: list[dict[str, Any]] | None,
        preparation_error: str | None = None,
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
        error = self._synchronize_head_error(preparation_error)
        if trajectories is not None and error is None:
            try:
                config = getattr(self.train_engine, "config", None)
                mb_spec = getattr(config, "mb_spec", None)
                packing_algorithm = getattr(mb_spec, "packing_algorithm", "ffd")
                redist = redistribute_trajectories(
                    trajectories,
                    group=self.train_engine.data_parallel_group,
                    packing_algorithm=packing_algorithm,
                )
                batch = redist.data
            except Exception as exc:
                batch = None
                error = f"{type(exc).__name__}: {exc}"
        else:
            batch = None

        error = self._synchronize_head_error(error)
        error = broadcast_tensor_container(
            error,
            src_rank=self.train_engine.current_data_parallel_head(),
            group=self.train_engine.context_and_model_parallel_group,
        )
        if error is not None:
            raise RuntimeError(f"Rollout batch preparation failed: {error}")

        current_platform.synchronize()
        dist.barrier(group=self.train_engine.cpu_group)

        batch = broadcast_tensor_container(
            batch,
            src_rank=self.train_engine.current_data_parallel_head(),
            group=self.train_engine.context_and_model_parallel_group,
        )

        current_platform.synchronize()
        dist.barrier(group=self.train_engine.cpu_group)

        return batch

    def _global_trajectory_count(self, local_count: int) -> int:
        if not dist.is_initialized():
            return local_count
        group = self.train_engine.data_parallel_group
        counts: list[int | None] = [None] * dist.get_world_size(group)
        dist.all_gather_object(counts, local_count, group=group)
        return sum(count for count in counts if count is not None)

    def _prepare_on_data_parallel_head(
        self,
        prepare: Callable[[], list[dict[str, Any]]],
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        if not self.train_engine.is_data_parallel_head():
            return None, None
        try:
            trajectories = prepare()
            return (
                tensor_container_to(trajectories, current_platform.current_device()),
                None,
            )
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
        min_usable_group_size: int = 1,
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

        trajectories, preparation_error = self._prepare_on_data_parallel_head(
            lambda: self.rollout_engine.rollout_batch(
                data,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                group_size=group_size,
                min_usable_group_size=min_usable_group_size,
                reward_normalization=reward_normalization,
                drop_incomplete_group=drop_incomplete_group,
            )
        )

        return self._broadcast_and_redistribute_trajectories(
            trajectories, preparation_error=preparation_error
        )

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
        min_usable_group_size: int = 1,
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

        def _prepare_until_dispatchable() -> list[dict[str, Any]]:
            trajectories: list[dict[str, Any]] = []
            min_global_batch_size = (
                dist.get_world_size(self.train_engine.data_parallel_group)
                if dist.is_initialized()
                else 1
            )
            while True:
                local_error = None
                prepared: list[dict[str, Any]] = []
                try:
                    prepared = self.rollout_engine.prepare_batch(
                        dataloader,
                        workflow=workflow,
                        workflow_kwargs=workflow_kwargs,
                        should_accept_fn=should_accept_fn,
                        group_size=group_size,
                        min_usable_group_size=min_usable_group_size,
                        dynamic_bs=dynamic_bs,
                        reward_normalization=reward_normalization,
                        drop_incomplete_group=drop_incomplete_group,
                    )
                except Exception as exc:
                    local_error = f"{type(exc).__name__}: {exc}"
                coordinated_error = self._synchronize_head_error(local_error)
                if coordinated_error is not None:
                    raise RuntimeError(coordinated_error)
                trajectories.extend(prepared)
                global_batch_size = self._global_trajectory_count(len(trajectories))
                if global_batch_size >= min_global_batch_size:
                    return trajectories
                if not dynamic_bs:
                    raise RuntimeError(
                        "Fixed rollout preparation produced only "
                        f"{global_batch_size} trainable groups for "
                        f"{min_global_batch_size} data-parallel ranks"
                    )

        trajectories, preparation_error = self._prepare_on_data_parallel_head(
            _prepare_until_dispatchable
        )

        return self._broadcast_and_redistribute_trajectories(
            trajectories, preparation_error=preparation_error
        )
