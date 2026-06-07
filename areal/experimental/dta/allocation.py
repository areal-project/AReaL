# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from areal.experimental.dta.dp import DTAPartitionResult, LB_by_DFS_and_TM
from areal.utils.data import extract_single_valid_token_sequence


class _TreeTokenOnlyTimeModel:
    def pred(self, stats: dict[str, Any]) -> float:
        return float(stats["n_tree_tokens"])


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
    items: list[dict[str, Any]]
    group_indices: list[list[int]]
    metrics: DTAMetrics


def _extract_token_sequences(
    trajectories: list[dict[str, Any]],
) -> list[torch.Tensor]:
    token_seqs: list[torch.Tensor] = []
    for idx, trajectory in enumerate(trajectories):
        try:
            seq = extract_single_valid_token_sequence(trajectory)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"Invalid trajectory format at index {idx} for DTA partitioning."
            ) from err
        token_seqs.append(seq)
    return token_seqs


def allocate_dta_trajectories(
    trajectories: list[dict[str, Any]], n_groups: int
) -> DTAAllocationResult:
    """Prepare sequence-level DTA trajectories and allocate them across DP groups."""
    from areal.utils.data import unpack_groups_to_sequences

    items = unpack_groups_to_sequences(trajectories)
    token_seqs = _extract_token_sequences(items)
    config = SimpleNamespace(K=n_groups, mode="backward", block_size=None)
    partition = LB_by_DFS_and_TM(token_seqs, _TreeTokenOnlyTimeModel(), config)
    return DTAAllocationResult(
        items=items,
        group_indices=partition.bins,
        metrics=_compute_dta_metrics_from_partition(partition),
    )


def split_dta_allocation(
    allocation: DTAAllocationResult,
) -> list[list[dict[str, Any]]]:
    return [
        [allocation.items[idx] for idx in group_indices]
        for group_indices in allocation.group_indices
    ]


def _compute_dta_metrics_from_partition(partition: DTAPartitionResult) -> DTAMetrics:
    all_stats = partition.token_trie.get_stats(mode="backward")
    n_total_tokens = float(all_stats["n_tokens"])
    n_tree_tokens_before = float(all_stats["n_tree_tokens"])

    n_tree_tokens_after = 0.0
    for leaf_group in partition.leaf_bins:
        if not leaf_group:
            continue
        group_trie = partition.compressed_trie.get_subtrie(set(leaf_group))
        group_stats = group_trie.get_stats(mode="backward")
        n_tree_tokens_after += float(group_stats["n_tree_tokens"])

    return _make_dta_metrics(
        n_total_tokens=n_total_tokens,
        n_tree_tokens_before=n_tree_tokens_before,
        n_tree_tokens_after=n_tree_tokens_after,
    )


def _make_dta_metrics(
    n_total_tokens: float, n_tree_tokens_before: float, n_tree_tokens_after: float
) -> DTAMetrics:
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
    return DTAMetrics(
        n_tokens=n_total_tokens,
        n_tree_tokens_before_allocation=n_tree_tokens_before,
        n_tree_tokens_after_allocation=n_tree_tokens_after,
        compression_ratio_before_allocation=compression_ratio_before,
        compression_ratio_after_allocation=compression_ratio_after,
    )
