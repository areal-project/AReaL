# SPDX-License-Identifier: Apache-2.0

"""Logprob and entropy computation utilities for packed rollout tries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import distributed as dist

from areal.models.tree_attn.mapped_logprobs import mapped_logprobs_entropy
from areal.utils.perf_tracer import trace_perf

if TYPE_CHECKING:
    from areal.models.tree_attn.tree import TrieNode


def tree_semantic_rows(
    trie: TrieNode,
    input_ids: torch.Tensor,
    *,
    rows_on_cpu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, slice]]:
    """Map each original trajectory position to its compact logit and label.

    The terminal row predicts the trajectory's first token, matching the
    existing sequence-aligned AReaL interface; its loss mask excludes it.
    """
    rows: list[int] = []
    label_positions: list[int] = []
    sequence_slices: dict[int, slice] = {}
    for sequence_id in trie.all_sequence_ids:
        path = [
            position
            for start, end in trie.get_sequence_tree_indices(sequence_id)
            for position in range(start, end + 1)
        ]
        start = len(rows)
        rows.extend(path)
        label_positions.extend(path[1:] + path[:1])
        sequence_slices[sequence_id] = slice(start, len(rows))

    device = input_ids.device
    row_indices = torch.tensor(
        rows, dtype=torch.long, device="cpu" if rows_on_cpu else device
    )
    label_indices = torch.tensor(label_positions, dtype=torch.long, device=device)
    labels = input_ids.reshape(-1).index_select(0, label_indices)
    return row_indices, labels, sequence_slices


@trace_perf("tree_attn._gather_packed_tree_logprobs")
def _gather_packed_tree_logprobs(
    logits: torch.Tensor,
    trie: TrieNode,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
    tp_group: dist.ProcessGroup | None = None,
) -> dict[int, torch.Tensor]:
    del chunk_size
    rows, labels, slices = tree_semantic_rows(trie, input_ids)
    logprobs, _ = mapped_logprobs_entropy(logits, rows, labels, temperature, tp_group)
    return {sequence_id: logprobs[span] for sequence_id, span in slices.items()}


@trace_perf("tree_attn.gather_packed_tree_logprobs")
def gather_packed_tree_logprobs(
    logits: torch.Tensor,
    trie: TrieNode,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
    tp_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    del chunk_size
    rows, labels, _ = tree_semantic_rows(trie, input_ids)
    return mapped_logprobs_entropy(logits, rows, labels, temperature, tp_group)[0]


@trace_perf("tree_attn.gather_packed_tree_logprobs_entropy")
def gather_packed_tree_logprobs_entropy(
    logits: torch.Tensor,
    trie: TrieNode,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
    tp_group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del chunk_size
    rows, labels, _ = tree_semantic_rows(trie, input_ids)
    return mapped_logprobs_entropy(logits, rows, labels, temperature, tp_group)


@trace_perf("tree_attn._gather_packed_tree_vocab_stats")
def _gather_packed_tree_vocab_stats(
    logits: torch.Tensor,
    trie: TrieNode,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Compute vocab min and max logits for all sequences in a packed tree.

    For tree training, vocab_min_logits and vocab_max_logits need to be unpacked
    from the tree structure back to per-sequence format.

    Unlike logprobs calculation which requires shifted labels, vocab stats only
    need the min/max values from logits at each prediction position. For a
    sequence with tree indices [(s0,e0), (s1,e1), ...], the prediction positions
    are: [s0:e0+1], [s1:e1+1], ...

    Parameters
    ----------
    logits : torch.Tensor
        Model output logits of shape (T, vocab_size). T is the padded tree size.
    trie : TrieNode
        Root TrieNode of the packed tree structure.

    Returns
    -------
    tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]
        Tuple of (vocab_min_dict, vocab_max_dict), where each dictionary maps
        sequence_id to the corresponding tensor of shape (seq_len - 1,).
    """
    vocab_min_results: dict[int, torch.Tensor] = {}
    vocab_max_results: dict[int, torch.Tensor] = {}
    device = logits.device
    dtype = torch.float

    # Compute vocab min/max for all positions once
    all_vocab_min = logits.detach().min(-1).values.float()  # (T,)
    all_vocab_max = logits.detach().max(-1).values.float()  # (T,)

    for seq_id in trie.all_sequence_ids:
        indices = trie.get_sequence_tree_indices(seq_id)
        if not indices:
            empty = torch.empty(0, device=device, dtype=dtype)
            vocab_min_results[seq_id] = empty
            vocab_max_results[seq_id] = empty
            continue

        # Gather vocab stats for each node segment [start, end] (inclusive)
        min_parts = [all_vocab_min[start : end + 1] for start, end in indices]
        max_parts = [all_vocab_max[start : end + 1] for start, end in indices]

        vocab_min_results[seq_id] = torch.cat(min_parts, dim=0)
        vocab_max_results[seq_id] = torch.cat(max_parts, dim=0)

    return vocab_min_results, vocab_max_results


@trace_perf("tree_attn.gather_packed_tree_vocab_stats")
def gather_packed_tree_vocab_stats(
    logits: torch.Tensor,
    trie: TrieNode,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute vocab min and max logits for all sequences in a packed tree.

    This is the public API that returns concatenated tensors instead of
    per-sequence dictionaries.

    Parameters
    ----------
    logits : torch.Tensor
        Model output logits of shape (T, vocab_size).
    trie : TrieNode
        Root TrieNode of the packed tree structure.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (vocab_min_logits, vocab_max_logits), each of shape
        (total_tokens - num_sequences,) where total_tokens is the sum of
        all sequence lengths.
    """
    # Handle empty/dummy trie
    if not trie.all_sequence_ids:
        empty = torch.empty(0, device=logits.device, dtype=torch.float)
        return empty, empty

    vocab_min_results, vocab_max_results = _gather_packed_tree_vocab_stats(logits, trie)

    # Pack results according to trie.all_sequence_ids
    vocab_min = torch.cat(
        [vocab_min_results[sid] for sid in trie.all_sequence_ids], dim=0
    )
    vocab_max = torch.cat(
        [vocab_max_results[sid] for sid in trie.all_sequence_ids], dim=0
    )
    return vocab_min, vocab_max


@trace_perf("tree_attn.merge_packed_tree_results")
def merge_packed_tree_results(
    results_list: list[dict[int, torch.Tensor]],
    batch_size: int,
    max_seq_len: int | None = None,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """Merge per-sequence results from multiple packed trees back to batch format.

    After computing logprobs (or other per-sequence values) for each microbatch,
    this function merges them back into a single tensor with the original batch
    ordering.

    Parameters
    ----------
    results_list : list[dict[int, torch.Tensor]]
        List of dictionaries from gather_packed_tree_logprobs,
        one per microbatch. Each dict maps sequence_id to tensor.
    batch_size : int
        Original batch size (number of sequences).
    max_seq_len : int or None, default=None
        Maximum sequence length for output tensor. If None,
        inferred from the maximum length in results.
    padding_value : float, default=0.0
        Value to use for padding shorter sequences.

    Returns
    -------
    torch.Tensor
        Tensor of shape (batch_size, max_seq_len) with merged results.
        Sequences are placed at their original positions (sequence_id).

    Raises
    ------
    ValueError
        If duplicate sequence_id is found across microbatches or if
        sequence_id exceeds batch_size.
    """
    # Combine all results from all microbatches
    combined: dict[int, torch.Tensor] = {}
    for results in results_list:
        for seq_id, tensor in results.items():
            if seq_id in combined:
                raise ValueError(
                    f"Duplicate sequence_id {seq_id} found across microbatches"
                )
            combined[seq_id] = tensor

    if not combined:
        device = torch.device("cpu")
        return torch.full((batch_size, max_seq_len or 0), padding_value, device=device)

    # Infer device and dtype from first tensor
    first_tensor = next(iter(combined.values()))
    device = first_tensor.device
    dtype = first_tensor.dtype

    # Determine max sequence length if not provided
    if max_seq_len is None:
        max_seq_len = max(t.shape[0] for t in combined.values()) if combined else 0

    # Create output tensor
    output = torch.full(
        (batch_size, max_seq_len), padding_value, dtype=dtype, device=device
    )

    # Place each sequence's results at the correct position
    for seq_id, tensor in combined.items():
        if seq_id >= batch_size:
            raise ValueError(f"sequence_id {seq_id} exceeds batch_size {batch_size}")
        seq_len = min(tensor.shape[0], max_seq_len)
        output[seq_id, :seq_len] = tensor[:seq_len]

    return output
