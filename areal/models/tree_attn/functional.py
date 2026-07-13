# SPDX-License-Identifier: Apache-2.0

"""Logprob and entropy computation utilities for packed tree structures.

This module provides functions to compute log probabilities and entropy
for sequences packed into tree structures with shared prefixes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import distributed as dist

from areal.utils.functional.vocab_parallel import (
    gather_logprobs,
    gather_logprobs_entropy,
)
from areal.utils.perf_tracer import trace_perf

if TYPE_CHECKING:
    from areal.models.tree_attn.tree import TrieNode


def _compute_internal_node_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    start_idx: int,
    end_idx: int,
    temperature: float = 1.0,
    chunk_size: int = 1024,
    tp_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Compute logprobs for internal predictions within a single trie node.

    For a node spanning [start_idx, end_idx], computes logprobs for predicting
    tokens at positions [start_idx+1, ..., end_idx] given positions
    [start_idx, ..., end_idx-1].

    Parameters
    ----------
    logits : torch.Tensor
        Full logits tensor of shape (T, vocab_size) or (T, vocab_size/tp).
    input_ids : torch.Tensor
        Full input IDs tensor of shape (T,).
    start_idx : int
        Start index of the node.
    end_idx : int
        End index of the node (inclusive).
    temperature : float, default=1.0
        Softmax temperature scaling.
    chunk_size : int, default=1024
        Maximum chunk size for memory-efficient processing.
    tp_group : dist.ProcessGroup or None, default=None
        Tensor parallel process group for vocab-parallel computation.

    Returns
    -------
    torch.Tensor
        Logprobs tensor of shape (end_idx - start_idx,) for internal
        predictions. Empty tensor if node has only one token.
    """
    num_internal = end_idx - start_idx
    if num_internal <= 0:
        return torch.empty(0, device=logits.device, dtype=logits.dtype)

    # Prediction positions and corresponding labels
    pred_start, pred_end = start_idx, end_idx
    label_start, label_end = start_idx + 1, end_idx + 1

    pred_logits = logits[pred_start:pred_end]
    labels = input_ids[label_start:label_end]

    return gather_logprobs(
        pred_logits, labels, temperature, tp_group=tp_group, chunk_size=chunk_size
    )


def _compute_internal_node_logprobs_entropy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    start_idx: int,
    end_idx: int,
    temperature: float = 1.0,
    chunk_size: int = 1024,
    tp_group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute logprobs and entropy for internal predictions within a node.

    Parameters
    ----------
    logits : torch.Tensor
        Full logits tensor of shape (T, vocab_size) or (T, vocab_size/tp).
    input_ids : torch.Tensor
        Full input IDs tensor of shape (T,).
    start_idx : int
        Start index of the node.
    end_idx : int
        End index of the node (inclusive).
    temperature : float, default=1.0
        Softmax temperature scaling.
    chunk_size : int, default=1024
        Maximum chunk size for memory-efficient processing.
    tp_group : dist.ProcessGroup or None, default=None
        Tensor parallel process group for vocab-parallel computation.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (logprobs, entropy) tensors, each of shape
        (end_idx - start_idx,).
    """
    num_internal = end_idx - start_idx
    if num_internal <= 0:
        empty = torch.empty(0, device=logits.device, dtype=logits.dtype)
        return empty, empty

    pred_start, pred_end = start_idx, end_idx
    label_start, label_end = start_idx + 1, end_idx + 1

    pred_logits = logits[pred_start:pred_end]
    labels = input_ids[label_start:label_end]

    return gather_logprobs_entropy(
        pred_logits, labels, temperature, tp_group=tp_group, chunk_size=chunk_size
    )


def _compute_transition_logprob(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    pred_pos: int,
    label_pos: int,
    temperature: float = 1.0,
    tp_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Compute logprob for a single transition between nodes.

    Parameters
    ----------
    logits : torch.Tensor
        Full logits tensor of shape (T, vocab_size) or (T, vocab_size/tp).
    input_ids : torch.Tensor
        Full input IDs tensor of shape (T,).
    pred_pos : int
        Position of the prediction logit (last position of parent node).
    label_pos : int
        Position of the label token (first position of child node).
    temperature : float, default=1.0
        Softmax temperature scaling.
    tp_group : dist.ProcessGroup or None, default=None
        Tensor parallel process group for vocab-parallel computation.

    Returns
    -------
    torch.Tensor
        Scalar logprob tensor.
    """
    pred_logit = logits[pred_pos : pred_pos + 1]  # (1, vocab_size)
    label = input_ids[label_pos : label_pos + 1]  # (1,)
    return gather_logprobs(
        pred_logit, label, temperature, tp_group=tp_group, chunk_size=1
    ).squeeze(0)


def _compute_transition_logprob_entropy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    pred_pos: int,
    label_pos: int,
    temperature: float = 1.0,
    tp_group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute logprob and entropy for a single transition between nodes.

    Parameters
    ----------
    logits : torch.Tensor
        Full logits tensor of shape (T, vocab_size) or (T, vocab_size/tp).
    input_ids : torch.Tensor
        Full input IDs tensor of shape (T,).
    pred_pos : int
        Position of the prediction logit (last position of parent node).
    label_pos : int
        Position of the label token (first position of child node).
    temperature : float, default=1.0
        Softmax temperature scaling.
    tp_group : dist.ProcessGroup or None, default=None
        Tensor parallel process group for vocab-parallel computation.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (logprob, entropy), both scalar tensors.
    """
    pred_logit = logits[pred_pos : pred_pos + 1]
    label = input_ids[label_pos : label_pos + 1]
    lp, ent = gather_logprobs_entropy(
        pred_logit, label, temperature, tp_group=tp_group, chunk_size=1
    )
    return lp.squeeze(0), ent.squeeze(0)


@trace_perf("tree_attn._gather_packed_tree_logprobs")
def _gather_packed_tree_logprobs(
    logits: torch.Tensor,
    trie: TrieNode,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
    tp_group: dist.ProcessGroup | None = None,
) -> dict[int, torch.Tensor]:
    """Compute log probabilities for all sequences in a packed tree.

    In a packed tree, multiple sequences share prefix tokens. The labels for
    computing log probabilities cannot be obtained by simply rolling input_ids.
    Instead, we need to recover each original sequence from the tree structure
    and compute logprobs based on the correct next-token labels.

    This implementation includes two optimizations:

    1. Chunked computation: Processes logprobs in chunks to reduce peak memory.
    2. Node-level caching: Caches internal node logprobs and transition logprobs
       since sequences sharing prefixes will have identical logprobs for those
       shared portions.

    Parameters
    ----------
    logits : torch.Tensor
        Model output logits of shape (T, vocab_size) or (T, vocab_size/tp)
        when tensor parallelism is enabled. T is the padded tree size.
        Output logits[i] corresponds to input_ids[i].
    trie : TrieNode
        Root TrieNode of the packed tree structure.
    input_ids : torch.Tensor
        Packed input IDs of shape (T,).
    temperature : float, default=1.0
        Softmax temperature scaling.
    chunk_size : int, default=1024
        Maximum chunk size for memory-efficient processing.
    tp_group : dist.ProcessGroup or None, default=None
        Tensor parallel process group for vocab-parallel computation.
        If provided with tp_size > 1, uses vocab-parallel computation.

    Returns
    -------
    dict[int, torch.Tensor]
        Dictionary mapping sequence_id to logprobs tensor.
        Each logprobs tensor has shape (seq_len - 1,) where seq_len is the
        length of that sequence. The logprobs are aligned such that logprobs[i]
        is the log probability of predicting the (i+1)-th token given tokens
        0..i.

    Examples
    --------
    For sequences [A, B, C, D] and [A, B, E, F] packed into a tree:

    - Shared prefix [A, B] at positions [0, 1]
    - Sequence 0: [A, B, C, D] at positions [0, 1, 2, 4]
    - Sequence 1: [A, B, E, F] at positions [0, 1, 3, 5]

    For sequence 0, logprobs are computed as:

    - logprobs[0] = log P(B | A) using logits[0]
    - logprobs[1] = log P(C | A, B) using logits[1]
    - logprobs[2] = log P(D | A, B, C) using logits[2]

    The logprob for P(B | A) is cached and reused for sequence 1.
    """
    results: dict[int, torch.Tensor] = {}
    device = logits.device
    dtype = torch.float  # logprobs should always be float
    input_ids = input_ids.squeeze(0)
    # Cached implementation with chunking
    # Cache for internal node logprobs: (start_idx, end_idx) -> tensor
    node_cache: dict[tuple[int, int], torch.Tensor] = {}
    # Cache for transition logprobs: (pred_pos, label_pos) -> scalar tensor
    transition_cache: dict[tuple[int, int], torch.Tensor] = {}

    for seq_id in trie.all_sequence_ids:
        indices = trie.get_sequence_tree_indices(seq_id)
        if not indices:
            results[seq_id] = torch.empty(0, device=device, dtype=dtype)
            continue

        logprob_parts: list[torch.Tensor] = []

        for i, (start, end) in enumerate(indices):
            # Compute or retrieve cached internal logprobs for this node
            node_key = (start, end)
            if node_key not in node_cache:
                node_cache[node_key] = _compute_internal_node_logprobs(
                    logits, input_ids, start, end, temperature, chunk_size, tp_group
                )
            internal_logprobs = node_cache[node_key]
            if internal_logprobs.numel() > 0:
                logprob_parts.append(internal_logprobs)

            # Compute or retrieve cached transition logprob to next node
            next_start = 0
            if i + 1 < len(indices):
                next_start, _ = indices[i + 1]
            trans_key = (end, next_start)
            if trans_key not in transition_cache:
                transition_cache[trans_key] = _compute_transition_logprob(
                    logits, input_ids, end, next_start, temperature, tp_group
                )
            logprob_parts.append(transition_cache[trans_key].unsqueeze(0))

        if logprob_parts:
            results[seq_id] = torch.cat(logprob_parts, dim=0)
        else:
            results[seq_id] = torch.empty(0, device=device, dtype=dtype)

    return results


@trace_perf("tree_attn._gather_packed_tree_logprobs_entropy")
def _gather_packed_tree_logprobs_entropy(
    logits: torch.Tensor,
    trie: TrieNode,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
    tp_group: dist.ProcessGroup | None = None,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Compute log probabilities and entropy for all sequences in a packed tree.

    Similar to gather_packed_tree_logprobs but also computes entropy.
    Includes chunked computation and node-level caching optimizations.

    Parameters
    ----------
    logits : torch.Tensor
        Model output logits of shape (T, vocab_size) or (T, vocab_size/tp)
        when tensor parallelism is enabled.
    trie : TrieNode
        Root TrieNode of the packed tree structure.
    input_ids : torch.Tensor
        Packed input IDs of shape (T,).
    temperature : float, default=1.0
        Softmax temperature scaling.
    chunk_size : int, default=1024
        Maximum chunk size for memory-efficient processing.
    tp_group : dist.ProcessGroup or None, default=None
        Tensor parallel process group for vocab-parallel computation.
        If provided with tp_size > 1, uses vocab-parallel computation.

    Returns
    -------
    tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]
        Tuple of (logprobs_dict, entropy_dict), where each dictionary maps
        sequence_id to the corresponding tensor of shape (seq_len - 1,).
    """
    logprobs_results: dict[int, torch.Tensor] = {}
    entropy_results: dict[int, torch.Tensor] = {}
    device = logits.device
    dtype = torch.float  # logprobs should always be float
    input_ids = input_ids.squeeze(0)
    # Cached implementation with chunking
    # Cache for internal node results: (start_idx, end_idx) -> (logprobs, entropy)
    node_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    # Cache for transition results: (pred_pos, label_pos) -> (logprob, entropy)
    transition_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    for seq_id in trie.all_sequence_ids:
        indices = trie.get_sequence_tree_indices(seq_id)
        if not indices:
            empty = torch.empty(0, device=device, dtype=dtype)
            logprobs_results[seq_id] = empty
            entropy_results[seq_id] = empty
            continue

        logprob_parts: list[torch.Tensor] = []
        entropy_parts: list[torch.Tensor] = []

        for i, (start, end) in enumerate(indices):
            # Compute or retrieve cached internal results for this node
            node_key = (start, end)
            if node_key not in node_cache:
                node_cache[node_key] = _compute_internal_node_logprobs_entropy(
                    logits, input_ids, start, end, temperature, chunk_size, tp_group
                )
            internal_logprobs, internal_entropy = node_cache[node_key]
            if internal_logprobs.numel() > 0:
                logprob_parts.append(internal_logprobs)
                entropy_parts.append(internal_entropy)

            # Compute or retrieve cached transition results to next node
            next_start = 0
            if i + 1 < len(indices):
                next_start, _ = indices[i + 1]
            trans_key = (end, next_start)
            if trans_key not in transition_cache:
                transition_cache[trans_key] = _compute_transition_logprob_entropy(
                    logits, input_ids, end, next_start, temperature, tp_group
                )
            trans_lp, trans_ent = transition_cache[trans_key]
            logprob_parts.append(trans_lp.unsqueeze(0))
            entropy_parts.append(trans_ent.unsqueeze(0))

        if logprob_parts:
            logprobs_results[seq_id] = torch.cat(logprob_parts, dim=0)
            entropy_results[seq_id] = torch.cat(entropy_parts, dim=0)
        else:
            empty = torch.empty(0, device=device, dtype=dtype)
            logprobs_results[seq_id] = empty
            entropy_results[seq_id] = empty

    return logprobs_results, entropy_results


@trace_perf("tree_attn.gather_packed_tree_logprobs")
def gather_packed_tree_logprobs(
    logits: torch.Tensor,
    trie: TrieNode,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
    tp_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    # Handle empty/dummy trie
    if not trie.all_sequence_ids:
        # logprobs/entropy should always be float
        return torch.empty(0, device=logits.device, dtype=torch.float)

    logprob_results = _gather_packed_tree_logprobs(
        logits,
        trie,
        input_ids,
        temperature,
        chunk_size,
        tp_group,
    )
    # Pack sequence logprobs according to trie.all_sequence_ids
    logprob = torch.cat([logprob_results[sid] for sid in trie.all_sequence_ids], dim=0)
    return logprob


@trace_perf("tree_attn.gather_packed_tree_logprobs_entropy")
def gather_packed_tree_logprobs_entropy(
    logits: torch.Tensor,
    trie: TrieNode,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
    tp_group: dist.ProcessGroup | None = None,
):
    # Handle empty/dummy trie
    if not trie.all_sequence_ids:
        # logprobs/entropy should always be float
        empty = torch.empty(0, device=logits.device, dtype=torch.float)
        return empty, empty

    logprob_results, entropy_results = _gather_packed_tree_logprobs_entropy(
        logits,
        trie,
        input_ids,
        temperature,
        chunk_size,
        tp_group,
    )
    # Pack sequence logprobs and entropy according to trie.all_sequence_ids
    logprob = torch.cat([logprob_results[sid] for sid in trie.all_sequence_ids], dim=0)
    entropy = torch.cat([entropy_results[sid] for sid in trie.all_sequence_ids], dim=0)
    return logprob, entropy


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
