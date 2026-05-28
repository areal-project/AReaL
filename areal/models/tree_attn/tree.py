# SPDX-License-Identifier: Apache-2.0

"""Tree-based sequence packing for efficient training with shared prefixes.

This module provides utilities for building trie structures from sequences,
packing them into tree-structured inputs for efficient training with shared
prefix computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import BlockMask

from areal.api.cli_args import MicroBatchSpec
from areal.models.tree_attn.constants import BLOCK_SIZE, USE_TRITON_TREE_ATTN
from areal.models.tree_attn.module_fsdp import create_block_mask_from_dense
from areal.models.tree_attn.triton_kernel import (
    TRITON_AVAILABLE,
    TreeAttentionData,
    precompute_tree_attention_data,
)
from areal.utils import logging, stats_tracker
from areal.utils.data import MicroBatchList
from areal.utils.perf_tracer import trace_perf, trace_scope

logger = logging.getLogger("TreeAttentionCore")


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class TrieNode:
    """A node in a compressed trie (prefix tree) for token sequences.

    Each node represents a contiguous run of tokens that share the same set of
    sequences. When sequences diverge, child nodes are created.

    For root nodes, start_idx and end_idx are -1, and the node contains no tokens.
    Root nodes use the `nodes` list to track all descendant nodes in pre-order.

    Attributes
    ----------
    tree_id : int
        Identifier of the tree this node belongs to.
    start_idx : int
        Starting index in the flattened tree representation (-1 for root).
    end_idx : int
        Ending index (inclusive) in the flattened tree representation (-1 for root).
    tokens : list[int]
        List of token IDs stored in this node (empty for root).
    sequence_ids : list[int]
        IDs of sequences that pass through this node.
    children : dict[int, TrieNode]
        Child nodes keyed by the first diverging token.
    ancestors : list[TrieNode]
        List of ancestor nodes from root to parent (empty for root).
    nodes : list[TrieNode]
        All descendant nodes in pre-order traversal (only used by root).
    """

    tree_id: int
    start_idx: int = -1
    end_idx: int = -1
    tokens: list[int] = field(default_factory=list)
    sequence_ids: list[int] = field(default_factory=list)
    children: dict[int, TrieNode] = field(default_factory=dict)
    ancestors: list[TrieNode] = field(default_factory=list)
    nodes: list[TrieNode] = field(default_factory=list)

    @property
    def is_root(self) -> bool:
        """Check if this is a root node."""
        return self.start_idx == -1 and self.end_idx == -1

    @property
    def num_tokens(self) -> int:
        """Number of tokens stored in this node (or total for root)."""
        if self.is_root:
            return sum(node.num_tokens for node in self.nodes)
        return len(self.tokens)

    @property
    def tree_indices(self) -> tuple[int, int]:
        """Return (start_idx, end_idx) tuple for this node's position in tree."""
        return (self.start_idx, self.end_idx)

    @property
    def all_sequence_ids(self) -> list[int]:
        """Get sorted list of all sequence IDs in this tree (for root nodes)."""
        if self.is_root:
            seq_ids: set[int] = set()
            for node in self.nodes:
                seq_ids.update(node.sequence_ids)
            return sorted(seq_ids)
        return sorted(set(self.sequence_ids))

    def get_all_tree_indices(self) -> list[tuple[int, int]]:
        """Get tree indices for this node and all ancestors."""
        indices = [ancestor.tree_indices for ancestor in self.ancestors]
        if not self.is_root:
            indices.append(self.tree_indices)
        return indices

    def get_sequence_tree_indices(self, seq_id: int) -> list[tuple[int, int]]:
        """Get all tree index ranges for a given sequence ID (for root nodes)."""
        indices = []
        for node in self.nodes:
            if seq_id in node.sequence_ids:
                indices.append(node.tree_indices)
        return indices


# =============================================================================
# Internal Trie Building Helpers
# =============================================================================


class _BuildNode:
    """Temporary node structure used during trie construction."""

    __slots__ = ("tree_id", "token_id", "node_id", "children", "is_end", "sequence_ids")

    def __init__(self, tree_id: int, token_id: int, node_id: int):
        self.tree_id = tree_id
        self.token_id = token_id
        self.node_id = node_id
        self.children: dict[int, _BuildNode] = {}
        self.is_end = False
        self.sequence_ids: list[int] = []


def _count_additional_nodes(root: _BuildNode, sequence: list[int]) -> int:
    """Count how many new nodes are needed to insert sequence into root."""
    current = root
    for idx, token in enumerate(sequence):
        child = current.children.get(token)
        if child is None:
            return len(sequence) - idx
        current = child
    return 0


def _insert_sequence(
    root: _BuildNode,
    all_nodes: list[_BuildNode],
    sequence: list[int],
    tree_id: int,
    sequence_id: int,
) -> None:
    """Insert a sequence into the trie, mutating it in-place."""
    current = root
    for token in sequence:
        if token not in current.children:
            node_id = len(all_nodes)
            current.children[token] = _BuildNode(tree_id, token, node_id)
            all_nodes.append(current.children[token])
        current.children[token].sequence_ids.append(sequence_id)
        current = current.children[token]
    current.is_end = True


@trace_perf("tree_attn._compress_trie")
def _compress_trie(root: _BuildNode) -> TrieNode:
    """Compress a trie by merging linear chains into single TrieNodes."""
    trie_root = TrieNode(tree_id=root.tree_id)

    def _compress_chain(
        node: _BuildNode,
        ancestors: list[TrieNode],
    ) -> TrieNode:
        """Compress a chain of nodes with single children into one TrieNode."""
        tokens: list[int] = []
        current = node
        start_id = node.node_id

        # Follow single-child chains
        while True:
            tokens.append(current.token_id)
            if len(current.children) != 1 or current.is_end:
                break
            # Get the single child
            next_child = next(iter(current.children.values()))
            # Verify consistency
            if current.sequence_ids != next_child.sequence_ids:
                raise ValueError(
                    f"Sequence IDs mismatch along chain: "
                    f"{current.sequence_ids} vs {next_child.sequence_ids}"
                )
            if next_child.node_id != current.node_id + 1:
                raise ValueError("Node IDs not consecutive along chain.")
            current = next_child

        # Create compressed node
        trie_node = TrieNode(
            tree_id=root.tree_id,
            start_idx=start_id,
            end_idx=current.node_id,
            tokens=tokens,
            sequence_ids=current.sequence_ids.copy(),
            ancestors=ancestors.copy(),
        )
        trie_root.nodes.append(trie_node)

        # Recursively compress children
        if current.children:
            for token, child in sorted(current.children.items()):
                trie_node.children[token] = _compress_chain(
                    child,
                    ancestors + [trie_node],
                )

        return trie_node

    # Compress all children of root
    if root.children:
        for token, child in sorted(root.children.items()):
            trie_root.children[token] = _compress_chain(child, [])

    return trie_root


def trie_to_parent_array(trie: TrieNode, max_tokens: int) -> torch.Tensor:
    """Build a parent array from TrieNode structure.

    The parent array `fa` is a 1D tensor where `fa[i]` is the index of the
    parent token of token `i`. For root tokens, the parent index is -1.
    In this tree structure, all tokens within a compressed trie node
    share the same parent, which is the last token of the parent node.

    Args:
        trie: The root TrieNode.
        max_tokens: Maximum number of tokens (length of the output tensor).

    Returns:
        torch.Tensor: Parent array of shape (1, max_tokens) with dtype int32.
    """
    fa = torch.full((1, max_tokens), -1, dtype=torch.int32)

    if not trie.nodes:  # Empty/dummy trie
        return fa

    for node in trie.nodes:
        parent_end_pos = -1
        if node.ancestors:  # Has parent
            parent_node = node.ancestors[-1]  # Last ancestor is parent
            parent_end_pos = parent_node.end_idx

        # First token in node attends to parent; internal tokens attend to previous token
        if node.start_idx >= 0 and node.start_idx < max_tokens:
            fa[0, node.start_idx] = parent_end_pos
        for pos in range(node.start_idx + 1, node.end_idx + 1):
            if pos < max_tokens:
                fa[0, pos] = pos - 1

    return fa


# =============================================================================
# Public API
# =============================================================================


@trace_perf("tree_attn.build_packed_tree_batch")
def build_packed_tree_batch(
    data: dict[str, Any],
    mb_spec: MicroBatchSpec,
    pad_to_maximum: bool = True,
    dp_group: dist.ProcessGroup | None = None,
    parallel_size: int = 1,
) -> MicroBatchList:
    """Build a MicroBatchList from input data using greedy trie packing.

    This function constructs tries from input sequences using a greedy packing
    strategy, then converts them into tree-packed format suitable for training.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary containing 'input_ids' and 'attention_mask' tensors
        describing the batch of sequences. Shape: [batch_size, seq_len].
    mb_spec : MicroBatchSpec
        MicroBatchSpec containing max_tokens_per_mb for tree packing.
        Note: n_mbs, granularity, and n_mbs_divisor are not used in tree
        training and will trigger warnings if set to non-default values.
    pad_to_maximum : bool, default=True
        If True, pad all trees to max_tokens_per_mb.
    dp_group : dist.ProcessGroup | None, default=None
        Data parallel process group. Default (None) is the world group.
        If torch.distributed is initialized, synchronizes the number of
        trees across all ranks by appending dummy trees to ranks with fewer
        trees.
    parallel_size : int, default=1
        Product of parallelism dimensions (e.g. TP, SP) that require
        BLOCK_SIZE alignment. The actual alignment is
        ``math.lcm(BLOCK_SIZE, parallel_size)``.

    Returns
    -------
    MicroBatchList
        MicroBatchList containing all packed tree data.

    Raises
    ------
    ValueError
        If max_tokens_per_mb is None or not positive, or if padding
        constraints are violated, or if a sequence exceeds max_tokens_per_mb.
    """
    # Warn about non-effective attributes
    if mb_spec.n_mbs != 1 or mb_spec.granularity != 1 or mb_spec.n_mbs_divisor != 1:
        logger.warning(
            "`n_mbs`, `granularity` and `n_mbs_divisor` is currently not effective for tree packing."
        )

    max_tokens_per_tree = mb_spec.max_tokens_per_mb
    if max_tokens_per_tree is None or max_tokens_per_tree <= 0:
        raise ValueError(
            "MicroBatchSpec.max_tokens_per_mb must be a positive value for tree training."
        )

    # Validate padding constraints for block masks
    if not pad_to_maximum:
        raise ValueError(
            "No padding is not supported for tree training. "
            "Block masks require padded sequences for efficient computation. "
            "Please set pad_to_maximum=True."
        )
    # Block masks operate at BLOCK_SIZE granularity. parallel_size comes from
    # TP/SP dimensions that may impose additional alignment requirements.
    # Currently parallel_size always divides BLOCK_SIZE (e.g. TP=1,2,4,8 vs
    # BLOCK_SIZE=128), so lcm == BLOCK_SIZE. If a future parallel_size doesn't
    # divide BLOCK_SIZE, lcm will enforce the stricter alignment automatically.
    block_align = math.lcm(BLOCK_SIZE, parallel_size)
    if pad_to_maximum and max_tokens_per_tree % block_align != 0:
        raise ValueError(
            f"max_tokens_per_tree ({max_tokens_per_tree}) must be a multiple of "
            f"{block_align} (lcm of BLOCK_SIZE={BLOCK_SIZE} and "
            f"parallel_size={parallel_size}) when pad_to_maximum=True."
        )

    # Build tries using greedy packing
    tries, num_tokens_list = _greedy_build_tries(data, max_tokens_per_tree)

    # Synchronize number of trees across dp_group.
    if dist.is_initialized():
        num_trees = len(tries)
        input_template: torch.Tensor = data["input_ids"]

        # All-gather tree counts from all ranks
        local_count = torch.tensor(
            [num_trees], dtype=torch.int64, device=input_template.device
        )
        world_size = dist.get_world_size(dp_group)
        all_counts = [
            torch.zeros(1, dtype=torch.int64, device=input_template.device)
            for _ in range(world_size)
        ]
        dist.all_gather(all_counts, local_count, group=dp_group)

        # Find the maximum tree count across all ranks
        max_num_trees = max(c.item() for c in all_counts)

        # If this rank has fewer trees, append dummy trees
        if num_trees < max_num_trees:
            num_dummy_trees = max_num_trees - num_trees
            for _ in range(num_dummy_trees):
                # Create an empty dummy trie
                dummy_tree_id = len(tries)
                dummy_trie = TrieNode(tree_id=dummy_tree_id)
                tries.append(dummy_trie)
                num_tokens_list.append(0)

    # Prepare templates and metadata
    input_template: torch.Tensor = data["input_ids"]
    mask_template: torch.Tensor = data["attention_mask"]

    # Directly track tree token ratio statistic
    original_num_tokens = mask_template.sum()
    total_tree_tokens = sum(num_tokens_list)
    ratio = total_tree_tokens / original_num_tokens
    stats_tracker.scalar(tree_token_ratio=ratio)

    sequence_lens = mask_template.sum(dim=1, dtype=torch.int32)

    # Identify packable keys (same shape as input_ids)
    packable_keys = {
        key
        for key, value in data.items()
        if key not in {"input_ids", "attention_mask"}
        and torch.is_tensor(value)
        and value.shape == input_template.shape
    }
    non_packable_keys = (
        set(data.keys()) - packable_keys - {"input_ids", "attention_mask"}
    )

    # Build packed outputs for each tree
    mbs: list[dict[str, Any]] = []
    padding_lengths: list[int] = []
    padded_to_lengths: list[int] = []

    for trie, num_tokens in zip(tries, num_tokens_list):
        # Compute padded size based on padding options
        padded_size = max_tokens_per_tree if pad_to_maximum else num_tokens

        # Pack input_ids
        with trace_scope("tree_attn.pack_input_ids"):
            input_ids = _pack_input_ids(
                trie,
                input_template,
                padded_size,
            )

        # Build dense attention mask (temporary, for position_ids computation)
        with trace_scope("tree_attn.build_attention_mask"):
            attention_mask = _build_attention_mask(
                trie,
                padded_size,
                mask_template.device,
            )

        # Compute position_ids (needs dense attention_mask)
        with trace_scope("tree_attn.get_position_ids"):
            position_ids = get_packed_tree_position_ids(
                input_ids,
                attention_mask,
            )

        # Release dense attention mask memory after position_ids are computed
        # Block mask will be lazily created in forward_backward_batch
        del attention_mask

        # Pack extra data
        with trace_scope("tree_attn.pack_extra_data"):
            extra_data = _pack_extra_data(
                trie,
                data,
                sequence_lens,
                packable_keys,
                non_packable_keys,
            )

        mb = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "trie_node": trie,
            **extra_data,
        }
        mbs.append(mb)
        padding_lengths.append(padded_size - num_tokens)
        padded_to_lengths.append(padded_size)

    # NOTE: mbs is padded data instead of original data
    # to avoid duplicate attention mask memory consumption
    batch = MicroBatchList(
        data=data,
        mb_spec=mb_spec,
        mbs=mbs,
        group_lens=[num for num in num_tokens_list],
        padded_mbs=mbs,
        padding_lengths=padding_lengths,
        padded_to_lengths=padded_to_lengths,
        _max_seqlen=max(padded_to_lengths),
    )
    return batch


@trace_perf("tree_attn._greedy_build_tries")
def _greedy_build_tries(
    data: dict[str, Any],
    max_tokens_per_tree: int,
) -> tuple[list[TrieNode], list[int]]:
    """Build tries using greedy packing strategy."""
    sequences = _extract_sequences(data)
    forests: list[dict[str, Any]] = []

    for seq_id, seq in enumerate(sequences):
        inserted = False

        # Try to insert into existing trees
        for tree_id, tree in enumerate(forests):
            additional = _count_additional_nodes(tree["root"], seq)
            if tree["nodes"] + additional <= max_tokens_per_tree:
                _insert_sequence(
                    tree["root"],
                    tree["all_nodes"],
                    seq,
                    tree_id,
                    seq_id,
                )
                tree["nodes"] += additional
                inserted = True
                break

        if inserted:
            continue

        # Create new tree
        if len(seq) > max_tokens_per_tree:
            raise ValueError(
                f"Sequence length {len(seq)} exceeds max_tokens_per_tree "
                f"{max_tokens_per_tree}; adjust limit or split sequences."
            )

        new_tree_id = len(forests)
        new_root = _BuildNode(new_tree_id, -1, -1)
        all_nodes: list[_BuildNode] = []
        _insert_sequence(new_root, all_nodes, seq, new_tree_id, seq_id)
        forests.append({"root": new_root, "all_nodes": all_nodes, "nodes": len(seq)})

    # Compress all trees
    tries = [_compress_trie(f["root"]) for f in forests]
    num_tokens_list = [f["nodes"] for f in forests]

    return tries, num_tokens_list


def _extract_sequences(data: dict[str, Any]) -> list[list[int]]:
    """Extract token sequences from padded input data."""
    assert "input_ids" in data, "Input data must contain 'input_ids'"
    assert "attention_mask" in data, "Input data must contain 'attention_mask'"

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]

    sequences = []
    for ids, mask in zip(input_ids, attention_mask):
        seq = ids[mask.bool()].tolist()
        sequences.append(seq)
    return sequences


def _pack_input_ids(
    trie: TrieNode,
    input_template: torch.Tensor,
    max_tokens: int,
) -> torch.Tensor:
    """Pack input_ids from trie structure into a flat tensor."""
    input_ids = torch.zeros(
        (max_tokens,),
        dtype=input_template.dtype,
        device=input_template.device,
    )

    for node in trie.nodes:
        # Find the first sequence that owns this node to get source tokens
        seq_id = node.sequence_ids[0]
        # Calculate position in original sequence
        seq_pos = sum(ancestor.num_tokens for ancestor in node.ancestors)
        # Copy tokens from source sequence
        tree_start, tree_end = node.tree_indices
        input_ids[tree_start : tree_end + 1] = input_template[seq_id][
            seq_pos : seq_pos + node.num_tokens
        ]

    return input_ids.unsqueeze(0)


# Block size for memory-efficient attention mask building.
# tril_indices for a block of this size uses ~32MB (2048*2048/2 * 2 tensors * 4 bytes).
_ATTN_MASK_BLOCK_SIZE = 2048


def _build_attention_mask(
    trie: TrieNode,
    max_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Build 2D attention mask from trie structure.

    Uses blockwise processing for large sequences to limit memory usage.
    For a sequence of length N, instead of creating tril_indices(N, N) which
    uses O(N^2) memory, we process in blocks of size B, each using O(B^2) memory.
    """
    mask = torch.zeros((max_tokens, max_tokens), dtype=torch.bool, device=device)

    for seq_id in trie.all_sequence_ids:
        # Get all tree index ranges for this sequence
        indices = trie.get_sequence_tree_indices(seq_id)
        if not indices:
            continue

        # Build position tensor from all segments
        position_chunks = [
            torch.arange(start, end + 1, device=device)
            for start, end in indices
            if end >= start
        ]
        if not position_chunks:
            continue

        positions = torch.cat(position_chunks, dim=0)
        seq_len = positions.numel()
        if seq_len == 0:
            continue

        # Apply causal mask in blocks to limit memory usage
        _apply_causal_mask_blockwise(mask, positions, seq_len, device)

    return mask


def _apply_causal_mask_blockwise(
    mask: torch.Tensor,
    positions: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> None:
    """Apply causal mask entries blockwise to limit memory usage.

    Instead of creating one huge tril_indices(seq_len, seq_len), we process
    the lower triangular matrix in blocks. For each block, we compute the
    valid (row, col) pairs where row >= col (causal) and set mask entries.

    The lower triangular matrix is divided into:
    1. Diagonal blocks: square blocks along the diagonal
    2. Off-diagonal blocks: rectangular blocks below the diagonal

    For a 6x6 matrix with block_size=2:
        [D0  .   . ]
        [B10 D1  . ]
        [B20 B21 D2]

    Where D0, D1, D2 are diagonal blocks (lower triangular within)
    and B10, B20, B21 are fully dense blocks.
    """
    block_size = _ATTN_MASK_BLOCK_SIZE
    num_blocks = (seq_len + block_size - 1) // block_size

    for block_row in range(num_blocks):
        row_start = block_row * block_size
        row_end = min((block_row + 1) * block_size, seq_len)
        row_len = row_end - row_start

        # Process diagonal block (lower triangular within block)
        # These are positions where block_row == block_col
        tril = torch.tril_indices(row_len, row_len, device=device, dtype=torch.int32)
        local_rows, local_cols = tril
        global_rows = row_start + local_rows
        global_cols = row_start + local_cols
        mask[positions[global_rows], positions[global_cols]] = True
        del tril, local_rows, local_cols, global_rows, global_cols

        # Process off-diagonal blocks (fully dense, all entries valid)
        # These are positions where block_row > block_col
        for block_col in range(block_row):
            col_start = block_col * block_size
            col_end = min((block_col + 1) * block_size, seq_len)

            # Create meshgrid for this block - all (row, col) pairs are valid
            # since row >= row_start > col_end > col for off-diagonal blocks
            block_rows = torch.arange(
                row_start, row_end, device=device, dtype=torch.int32
            )
            block_cols = torch.arange(
                col_start, col_end, device=device, dtype=torch.int32
            )

            # Use broadcasting to set all entries in this block
            # positions[block_rows] gives row indices, positions[block_cols] gives col indices
            row_positions = positions[block_rows].unsqueeze(1)  # [row_len, 1]
            col_positions = positions[block_cols].unsqueeze(0)  # [1, col_len]
            mask[row_positions, col_positions] = True
            del block_rows, block_cols, row_positions, col_positions


def _pack_extra_data(
    trie: TrieNode,
    data: dict[str, Any],
    sequence_lens: torch.Tensor,
    packable_keys: set[str],
    non_packable_keys: set[str],
) -> dict[str, Any]:
    """Pack additional tensor data according to trie structure."""
    extra_data: dict[str, Any] = {}
    seq_ids = trie.all_sequence_ids
    lens = [sequence_lens[sid].item() for sid in seq_ids]

    # Pack tensors according to the order in trie.all_sequence_ids
    for key in packable_keys:
        value = data[key]
        packed = torch.empty(
            (sum(lens), *value.shape[2:]),
            dtype=value.dtype,
            device=value.device,
        )
        cursor = 0
        for length, seq_id in zip(lens, seq_ids):
            packed[cursor : cursor + length] = value[seq_id][:length]
            cursor += length
        extra_data[key] = packed

    # Copy non-packable data as-is
    for key in non_packable_keys:
        extra_data[key] = data[key]

    return extra_data


def get_packed_tree_position_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Generate position IDs for packed tree inputs.
    Position IDs are computed from the attention mask by counting the number
    of ancestors each token can attend to (minus 1 for 0-indexing).
    """
    input_ids = input_ids.squeeze()
    if input_ids.ndim != 1:
        raise ValueError("Packed tree 'input_ids' must be a 1D tensor after squeezing.")
    if attention_mask.ndim != 2 or attention_mask.shape[0] != attention_mask.shape[1]:
        raise ValueError("Packed tree attention_mask must be a square matrix.")
    if attention_mask.shape[0] != input_ids.shape[0]:
        raise ValueError("Packed tree attention_mask must align with input_ids length.")

    if attention_mask.shape[0] == 0:
        position_ids = torch.empty(0, dtype=torch.long, device=attention_mask.device)
    else:
        ancestor_counts = attention_mask.bool().sum(dim=-1, dtype=torch.long)
        position_ids = torch.clamp_min(ancestor_counts - 1, 0)

    return position_ids.unsqueeze(0)


@trace_perf("tree_attn.build_block_mask_from_trie")
def build_block_mask_from_trie(
    trie: TrieNode,
    padded_size: int,
    device: torch.device,
) -> BlockMask:
    """Lazily build a block mask from a trie node.

    This function builds the dense attention mask from the trie structure and
    converts it to a block mask for use with flex attention. It should be called
    just before the forward pass to minimize memory usage.

    Parameters
    ----------
    trie : TrieNode
        The root trie node containing the tree structure.
    padded_size : int
        The padded sequence length.
    device : torch.device
        Device to create the block mask on.

    Returns
    -------
    BlockMask
        The created block mask for use with flex_attention.
    """
    # Handle dummy trie (empty tree for DP synchronization)
    if not trie.all_sequence_ids:
        # Create a minimal valid block mask for empty trees
        dummy_mask = torch.zeros(
            (padded_size, padded_size), dtype=torch.bool, device=device
        )
        return create_block_mask_from_dense(dummy_mask, padded_size, device)

    with trace_scope("tree_attn.build_attention_mask"):
        attention_mask = _build_attention_mask(trie, padded_size, device)

    with trace_scope("tree_attn.create_block_mask"):
        block_mask = create_block_mask_from_dense(attention_mask, padded_size, device)

    # Release dense attention mask memory
    del attention_mask

    return block_mask


@trace_perf("tree_attn.build_attention_mask_from_trie")
def build_attention_mask_from_trie(
    trie: TrieNode,
    padded_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a dense attention mask tensor from a trie node.

    This function builds the dense attention mask from the trie structure.
    Unlike build_block_mask_from_trie, this returns a torch.Tensor that can
    be saved by gradient checkpointing mechanisms.

    This is useful for Megatron engine where gradient checkpointing requires
    all forward arguments to be tensors (BlockMask cannot be saved by
    save_for_backward). The BlockMask can be created inside the attention
    module from this dense tensor.

    Parameters
    ----------
    trie : TrieNode
        The root trie node containing the tree structure.
    padded_size : int
        The padded sequence length.
    device : torch.device
        Device to create the attention mask on.

    Returns
    -------
    torch.Tensor
        Dense attention mask of shape (padded_size, padded_size) with dtype bool.
    """
    # Handle dummy trie (empty tree for DP synchronization)
    if not trie.all_sequence_ids:
        return torch.zeros((padded_size, padded_size), dtype=torch.bool, device=device)

    with trace_scope("tree_attn.build_attention_mask"):
        attention_mask = _build_attention_mask(trie, padded_size, device)

    return attention_mask


@trace_perf("tree_attn.build_triton_attn_data_from_trie")
def build_triton_attn_data_from_trie(
    trie: TrieNode,
    padded_size: int,
) -> TreeAttentionData:
    """Lazily build Triton tree attention data from a trie node.

    This function builds the parent array from the trie structure and
    precomputes packed masks and sparse block indices for Triton kernels.

    Parameters
    ----------
    trie : TrieNode
        The root trie node containing the tree structure.
    padded_size : int
        The padded sequence length.

    Returns
    -------
    TreeAttentionData
        The precomputed Triton attention data.
    """
    # Handle dummy trie (empty tree for DP synchronization)
    if not trie.all_sequence_ids:
        num_words = (padded_size + 63) >> 6
        num_q_blocks = (padded_size + 128 - 1) // 128
        num_kv_blocks = num_words
        packed_mask = torch.zeros((1, padded_size, num_words), dtype=torch.int64)
        kv_indices = torch.zeros((0,), dtype=torch.int32)
        kv_offsets = torch.zeros((1, num_q_blocks + 1), dtype=torch.int32)
        q_indices = torch.zeros((0,), dtype=torch.int32)
        q_offsets = torch.zeros((1, num_kv_blocks + 1), dtype=torch.int32)
        return TreeAttentionData(
            packed_mask=packed_mask,
            kv_indices=kv_indices,
            kv_offsets=kv_offsets,
            q_indices=q_indices,
            q_offsets=q_offsets,
        )

    with trace_scope("tree_attn.precompute_triton_data"):
        fa = trie_to_parent_array(trie, padded_size)
        triton_attn_data = precompute_tree_attention_data(fa)
    return triton_attn_data


def build_tree_attn_kwargs(
    trie: TrieNode,
    padded_size: int,
    device: torch.device,
    *,
    dense_mask: bool = False,
) -> dict[str, Any]:
    """Build tree attention kwargs dict for HF model forwarding.

    Selects Triton or flex attention backend automatically.
    Returns a dict to be merged into model ``**kwargs``.

    Parameters
    ----------
    trie : TrieNode
        The root trie node.
    padded_size : int
        Padded sequence length.
    device : torch.device
        Device for mask tensors.
    dense_mask : bool, default=False
        If True, the non-Triton path returns a dense ``attention_mask`` tensor
        (for Megatron, which needs tensors for gradient checkpointing).
        If False, returns a ``tree_block_mask`` BlockMask (for FSDP).
    """
    if USE_TRITON_TREE_ATTN and TRITON_AVAILABLE and not dense_mask:
        return {"tree_triton_data": build_triton_attn_data_from_trie(trie, padded_size)}
    if dense_mask:
        return {
            "attention_mask": build_attention_mask_from_trie(trie, padded_size, device)
        }
    return {"tree_block_mask": build_block_mask_from_trie(trie, padded_size, device)}
