# SPDX-License-Identifier: Apache-2.0

# The following code is adapted with minor modifications from
# https://github.com/Whisper-6/DynamicTreeAttn/blob/main/trie.py.
import random
from dataclasses import dataclass, field
from math import ceil


def _get_stats(
    lens: list[int], lcp_lens: list[int], mode: str, block_size: int | None = None
) -> dict:
    n_tree_tokens = sum(lens) - sum(lcp_lens)
    sum_depth = 0
    for i in range(len(lens)):
        start = lcp_lens[i - 1] if i > 0 else 0
        end = lens[i]
        sum_depth += (start + end - 1) * (end - start) // 2

    if mode == "forward":
        sum_prefix_len = sum(lcp_lens)

        return {
            "n_leaf_sequences": len(lens),
            "n_tree_tokens": n_tree_tokens,
            "sum_prefix_len": sum_prefix_len,
            "sum_depth": sum_depth,
        }

    elif mode == "backward":
        sum_prefix_len = 0
        n_f1_tokens = 0
        for i in range(len(lens)):
            start = lcp_lens[i] if i < len(lcp_lens) else 0
            end = lens[i]
            pop_len = end - start
            f1_start = lcp_lens[i - 1] if i > 0 else 0

            if block_size is None or pop_len <= block_size:
                f1_end = start
                sum_prefix_len += start
            else:
                n_blocks = ceil(pop_len / block_size)
                block_size_actual = ceil(pop_len / n_blocks)
                f1_end = end - block_size_actual
                for b in range(n_blocks):
                    pop_start = max(end - (b + 1) * block_size_actual, start)
                    sum_prefix_len += pop_start

            n_f1_tokens += max(f1_end - f1_start, 0)

        return {
            "n_leaf_sequences": len(lens),
            "n_tree_tokens": n_tree_tokens,
            "sum_prefix_len": sum_prefix_len,
            "sum_depth": sum_depth,
            "n_f1_tokens": n_f1_tokens,
        }

    else:
        raise ValueError(f"Unsupported mode: {mode}")


@dataclass(slots=True)
class CTNode:
    """Node in the compressed trie."""

    depth: int = 0  # Depth of this node.
    seq_id: int = -1  # Sequence index; -1 indicates an internal node.
    chain_tail_depth: int = 0  # Tail depth of the prioritized chain.
    child_ids: list[int] = field(default_factory=list)  # IDs of child nodes.


class CompressedTrie:
    """Compressed trie used to plan traversal order."""

    def __init__(self, lens: list[int], lcp_lens: list[int]):
        """
        Initialize the compressed trie.

        Args:
            lens: Length of each sequence, sorted in lexicographic order.
            lcp_lens: LCP length between adjacent sequences, where
                len(lcp_lens) == max(len(lens) - 1, 0). An empty `lens`
                produces a degenerate trie that contains only the root node.
        """
        expected_lcp = max(len(lens) - 1, 0)
        if len(lcp_lens) != expected_lcp:
            raise ValueError(
                f"len(lcp_lens) must be {expected_lcp}, got {len(lcp_lens)}"
            )

        self.nodes: list[CTNode] = []  # Stores all trie nodes.
        self._build(lens, lcp_lens)

        self.lca_depth = None
        self.order = None
        self.lens = None
        self.lcp_lens = None

    def _new_node(self, depth: int, seq_id: int = -1) -> int:
        """Create a new node and return its ID."""
        self.nodes.append(CTNode(depth=depth, seq_id=seq_id))
        return len(self.nodes) - 1

    def _build(self, lens: list[int], lcp_lens: list[int]):
        """Build the compressed trie."""

        n_seqs = len(lens)
        # Create the root node.
        root_id = self._new_node(depth=0, seq_id=-1)
        stack = [(root_id, 0)]  # Stack entries are (node_id, depth).
        nodes = self.nodes

        for seq_id in range(n_seqs):
            len_i = lens[seq_id]
            lcp = lcp_lens[seq_id - 1] if seq_id > 0 else 0

            if len(stack) >= 2:
                while stack[-2][1] > lcp:
                    # Pop a child node and connect it to its parent.
                    child_id = stack.pop()[0]
                    parent_id = stack[-1][0]
                    nodes[parent_id].child_ids.append(child_id)

                child_id = stack.pop()[0]
                if stack[-1][1] < lcp:
                    lcp_node_id = self._new_node(depth=lcp, seq_id=-1)
                    stack.append((lcp_node_id, lcp))
                parent_id = stack[-1][0]
                nodes[parent_id].child_ids.append(child_id)
            else:
                if stack[-1][1] < lcp:
                    lcp_node_id = self._new_node(depth=lcp, seq_id=-1)
                    stack.append((lcp_node_id, lcp))

            # Create a new leaf node.
            parent_id = stack[-1][0]
            cur_node_id = self._new_node(depth=len_i, seq_id=seq_id)
            stack.append((cur_node_id, len_i))

        while len(stack) >= 2:
            child_id = stack.pop()[0]
            parent_id = stack[-1][0]
            nodes[parent_id].child_ids.append(child_id)

    def dfs_chain(self, node_id: int, child_order_func) -> int:
        """Compute `chain_tail_depth` for each node."""
        node = self.nodes[node_id]

        # Leaf node.
        if node.seq_id != -1:
            node.chain_tail_depth = node.depth
            return

        for child_id in node.child_ids:
            self.dfs_chain(child_id, child_order_func)

        child_ids = child_order_func(node_id)
        if not child_ids:
            # Only reachable for the root of an empty trie. The value never
            # propagates anywhere since the subtree carries no leaves.
            node.chain_tail_depth = node.depth
            return
        node.chain_tail_depth = self.nodes[child_ids[0]].chain_tail_depth

    def dfs_get_lens(self, node_id: int, seq_set: set[int]):
        node = self.nodes[node_id]

        if node.seq_id != -1:
            if node.seq_id in seq_set:
                self.lens.append(node.depth)
                self.lcp_lens.append(self.lca_depth)
                self.lca_depth = node.depth
            return

        for child_id in node.child_ids:
            self.lca_depth = min(self.lca_depth, node.depth)
            self.dfs_get_lens(child_id, seq_set)

    def get_lens(self, seq_set: set[int]):
        self.lens = []
        self.lcp_lens = []
        self.lca_depth = 0
        self.dfs_get_lens(0, seq_set)
        return self.lens, self.lcp_lens[1:]

    def dfs_get_order(self, node_id: int, child_order_func):
        node = self.nodes[node_id]

        # Leaf node: record the sequence index.
        if node.seq_id != -1:
            self.order.append(node.seq_id)
            self.lens.append(node.depth)
            self.lcp_lens.append(self.lca_depth)
            self.lca_depth = node.depth
            return

        # Get child traversal order from the given strategy.
        child_ids = child_order_func(node_id)

        # Recursively traverse children.
        for child_id in child_ids:
            self.lca_depth = min(self.lca_depth, node.depth)
            self.dfs_get_order(child_id, child_order_func)

    def _get_child_order_forward(self, node_id: int) -> list[int]:
        node = self.nodes[node_id]
        return sorted(
            node.child_ids, key=lambda child_id: self.nodes[child_id].chain_tail_depth
        )

    def _get_child_order_backward(self, node_id: int) -> list[int]:
        node = self.nodes[node_id]
        return sorted(
            node.child_ids,
            key=lambda child_id: (
                1 if self.nodes[child_id].child_ids else 0,
                self.nodes[child_id].chain_tail_depth,
            ),
        )

    def _get_child_order_random(
        self, node_id: int, seed: int | None = None
    ) -> list[int]:
        node = self.nodes[node_id]
        child_ids = node.child_ids.copy()

        if seed is not None:
            local_random = random.Random(seed)
            local_random.shuffle(child_ids)
        else:
            random.shuffle(child_ids)

        return child_ids

    def get_order(self, child_order_func):
        """Get sequence order from DFS with a custom child-order strategy."""
        self.dfs_chain(0, child_order_func)
        self.order = []
        self.lens = []
        self.lcp_lens = []
        self.lca_depth = 0
        self.dfs_get_order(0, child_order_func)

    def get_order_forward(self):
        """Get sequence order from DFS using main-Ld-priority traversal."""
        self.get_order(self._get_child_order_forward)
        return self.order, self.lens, self.lcp_lens[1:]

    def get_order_backward(self):
        """Get sequence order from DFS for backward-style pop traversal."""
        self.get_order(self._get_child_order_backward)
        return self.order[::-1], self.lens[::-1], self.lcp_lens[1:][::-1]

    def get_order_random(self, seed: int | None = None):
        """Get sequence order from DFS after randomizing child edges."""
        self.get_order(lambda node_id: self._get_child_order_random(node_id, seed))
        return self.order


def _get_subtrie(trie, seq_set: set[int]) -> CompressedTrie:
    lens, lcp_lens = trie.get_lens(seq_set)
    return CompressedTrie(lens, lcp_lens)


# -------- Test --------


def test_compressed_trie():
    lens1 = [5, 4, 3, 2]
    lcp_lens1 = [3, 2, 1]

    trie1 = CompressedTrie(lens1, lcp_lens1)

    order, lens, lcp_lens = trie1.get_order_forward()
    print(order, lens, lcp_lens)

    order, lens, lcp_lens = trie1.get_order_backward()
    print(order, lens, lcp_lens)

    order, lens, lcp_lens = trie1.get_order_random()
    print(order, lens, lcp_lens)


if __name__ == "__main__":
    test_compressed_trie()
