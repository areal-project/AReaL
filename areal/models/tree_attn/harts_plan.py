# SPDX-License-Identifier: Apache-2.0

"""Minimum-call linear-attention schedule for a compact rollout trie.

The recurrence and state-readiness rule follow HARTS, Section 3.3. A plan
contains compact output rows exactly once; replay rows are consumed only by
the linear-attention core and never produce semantic outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from areal.models.tree_attn.tree import TrieNode


@dataclass
class LinearAttentionSequence:
    round_idx: int
    compact_indices: list[int] = field(default_factory=list)
    replay_indices: list[int] = field(default_factory=list)
    initial_state_token: int | None = None


@dataclass
class LinearAttentionPlan:
    rounds: list[list[LinearAttentionSequence]]
    boundary_state_tokens: set[int]

    @property
    def replay_tokens(self) -> int:
        return sum(
            len(sequence.replay_indices)
            for round_sequences in self.rounds
            for sequence in round_sequences
        )


def plan_linear_attention(trie: TrieNode, chunk_size: int) -> LinearAttentionPlan:
    """Schedule each compact segment once with minimum sequential call depth.

    ``initial_state_token`` names the last compact token at a recovered chunk
    boundary. ``None`` denotes the zero state. Sequences in one round can be
    concatenated into one variable-length linear-attention call.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not trie.nodes:
        return LinearAttentionPlan([], set())

    # The compressed trie is preordered, so reverse order visits children first.
    entry_depth: dict[int, int] = {}
    parent_token = [-1] * trie.num_tokens
    pre_calls: dict[int, int] = {}
    in_calls: dict[int, int] = {}
    direct_child: dict[int, int] = {}
    for node in trie.nodes:
        parent = node.ancestors[-1] if node.ancestors else None
        entry_depth[node.start_idx] = (
            entry_depth[parent.start_idx] + parent.num_tokens if parent else 0
        )
        parent_token[node.start_idx] = parent.end_idx if parent else -1
        for token in range(node.start_idx + 1, node.end_idx + 1):
            parent_token[token] = token - 1

    for node in reversed(trie.nodes):
        key = node.start_idx
        children = list(node.children.values())
        if not children:
            pre_calls[key] = in_calls[key] = 1
            continue

        # Largest and second-largest side-branch costs keep the DP linear.
        child_costs = [pre_calls[child.start_idx] for child in children]
        largest = max(child_costs)
        largest_count = child_costs.count(largest)
        second = max((cost for cost in child_costs if cost != largest), default=0)
        best_child = min(
            children,
            key=lambda child: max(
                in_calls[child.start_idx],
                1
                + (
                    second
                    if pre_calls[child.start_idx] == largest and largest_count == 1
                    else largest
                ),
            ),
        )
        direct_child[key] = best_child.start_idx
        side_cost = (
            second
            if pre_calls[best_child.start_idx] == largest and largest_count == 1
            else largest
        )
        in_calls[key] = max(in_calls[best_child.start_idx], 1 + side_cost)
        anchor = entry_depth[key] // chunk_size * chunk_size
        end_depth = entry_depth[key] + node.num_tokens
        pre_calls[key] = (
            max(child_costs) if end_depth < anchor + chunk_size else in_calls[key]
        )

    rounds: list[list[LinearAttentionSequence]] = []
    boundary_state_tokens: set[int] = set()

    def add_sequence(round_idx: int, replay: list[int], source: int | None):
        while len(rounds) <= round_idx:
            rounds.append([])
        sequence = LinearAttentionSequence(
            round_idx, replay_indices=replay, initial_state_token=source
        )
        rounds[round_idx].append(sequence)
        if source is not None:
            boundary_state_tokens.add(source)
        return sequence

    def recover(node: TrieNode, ready: bool, sequence: LinearAttentionSequence):
        sequence.compact_indices.extend(range(node.start_idx, node.end_idx + 1))
        children = list(node.children.values())
        if not children:
            return

        start = entry_depth[node.start_idx]
        end = start + node.num_tokens
        anchor = start // chunk_size * chunk_size
        same_round = ready and end < anchor + chunk_size
        direct = direct_child[node.start_idx]
        boundary = end // chunk_size * chunk_size
        if same_round:
            boundary = anchor
        replay = []
        token = node.end_idx
        for _ in range(end - boundary):
            replay.append(token)
            token = parent_token[token]
        replay.reverse()
        source = token if boundary else None

        for child in children:
            if child.start_idx == direct:
                recover(child, same_round, sequence)
            else:
                side = add_sequence(
                    sequence.round_idx + (not same_round), replay.copy(), source
                )
                recover(child, True, side)

    for root_child in trie.children.values():
        recover(root_child, True, add_sequence(0, [], None))
    return LinearAttentionPlan(rounds, boundary_state_tokens)
