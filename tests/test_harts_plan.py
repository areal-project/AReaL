"""Boundary and coverage tests for HARTS linear-attention planning."""

import pytest
import torch

from areal.models.tree_attn.harts_plan import plan_linear_attention
from areal.models.tree_attn.tree import _greedy_build_tries


def _trie(sequences: list[list[int]]):
    length = max(map(len, sequences))
    input_ids = torch.zeros((len(sequences), length), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence)
        attention_mask[row, : len(sequence)] = True
    tries, _ = _greedy_build_tries(
        {"input_ids": input_ids, "attention_mask": attention_mask}, 1024
    )
    return tries[0]


@pytest.mark.parametrize(
    ("fork_depth", "expected_rounds", "expected_replay"),
    [(7, 1, 7), (8, 2, 0), (9, 2, 1)],
)
def test_linear_attention_plan_at_chunk_boundary_preserves_minimum_calls(
    fork_depth: int, expected_rounds: int, expected_replay: int
):
    """A side branch waits only when its anchor is produced in this call."""
    prefix = list(range(1, fork_depth + 1))
    trie = _trie([prefix + [101, 102], prefix + [201, 202]])

    plan = plan_linear_attention(trie, chunk_size=8)

    assert len(plan.rounds) == expected_rounds
    assert plan.replay_tokens == expected_replay
    sequences = [
        sequence for round_sequences in plan.rounds for sequence in round_sequences
    ]
    assert sorted(
        idx for sequence in sequences for idx in sequence.compact_indices
    ) == list(range(trie.num_tokens))
    assert sum(sequence.initial_state_token is not None for sequence in sequences) == (
        0 if fork_depth < 8 else 1
    )


def test_linear_attention_plan_nested_forks_batches_ready_branches():
    """The nested B-1 and B+1 example needs two packed calls."""
    prefix = list(range(1, 8))
    trie = _trie(
        [
            prefix + [101, 102, 103],
            prefix + [101, 102, 201],
            prefix + [301],
        ]
    )

    plan = plan_linear_attention(trie, chunk_size=8)

    assert [len(round_sequences) for round_sequences in plan.rounds] == [2, 1]
    assert plan.replay_tokens == 8
    assert len(plan.boundary_state_tokens) == 1
    assert sorted(
        idx
        for round_sequences in plan.rounds
        for sequence in round_sequences
        for idx in sequence.compact_indices
    ) == list(range(trie.num_tokens))
