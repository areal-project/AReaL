"""Forward and gradient equivalence for the differentiable tree GDN schedule."""

import pytest
import torch

from areal.api.cli_args import MicroBatchSpec
from areal.models.tree_attn.harts_core import run_linear_attention_plan
from areal.models.tree_attn.harts_plan import plan_linear_attention
from areal.models.tree_attn.harts_runtime import (
    _TOKEN_MULTIPLICITY,
    TreeTopKRouter,
    weighted_router_stats,
)
from areal.models.tree_attn.tree import _greedy_build_tries, build_packed_tree_batch


def _trie(sequences):
    width = max(map(len, sequences))
    input_ids = torch.zeros((len(sequences), width), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence)
        attention_mask[row, : len(sequence)] = True
    tries, _ = _greedy_build_tries(
        {"input_ids": input_ids, "attention_mask": attention_mask}, 1024
    )
    return tries[0]


def _recurrent_kernel(q, k, v, g, beta, *, initial_state, cu_seqlens_cpu, **kwargs):
    del kwargs
    offsets = cu_seqlens_cpu.tolist()
    outputs = []
    final_states = []
    for sequence, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        state = initial_state[sequence]
        for row in range(start, end):
            state = g[0, row].exp().unsqueeze(-1).unsqueeze(-1) * state
            state = state + beta[0, row].unsqueeze(-1).unsqueeze(-1) * (
                k[0, row].unsqueeze(-1) * v[0, row].unsqueeze(-2)
            )
            outputs.append((q[0, row].unsqueeze(-1) * state).sum(dim=-2))
        final_states.append(state)
    return torch.stack(outputs).unsqueeze(0), torch.stack(final_states)


@pytest.mark.parametrize(
    ("fork_depth", "branch_case", "expected_calls"),
    [
        (7, "single", 1),
        (8, "single", 2),
        (9, "single", 2),
        (7, "nested", 2),
        (8, "stacked_boundaries", 2),
    ],
)
def test_tree_gdn_matches_independent_sequences_and_gradients(
    fork_depth, branch_case, expected_calls
):
    prefix = list(range(1, fork_depth + 1))
    if branch_case == "stacked_boundaries":
        middle = list(range(20, 29))
        sequences = [prefix + middle + [101], prefix + middle + [201], prefix + [301]]
    elif branch_case == "nested":
        sequences = [prefix + [101, 102, 103], prefix + [101, 102, 201], prefix + [301]]
    else:
        sequences = [prefix + [101, 102], prefix + [201, 202]]
    trie = _trie(sequences)
    plan = plan_linear_attention(trie, chunk_size=8)
    torch.manual_seed(7)
    data = [
        torch.randn((1, trie.num_tokens, 1, 1), dtype=torch.double, requires_grad=True)
        for _ in range(3)
    ]
    g = torch.randn((1, trie.num_tokens, 1), dtype=torch.double, requires_grad=True)
    beta = torch.sigmoid(
        torch.randn((1, trie.num_tokens, 1), dtype=torch.double)
    ).requires_grad_()
    inputs = [*data, g, beta]
    calls = 0

    def counted_kernel(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _recurrent_kernel(*args, **kwargs)

    actual = run_linear_attention_plan(*inputs, plan, counted_kernel)
    assert calls == expected_calls
    tree_rows = []
    independent = []
    for sequence_id in trie.all_sequence_ids:
        rows = [
            row
            for start, end in trie.get_sequence_tree_indices(sequence_id)
            for row in range(start, end + 1)
        ]
        tree_rows.extend(rows)
        positions = torch.tensor(rows)
        sliced = [tensor.index_select(1, positions) for tensor in inputs]
        reference, _ = _recurrent_kernel(
            *sliced,
            initial_state=torch.zeros((1, 1, 1, 1), dtype=torch.double),
            cu_seqlens_cpu=torch.tensor([0, len(rows)]),
        )
        independent.append(reference)
    expected = torch.cat(independent, dim=1)
    torch.testing.assert_close(actual[:, tree_rows], expected, rtol=0, atol=1e-12)

    weights = torch.arange(1, len(tree_rows) + 1, dtype=torch.double).view(1, -1, 1, 1)
    actual_grad = torch.autograd.grad((actual[:, tree_rows] * weights).sum(), inputs)
    expected_grad = torch.autograd.grad((expected * weights).sum(), inputs)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want, rtol=0, atol=1e-11)


def test_router_multiplicity_matches_unpacked_trajectory_statistics():
    scores = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.9, 0.1]], requires_grad=True)
    routing_map = torch.tensor([[False, True], [True, False], [True, False]])
    multiplicity = torch.tensor([2.0, 1.0, 1.0])
    expanded = torch.tensor([0, 0, 1, 2])

    weighted_scores, counts, token_count = weighted_router_stats(
        scores, routing_map, multiplicity
    )
    expected_scores = scores.index_select(0, expanded)
    expected_counts = routing_map.index_select(0, expanded).float().sum(dim=0)
    torch.testing.assert_close(weighted_scores.sum(dim=0), expected_scores.sum(dim=0))
    torch.testing.assert_close(counts, expected_counts)
    assert token_count == 4
    actual_grad = torch.autograd.grad(weighted_scores.sum(), scores, retain_graph=True)[
        0
    ]
    expected_grad = torch.autograd.grad(expected_scores.sum(), scores)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_tree_expert_bias_counts_semantic_tokens_once():
    class Router:
        enable_expert_bias = True
        local_tokens_per_expert = torch.zeros(2)

    router = Router()
    routing_map = torch.tensor([[True, False], [False, True], [True, False]])
    padding_mask = torch.tensor([[False], [True], [False]])
    token = _TOKEN_MULTIPLICITY.set(torch.tensor([3.0, 2.0, 1.0]))
    try:
        TreeTopKRouter._apply_expert_bias(router, routing_map, padding_mask)
        with torch.no_grad():
            TreeTopKRouter._apply_expert_bias(router, routing_map, padding_mask)
    finally:
        _TOKEN_MULTIPLICITY.reset(token)
    torch.testing.assert_close(router.local_tokens_per_expert, torch.tensor([4.0, 0.0]))


def test_compact_tree_padding_uses_distinct_tokens_not_capacity():
    input_ids = torch.tensor([list(range(1, 11)) + [101], list(range(1, 11)) + [201]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    batch = build_packed_tree_batch(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        MicroBatchSpec(max_tokens_per_mb=1024),
        compact_padding=True,
    )

    assert batch.group_lens == [12]
    assert batch.padded_to_lengths == [128]
    assert batch.padding_lengths == [116]
