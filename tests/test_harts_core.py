"""Forward and gradient equivalence for the differentiable tree GDN schedule."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask

from areal.api.cli_args import MicroBatchSpec
from areal.models.tree_attn import module_megatron
from areal.models.tree_attn import tree as tree_module
from areal.models.tree_attn.harts_core import run_linear_attention_plan
from areal.models.tree_attn.harts_plan import plan_linear_attention
from areal.models.tree_attn.harts_runtime import (
    _TOKEN_MULTIPLICITY,
    TreeTopKRouter,
    register_tree_mask,
    unregister_tree_masks,
    weighted_router_stats,
)
from areal.models.tree_attn.module_fsdp import create_block_mask_from_compact
from areal.models.tree_attn.tree import (
    TrieNode,
    _build_attention_mask,
    _greedy_build_tries,
    build_compact_attention_mask_from_trie,
    build_packed_tree_batch,
    get_packed_tree_position_ids,
    get_packed_tree_position_ids_from_trie,
)


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


@pytest.mark.parametrize(
    "sequences",
    [
        [[1, 2, 3, 4]],
        [[1, 2, 3], [1, 2, 4]],
        [[1, 2], [1, 2, 3]],
        [[1, 2, 3], [4, 5, 6]],
        [[1, 2, 3, 4], [1, 2, 5, 6], [1, 7, 8]],
    ],
)
def test_tree_position_ids_from_trie_match_dense_mask(sequences):
    """Trie depths preserve positions across forks, endings, roots, and padding."""
    trie = _trie(sequences)
    padded_size = 128
    reference = get_packed_tree_position_ids(
        torch.zeros((1, padded_size), dtype=torch.long),
        _build_attention_mask(trie, padded_size, torch.device("cpu")),
    )

    actual = get_packed_tree_position_ids_from_trie(
        trie, padded_size, torch.device("cpu")
    )

    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_tree_position_ids_from_empty_trie_are_zero():
    """A dummy DP microbatch has valid zero position IDs."""
    actual = get_packed_tree_position_ids_from_trie(
        TrieNode(tree_id=0), 128, torch.device("cpu")
    )
    torch.testing.assert_close(actual, torch.zeros((1, 128), dtype=torch.long))


@pytest.mark.parametrize(
    "sequences",
    [
        [[1, 2, 3, 4]],
        [[1, 2, 3], [1, 2, 4]],
        [[1, 2], [1, 2, 3]],
        [[1, 2, 3], [4, 5, 6]],
        [[1, 2, 3, 4], [1, 2, 5, 6], [1, 7, 8]],
        [[1, 2, 9], [1, 2, 3, 4], [1, 2, 5, 6]],
    ],
)
def test_compact_tree_mask_matches_dense_attention(sequences):
    """Ancestry intervals preserve causal attention across forks and padding."""
    trie = _trie(sequences)
    size = 128
    descriptor = build_compact_attention_mask_from_trie(trie, size, torch.device("cpu"))
    block_mask = create_block_mask_from_compact(descriptor, size, torch.device("cpu"))
    actual = create_mask(
        block_mask.mask_mod, 1, 1, size, size, device=torch.device("cpu")
    )[0, 0]
    expected = _build_attention_mask(trie, size, torch.device("cpu"))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert descriptor.shape == (size, 2)
    assert descriptor.dtype == torch.int32


def test_compact_tree_mask_empty_trie_masks_all_tokens():
    descriptor = build_compact_attention_mask_from_trie(
        TrieNode(tree_id=0), 128, torch.device("cpu")
    )
    assert torch.all(descriptor == -1)
    block_mask = create_block_mask_from_compact(descriptor, 128, torch.device("cpu"))
    actual = create_mask(block_mask.mask_mod, 1, 1, 128, 128, device="cpu")
    assert not torch.any(actual)


def test_packed_tree_batch_computes_positions_without_dense_mask(monkeypatch):
    """Packing long trees does not allocate a quadratic position helper mask."""

    def reject_dense_mask(*args, **kwargs):
        raise AssertionError("dense attention mask built during tree packing")

    monkeypatch.setattr(tree_module, "_build_attention_mask", reject_dense_mask)
    input_ids = torch.tensor([[1, 2, 3], [1, 2, 4]])
    batch = build_packed_tree_batch(
        {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)},
        MicroBatchSpec(max_tokens_per_mb=128),
        compact_padding=True,
    )

    torch.testing.assert_close(
        batch.padded_mbs[0]["position_ids"][:, :4],
        torch.tensor([[0, 1, 2, 2]]),
        rtol=0,
        atol=0,
    )
    assert torch.count_nonzero(batch.padded_mbs[0]["position_ids"][:, 4:]) == 0


def test_megatron_tree_block_mask_reused_until_backward_cleanup(monkeypatch):
    """Attention layers and checkpoint replay share one BlockMask conversion."""
    mask = torch.ones((4, 4), dtype=torch.bool)
    trie = _trie([[1, 2, 3, 4]])
    pointer = register_tree_mask(mask, trie, plan_linear_attention(trie, 4))
    conversions = []

    def convert(dense, seq_len, device):
        conversions.append((dense, seq_len, device))
        return object()

    monkeypatch.setattr(module_megatron, "create_block_mask_from_dense", convert)
    monkeypatch.setattr(
        module_megatron, "_flex_attention", lambda query, key, value, **kwargs: query
    )
    layer = module_megatron.PytorchFlexAttention(
        config=SimpleNamespace(context_parallel_size=1),
        layer_number=1,
        attn_mask_type=None,
        attention_type="self",
    )
    qkv = torch.randn((4, 1, 2, 4))
    other_mask = torch.ones_like(mask)
    other_pointer = register_tree_mask(other_mask, trie, plan_linear_attention(trie, 4))
    try:
        layer(qkv, qkv, qkv, mask, None)
        layer(qkv, qkv, qkv, mask, None)
        assert len(conversions) == 1
        layer(qkv, qkv, qkv, other_mask, None)
        layer(qkv, qkv, qkv, mask, None)
        assert len(conversions) == 3
    finally:
        unregister_tree_masks([pointer, other_pointer])

    layer(qkv, qkv, qkv, mask, None)
    assert len(conversions) == 4


def _recurrent_kernel(q, k, v, g, beta, *, initial_state, cu_seqlens_cpu, **kwargs):
    output_final_state = kwargs.get("output_final_state", False)
    offsets = cu_seqlens_cpu.tolist()
    if initial_state is None:
        initial_state = q.new_zeros(
            (len(offsets) - 1, q.shape[2], k.shape[-1], v.shape[-1])
        )
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
    states = torch.stack(final_states) if output_final_state else None
    return torch.stack(outputs).unsqueeze(0), states


def test_tree_gdn_requests_only_needed_boundary_states():
    """FLA state reads and writes follow the tree call dependencies."""
    prefix = list(range(1, 9))
    trie = _trie([prefix + [101], prefix + [201]])
    plan = plan_linear_attention(trie, chunk_size=8)
    count = trie.num_tokens
    inputs = [torch.randn((1, count, 1, 1)) for _ in range(3)]
    inputs += [torch.randn((1, count, 1)), torch.rand((1, count, 1))]
    state_io = []

    def recorded_kernel(*args, **kwargs):
        state_io.append(
            (kwargs["initial_state"] is not None, kwargs["output_final_state"])
        )
        return _recurrent_kernel(*args, **kwargs)

    run_linear_attention_plan(*inputs, plan, recorded_kernel)

    assert state_io == [(False, True), (True, False)]


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
