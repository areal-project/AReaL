"""Causal convolution over compact tree coordinates."""

import pytest
import torch
import torch.nn.functional as F

from areal.models.tree_attn.causal_conv import (
    _tree_conv_ancestry,
    tree_causal_conv1d,
)
from areal.models.tree_attn.tree import _greedy_build_tries


def test_tree_causal_conv_branches_match_trajectory_outputs_and_gradients():
    """Forks read ancestors and shared rows collect every consumer gradient."""
    tokens = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 6, 7], [1, 2, 8, 9, 0]])
    mask = tokens != 0
    trie = _greedy_build_tries({"input_ids": tokens, "attention_mask": mask}, 64)[0][0]
    paths = [
        [
            idx
            for start, end in trie.get_sequence_tree_indices(sequence_id)
            for idx in range(start, end + 1)
        ]
        for sequence_id in trie.all_sequence_ids
    ]
    semantic_rows = torch.tensor([idx for path in paths for idx in path])

    generator = torch.Generator().manual_seed(23)
    projected = torch.randn(trie.num_tokens, 5, generator=generator)
    weight = torch.randn(5, 1, 4, generator=generator)
    bias = torch.randn(5, generator=generator)
    semantic_weights = torch.randn(len(semantic_rows), 5, generator=generator)
    actual_projected = projected.clone().requires_grad_()
    reference_projected = projected.clone().requires_grad_()

    actual = tree_causal_conv1d(actual_projected, weight, bias, trie, F.silu)
    (actual[semantic_rows] * semantic_weights).sum().backward()

    reference = []
    for path in paths:
        selected = reference_projected[path].t().unsqueeze(0)
        convolved = F.conv1d(selected, weight, bias, padding=3, groups=5)[
            ..., : len(path)
        ]
        reference.append(F.silu(convolved).squeeze(0).t())
    expected = torch.cat(reference)
    (expected * semantic_weights).sum().backward()

    torch.testing.assert_close(actual[semantic_rows], expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        actual_projected.grad, reference_projected.grad, rtol=1e-5, atol=1e-6
    )


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compiled_tree_causal_conv_matches_eager_output_and_gradients():
    """The fused CUDA path preserves branch accumulation and weight gradients."""
    tokens = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 6, 7]])
    trie = _greedy_build_tries(
        {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}, 128
    )[0][0]
    generator = torch.Generator(device="cuda").manual_seed(23)
    projected = torch.randn(
        (trie.num_tokens, 32), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    weight = torch.randn(
        (32, 1, 4), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    bias = torch.randn((32,), device="cuda", dtype=torch.bfloat16, generator=generator)
    output_weight = torch.randn(
        projected.shape, device="cuda", dtype=torch.float32, generator=generator
    )
    results = []
    for activation in (lambda value: F.silu(value), F.silu):
        inputs = [
            tensor.detach().clone().requires_grad_()
            for tensor in (projected, weight, bias)
        ]
        output = tree_causal_conv1d(*inputs, trie, activation)
        (output.float() * output_weight).sum().backward()
        results.append((output.detach(), *(tensor.grad.detach() for tensor in inputs)))

    for index, (eager, compiled) in enumerate(zip(*results)):
        # Fused BF16 reduction can shift the accumulated input gradient by a
        # few ULPs; outputs and parameter gradients retain the tighter bound.
        atol = 4e-2 if index == 1 else 1e-2
        torch.testing.assert_close(compiled, eager, rtol=1e-2, atol=atol)

    first = _tree_conv_ancestry(trie, trie.num_tokens, 4, projected.device)
    second = _tree_conv_ancestry(trie, trie.num_tokens, 4, projected.device)
    assert first.data_ptr() == second.data_ptr()
