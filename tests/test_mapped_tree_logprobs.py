"""Semantic logprob checks for shared compact rows."""

import torch

from areal.models.tree_attn.functional import (
    gather_packed_tree_logprobs_entropy,
    tree_semantic_rows,
)
from areal.models.tree_attn.mapped_logprobs import mapped_logprobs_entropy
from areal.models.tree_attn.tree import _greedy_build_tries


def test_mapped_logprobs_different_fork_labels_accumulate_gradients():
    """Two branch labels reuse one logit without losing either gradient."""
    rows = torch.tensor([0, 1, 1, 2, 1])
    labels = torch.tensor([2, 3, 4, 1, 0])
    weights = torch.tensor([0.2, 0.5, -0.3, 1.0, 0.7])
    logits = torch.randn(4, 7, generator=torch.Generator().manual_seed(7))
    actual_logits = logits.clone().requires_grad_()
    reference_logits = logits.clone().requires_grad_()

    logprobs, entropy = mapped_logprobs_entropy(actual_logits, rows, labels, 0.8)
    actual_loss = (logprobs * weights + entropy * 0.01).sum()
    actual_loss.backward()

    expanded = reference_logits[rows].float() / 0.8
    reference_logprobs = expanded.log_softmax(dim=-1)
    expected_logprobs = reference_logprobs.gather(1, labels[:, None]).squeeze(1)
    expected_entropy = -(reference_logprobs.exp() * reference_logprobs).sum(-1)
    reference_loss = (expected_logprobs * weights + expected_entropy * 0.01).sum()
    reference_loss.backward()

    torch.testing.assert_close(logprobs, expected_logprobs, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(entropy, expected_entropy, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        actual_logits.grad, reference_logits.grad, rtol=1e-5, atol=1e-6
    )


def test_tree_semantic_rows_at_fork_preserve_distinct_targets():
    """Trie paths map one shared parent logit to both branch labels."""
    sequences = torch.tensor([[1, 2, 3, 4], [1, 2, 5, 6]])
    tries, _ = _greedy_build_tries(
        {
            "input_ids": sequences,
            "attention_mask": torch.ones_like(sequences, dtype=torch.bool),
        },
        16,
    )
    trie = tries[0]
    packed = torch.tensor(
        [token for node in trie.nodes for token in node.tokens], dtype=torch.long
    )
    rows, labels, slices = tree_semantic_rows(trie, packed)
    assert rows[slices[0]][1] == rows[slices[1]][1]
    assert labels[slices[0]][1] != labels[slices[1]][1]

    logits = torch.randn(trie.num_tokens, 8, requires_grad=True)
    logprobs, entropy = gather_packed_tree_logprobs_entropy(logits, trie, packed)
    expected_logprobs = (
        logits[rows].log_softmax(-1).gather(1, labels[:, None]).squeeze(1)
    )
    expected_entropy = -(
        logits[rows].log_softmax(-1).exp() * logits[rows].log_softmax(-1)
    ).sum(-1)
    torch.testing.assert_close(logprobs, expected_logprobs, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(entropy, expected_entropy, rtol=1e-5, atol=1e-6)
