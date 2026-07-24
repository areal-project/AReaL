# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from areal.utils.functional.vocab_parallel import (
    _inplace_vocab_parallel_logprobs_entropy,
    gather_logprobs_entropy,
)
from areal.utils.functional.vocab_parallel_kernels import (
    reusable_vocab_parallel_logits,
)

CUDA_AVAILABLE = torch.cuda.is_available()


def _reference(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = torch.log_softmax(logits / temperature, dim=-1)
    probabilities = log_probs.exp()
    selected = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    entropy = -(probabilities * log_probs).sum(dim=-1)
    return selected, entropy


def _autograd_node_names(output: torch.Tensor) -> set[str]:
    names: set[str] = set()
    pending = [output.grad_fn]
    while pending:
        node = pending.pop()
        if node is None or type(node).__name__ in names:
            continue
        names.add(type(node).__name__)
        pending.extend(next_node for next_node, _ in node.next_functions)
    return names


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for Triton kernels")
@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.5])
def test_inplace_vocab_parallel_matches_reference_and_reuses_storage(
    temperature: float,
):
    """The fused path should preserve results while reusing logits storage."""
    torch.manual_seed(123)
    device = torch.device("cuda")
    original = torch.randn(17, 1537, dtype=torch.float32, device=device)
    labels = torch.randint(0, original.size(-1), (17,), device=device)
    logprob_weights = torch.randn(17, device=device)
    logprob_weights[::3] = 0.0

    reference_logits = original.clone().requires_grad_(True)
    expected_logprobs, expected_entropy = _reference(
        reference_logits, labels, temperature
    )
    expected_loss = (expected_logprobs * logprob_weights).sum()
    expected_loss.backward()

    logits_leaf = original.clone().requires_grad_(True)
    logits = logits_leaf + 0.0
    storage_ptr = logits.data_ptr()
    logprobs, entropy = _inplace_vocab_parallel_logprobs_entropy(
        logits,
        labels,
        tp_group=None,
        temperature=temperature,
        chunk_size=5,
    )

    assert logits.data_ptr() == storage_ptr
    torch.testing.assert_close(
        logits.sum(dim=-1),
        torch.ones(17, device=device),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(logprobs, expected_logprobs, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(entropy, expected_entropy, rtol=1e-5, atol=2e-5)
    assert not entropy.requires_grad
    assert "SliceBackward0" not in _autograd_node_names(logprobs)

    loss = (logprobs * logprob_weights).sum()
    loss.backward()

    torch.testing.assert_close(
        logits_leaf.grad, reference_logits.grad, rtol=1e-5, atol=2e-5
    )
    torch.testing.assert_close(
        logits_leaf.grad[::3],
        torch.zeros_like(logits_leaf.grad[::3]),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for Triton kernels")
@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf"), float("nan")])
def test_inplace_vocab_parallel_rejects_invalid_temperature(temperature: float):
    """Temperature must define a valid softmax distribution."""
    logits = torch.randn(2, 8, device="cuda")
    labels = torch.tensor([0, 1], device="cuda")

    with pytest.raises(ValueError, match="temperature must be positive"):
        _inplace_vocab_parallel_logprobs_entropy(
            logits, labels, tp_group=None, temperature=temperature
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for Triton kernels")
def test_reuse_logits_accepts_full_storage_squeeze_view():
    """Megatron's full-storage squeeze view should reuse its owning FP32 buffer."""
    torch.manual_seed(321)
    leaf = torch.randn(1, 7, 33, device="cuda", requires_grad=True)
    owner = leaf + 0.0
    logits = owner.squeeze(0)
    labels = torch.randint(0, 33, (7,), device="cuda")
    original = logits.detach().clone()

    assert reusable_vocab_parallel_logits(logits) is owner
    logprobs, entropy = gather_logprobs_entropy(
        logits, labels, reuse_logits=True, chunk_size=3
    )

    expected_logprobs, expected_entropy = _reference(original, labels, 1.0)
    torch.testing.assert_close(logprobs, expected_logprobs, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(entropy, expected_entropy, rtol=1e-5, atol=2e-5)
    assert owner.data_ptr() == logits.data_ptr()
    assert not entropy.requires_grad
    logprobs.sum().backward()
    assert leaf.grad is not None


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for Triton kernels")
def test_reuse_logits_rejects_partial_view():
    """A partial view must not be mutated through the destructive fast path."""
    leaf = torch.randn(1, 7, 33, device="cuda", requires_grad=True)
    owner = leaf + 0.0
    partial_view = owner[:, :-1].squeeze(0)

    assert reusable_vocab_parallel_logits(partial_view) is None
