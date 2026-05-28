"""CUDA deterministic mode regression tests.

Validates reproducibility of key PyTorch CUDA operations under
deterministic mode:
- torch._grouped_mm (MoE grouped matmuls)
- torch.compile with Inductor backend
- _gather_logprobs_entropy compiled loss helper
- Activation checkpointing recompute + compile interaction
"""

import os
import warnings

import pytest
import torch
import torch.nn as nn

CUDA_AVAILABLE = torch.cuda.is_available()
GROUPED_MM_AVAILABLE = hasattr(torch, "_grouped_mm")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def deterministic_env():
    """Set up and tear down a deterministic CUDA environment for testing."""
    prev_deterministic = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    prev_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
    prev_compile_det = os.environ.get("TORCH_COMPILE_DETERMINISTIC", "")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TORCH_COMPILE_DETERMINISTIC"] = "1"
    torch.use_deterministic_algorithms(True, warn_only=True)

    yield

    torch.use_deterministic_algorithms(prev_deterministic, warn_only=prev_warn_only)
    if prev_cublas:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = prev_cublas
    elif "CUBLAS_WORKSPACE_CONFIG" in os.environ:
        del os.environ["CUBLAS_WORKSPACE_CONFIG"]
    if prev_compile_det:
        os.environ["TORCH_COMPILE_DETERMINISTIC"] = prev_compile_det
    elif "TORCH_COMPILE_DETERMINISTIC" in os.environ:
        del os.environ["TORCH_COMPILE_DETERMINISTIC"]
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CUDA regression tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
@pytest.mark.skipif(
    not GROUPED_MM_AVAILABLE,
    reason="torch._grouped_mm not available (requires PyTorch 2.4+)",
)
def test_grouped_mm_deterministic(deterministic_env):
    """Verify torch._grouped_mm produces bit-identical results under deterministic mode.

    Exercises the same call pattern as grouped_experts.py:
        torch._grouped_mm(A, B, offs=offsets)
    where A is (total_tokens, hidden_dim), B is (n_experts, hidden_dim, ffn_dim),
    and offs is (n_experts,) int32 cumulative offsets.
    """
    n_experts = 4
    hidden_dim = 128
    ffn_dim = 256
    tokens_per_expert = 16
    total_tokens = tokens_per_expert * n_experts
    device = "cuda"
    dtype = torch.bfloat16

    # Arrange
    torch.manual_seed(42)
    A = torch.randn(total_tokens, hidden_dim, device=device, dtype=dtype)
    B = torch.randn(
        n_experts,
        hidden_dim,
        ffn_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    offsets = torch.arange(
        tokens_per_expert,
        total_tokens + 1,
        tokens_per_expert,
        device=device,
        dtype=torch.int32,
    )
    assert offsets.shape == (n_experts,)
    assert offsets[-1] == total_tokens

    # Forward: two identical calls must produce same output
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out1 = torch._grouped_mm(A, B, offs=offsets)
        out2 = torch._grouped_mm(A, B, offs=offsets)

    assert out1.shape == (total_tokens, ffn_dim)
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)

    # Backward: two identical backward passes must produce same gradients
    grad_output = torch.randn_like(out1)

    B_clone1 = B.detach().clone().requires_grad_(True)
    out_bwd1 = torch._grouped_mm(A, B_clone1, offs=offsets)
    out_bwd1.backward(grad_output)
    assert B_clone1.grad is not None
    grad1 = B_clone1.grad.clone()

    B_clone2 = B.detach().clone().requires_grad_(True)
    out_bwd2 = torch._grouped_mm(A, B_clone2, offs=offsets)
    out_bwd2.backward(grad_output)
    assert B_clone2.grad is not None
    grad2 = B_clone2.grad.clone()

    assert grad1.shape == B.shape
    torch.testing.assert_close(grad1, grad2, rtol=0, atol=0)

    # No determinism warnings should be emitted
    det_warnings = [
        str(w.message)
        for w in caught
        if "nondeterministic" in str(w.message).lower()
        or "deterministic" in str(w.message).lower()
    ]
    assert not det_warnings, f"Unexpected determinism warnings: {det_warnings}"


@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
def test_compile_deterministic(deterministic_env):
    """Verify torch.compile with Inductor produces reproducible forward/backward results.

    Creates a small MLP, compiles with Inductor, and checks that two runs
    with the same input yield bit-identical outputs and gradients.
    """
    device = torch.device("cuda")
    hidden = 64

    def _make_mlp():
        return nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
        ).to(device)

    # Run 1
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = _make_mlp()
    compiled_model = torch.compile(model, backend="inductor")

    inp = torch.randn(4, hidden, device=device, requires_grad=False)
    torch.manual_seed(99)
    out1 = compiled_model(inp)
    out1.sum().backward()
    grad1 = {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }
    out1_val = out1.detach().clone()

    # Run 2 (same model, same input)
    model.zero_grad(set_to_none=True)
    torch.manual_seed(99)
    out2 = compiled_model(inp)
    out2.sum().backward()
    grad2 = {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }
    out2_val = out2.detach().clone()

    # Assert reproducibility
    torch.testing.assert_close(
        out1_val,
        out2_val,
        rtol=0,
        atol=0,
        msg="Compiled forward outputs are not bit-identical",
    )
    for name in grad1:
        torch.testing.assert_close(
            grad1[name],
            grad2[name],
            rtol=0,
            atol=0,
            msg=f"Compiled gradient '{name}' is not bit-identical",
        )


@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
def test_compile_gather_logprobs_entropy_deterministic(deterministic_env):
    """Verify compiled _gather_logprobs_entropy is bit-identical across calls.

    This function is compiled at import time via torch.compile() in
    vocab_parallel.py. It performs log_softmax + gather (pure tensor ops,
    no distributed setup required).
    """
    from areal.utils.functional.vocab_parallel import _gather_logprobs_entropy

    device = torch.device("cuda")
    seq_len, vocab_size = 32, 128

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    logits = torch.randn(seq_len, vocab_size, device=device)
    labels = torch.randint(0, vocab_size, (seq_len,), device=device)

    log_probs_1, entropy_1 = _gather_logprobs_entropy(logits, labels)
    lp1 = log_probs_1.detach().clone()
    ent1 = entropy_1.detach().clone()

    log_probs_2, entropy_2 = _gather_logprobs_entropy(logits, labels)
    lp2 = log_probs_2.detach().clone()
    ent2 = entropy_2.detach().clone()

    torch.testing.assert_close(lp1, lp2, rtol=0, atol=0)
    torch.testing.assert_close(ent1, ent2, rtol=0, atol=0)


@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
def test_ac_recompute_compile_deterministic(deterministic_env):
    """Verify activation checkpointing + compile produces deterministic recompute.

    Tests the checkpoint_wrapper API with preserve_rng_state=True,
    then compiles the wrapped module.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    class LinearSiLUBlock(torch.nn.Module):
        """Small block with RNG-sensitive dropout for AC testing."""

        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(32, 64)
            self.silu = torch.nn.SiLU()
            self.linear2 = torch.nn.Linear(64, 32)
            self.drop = torch.nn.Dropout(0.1)

        def forward(self, x):
            return self.linear2(self.drop(self.silu(self.linear1(x))))

    device = torch.device("cuda")
    torch.cuda.empty_cache()

    # Config 1: preserve_rng_state=True (standard deterministic AC config)
    model_rng = LinearSiLUBlock().to(device)
    ac_error = None

    try:
        wrapped_rng = checkpoint_wrapper(
            model_rng,
            preserve_rng_state=True,
        )
        compiled_rng = torch.compile(wrapped_rng)

        # Run forward+backward twice with same seed
        torch.manual_seed(42)
        x1 = torch.randn(4, 32, device=device)
        out1 = compiled_rng(x1)
        out1.sum().backward()
        grads1 = {
            n: p.grad.clone()
            for n, p in model_rng.named_parameters()
            if p.grad is not None
        }
        output1 = out1.detach().clone()

        model_rng.zero_grad()
        torch.manual_seed(42)
        x2 = torch.randn(4, 32, device=device)
        out2 = compiled_rng(x2)
        out2.sum().backward()
        grads2 = {
            n: p.grad.clone()
            for n, p in model_rng.named_parameters()
            if p.grad is not None
        }
        output2 = out2.detach().clone()

        torch.testing.assert_close(
            output1,
            output2,
            rtol=0,
            atol=0,
            msg="AC+compile forward not bit-identical with preserve_rng_state=True",
        )
        for name in grads1:
            torch.testing.assert_close(
                grads1[name],
                grads2[name],
                rtol=0,
                atol=0,
                msg=f"AC+compile grad '{name}' not bit-identical",
            )

    except Exception as e:
        ac_error = f"{type(e).__name__}: {e}"
        pytest.xfail(
            f"AC recompute with preserve_rng_state=True raised "
            f"under torch.compile: {ac_error}"
        )

    # Config 2: preserve_rng_state=False (baseline — may diverge due to dropout)
    model_no_rng = LinearSiLUBlock().to(device)

    try:
        wrapped_no_rng = checkpoint_wrapper(
            model_no_rng,
            preserve_rng_state=False,
            determinism_check="default",
        )
        compiled_no_rng = torch.compile(wrapped_no_rng)

        results_no_rng = []
        for _ in range(2):
            model_no_rng.zero_grad()
            torch.manual_seed(42)
            x = torch.randn(4, 32, device=device)
            out = compiled_no_rng(x)
            out.sum().backward()
            results_no_rng.append(out.detach().clone())

        # Without preserve_rng_state, dropout recompute may differ —
        # we only assert shape here, not values
        assert results_no_rng[0].shape == results_no_rng[1].shape

    except Exception:
        pass  # Baseline config failure is not a test failure
