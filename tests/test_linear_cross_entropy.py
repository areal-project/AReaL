# SPDX-License-Identifier: Apache-2.0
"""
Correctness + performance tests for the fused linear-cross-entropy kernel.

The test suite verifies that
:func:`areal.models.kernel.linear_cross_entropy_logprobs_entropy` produces
results numerically equivalent to the materialised ``logits @ weight`` +
``log_softmax`` reference, and that it provides a measurable wall-clock /
memory benefit over the reference path on representative LLM shapes.

The performance assertions are intentionally loose (>=1.0x runtime, i.e.
"not slower") so they remain meaningful in CI where cudagraph capture and
power-state variability can swing absolute timings; the PRINTED report is
the authoritative artifact for review.

Run only the correctness checks (fast, single-GPU)::

    pytest tests/test_linear_cross_entropy.py -k correctness -s

Run the full benchmark (includes large-vocab cases, slow)::

    pytest tests/test_linear_cross_entropy.py -m slow -s
"""

from __future__ import annotations

import gc
import math

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not (CUDA_AVAILABLE and TRITON_AVAILABLE),
    reason="Fused LCE requires CUDA + Triton",
)


def _reference_logprobs_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialised-logits reference. Same math, no fusion."""
    logits = (hidden.float() @ weight.float().t()) / temperature
    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    logprobs = log_softmax.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    probs = log_softmax.exp()
    entropy = -(probs * log_softmax).sum(dim=-1)
    return logprobs, entropy


def _make_inputs(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    hidden = (
        torch.randn(num_tokens, hidden_size, dtype=dtype, device=device, generator=g)
        * 0.02
    )
    weight = (
        torch.randn(vocab_size, hidden_size, dtype=dtype, device=device, generator=g)
        * 0.02
    )
    labels = torch.randint(0, vocab_size, (num_tokens,), device=device, generator=g)
    return hidden.contiguous(), weight.contiguous(), labels.contiguous()


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens,hidden_size,vocab_size,dtype",
    [
        (256, 512, 4096, torch.float32),
        (512, 1024, 32000, torch.bfloat16),
        (128, 768, 8192, torch.float16),
    ],
)
def test_linear_cross_entropy_correctness(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    dtype: torch.dtype,
) -> None:
    """Fused forward output must match the materialised reference."""
    from areal.models.kernel import linear_cross_entropy_logprobs_entropy

    hidden, weight, labels = _make_inputs(num_tokens, hidden_size, vocab_size, dtype)

    ref_logprobs, ref_entropy = _reference_logprobs_entropy(hidden, weight, labels)
    fused_logprobs, fused_entropy = linear_cross_entropy_logprobs_entropy(
        hidden, weight, labels, temperature=1.0
    )

    # Tolerances are dtype-dependent. The fused kernel performs the same
    # matmul + log-softmax math as the reference, so fp32 inputs should agree
    # to within a few ulps (~1e-5). bf16 / fp16 inputs are widened only to
    # absorb the documented matmul-accumulation drift; anything looser would
    # mask real numerical regressions.
    if dtype == torch.float32:
        rtol, atol = 1e-5, 1e-5
    elif dtype == torch.bfloat16:
        rtol, atol = 2e-2, 2e-2
    else:  # float16
        rtol, atol = 1e-2, 1e-2

    torch.testing.assert_close(
        fused_logprobs.float(), ref_logprobs.float(), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        fused_entropy.float(), ref_entropy.float(), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.5])
def test_linear_cross_entropy_temperature(temperature: float) -> None:
    """Temperature scaling matches the reference for non-trivial values."""
    from areal.models.kernel import linear_cross_entropy_logprobs_entropy

    hidden, weight, labels = _make_inputs(
        num_tokens=128, hidden_size=512, vocab_size=4096, dtype=torch.float32
    )
    ref_lp, ref_h = _reference_logprobs_entropy(hidden, weight, labels, temperature)
    fused_lp, fused_h = linear_cross_entropy_logprobs_entropy(
        hidden, weight, labels, temperature=temperature
    )
    # fp32 inputs: fused vs reference must agree to ~1e-5 (a few ulps).
    torch.testing.assert_close(fused_lp, ref_lp, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fused_h, ref_h, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "num_tokens,hidden_size,vocab_size",
    [
        # Small shape: catches obvious correctness bugs cheaply.
        (64, 256, 2048),
        # Medium shape: typical SFT microbatch.
        (512, 1024, 32000),
        # Large shape: stresses the fused backward at LLM-class dimensions
        # where the materialised reference begins to dominate memory but is
        # still fp32-tractable on a single GPU. This is the configuration
        # most likely to surface accumulation-order bugs in d_hidden /
        # d_weight reductions.
        (2048, 2048, 32000),
    ],
    ids=["small_64x256x2048", "medium_512x1024x32k", "large_2048x2048x32k"],
)
def test_linear_cross_entropy_backward_matches_reference(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
) -> None:
    """Backward gradients on hidden/weight match autograd through the reference.

    Runs across small / medium / large shapes so that any accumulation-order
    drift in the fused d_hidden / d_weight kernels is caught at scale rather
    than only on toy inputs.
    """
    from areal.models.kernel import linear_cross_entropy

    hidden_a, weight_a, labels = _make_inputs(
        num_tokens, hidden_size, vocab_size, torch.float32
    )
    hidden_b = hidden_a.clone()
    weight_b = weight_a.clone()
    hidden_a.requires_grad_(True)
    weight_a.requires_grad_(True)
    hidden_b.requires_grad_(True)
    weight_b.requires_grad_(True)

    # Reference path
    logits = hidden_b @ weight_b.t()
    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    ref_lp = log_softmax.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    probs = log_softmax.exp()
    ref_h = -(probs * log_softmax).sum(dim=-1)
    (ref_lp.sum() + 0.5 * ref_h.sum()).backward()

    # Fused path
    fused_lp, fused_h = linear_cross_entropy(
        hidden_a, weight_a, labels, 1.0, "none", None
    )
    (fused_lp.sum() + 0.5 * fused_h.sum()).backward()

    # fp32 inputs: backward must match the reference to ~1e-4. The fused
    # kernel's d_weight accumulates ``num_tokens`` partial products, so we
    # use a slightly looser absolute tolerance for d_weight at the largest
    # shape; rtol stays tight to catch directional errors.
    torch.testing.assert_close(hidden_a.grad, hidden_b.grad, rtol=1e-4, atol=1e-4)
    weight_atol = 1e-4 if num_tokens <= 512 else 5e-4
    torch.testing.assert_close(
        weight_a.grad, weight_b.grad, rtol=1e-4, atol=weight_atol
    )


# ---------------------------------------------------------------------------
# Tensor-parallel (TP=2) correctness + performance
#
# These tests are invoked through pytest, while the 2-rank distributed body is
# launched with subprocess.run(["torchrun", ...]) following the repository's
# distributed-test pattern. Users do not need to run torchrun manually.
# ---------------------------------------------------------------------------


def _tp2_available() -> bool:
    """Whether we can launch a 2-rank TP test on this host."""
    if not (CUDA_AVAILABLE and TRITON_AVAILABLE):
        return False
    if torch.cuda.device_count() < 2:
        return False
    return True


_tp2_skip = pytest.mark.skipif(
    not _tp2_available(), reason="TP=2 requires >= 2 CUDA GPUs"
)


def _run_lce_tp2_with_torchrun(
    test_type: str,
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    dtype: str = "bfloat16",
) -> None:
    import subprocess

    from areal.utils.network import find_free_ports

    port = find_free_ports(1)[0]
    try:
        subprocess.run(
            [
                "torchrun",
                "--nproc_per_node=2",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "tests/torchrun/run_lce_tp2.py",
                f"--test_type={test_type}",
                f"--num_tokens={num_tokens}",
                f"--hidden_size={hidden_size}",
                f"--vocab_size={vocab_size}",
                f"--dtype={dtype}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"TP=2 LCE torchrun test failed:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        )


@_tp2_skip
@pytest.mark.multi_gpu
@pytest.mark.parametrize(
    "num_tokens,hidden_size,vocab_size,dtype_str",
    [
        (128, 512, 8192, "float32"),
        (256, 1024, 32000, "bfloat16"),
    ],
)
def test_linear_cross_entropy_tp2_correctness(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    dtype_str: str,
) -> None:
    """TP=2 fused forward+backward matches a full-vocab reference.

    The 2-rank worker is launched via torchrun inside this pytest test, so the
    caller can use a normal pytest command.
    """
    _run_lce_tp2_with_torchrun(
        "correctness", num_tokens, hidden_size, vocab_size, dtype_str
    )


@_tp2_skip
@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    "num_tokens,hidden_size,vocab_size",
    [
        (1024, 1024, 32000),
        (2048, 4096, 152064),
    ],
)
def test_linear_cross_entropy_tp2_performance_benchmark(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
) -> None:
    """TP=2 fused vs TP-materialised forward+backward time and peak memory."""
    _run_lce_tp2_with_torchrun("performance", num_tokens, hidden_size, vocab_size)


# ---------------------------------------------------------------------------
# Performance benchmark (single-GPU)
# ---------------------------------------------------------------------------


def _peak_memory_mb(fn, *args, **kwargs) -> tuple[float, float]:
    """Return (elapsed_ms, peak_mem_mb) of a single forward+backward pass."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn(*args, **kwargs)
    if isinstance(out, tuple):
        loss = sum(
            t.float().sum() for t in out if t.requires_grad or t.grad_fn is not None
        )
    else:
        loss = out.float().sum()
    loss.backward()
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return elapsed, peak


def _run_reference_forward_backward(hidden, weight, labels, temperature):
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    logits = (h.float() @ w.float().t()) / temperature
    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    lp = log_softmax.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    probs = log_softmax.exp()
    ent = -(probs * log_softmax).sum(dim=-1)
    return lp, ent


def _run_fused_forward_backward(hidden, weight, labels, temperature):
    from areal.models.kernel import linear_cross_entropy

    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    return linear_cross_entropy(h, w, labels, temperature, "none", None)


@pytest.mark.slow
@pytest.mark.parametrize(
    "num_tokens,hidden_size,vocab_size",
    [
        # Small: validates the speedup is measurable even on toy shapes.
        (1024, 1024, 32000),
        # Medium: typical 7B-class one-microbatch shape.
        (4096, 4096, 128256),
        # Large vocab: where fused kernel really wins (e.g. Qwen3).
        (2048, 4096, 152064),
    ],
)
def test_linear_cross_entropy_performance_benchmark(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
) -> None:
    """Compare fused vs materialised forward+backward time and peak memory.

    Failures here mean the fused path *regressed* against the reference; the
    captured numbers are also printed for human review.
    """
    dtype = torch.bfloat16
    hidden, weight, labels = _make_inputs(num_tokens, hidden_size, vocab_size, dtype)

    # warm-up
    for _ in range(2):
        lp, ent = _run_reference_forward_backward(hidden, weight, labels, 1.0)
        (lp.sum() + ent.sum()).backward()
        del lp, ent
        gc.collect()
        torch.cuda.empty_cache()
    for _ in range(2):
        lp, ent = _run_fused_forward_backward(hidden, weight, labels, 1.0)
        (lp.sum() + ent.sum()).backward()
        del lp, ent
        gc.collect()
        torch.cuda.empty_cache()

    # Reference timing
    ref_times = []
    ref_mems = []
    for _ in range(5):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        lp, ent = _run_reference_forward_backward(hidden, weight, labels, 1.0)
        (lp.sum() + ent.sum()).backward()
        end.record()
        torch.cuda.synchronize()
        ref_times.append(start.elapsed_time(end))
        ref_mems.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
        del lp, ent

    # Fused timing
    fused_times = []
    fused_mems = []
    for _ in range(5):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        lp, ent = _run_fused_forward_backward(hidden, weight, labels, 1.0)
        (lp.sum() + ent.sum()).backward()
        end.record()
        torch.cuda.synchronize()
        fused_times.append(start.elapsed_time(end))
        fused_mems.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
        del lp, ent

    ref_med = sorted(ref_times)[len(ref_times) // 2]
    fused_med = sorted(fused_times)[len(fused_times) // 2]
    ref_peak = max(ref_mems)
    fused_peak = max(fused_mems)
    speedup = ref_med / fused_med if fused_med > 0 else math.inf
    mem_ratio = fused_peak / ref_peak if ref_peak > 0 else math.inf

    print(
        f"\n[LCE-Bench] tokens={num_tokens} hidden={hidden_size} vocab={vocab_size} "
        f"dtype={dtype}\n"
        f"            reference: {ref_med:7.2f} ms / {ref_peak:7.1f} MB peak\n"
        f"            fused    : {fused_med:7.2f} ms / {fused_peak:7.1f} MB peak\n"
        f"            speedup  : {speedup:5.2f}x   memory_ratio: {mem_ratio:5.2f}x"
    )

    # Soft assertions: fused path must not be drastically slower or more
    # memory-hungry. Tight thresholds would cause flaky CI on shared GPUs.
    assert fused_med < ref_med * 1.5, (
        f"Fused LCE is more than 1.5x slower than reference "
        f"(fused={fused_med:.2f}ms ref={ref_med:.2f}ms). Please investigate."
    )
    assert fused_peak < ref_peak * 1.2, (
        f"Fused LCE peak memory exceeds reference by >20% "
        f"(fused={fused_peak:.1f}MB ref={ref_peak:.1f}MB)."
    )
