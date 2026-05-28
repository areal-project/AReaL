import os

import torch
import torch.distributed as dist

from areal.infra.platforms import current_platform
from areal.utils.functional.vocab_parallel import (
    _vocab_parallel_logprobs,
    _vocab_parallel_logprobs_entropy,
)


def reference_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Reference implementation: compute logprobs from full logits."""
    log_softmax = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    logprobs = log_softmax.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return logprobs


def reference_logprobs_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation: compute both logprobs and entropy."""
    log_softmax = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    probs = log_softmax.exp()
    logprobs = log_softmax.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    entropy = -(probs * log_softmax).sum(dim=-1)
    return logprobs, entropy


def setup_distributed_environment():
    if dist.is_initialized():
        return
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )
    current_platform.set_device(rank)


def get_tp_group() -> dist.ProcessGroup:
    """Get the tensor parallel process group."""
    return dist.distributed_c10d._get_default_group()


def test_vocab_parallel_logprobs():
    """Test _vocab_parallel_logprobs with actual TP distribution."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = current_platform.current_device()

    batch_size, seq_len, vocab_size = 4, 16, 1024
    assert vocab_size % world_size == 0, "vocab_size must be divisible by world_size"
    partition_size = vocab_size // world_size

    # Generate same data on all ranks (seeded)
    torch.manual_seed(42)
    full_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Shard logits for this rank
    start_idx = rank * partition_size
    end_idx = start_idx + partition_size
    local_logits = full_logits[..., start_idx:end_idx].clone()

    # Compute vocab parallel logprobs
    result = _vocab_parallel_logprobs(local_logits, labels, get_tp_group())

    # Compute reference
    expected = reference_logprobs(full_logits, labels)

    # Verify
    if not torch.allclose(result, expected, atol=1e-5, rtol=1e-5):
        max_diff = (result - expected).abs().max().item()
        raise ValueError(
            f"[Rank {rank}] _vocab_parallel_logprobs mismatch! Max diff: {max_diff}"
        )

    if rank == 0:
        print("✓ test_vocab_parallel_logprobs passed")


def test_vocab_parallel_logprobs_entropy():
    """Test _vocab_parallel_logprobs_entropy with actual TP distribution."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = current_platform.current_device()

    batch_size, seq_len, vocab_size = 4, 16, 1024
    partition_size = vocab_size // world_size

    # Generate same data on all ranks (seeded)
    torch.manual_seed(123)
    full_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Shard logits for this rank
    start_idx = rank * partition_size
    end_idx = start_idx + partition_size
    local_logits = full_logits[..., start_idx:end_idx].clone()

    # Compute vocab parallel
    logprobs, entropy = _vocab_parallel_logprobs_entropy(
        local_logits, labels, get_tp_group()
    )

    # Compute reference
    expected_logprobs, expected_entropy = reference_logprobs_entropy(
        full_logits, labels
    )

    # Verify logprobs
    if not torch.allclose(logprobs, expected_logprobs, atol=1e-5, rtol=1e-5):
        max_diff = (logprobs - expected_logprobs).abs().max().item()
        raise ValueError(f"[Rank {rank}] logprobs mismatch! Max diff: {max_diff}")

    # Verify entropy
    if not torch.allclose(entropy, expected_entropy, atol=1e-5, rtol=1e-5):
        max_diff = (entropy - expected_entropy).abs().max().item()
        raise ValueError(f"[Rank {rank}] entropy mismatch! Max diff: {max_diff}")

    if rank == 0:
        print("✓ test_vocab_parallel_logprobs_entropy passed")


def test_vocab_parallel_with_temperature():
    """Test vocab parallel functions with temperature scaling."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = current_platform.current_device()

    batch_size, seq_len, vocab_size = 2, 8, 512
    partition_size = vocab_size // world_size
    temperature = 0.7

    torch.manual_seed(456)
    full_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    start_idx = rank * partition_size
    end_idx = start_idx + partition_size
    local_logits = full_logits[..., start_idx:end_idx].clone()

    # Compute with temperature
    result = _vocab_parallel_logprobs(
        local_logits, labels, get_tp_group(), temperature=temperature
    )

    # Reference with temperature applied
    expected = reference_logprobs(full_logits / temperature, labels)

    if not torch.allclose(result, expected, atol=1e-5, rtol=1e-5):
        max_diff = (result - expected).abs().max().item()
        raise ValueError(
            f"[Rank {rank}] temperature test mismatch! Max diff: {max_diff}"
        )

    if rank == 0:
        print("✓ test_vocab_parallel_with_temperature passed")


def test_vocab_parallel_numerical_stability():
    """Test numerical stability with large logit values."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = current_platform.current_device()

    batch_size, seq_len, vocab_size = 2, 8, 512
    partition_size = vocab_size // world_size

    # Large logits that could cause overflow without proper handling
    torch.manual_seed(789)
    full_logits = torch.randn(batch_size, seq_len, vocab_size, device=device) * 100
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    start_idx = rank * partition_size
    end_idx = start_idx + partition_size
    local_logits = full_logits[..., start_idx:end_idx].clone()

    logprobs, entropy = _vocab_parallel_logprobs_entropy(
        local_logits, labels, get_tp_group()
    )

    # Check no NaN or Inf
    if torch.isnan(logprobs).any() or torch.isinf(logprobs).any():
        raise ValueError(f"[Rank {rank}] logprobs has NaN or Inf!")
    if torch.isnan(entropy).any() or torch.isinf(entropy).any():
        raise ValueError(f"[Rank {rank}] entropy has NaN or Inf!")

    # Verify against reference
    expected_logprobs, expected_entropy = reference_logprobs_entropy(
        full_logits, labels
    )

    if not torch.allclose(logprobs, expected_logprobs, atol=1e-4, rtol=1e-4):
        max_diff = (logprobs - expected_logprobs).abs().max().item()
        raise ValueError(
            f"[Rank {rank}] numerical stability logprobs mismatch! Max diff: {max_diff}"
        )

    if not torch.allclose(entropy, expected_entropy, atol=1e-4, rtol=1e-4):
        max_diff = (entropy - expected_entropy).abs().max().item()
        raise ValueError(
            f"[Rank {rank}] numerical stability entropy mismatch! Max diff: {max_diff}"
        )

    if rank == 0:
        print("✓ test_vocab_parallel_numerical_stability passed")


def test_vocab_parallel_gradient():
    """Test gradient computation with vocab parallel."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = current_platform.current_device()

    batch_size, seq_len, vocab_size = 2, 4, 128
    partition_size = vocab_size // world_size

    torch.manual_seed(999)
    full_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    start_idx = rank * partition_size
    end_idx = start_idx + partition_size

    # Test gradient for _vocab_parallel_logprobs
    local_logits = full_logits[..., start_idx:end_idx].clone().requires_grad_(True)
    result = _vocab_parallel_logprobs(local_logits, labels, get_tp_group())
    result.sum().backward()

    assert local_logits.grad is not None, "Gradient should not be None"
    assert local_logits.grad.shape == local_logits.shape, "Gradient shape mismatch"
    assert not torch.isnan(local_logits.grad).any(), "Gradient has NaN"

    # Test gradient for _vocab_parallel_logprobs_entropy
    local_logits2 = full_logits[..., start_idx:end_idx].clone().requires_grad_(True)
    logprobs, entropy = _vocab_parallel_logprobs_entropy(
        local_logits2, labels, get_tp_group()
    )
    (logprobs.sum() + entropy.sum()).backward()

    assert local_logits2.grad is not None, "Gradient should not be None"
    assert not torch.isnan(local_logits2.grad).any(), "Gradient has NaN"

    if rank == 0:
        print("✓ test_vocab_parallel_gradient passed")


def test_vocab_parallel_gradient_correctness():
    """Verify gradient correctness by comparing with reference implementation.

    Note: We don't use torch.autograd.gradcheck because the vocab parallel
    implementation uses in-place operations for memory efficiency, which
    doesn't support multiple backward calls that gradcheck requires.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = current_platform.current_device()

    batch_size, seq_len, vocab_size = 2, 4, 128
    partition_size = vocab_size // world_size

    torch.manual_seed(42)
    full_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    start_idx = rank * partition_size
    end_idx = start_idx + partition_size

    # Test gradient correctness for _vocab_parallel_logprobs
    local_logits = full_logits[..., start_idx:end_idx].clone().requires_grad_(True)
    result = _vocab_parallel_logprobs(local_logits, labels, get_tp_group())
    result.sum().backward()
    vocab_parallel_grad = local_logits.grad.clone()

    # Compute reference gradient using full logits
    full_logits_ref = full_logits.clone().requires_grad_(True)
    ref_result = reference_logprobs(full_logits_ref, labels)
    ref_result.sum().backward()
    ref_grad = full_logits_ref.grad[..., start_idx:end_idx]

    if not torch.allclose(vocab_parallel_grad, ref_grad, atol=1e-5, rtol=1e-5):
        max_diff = (vocab_parallel_grad - ref_grad).abs().max().item()
        raise ValueError(
            f"[Rank {rank}] logprobs gradient mismatch! Max diff: {max_diff}"
        )

    # Test gradient correctness for _vocab_parallel_logprobs_entropy
    local_logits2 = full_logits[..., start_idx:end_idx].clone().requires_grad_(True)
    logprobs, entropy = _vocab_parallel_logprobs_entropy(
        local_logits2, labels, get_tp_group()
    )
    (logprobs.sum() + entropy.sum()).backward()
    vocab_parallel_grad2 = local_logits2.grad.clone()

    # Compute reference gradient for logprobs + entropy
    full_logits_ref2 = full_logits.clone().requires_grad_(True)
    ref_logprobs, ref_entropy = reference_logprobs_entropy(full_logits_ref2, labels)
    (ref_logprobs.sum() + ref_entropy.sum()).backward()
    ref_grad2 = full_logits_ref2.grad[..., start_idx:end_idx]

    if not torch.allclose(vocab_parallel_grad2, ref_grad2, atol=1e-5, rtol=1e-5):
        max_diff = (vocab_parallel_grad2 - ref_grad2).abs().max().item()
        raise ValueError(
            f"[Rank {rank}] logprobs+entropy gradient mismatch! Max diff: {max_diff}"
        )

    if rank == 0:
        print("✓ test_vocab_parallel_gradient_correctness passed")


def test_vocab_parallel_different_shapes():
    """Test with different input shapes (1D, 2D, 3D)."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = current_platform.current_device()

    vocab_size = 512
    partition_size = vocab_size // world_size

    # Test 1D input (packed sequences)
    torch.manual_seed(111)
    total_tokens = 64
    full_logits_1d = torch.randn(total_tokens, vocab_size, device=device)
    labels_1d = torch.randint(0, vocab_size, (total_tokens,), device=device)

    start_idx = rank * partition_size
    end_idx = start_idx + partition_size
    local_logits_1d = full_logits_1d[..., start_idx:end_idx].clone()

    result_1d = _vocab_parallel_logprobs(local_logits_1d, labels_1d, get_tp_group())
    expected_1d = reference_logprobs(full_logits_1d, labels_1d)

    if not torch.allclose(result_1d, expected_1d, atol=1e-5, rtol=1e-5):
        raise ValueError(f"[Rank {rank}] 1D input mismatch!")

    # Test 2D input
    torch.manual_seed(222)
    seq_len = 32
    full_logits_2d = torch.randn(seq_len, vocab_size, device=device)
    labels_2d = torch.randint(0, vocab_size, (seq_len,), device=device)

    local_logits_2d = full_logits_2d[..., start_idx:end_idx].clone()
    result_2d = _vocab_parallel_logprobs(local_logits_2d, labels_2d, get_tp_group())
    expected_2d = reference_logprobs(full_logits_2d, labels_2d)

    if not torch.allclose(result_2d, expected_2d, atol=1e-5, rtol=1e-5):
        raise ValueError(f"[Rank {rank}] 2D input mismatch!")

    # Test 3D input (batch, seq, vocab)
    torch.manual_seed(333)
    batch_size, seq_len = 4, 16
    full_logits_3d = torch.randn(batch_size, seq_len, vocab_size, device=device)
    labels_3d = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    local_logits_3d = full_logits_3d[..., start_idx:end_idx].clone()
    result_3d = _vocab_parallel_logprobs(local_logits_3d, labels_3d, get_tp_group())
    expected_3d = reference_logprobs(full_logits_3d, labels_3d)

    if not torch.allclose(result_3d, expected_3d, atol=1e-5, rtol=1e-5):
        raise ValueError(f"[Rank {rank}] 3D input mismatch!")

    if rank == 0:
        print("✓ test_vocab_parallel_different_shapes passed")


def run_all_tests():
    """Run all tensor parallel tests."""
    rank = dist.get_rank()

    if rank == 0:
        print(f"Running tensor parallel tests with {dist.get_world_size()} ranks...")
        print("-" * 60)

    dist.barrier()

    test_vocab_parallel_logprobs()
    dist.barrier()

    test_vocab_parallel_logprobs_entropy()
    dist.barrier()

    test_vocab_parallel_with_temperature()
    dist.barrier()

    test_vocab_parallel_numerical_stability()
    dist.barrier()

    test_vocab_parallel_gradient()
    dist.barrier()

    test_vocab_parallel_gradient_correctness()
    dist.barrier()

    test_vocab_parallel_different_shapes()
    dist.barrier()

    if rank == 0:
        print("-" * 60)
        print("All tensor parallel tests passed! ✓")


if __name__ == "__main__":
    setup_distributed_environment()
    try:
        run_all_tests()
    finally:
        dist.destroy_process_group()
