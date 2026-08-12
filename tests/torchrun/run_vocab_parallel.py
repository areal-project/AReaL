import os

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer import TransformerConfig

from areal.infra.platforms import current_platform
from areal.models.mcore import vocab_parallel_head as lm_head_module
from areal.models.mcore.vocab_parallel_head import (
    chunked_lm_head_logprobs_entropy,
    replace_output_layer_with_areal_lm_head,
)
from areal.utils.functional.vocab_parallel import (
    _inplace_vocab_parallel_logprobs_entropy,
    _vocab_parallel_logprobs,
    _vocab_parallel_logprobs_entropy,
    gather_logprobs_entropy,
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


def test_inplace_vocab_parallel_logprobs_entropy():
    """Test the destructive fused path with real TP collectives."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = current_platform.current_device()
    num_tokens, vocab_size = 13, 3072
    partition_size = vocab_size // world_size
    temperature = 0.7

    torch.manual_seed(444)
    full_logits = torch.randn(num_tokens, vocab_size, device=device)
    labels = torch.randint(0, vocab_size, (num_tokens,), device=device)
    logprob_weights = torch.randn(num_tokens, device=device)
    start_idx = rank * partition_size
    end_idx = start_idx + partition_size

    full_logits_ref = full_logits.clone().requires_grad_(True)
    ref_logprobs, ref_entropy = reference_logprobs_entropy(
        full_logits_ref / temperature, labels
    )
    (ref_logprobs * logprob_weights).sum().backward()

    local_logits_leaf = (
        full_logits[..., start_idx:end_idx].clone().contiguous().requires_grad_(True)
    )
    local_logits = local_logits_leaf + 0.0
    storage_ptr = local_logits.data_ptr()
    all_reduce_calls = 0
    original_all_reduce = dist.all_reduce

    def counted_all_reduce(*args, **kwargs):
        nonlocal all_reduce_calls
        all_reduce_calls += 1
        return original_all_reduce(*args, **kwargs)

    dist.all_reduce = counted_all_reduce
    try:
        logprobs, entropy = _inplace_vocab_parallel_logprobs_entropy(
            local_logits,
            labels,
            get_tp_group(),
            temperature=temperature,
            chunk_size=5,
        )
    finally:
        dist.all_reduce = original_all_reduce
    (logprobs * logprob_weights).sum().backward()

    assert local_logits.data_ptr() == storage_ptr
    assert all_reduce_calls == 2 * ((num_tokens + 4) // 5)
    assert not entropy.requires_grad
    torch.testing.assert_close(logprobs, ref_logprobs, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(entropy, ref_entropy, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(
        local_logits_leaf.grad,
        full_logits_ref.grad[..., start_idx:end_idx],
        rtol=1e-5,
        atol=2e-5,
    )

    if rank == 0:
        print("✓ test_inplace_vocab_parallel_logprobs_entropy passed")


class _HeadContainer(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.output_layer = head


def test_areal_lm_head_tensor_and_sequence_parallel():
    """Test native and FP32 output with TP, SP, and an externally tied weight."""
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=dist.get_world_size()
        )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = current_platform.current_device()
    hidden_size = 16
    local_vocab_size = 32
    local_sequence = 3
    tp_group = get_tp_group()

    for sequence_parallel in (False, True):
        for fp32_output in (False, True):
            for weight_requires_grad in (False, True):
                _test_areal_lm_head_tensor_and_sequence_parallel_case(
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    hidden_size=hidden_size,
                    local_vocab_size=local_vocab_size,
                    local_sequence=local_sequence,
                    tp_group=tp_group,
                    sequence_parallel=sequence_parallel,
                    fp32_output=fp32_output,
                    weight_requires_grad=weight_requires_grad,
                )

    if rank == 0:
        print("✓ test_areal_lm_head_tensor_and_sequence_parallel passed")


def _test_areal_lm_head_tensor_and_sequence_parallel_case(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    hidden_size: int,
    local_vocab_size: int,
    local_sequence: int,
    tp_group: dist.ProcessGroup,
    sequence_parallel: bool,
    fp32_output: bool,
    weight_requires_grad: bool,
) -> None:
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=4,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=world_size,
        sequence_parallel=sequence_parallel,
        gradient_accumulation_fusion=False,
    )
    head = ColumnParallelLinear(
        hidden_size,
        local_vocab_size * world_size,
        config=config,
        init_method=lambda tensor: tensor,
        bias=False,
        gather_output=False,
        skip_weight_param_allocation=True,
        tp_group=tp_group,
    )
    container = _HeadContainer(head)
    replace_output_layer_with_areal_lm_head(
        container,
        fp32_output=fp32_output,
    )

    torch.manual_seed(1000 + rank)
    local_input = torch.randn(
        local_sequence,
        1,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    tied_weight = torch.randn(
        local_vocab_size,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=weight_requires_grad,
    )
    output, _ = container.output_layer(local_input, weight=tied_weight)

    if sequence_parallel:
        gathered_inputs = [torch.empty_like(local_input) for _ in range(world_size)]
        dist.all_gather(gathered_inputs, local_input.detach(), group=tp_group)
        total_input = torch.cat(gathered_inputs, dim=0)
    else:
        total_input = local_input.detach()
    if fp32_output:
        expected_output = torch.mm(
            total_input.reshape(-1, hidden_size),
            tied_weight.detach().t(),
            out_dtype=torch.float32,
        ).view(*total_input.shape[:-1], local_vocab_size)
    else:
        expected_output = torch.matmul(
            total_input,
            tied_weight.detach().t(),
        )
    torch.testing.assert_close(output, expected_output, rtol=0.0, atol=0.0)

    torch.manual_seed(2000 + rank)
    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    grad_output_bf16 = grad_output.to(torch.bfloat16)
    expected_full_dgrad = grad_output_bf16.matmul(tied_weight.detach())

    if sequence_parallel:
        expected_input_grad = torch.empty_like(local_input)
        dist.reduce_scatter_tensor(
            expected_input_grad,
            expected_full_dgrad,
            group=tp_group,
        )
    else:
        expected_input_grad = expected_full_dgrad
        dist.all_reduce(expected_input_grad, group=tp_group)

    torch.testing.assert_close(
        local_input.grad, expected_input_grad, rtol=0.0, atol=0.0
    )
    if weight_requires_grad:
        expected_weight_grad = (
            grad_output_bf16.reshape(-1, local_vocab_size)
            .t()
            .matmul(total_input.reshape(-1, hidden_size))
        )
        torch.testing.assert_close(
            tied_weight.grad, expected_weight_grad, rtol=0.0, atol=0.0
        )
    else:
        assert tied_weight.grad is None


def test_areal_lm_head_packed_dlogits_tensor_and_sequence_parallel():
    """Test fused-loss storage packing through MCore TP/SP backward."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = current_platform.current_device()
    hidden_size = 16
    local_vocab_size = 32
    local_sequence = 3
    tp_group = get_tp_group()

    class HeadContainer(torch.nn.Module):
        def __init__(self, head):
            super().__init__()
            self.output_layer = head

    for sequence_parallel in (False, True):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            params_dtype=torch.bfloat16,
            tensor_model_parallel_size=world_size,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=False,
        )
        head = ColumnParallelLinear(
            hidden_size,
            local_vocab_size * world_size,
            config=config,
            init_method=lambda tensor: tensor,
            bias=False,
            gather_output=False,
            skip_weight_param_allocation=True,
            tp_group=tp_group,
        )
        container = HeadContainer(head)
        replace_output_layer_with_areal_lm_head(container, fp32_output=True)

        torch.manual_seed(3000 + rank)
        local_input = torch.randn(
            local_sequence,
            1,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
            requires_grad=True,
        )
        tied_weight = torch.randn(
            local_vocab_size,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
            requires_grad=True,
        )
        output, _ = container.output_layer(local_input, weight=tied_weight)

        gathered_logits = [torch.empty_like(output) for _ in range(world_size)]
        dist.all_gather(gathered_logits, output.detach(), group=tp_group)
        reference_logits = torch.cat(gathered_logits, dim=-1).requires_grad_(True)
        torch.manual_seed(4000)
        labels = torch.randint(
            0,
            local_vocab_size * world_size,
            output.shape[:-1],
            device=device,
        )
        loss_weights = torch.randn(output.shape[:-1], device=device)
        reference_logprobs = (
            torch.log_softmax(reference_logits, dim=-1)
            .gather(-1, labels.unsqueeze(-1))
            .squeeze(-1)
        )
        (reference_logprobs * loss_weights).sum().backward()

        pack_calls = 0
        original_pack = lm_head_module._pack_fp32_to_half_inplace

        def counted_pack(*args, **kwargs):
            nonlocal pack_calls
            pack_calls += 1
            return original_pack(*args, **kwargs)

        lm_head_module._pack_fp32_to_half_inplace = counted_pack
        try:
            logprobs, entropy = gather_logprobs_entropy(
                output,
                labels,
                tp_group=tp_group,
                reuse_logits=True,
                chunk_size=2,
            )
            (logprobs * loss_weights).sum().backward()
        finally:
            lm_head_module._pack_fp32_to_half_inplace = original_pack

        assert pack_calls == 1
        assert not entropy.requires_grad
        torch.testing.assert_close(logprobs, reference_logprobs, rtol=1e-5, atol=2e-5)
        local_dlogits = reference_logits.grad[
            ..., rank * local_vocab_size : (rank + 1) * local_vocab_size
        ].to(torch.bfloat16)
        expected_full_dgrad = local_dlogits.matmul(tied_weight.detach())

        if sequence_parallel:
            expected_input_grad = torch.empty_like(local_input)
            dist.reduce_scatter_tensor(
                expected_input_grad,
                expected_full_dgrad,
                group=tp_group,
            )
            gathered_inputs = [torch.empty_like(local_input) for _ in range(world_size)]
            dist.all_gather(gathered_inputs, local_input.detach(), group=tp_group)
            total_input = torch.cat(gathered_inputs, dim=0)
        else:
            expected_input_grad = expected_full_dgrad
            dist.all_reduce(expected_input_grad, group=tp_group)
            total_input = local_input.detach()

        expected_weight_grad = (
            local_dlogits.reshape(-1, local_vocab_size)
            .t()
            .matmul(total_input.reshape(-1, hidden_size))
        )
        torch.testing.assert_close(
            local_input.grad, expected_input_grad, rtol=1e-5, atol=2e-5
        )
        torch.testing.assert_close(
            tied_weight.grad, expected_weight_grad, rtol=1e-5, atol=2e-5
        )

    if rank == 0:
        print("✓ test_areal_lm_head_packed_dlogits_tensor_and_sequence_parallel passed")


def test_chunked_lm_head_tensor_and_sequence_parallel():
    """Compare chunked LM Head logprobs and gradients under TP and SP."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = current_platform.current_device()
    hidden_size = 16
    local_vocab_size = 32
    local_sequence = 4
    tp_group = get_tp_group()

    for sequence_parallel in (False, True):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            params_dtype=torch.bfloat16,
            tensor_model_parallel_size=world_size,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=False,
        )
        head = ColumnParallelLinear(
            hidden_size,
            local_vocab_size * world_size,
            config=config,
            init_method=lambda tensor: tensor,
            bias=False,
            gather_output=False,
            skip_weight_param_allocation=True,
            tp_group=tp_group,
        )
        container = _HeadContainer(head)
        replace_output_layer_with_areal_lm_head(container, fp32_output=True)

        torch.manual_seed(5000 + rank)
        local_input = torch.randn(
            local_sequence,
            1,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
            requires_grad=True,
        )
        tied_weight = torch.randn(
            local_vocab_size,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
            requires_grad=True,
        )
        total_sequence = (
            local_sequence * world_size if sequence_parallel else local_sequence
        )
        torch.manual_seed(6000)
        labels = torch.randint(
            local_vocab_size * world_size,
            (total_sequence, 1),
            device=device,
        )
        loss_weights = torch.randn(total_sequence, device=device)

        output = chunked_lm_head_logprobs_entropy(
            container.output_layer,
            local_input,
            tied_weight,
            labels,
            temperature=0.7,
            chunk_size=3,
        )

        if sequence_parallel:
            gathered_inputs = [torch.empty_like(local_input) for _ in range(world_size)]
            dist.all_gather(gathered_inputs, local_input.detach(), group=tp_group)
            total_input = torch.cat(gathered_inputs, dim=0)
        else:
            total_input = local_input.detach()
        local_logits = torch.mm(
            total_input.reshape(-1, hidden_size),
            tied_weight.detach().t(),
            out_dtype=torch.float32,
        )
        gathered_logits = [torch.empty_like(local_logits) for _ in range(world_size)]
        dist.all_gather(gathered_logits, local_logits, group=tp_group)
        reference_logits = torch.cat(gathered_logits, dim=-1).requires_grad_(True)
        scaled_logits = reference_logits / 0.7
        reference_logprobs = (
            torch.log_softmax(scaled_logits, dim=-1)
            .gather(-1, labels.reshape(-1, 1))
            .squeeze(-1)
        )
        reference_entropy = -torch.sum(
            torch.softmax(scaled_logits, dim=-1)
            * torch.log_softmax(scaled_logits, dim=-1),
            dim=-1,
        )
        torch.testing.assert_close(
            output.logprobs, reference_logprobs, rtol=1e-5, atol=2e-5
        )
        torch.testing.assert_close(
            output.entropy, reference_entropy, rtol=1e-5, atol=2e-5
        )

        (reference_logprobs * loss_weights).sum().backward()
        (output.logprobs * loss_weights).sum().backward()
        local_dlogits = reference_logits.grad[
            :, rank * local_vocab_size : (rank + 1) * local_vocab_size
        ].to(torch.bfloat16)
        expected_full_dgrad = local_dlogits.matmul(tied_weight.detach()).view_as(
            total_input
        )
        if sequence_parallel:
            expected_input_grad = torch.empty_like(local_input)
            dist.reduce_scatter_tensor(
                expected_input_grad, expected_full_dgrad, group=tp_group
            )
        else:
            expected_input_grad = expected_full_dgrad
            dist.all_reduce(expected_input_grad, group=tp_group)
        expected_weight_grad = local_dlogits.t().matmul(
            total_input.reshape(-1, hidden_size)
        )

        torch.testing.assert_close(
            local_input.grad, expected_input_grad, rtol=1e-5, atol=2e-5
        )
        torch.testing.assert_close(
            tied_weight.grad, expected_weight_grad, rtol=5e-3, atol=3e-2
        )

    if rank == 0:
        print("✓ test_chunked_lm_head_tensor_and_sequence_parallel passed")


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

    test_inplace_vocab_parallel_logprobs_entropy()
    dist.barrier()

    test_areal_lm_head_tensor_and_sequence_parallel()
    dist.barrier()

    test_areal_lm_head_packed_dlogits_tensor_and_sequence_parallel()
    dist.barrier()

    test_chunked_lm_head_tensor_and_sequence_parallel()
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
