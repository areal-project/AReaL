#!/usr/bin/env python3
"""Qwen3 parallelization tests for dense and MoE models.

This script tests forward/backward passes with various parallelization strategies:
- Dense model: TP only (no EP)
- MoE model: TP, EP, or TP+EP

Supports torch.compile for both model types.

Run with:
    torchrun --nproc_per_node=2 tests/experimental/archon/torchrun/run_qwen3_parallelize.py \
        --test_type=dense_tp_forward_backward --output=/tmp/result.out

    torchrun --nproc_per_node=2 tests/experimental/archon/torchrun/run_qwen3_parallelize.py \
        --test_type=moe_tp_compile_forward_backward --output=/tmp/result.out

Supported test types:
    Dense (TP only):
    - dense_tp_forward_backward: Dense model with TP
    - dense_tp_compile_forward_backward: Dense model with TP + torch.compile

    MoE:
    - moe_tp_forward_backward: MoE model with TP
    - moe_ep_forward_backward: MoE model with EP only
    - moe_tp_ep_forward_backward: MoE model with TP + EP
    - moe_tp_compile_forward_backward: MoE model with TP + torch.compile
    - moe_tp_ep_compile_forward_backward: MoE model with TP + EP + torch.compile
"""

import argparse

import torch
import torch.distributed as dist

from tests.experimental.archon.torchrun.dist_utils import (
    create_dense_model_args,
    create_moe_model_args,
    create_test_input,
    print_rank0,
    validate_gradients,
    validate_no_nan,
    write_result,
)

from areal.experimental.models.archon import ArchonParallelDims
from areal.experimental.models.archon.qwen3 import Qwen3Model, parallelize_qwen3

# =============================================================================
# Parallel Configuration Helpers
# =============================================================================


def get_parallel_dims_tp_only(world_size: int, tp: int) -> ArchonParallelDims:
    """Get parallel dims for TP only configuration (for dense models)."""
    dp_shard = world_size // tp
    return ArchonParallelDims(
        dp_shard=dp_shard,
        tp=tp,
        cp=1,
        ep=1,
        world_size=world_size,
        device_type="cuda",
    )


def get_parallel_dims_ep_only(world_size: int, ep: int) -> ArchonParallelDims:
    """Get parallel dims for EP only configuration (for MoE models)."""
    return ArchonParallelDims(
        dp_shard=world_size,
        tp=1,
        cp=1,
        ep=ep,
        world_size=world_size,
        device_type="cuda",
    )


def get_parallel_dims_tp_ep(world_size: int, tp: int, ep: int) -> ArchonParallelDims:
    """Get parallel dims for TP + EP configuration (for MoE models)."""
    dp_shard = world_size // tp
    return ArchonParallelDims(
        dp_shard=dp_shard,
        tp=tp,
        cp=1,
        ep=ep,
        world_size=world_size,
        device_type="cuda",
    )


# =============================================================================
# Model Creation Helpers
# =============================================================================


def create_and_parallelize_model(
    model_args,
    parallel_dims: ArchonParallelDims,
    device: torch.device,
    enable_compile: bool = False,
    seed: int = 42,
) -> Qwen3Model:
    """Create model, initialize weights, and apply parallelization."""
    torch.manual_seed(seed)
    model = Qwen3Model(model_args)
    model.init_weights()
    model.init_buffers(buffer_device=device)
    model = model.to(device)

    parallelize_qwen3(
        model=model,
        parallel_dims=parallel_dims,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        loss_parallel=True,
        cpu_offload=False,
        reshard_after_forward_policy="default",
        ac_config=None,
        enable_compile=enable_compile,
    )

    return model


# =============================================================================
# Test Core Logic
# =============================================================================


def run_forward_backward_test(
    model: Qwen3Model,
    model_args,
    device: torch.device,
    test_name: str,
    multi_microbatch: bool = False,
) -> bool:
    """Run forward and backward pass, validate output and gradients.

    Args:
        model: Parallelized model to test.
        model_args: Model arguments (for vocab_size).
        device: Device to run on.
        test_name: Name of the test for logging.
        multi_microbatch: If True, simulate real training with multiple microbatches
            of different sequence lengths within a single training step. This triggers
            dynamic shape handling in torch.compile because the compiled model is
            called multiple times with different input shapes before optimizer.step().

    Returns:
        True if test passed, False otherwise.
    """
    if multi_microbatch:
        # Simulate real training: multiple microbatches with same total length
        # but different cu_seqlens (different number of sub-sequences).
        # This tests cu_seqlens dynamic shape while keeping total tokens fixed.
        microbatch_configs = [
            (4, 16),  # 64 tokens - 4 sequences of 16 tokens each
            (8, 8),  # 64 tokens - 8 sequences of 8 tokens each (different cu_seqlens)
        ]
    else:
        microbatch_configs = [(4, 8)]

    # Zero gradients once at the beginning (like optimizer_zero_grad)
    model.zero_grad()

    # Process all microbatches, accumulating gradients
    for mb_idx, (num_seqs, seq_len_per_seq) in enumerate(microbatch_configs):
        tokens, positions, cu_seqlens, max_seqlen = create_test_input(
            num_seqs=num_seqs,
            seq_len_per_seq=seq_len_per_seq,
            vocab_size=model_args.vocab_size,
            device=device,
            seed=123 + mb_idx,
        )

        mb_name = f"{test_name}[mb={mb_idx}, len={num_seqs * seq_len_per_seq}]"
        print_rank0(
            f"  {mb_name}: Input tokens={tokens.shape}, max_seqlen={max_seqlen}"
        )

        # Forward pass
        print_rank0(f"  {mb_name}: Running forward pass...")
        output = model(tokens, positions, cu_seqlens, max_seqlen)
        print_rank0(f"  {mb_name}: Output shape: {output.shape}")

        # Validate output
        if not validate_no_nan(output):
            print_rank0(f"  {mb_name}: FAILED (output contains NaN/Inf)")
            return False
        print_rank0(f"  {mb_name}: Output validation: OK (no NaN/Inf)")

        # Backward pass - accumulate gradients across microbatches
        print_rank0(f"  {mb_name}: Running backward pass...")
        loss = output.sum()
        loss.backward()
        print_rank0(f"  {mb_name}: Loss: {loss.item()}")

    # Validate gradients after all microbatches
    grad_ok, errors = validate_gradients(model)
    if not grad_ok:
        for err in errors:
            print_rank0(f"  ERROR: {err}")
        print_rank0(f"  {test_name}: FAILED (gradient validation)")
        return False
    print_rank0(f"  {test_name}: Gradient validation: OK")

    print_rank0(f"  {test_name}: PASSED")
    return True


# =============================================================================
# Dense Model Tests
# =============================================================================


def run_dense_tp_forward_backward(output: str | None = None) -> bool:
    """Run dense model with TP forward/backward test."""
    print_rank0("\n=== Dense TP Forward/Backward Test ===")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # Create model args
    model_args = create_dense_model_args()
    print_rank0(f"  Model: dense, dim={model_args.dim}, layers={model_args.n_layers}")

    # Create parallel dims (TP = world_size)
    parallel_dims = get_parallel_dims_tp_only(world_size, tp=world_size)
    print_rank0(f"  Parallelism: TP={parallel_dims.tp}, DP={parallel_dims.dp_shard}")

    # Create and parallelize model
    model = create_and_parallelize_model(
        model_args, parallel_dims, device, enable_compile=False
    )

    # Run test
    success = run_forward_backward_test(model, model_args, device, "dense_tp")

    dist.barrier()
    if rank == 0 and output:
        write_result(output, success)

    return success


def run_dense_tp_compile_forward_backward(output: str | None = None) -> bool:
    """Run dense model with TP + torch.compile forward/backward test."""
    print_rank0("\n=== Dense TP + Compile Forward/Backward Test ===")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # Create model args
    model_args = create_dense_model_args()
    print_rank0(f"  Model: dense, dim={model_args.dim}, layers={model_args.n_layers}")

    # Create parallel dims (TP = world_size)
    parallel_dims = get_parallel_dims_tp_only(world_size, tp=world_size)
    print_rank0(
        f"  Parallelism: TP={parallel_dims.tp}, DP={parallel_dims.dp_shard}, compile=True"
    )

    # Create and parallelize model with compile
    model = create_and_parallelize_model(
        model_args, parallel_dims, device, enable_compile=True
    )

    # Run test
    success = run_forward_backward_test(model, model_args, device, "dense_tp_compile")

    dist.barrier()
    if rank == 0 and output:
        write_result(output, success)

    return success


# =============================================================================
# MoE Model Tests
# =============================================================================


def run_moe_tp_forward_backward(output: str | None = None) -> bool:
    """Run MoE model with TP forward/backward test."""
    print_rank0("\n=== MoE TP Forward/Backward Test ===")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # Create model args
    num_experts = 4
    model_args = create_moe_model_args(num_experts=num_experts)
    print_rank0(
        f"  Model: MoE, dim={model_args.dim}, layers={model_args.n_layers}, experts={num_experts}"
    )

    # Create parallel dims (TP = world_size, EP = 1)
    parallel_dims = get_parallel_dims_tp_only(world_size, tp=world_size)
    print_rank0(f"  Parallelism: TP={parallel_dims.tp}, EP={parallel_dims.ep}")

    # Create and parallelize model
    model = create_and_parallelize_model(
        model_args, parallel_dims, device, enable_compile=False
    )

    # Run test
    success = run_forward_backward_test(model, model_args, device, "moe_tp")

    dist.barrier()
    if rank == 0 and output:
        write_result(output, success)

    return success


def run_moe_ep_forward_backward(output: str | None = None) -> bool:
    """Run MoE model with EP only forward/backward test."""
    print_rank0("\n=== MoE EP Forward/Backward Test ===")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # Create model args (experts must be divisible by EP)
    num_experts = world_size * 2  # 4 experts for 2 GPUs
    model_args = create_moe_model_args(num_experts=num_experts)
    print_rank0(
        f"  Model: MoE, dim={model_args.dim}, layers={model_args.n_layers}, experts={num_experts}"
    )

    # Create parallel dims (EP = world_size, TP = 1)
    parallel_dims = get_parallel_dims_ep_only(world_size, ep=world_size)
    print_rank0(f"  Parallelism: TP={parallel_dims.tp}, EP={parallel_dims.ep}")

    # Create and parallelize model
    model = create_and_parallelize_model(
        model_args, parallel_dims, device, enable_compile=False
    )

    # Run test
    success = run_forward_backward_test(model, model_args, device, "moe_ep")

    dist.barrier()
    if rank == 0 and output:
        write_result(output, success)

    return success


def run_moe_tp_ep_forward_backward(output: str | None = None) -> bool:
    """Run MoE model with TP + EP forward/backward test."""
    print_rank0("\n=== MoE TP+EP Forward/Backward Test ===")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # Create model args (experts must be divisible by EP)
    num_experts = world_size * 2  # 4 experts for 2 GPUs
    model_args = create_moe_model_args(num_experts=num_experts)
    print_rank0(
        f"  Model: MoE, dim={model_args.dim}, layers={model_args.n_layers}, experts={num_experts}"
    )

    # Create parallel dims (TP = world_size, EP = world_size)
    parallel_dims = get_parallel_dims_tp_ep(world_size, tp=world_size, ep=world_size)
    print_rank0(f"  Parallelism: TP={parallel_dims.tp}, EP={parallel_dims.ep}")

    # Create and parallelize model
    model = create_and_parallelize_model(
        model_args, parallel_dims, device, enable_compile=False
    )

    # Run test
    success = run_forward_backward_test(model, model_args, device, "moe_tp_ep")

    dist.barrier()
    if rank == 0 and output:
        write_result(output, success)

    return success


def run_moe_tp_compile_forward_backward(output: str | None = None) -> bool:
    """Run MoE model with TP + torch.compile forward/backward test."""
    print_rank0("\n=== MoE TP + Compile Forward/Backward Test ===")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # Create model args
    num_experts = 4
    model_args = create_moe_model_args(num_experts=num_experts)
    print_rank0(
        f"  Model: MoE, dim={model_args.dim}, layers={model_args.n_layers}, experts={num_experts}"
    )

    # Create parallel dims (TP = world_size)
    parallel_dims = get_parallel_dims_tp_only(world_size, tp=world_size)
    print_rank0(
        f"  Parallelism: TP={parallel_dims.tp}, EP={parallel_dims.ep}, compile=True"
    )

    # Create and parallelize model with compile
    model = create_and_parallelize_model(
        model_args, parallel_dims, device, enable_compile=True
    )

    # Run test
    success = run_forward_backward_test(model, model_args, device, "moe_tp_compile")

    dist.barrier()
    if rank == 0 and output:
        write_result(output, success)

    return success


def run_moe_tp_ep_compile_forward_backward(output: str | None = None) -> bool:
    """Run MoE model with TP + EP + torch.compile forward/backward test.

    This test covers the scenario where TP, EP, and torch.compile are all enabled,
    which is common in production MoE training (e.g., Qwen3-30B-A3B with d4t2e2).

    IMPORTANT: This test requires 4 GPUs to properly test the FSDP + TP + EP + compile
    combination. With 4 GPUs and dp_shard=2, tp=2, ep=2:
    - dp_shard_mod_ep = dp_shard * cp * tp / ep = 2 * 1 * 2 / 2 = 2 (> 1)
    - This enables proper FSDP sharding which is essential for reproducing
      production issues like DTensor sharding propagation failures.

    The test uses multi_microbatch=True to simulate real training where multiple
    microbatches with different sequence lengths are processed in a single training
    step. This triggers dynamic shape handling in torch.compile.
    """
    print_rank0("\n=== MoE TP + EP + Compile Forward/Backward Test ===")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # Require 4 GPUs for proper FSDP + TP + EP testing
    # With 4 GPUs: dp_shard=2, tp=2, ep=2 → dp_shard_mod_ep=2 (enables FSDP sharding)
    if world_size < 4:
        print_rank0(
            f"  WARNING: This test requires 4 GPUs for proper FSDP sharding, "
            f"but only {world_size} GPUs available. Running with limited config."
        )

    # Create model args (experts must be divisible by EP)
    # Use 4 experts for ep=2, so each EP rank gets 2 experts
    num_experts = 4
    tp = 2
    ep = 2

    model_args = create_moe_model_args(num_experts=num_experts)
    print_rank0(
        f"  Model: MoE, dim={model_args.dim}, layers={model_args.n_layers}, experts={num_experts}"
    )

    # Create parallel dims
    # For 4 GPUs: dp_shard=2, tp=2, ep=2
    # For 2 GPUs: dp_shard=1, tp=2, ep=2 (limited test)
    parallel_dims = get_parallel_dims_tp_ep(world_size, tp=tp, ep=ep)
    print_rank0(
        f"  Parallelism: TP={parallel_dims.tp}, EP={parallel_dims.ep}, "
        f"DP={parallel_dims.dp_shard}, compile=True"
    )

    # Create and parallelize model with compile
    model = create_and_parallelize_model(
        model_args, parallel_dims, device, enable_compile=True
    )

    # Run test with multi_microbatch=True to trigger dynamic shape handling
    # This simulates real training where different microbatches have different lengths
    success = run_forward_backward_test(
        model, model_args, device, "moe_tp_ep_compile", multi_microbatch=True
    )

    dist.barrier()
    if rank == 0 and output:
        write_result(output, success)

    return success


# =============================================================================
# Main
# =============================================================================

TEST_REGISTRY = {
    # Dense tests
    "dense_tp_forward_backward": run_dense_tp_forward_backward,
    "dense_tp_compile_forward_backward": run_dense_tp_compile_forward_backward,
    # MoE tests
    "moe_tp_forward_backward": run_moe_tp_forward_backward,
    "moe_ep_forward_backward": run_moe_ep_forward_backward,
    "moe_tp_ep_forward_backward": run_moe_tp_ep_forward_backward,
    "moe_tp_compile_forward_backward": run_moe_tp_compile_forward_backward,
    "moe_tp_ep_compile_forward_backward": run_moe_tp_ep_compile_forward_backward,
}


def main():
    parser = argparse.ArgumentParser(description="Qwen3 Parallelize Tests")
    parser.add_argument(
        "--test_type",
        type=str,
        required=True,
        choices=list(TEST_REGISTRY.keys()),
        help="Type of test to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for test result (Passed/Failed)",
    )
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()

    torch.cuda.set_device(rank)

    print_rank0("=" * 60)
    print_rank0(f"Running Qwen3 Parallelize Test: {args.test_type}")
    print_rank0("=" * 60)

    try:
        test_fn = TEST_REGISTRY[args.test_type]
        success = test_fn(args.output)

        dist.barrier()

        if success:
            print_rank0(f"\n{'=' * 60}")
            print_rank0(f"Qwen3 Parallelize Test {args.test_type}: PASSED")
            print_rank0("=" * 60)
        else:
            print_rank0(f"\n{'=' * 60}")
            print_rank0(f"Qwen3 Parallelize Test {args.test_type}: FAILED")
            print_rank0("=" * 60)
            if rank == 0 and args.output:
                write_result(args.output, False)

    except Exception as e:
        print(f"Rank {rank} failed with: {e}")
        import traceback

        traceback.print_exc()
        if rank == 0 and args.output:
            write_result(args.output, False)
        raise

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
