"""Test suite for ulysses all_to_all_tensor using all_to_all_single_autograd.

This test verifies:
1. Correctness: new implementation matches reference dist.all_to_all
2. Backward: autograd backward pass works correctly
3. Compile: torch.compile compatibility (no graph breaks)
"""

import argparse
import os

import torch
import torch.distributed as dist

from areal.infra.platforms import current_platform
from areal.models.fsdp.ulysses import (
    all_to_all_tensor,
    set_ulysses_sequence_parallel_group,
)
from areal.utils.constants import DIST_GROUP_DEFAULT_TIMEOUT


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


def reference_all_to_all(x, scatter_dim, gather_dim, group):
    """Reference implementation using dist.all_to_all for comparison."""
    world_size = dist.get_world_size(group)
    input_list = list(torch.chunk(x, world_size, dim=scatter_dim))
    input_list = [t.contiguous() for t in input_list]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


def test_correctness():
    """Test that new implementation matches reference."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"{current_platform.device_type}:{rank}")

    sp_group = dist.new_group(
        ranks=list(range(world_size)),
        timeout=DIST_GROUP_DEFAULT_TIMEOUT,
        backend="nccl",
    )
    set_ulysses_sequence_parallel_group(sp_group)

    # Test case: [bsz, seq/n, h, d] -> [bsz, seq, h/n, d]
    bsz, seq_per_rank, num_heads, head_dim = 2, 64, 8, 32

    # Create input tensor with same seed across ranks for reproducibility
    torch.manual_seed(42)
    x_base = torch.randn(bsz, seq_per_rank, num_heads, head_dim, device=device)

    # Broadcast from rank 0 to ensure all ranks have same input
    dist.broadcast(x_base, src=0)

    # Test gather_seq_scatter_heads: scatter on heads (dim=2), gather on seq (dim=1)
    scatter_dim, gather_dim = 2, 1

    # Reference
    x_ref = x_base.clone()
    y_ref = reference_all_to_all(x_ref, scatter_dim, gather_dim, sp_group)

    # New implementation
    x_new = x_base.clone()
    y_new = all_to_all_tensor(x_new, scatter_dim, gather_dim, sp_group)

    # Compare
    if not torch.allclose(y_ref, y_new, atol=1e-6):
        max_diff = (y_ref - y_new).abs().max().item()
        raise AssertionError(
            f"Rank {rank}: Correctness test FAILED! Max diff: {max_diff}"
        )

    print(f"Rank {rank}: Forward correctness test PASSED")

    # Test reverse direction: gather_heads_scatter_seq
    scatter_dim, gather_dim = 1, 2

    torch.manual_seed(43)
    x_base2 = torch.randn(
        bsz, seq_per_rank * world_size, num_heads // world_size, head_dim, device=device
    )
    dist.broadcast(x_base2, src=0)

    y_ref2 = reference_all_to_all(x_base2.clone(), scatter_dim, gather_dim, sp_group)
    y_new2 = all_to_all_tensor(x_base2.clone(), scatter_dim, gather_dim, sp_group)

    if not torch.allclose(y_ref2, y_new2, atol=1e-6):
        max_diff = (y_ref2 - y_new2).abs().max().item()
        raise AssertionError(
            f"Rank {rank}: Reverse correctness test FAILED! Max diff: {max_diff}"
        )

    print(f"Rank {rank}: Reverse correctness test PASSED")
    dist.barrier()


def test_backward():
    """Test autograd backward pass."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"{current_platform.device_type}:{rank}")

    sp_group = dist.new_group(
        ranks=list(range(world_size)),
        timeout=DIST_GROUP_DEFAULT_TIMEOUT,
        backend="nccl",
    )
    set_ulysses_sequence_parallel_group(sp_group)

    bsz, seq_per_rank, num_heads, head_dim = 2, 64, 8, 32

    torch.manual_seed(42 + rank)
    x = torch.randn(
        bsz, seq_per_rank, num_heads, head_dim, device=device, requires_grad=True
    )

    scatter_dim, gather_dim = 2, 1

    # Forward
    y = all_to_all_tensor(x, scatter_dim, gather_dim, sp_group)

    # Backward
    grad_output = torch.ones_like(y)
    y.backward(grad_output)

    # Check gradient exists and is finite
    if x.grad is None:
        raise AssertionError(f"Rank {rank}: Backward test FAILED - no gradient!")

    if not torch.isfinite(x.grad).all():
        raise AssertionError(
            f"Rank {rank}: Backward test FAILED - gradient has NaN/Inf!"
        )

    # Verify gradient shape matches input shape
    if x.grad.shape != x.shape:
        raise AssertionError(
            f"Rank {rank}: Backward test FAILED - grad shape {x.grad.shape} != input shape {x.shape}!"
        )

    print(f"Rank {rank}: Backward test PASSED, grad shape: {x.grad.shape}")
    dist.barrier()


def test_compile():
    """Test torch.compile compatibility."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"{current_platform.device_type}:{rank}")

    sp_group = dist.new_group(
        ranks=list(range(world_size)),
        timeout=DIST_GROUP_DEFAULT_TIMEOUT,
        backend="nccl",
    )
    set_ulysses_sequence_parallel_group(sp_group)

    bsz, seq_per_rank, num_heads, head_dim = 2, 64, 8, 32
    scatter_dim, gather_dim = 2, 1

    # Define function to compile
    def forward_fn(x):
        return all_to_all_tensor(x, scatter_dim, gather_dim, sp_group)

    # Compile with inductor backend
    compiled_fn = torch.compile(forward_fn, backend="inductor")

    # Run multiple iterations to check for recompilation
    for i in range(3):
        torch.manual_seed(42 + rank + i)
        x = torch.randn(bsz, seq_per_rank, num_heads, head_dim, device=device)

        # Run compiled function
        y_compiled = compiled_fn(x)

        # Verify output matches eager mode
        y_eager = forward_fn(x.clone())

        if not torch.allclose(y_compiled, y_eager, atol=1e-5):
            max_diff = (y_compiled - y_eager).abs().max().item()
            raise AssertionError(
                f"Rank {rank}: Compile test FAILED - output mismatch! Max diff: {max_diff}"
            )

    print(f"Rank {rank}: Compile test PASSED")
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_name",
        type=str,
        required=True,
        choices=["correctness", "backward", "compile", "all"],
    )
    args = parser.parse_args()

    setup_distributed_environment()

    if args.test_name == "all":
        test_correctness()
        test_backward()
        test_compile()
    else:
        test_fn = {
            "correctness": test_correctness,
            "backward": test_backward,
            "compile": test_compile,
        }[args.test_name]
        test_fn()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
