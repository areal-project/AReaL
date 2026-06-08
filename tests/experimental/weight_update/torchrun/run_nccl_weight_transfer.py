#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import torch
import torch.distributed as dist

from tests.experimental.weight_update.torchrun.dist_utils import (
    print_rank0,
    write_result,
)

from areal.experimental.weight_update.nccl_group import (
    init_weights_update_group,
    setup_batch_isend_irecv,
)
from areal.infra.platforms import current_platform


def run_nccl_group_init(output=None):
    """Test: All ranks create and destroy a custom NCCL process group."""
    print(1, flush=True)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(2, flush=True)

    print_rank0("=== NCCL Group Init Test ===")

    # Use the gateway's master_addr/port pattern
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    # Use a different port from the main group to avoid conflicts
    from areal.utils.network import find_free_ports

    if rank == 0:
        ports = find_free_ports(1)
        port_tensor = torch.tensor(ports, dtype=torch.long, device=f"cuda:{rank}")
    else:
        port_tensor = torch.zeros(1, dtype=torch.long, device=f"cuda:{rank}")
    dist.broadcast(port_tensor, src=0)
    master_port = int(port_tensor[0].item())

    try:
        group = init_weights_update_group(
            master_address=master_addr,
            master_port=master_port,
            rank=rank,
            world_size=world_size,
            group_name="awex_test_nccl_init",
            backend="nccl",
            role="test",
        )
        print_rank0(f"  Group created successfully with {world_size} ranks")

        # Verify barrier works
        dist.barrier(group=group)
        print_rank0("  Barrier completed")

        # Cleanup
        dist.destroy_process_group(group)
        print_rank0("  Group destroyed")

        success = True
    except Exception as e:
        print_rank0(f"  FAILED: {e}")
        success = False

    dist.barrier()
    if rank == 0 and output:
        write_result(output, success)
    return success


def run_batch_isend_irecv(output=None):
    """Test: P2P communication via batch_isend_irecv on custom group."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print_rank0("=== Batch Isend/Irecv Test ===")

    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    from areal.utils.network import find_free_ports

    if rank == 0:
        ports = find_free_ports(1)
        port_tensor = torch.tensor(ports, dtype=torch.long, device=f"cuda:{rank}")
    else:
        port_tensor = torch.zeros(1, dtype=torch.long, device=f"cuda:{rank}")
    dist.broadcast(port_tensor, src=0)
    master_port = int(port_tensor[0].item())

    try:
        group = init_weights_update_group(
            master_address=master_addr,
            master_port=master_port,
            rank=rank,
            world_size=world_size,
            group_name="awex_test_p2p",
            backend="nccl",
            role="test",
        )

        # This function sends/receives test tensors and verifies correctness
        setup_batch_isend_irecv(group, rank, world_size)
        print_rank0("  P2P communication completed and verified")

        dist.destroy_process_group(group)
        success = True
    except Exception as e:
        print_rank0(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        success = False

    dist.barrier()
    if rank == 0 and output:
        write_result(output, success)
    return success


def run_weight_transfer_lifecycle(output=None):
    """Test: Full weight transfer lifecycle - training ranks send, inference ranks receive."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print_rank0("=== Weight Transfer Lifecycle Test ===")

    infer_world_size = world_size // 2
    is_inference = rank < infer_world_size

    print_rank0(
        f"  Inference ranks: 0..{infer_world_size - 1}, Training ranks: {infer_world_size}..{world_size - 1}"
    )

    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    from areal.utils.network import find_free_ports

    if rank == 0:
        ports = find_free_ports(1)
        port_tensor = torch.tensor(ports, dtype=torch.long, device=f"cuda:{rank}")
    else:
        port_tensor = torch.zeros(1, dtype=torch.long, device=f"cuda:{rank}")
    dist.broadcast(port_tensor, src=0)
    master_port = int(port_tensor[0].item())

    try:
        group = init_weights_update_group(
            master_address=master_addr,
            master_port=master_port,
            rank=rank,
            world_size=world_size,
            group_name="awex_test_transfer",
            backend="nccl",
            role="inference" if is_inference else "training",
        )

        # Create test tensors simulating model weights
        param_shapes = [(512, 256), (256,), (1024, 512)]
        device = torch.device(f"cuda:{current_platform.current_device()}")

        if not is_inference:
            # Training side: create "model weights" with deterministic values
            torch.manual_seed(42 + rank)
            params = {
                f"param_{i}": torch.randn(shape, device=device)
                for i, shape in enumerate(param_shapes)
            }
        else:
            # Inference side: create empty receive buffers
            params = {
                f"param_{i}": torch.zeros(shape, device=device)
                for i, shape in enumerate(param_shapes)
            }

        # Pair each inference rank with a training rank for P2P transfer
        if is_inference:
            partner_rank = rank + infer_world_size
            if partner_rank < world_size:
                ops = []
                for name in sorted(params.keys()):
                    ops.append(
                        dist.P2POp(dist.irecv, params[name], partner_rank, group=group)
                    )
                if ops:
                    reqs = dist.batch_isend_irecv(ops)
                    for req in reqs:
                        req.wait()
        else:
            partner_rank = rank - infer_world_size
            if partner_rank >= 0 and partner_rank < infer_world_size:
                ops = []
                for name in sorted(params.keys()):
                    ops.append(
                        dist.P2POp(dist.isend, params[name], partner_rank, group=group)
                    )
                if ops:
                    reqs = dist.batch_isend_irecv(ops)
                    for req in reqs:
                        req.wait()

        current_platform.synchronize()
        dist.barrier(group=group)

        # Verify: inference ranks should have received training weights
        success = True
        if is_inference:
            partner_rank = rank + infer_world_size
            if partner_rank < world_size:
                torch.manual_seed(42 + partner_rank)
                for i, shape in enumerate(param_shapes):
                    expected = torch.randn(shape, device=device)
                    actual = params[f"param_{i}"]
                    if not torch.allclose(actual, expected, rtol=1e-6, atol=1e-6):
                        max_diff = (actual - expected).abs().max().item()
                        print_rank0(f"  MISMATCH param_{i}: max_diff={max_diff}")
                        success = False
            print_rank0(
                f"  Rank {rank} verification: {'PASSED' if success else 'FAILED'}"
            )

        # All-reduce success flag
        success_tensor = torch.tensor(
            [1 if success else 0], dtype=torch.int, device=device
        )
        dist.all_reduce(success_tensor, op=dist.ReduceOp.MIN, group=group)
        success = bool(success_tensor.item())

        dist.destroy_process_group(group)
        print_rank0(f"  Overall: {'PASSED' if success else 'FAILED'}")

    except Exception as e:
        print_rank0(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        success = False

    dist.barrier()
    if rank == 0 and output:
        write_result(output, success)
    return success


TEST_REGISTRY = {
    "nccl_group_init": run_nccl_group_init,
    "batch_isend_irecv": run_batch_isend_irecv,
    "weight_transfer_lifecycle": run_weight_transfer_lifecycle,
}


def main():
    parser = argparse.ArgumentParser(description="NCCL Weight Transfer Tests")
    parser.add_argument(
        "--test_type",
        type=str,
        required=True,
        choices=list(TEST_REGISTRY.keys()),
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    print_rank0("=" * 60)
    print_rank0(f"Running: {args.test_type}")
    print_rank0("=" * 60)

    try:
        test_fn = TEST_REGISTRY[args.test_type]
        success = test_fn(args.output)

        dist.barrier()
        if success:
            print_rank0(f"\n{args.test_type}: PASSED")
        else:
            print_rank0(f"\n{args.test_type}: FAILED")
            if rank == 0 and args.output:
                write_result(args.output, False)
    except Exception as e:
        print(f"Rank {rank} failed: {e}")
        import traceback

        traceback.print_exc()
        if rank == 0 and args.output:
            write_result(args.output, False)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
