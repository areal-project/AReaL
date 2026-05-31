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

from areal.experimental.weight_update.nccl_group import init_weights_update_group
from areal.infra.platforms import current_platform
from areal.utils.kernel.fp8_kernel import scaled_fp8_blockwise


def run_fp8_weight_transfer(output=None):
    """Test: FP8 block-wise quantized weight transfer from training to inference via NCCL.

    Rank 0 (training side) quantizes a BF16 weight to FP8 using block-wise
    quantization, then broadcasts both the FP8 weight and the per-block scale
    tensor to rank 1 (inference side) over a custom NCCL process group.
    A non-quantized 1D norm weight is also broadcast and verified.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print_rank0("=== FP8 Weight Transfer Test ===")

    # Use a different port from the main group to avoid conflicts
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    from areal.utils.network import find_free_ports

    if rank == 0:
        ports = find_free_ports(1)
        port_tensor = torch.tensor(ports, dtype=torch.long, device=f"cuda:{rank}")
    else:
        port_tensor = torch.zeros(1, dtype=torch.long, device=f"cuda:{rank}")
    dist.broadcast(port_tensor, src=0)
    master_port = int(port_tensor[0].item())

    # For this test: rank 0 = training, rank 1 = inference
    is_inference = rank == 1

    try:
        group = init_weights_update_group(
            master_address=master_addr,
            master_port=master_port,
            rank=rank,
            world_size=world_size,
            group_name="awex_test_fp8_transfer",
            backend="nccl",
            role="inference" if is_inference else "training",
        )
        print_rank0(f"  Group created successfully with {world_size} ranks")

        device = torch.device(f"cuda:{current_platform.current_device()}")

        # -------------------------------------------------------------------
        # Prepare tensors
        # -------------------------------------------------------------------
        weight_shape = (256, 512)
        block_size = [128, 128]
        scale_shape = (2, 4)  # ceil(256/128) x ceil(512/128)
        norm_shape = (256,)

        if not is_inference:
            # Training side: create deterministic BF16 weights and quantize
            torch.manual_seed(42)

            # 2D weight tensor -> quantize to FP8
            q_proj_weight_bf16 = torch.randn(
                weight_shape, dtype=torch.bfloat16, device=device
            )
            fp8_weight, scale_inv = scaled_fp8_blockwise(
                q_proj_weight_bf16, weight_block_size=block_size
            )

            # 1D norm weight -> NOT quantized, stays BF16
            norm_weight_bf16 = torch.randn(
                norm_shape, dtype=torch.bfloat16, device=device
            )

            tensors_to_send = {
                "layers.0.q_proj.weight": fp8_weight,
                "layers.0.q_proj.weight_scale_inv": scale_inv,
                "layers.0.norm.weight": norm_weight_bf16,
            }
        else:
            # Inference side: create receive buffers with matching shapes/dtypes
            tensors_to_send = {
                "layers.0.q_proj.weight": torch.zeros(
                    weight_shape, dtype=torch.float8_e4m3fn, device=device
                ),
                "layers.0.q_proj.weight_scale_inv": torch.zeros(
                    scale_shape, dtype=torch.float32, device=device
                ),
                "layers.0.norm.weight": torch.zeros(
                    norm_shape, dtype=torch.bfloat16, device=device
                ),
            }

        # -------------------------------------------------------------------
        # Broadcast from rank 0 (training) to all other ranks (inference)
        # -------------------------------------------------------------------
        for name in sorted(tensors_to_send.keys()):
            tensor = tensors_to_send[name]
            dist.broadcast(tensor, src=0, group=group)

        current_platform.synchronize()
        dist.barrier(group=group)

        # -------------------------------------------------------------------
        # Verify: inference side checks received data matches expected
        # -------------------------------------------------------------------
        success = True
        if is_inference:
            # Re-create expected values on rank 1 to compare
            torch.manual_seed(42)
            expected_q_proj = torch.randn(
                weight_shape, dtype=torch.bfloat16, device=device
            )
            expected_fp8, expected_scale = scaled_fp8_blockwise(
                expected_q_proj, weight_block_size=block_size
            )
            expected_norm = torch.randn(norm_shape, dtype=torch.bfloat16, device=device)

            # Verify FP8 weight
            received_fp8 = tensors_to_send["layers.0.q_proj.weight"]
            if not torch.equal(received_fp8, expected_fp8):
                print_rank0(
                    "  MISMATCH layers.0.q_proj.weight: FP8 weight does not match"
                )
                success = False
            else:
                print_rank0(
                    f"  OK layers.0.q_proj.weight: shape={list(received_fp8.shape)}, dtype={received_fp8.dtype}"
                )

            # Verify scale
            received_scale = tensors_to_send["layers.0.q_proj.weight_scale_inv"]
            if not torch.equal(received_scale, expected_scale):
                print_rank0(
                    "  MISMATCH layers.0.q_proj.weight_scale_inv: scale does not match"
                )
                success = False
            else:
                print_rank0(
                    f"  OK layers.0.q_proj.weight_scale_inv: shape={list(received_scale.shape)}, dtype={received_scale.dtype}"
                )

            # Verify norm weight (1D, BF16, not quantized)
            received_norm = tensors_to_send["layers.0.norm.weight"]
            if not torch.equal(received_norm, expected_norm):
                print_rank0(
                    "  MISMATCH layers.0.norm.weight: norm weight does not match"
                )
                success = False
            else:
                print_rank0(
                    f"  OK layers.0.norm.weight: shape={list(received_norm.shape)}, dtype={received_norm.dtype}"
                )

            print_rank0(
                f"  Rank {rank} verification: {'PASSED' if success else 'FAILED'}"
            )

        # All-reduce success flag so all ranks agree
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
    "fp8_weight_transfer": run_fp8_weight_transfer,
}


def main():
    parser = argparse.ArgumentParser(description="FP8 NCCL Weight Transfer Tests")
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
