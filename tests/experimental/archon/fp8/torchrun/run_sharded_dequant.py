#!/usr/bin/env python3
"""Sharded FP8 dequantization correctness tests.

Validates that ``dequant_fp8_state_dict`` produces identical results whether
the FP8 weights are plain tensors (single-device) or DTensors (FSDP-sharded).

This is the core correctness guarantee for the local-shard dequant approach:
each rank dequantizes only its own FSDP shard, and the concatenation of all
shards matches a single-device full dequantization.

Run::

    torchrun --nproc_per_node=2 tests/experimental/archon/fp8/torchrun/run_sharded_dequant.py --output result.out
"""

from __future__ import annotations

import argparse
import traceback

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard, distribute_tensor, init_device_mesh

from tests.experimental.archon.torchrun.dist_utils import print_rank0

from areal.experimental.models.archon.fp8_checkpoint import (
    dequant_fp8_state_dict,
    weight_dequant_cpu,
)


# ---------------------------------------------------------------------------
# Test 1: Basic sharded dequant matches full dequant
# ---------------------------------------------------------------------------
def test_sharded_dequant_matches_full():
    """2-GPU sharded dequant produces the same result as single-device full dequant.

    Setup:
    - M=512, N=256, block_size=128 → 4×2 scale blocks
    - Shard(0) over 2 GPUs → each rank gets (256, 256)
    - start_row = 0 / 256, both 128-aligned ✓

    Verifies:
    - Output is DTensor with Shard(0) placement
    - Output dtype is bfloat16
    - scale_inv key is removed
    - full_tensor() matches single-device weight_dequant_cpu result
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mesh = init_device_mesh("cuda", (world_size,))

    M, N = 512, 256
    torch.manual_seed(42)
    weight_bf16 = torch.randn(M, N, dtype=torch.bfloat16)
    weight_fp8 = weight_bf16.to(torch.float8_e4m3fn)
    scale_inv = torch.rand(M // 128, N // 128, dtype=torch.float32) + 0.5

    # Ground truth: single-device full dequant on CPU
    expected_full = weight_dequant_cpu(weight_fp8, scale_inv)

    # Simulate DCP output: FP8 weight as DTensor Shard(0), scale as plain tensor
    weight_fp8_dt = distribute_tensor(weight_fp8.cuda(), mesh, [Shard(0)])

    sd = {
        "model.layers.0.q_proj.weight": weight_fp8_dt,
        "model.layers.0.q_proj.weight_scale_inv": scale_inv.clone(),
    }

    result_sd = dequant_fp8_state_dict(sd, target_dtype=torch.bfloat16)

    w = result_sd["model.layers.0.q_proj.weight"]

    # Structure assertions
    assert isinstance(w, DTensor), f"Expected DTensor, got {type(w)}"
    assert w.placements == (Shard(0),), f"Expected (Shard(0),), got {w.placements}"
    assert w.shape == torch.Size([M, N]), f"Global shape mismatch: {w.shape}"
    assert w.dtype == torch.bfloat16, f"Expected bfloat16, got {w.dtype}"

    # scale key must be removed
    assert "model.layers.0.q_proj.weight_scale_inv" not in result_sd

    # Numerical check: gather back to full tensor and compare
    result_full = w.full_tensor().cpu()
    if rank == 0:
        torch.testing.assert_close(result_full, expected_full, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Test 2: Mixed FP8 + BF16 DTensors in the same state dict
# ---------------------------------------------------------------------------
def test_mixed_fp8_and_bf16_dtensors():
    """Non-FP8 DTensors (norms, embeddings) pass through untouched.

    Verifies that dequant only touches FP8 keys and leaves BF16 DTensors
    as-is — critical for the from_hf() DTensor-aware path to work.
    """
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    torch.manual_seed(42)
    # FP8 weight + scale
    fp8_w = torch.randn(256, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    fp8_dt = distribute_tensor(fp8_w.cuda(), mesh, [Shard(0)])
    scale = torch.ones(2, 1, dtype=torch.float32)

    # BF16 norm weight (also a DTensor from DCP, as in production)
    norm_w = torch.randn(128, dtype=torch.bfloat16)
    # 1-D tensor, Shard(0) → each rank gets half
    norm_dt = distribute_tensor(norm_w.cuda(), mesh, [Shard(0)])

    sd = {
        "model.layers.0.q_proj.weight": fp8_dt,
        "model.layers.0.q_proj.weight_scale_inv": scale,
        "model.layers.0.norm.weight": norm_dt,
    }

    result = dequant_fp8_state_dict(sd, target_dtype=torch.bfloat16)

    # FP8 key dequantized to BF16 DTensor
    q_proj = result["model.layers.0.q_proj.weight"]
    assert isinstance(q_proj, DTensor), f"Expected DTensor, got {type(q_proj)}"
    assert q_proj.dtype == torch.bfloat16

    # BF16 norm untouched: still DTensor, same object identity
    norm = result["model.layers.0.norm.weight"]
    assert isinstance(norm, DTensor), f"Norm should remain DTensor, got {type(norm)}"
    assert norm.dtype == torch.bfloat16
    # Verify it's the exact same object (not copied/converted)
    assert norm is norm_dt, "Non-FP8 DTensor should not be touched"


# ---------------------------------------------------------------------------
# Test 3: Non-128-aligned shard boundary (block boundary crosses shard)
# ---------------------------------------------------------------------------
def test_non_aligned_shard_boundary():
    """Shard boundary not aligned to block_size=128.

    M=384, R=2 → local_M=192, start_rows=[0, 192].
    Rank 1's start_row=192 is NOT a multiple of 128.
    Scale block layout: ceil(384/128) = 3 block-rows.
    Rank 0: global rows [0,192) → blocks [0,1) fully + block [1] partially (rows 128-191)
    Rank 1: global rows [192,384) → block [1] partially (rows 192-255) + block [2] fully

    This test verifies correctness when a single 128-block is split across
    two FSDP shards.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mesh = init_device_mesh("cuda", (world_size,))

    M, N = 384, 256
    torch.manual_seed(42)
    weight_bf16 = torch.randn(M, N, dtype=torch.bfloat16)
    weight_fp8 = weight_bf16.to(torch.float8_e4m3fn)
    scale_inv = (
        torch.rand(3, 2, dtype=torch.float32) + 0.5
    )  # ceil(384/128)=3, ceil(256/128)=2

    # Ground truth
    expected_full = weight_dequant_cpu(weight_fp8, scale_inv)

    fp8_dt = distribute_tensor(weight_fp8.cuda(), mesh, [Shard(0)])
    sd = {
        "w": fp8_dt,
        "w_scale_inv": scale_inv.clone(),
    }

    result_sd = dequant_fp8_state_dict(sd, target_dtype=torch.bfloat16)

    w = result_sd["w"]
    assert isinstance(w, DTensor)
    assert w.shape == torch.Size([M, N])

    result_full = w.full_tensor().cpu()
    if rank == 0:
        torch.testing.assert_close(result_full, expected_full, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Test 4: Multiple scale patterns (different scales per block)
# ---------------------------------------------------------------------------
def test_multi_block_scale_correctness():
    """Each 128×128 block has a distinct scale — verify per-block correctness.

    Uses ones as FP8 data so expected output = scale_inv value per block.
    This catches any scale indexing / slicing errors directly.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mesh = init_device_mesh("cuda", (world_size,))

    M, N = 256, 256  # 2×2 blocks
    weight_fp8 = torch.ones(M, N, dtype=torch.float8_e4m3fn)
    # Distinct scales: [[1, 2], [3, 4]]
    scale_inv = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    expected_full = weight_dequant_cpu(weight_fp8, scale_inv)

    fp8_dt = distribute_tensor(weight_fp8.cuda(), mesh, [Shard(0)])
    sd = {
        "w": fp8_dt,
        "w_scale_inv": scale_inv.clone(),
    }

    result_sd = dequant_fp8_state_dict(sd, target_dtype=torch.bfloat16)
    result_full = result_sd["w"].full_tensor().cpu()

    if rank == 0:
        # Check per-block mean values match expected scales
        # block [0,0]: 1.0*1.0 = 1.0, block [0,1]: 1.0*2.0 = 2.0
        # block [1,0]: 1.0*3.0 = 3.0, block [1,1]: 1.0*4.0 = 4.0
        torch.testing.assert_close(result_full, expected_full, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ALL_TESTS = [
    test_sharded_dequant_matches_full,
    test_mixed_fp8_and_bf16_dtensors,
    test_non_aligned_shard_boundary,
    test_multi_block_scale_correctness,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    print_rank0(f"\n=== Sharded FP8 dequant tests ({dist.get_world_size()} GPUs) ===")

    passed, failed = 0, 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            print_rank0(f"  {test_fn.__name__}: PASSED")
            passed += 1
        except Exception as e:
            failed += 1
            print_rank0(f"  {test_fn.__name__}: FAILED — {e}")
            traceback.print_exc()

    dist.barrier()
    success = failed == 0
    print_rank0(f"\nResults: {passed} passed, {failed} failed")

    if rank == 0 and args.output:
        with open(args.output, "w") as f:
            f.write("Passed" if success else "Failed")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
