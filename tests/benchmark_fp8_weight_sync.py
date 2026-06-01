#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Benchmark FP8 vs BF16 weight sync performance.

Usage:
    # Single GPU: quantization only
    python tests/benchmark_fp8_weight_sync.py

    # Multi GPU (2 GPUs): quantization + NCCL broadcast
    torchrun --nproc_per_node=2 tests/benchmark_fp8_weight_sync.py
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from areal.infra.platforms import current_platform
from areal.utils.kernel.fp8_kernel import scaled_fp8_blockwise


def benchmark_quantization(
    shape: tuple[int, int],
    block_size: list[int],
    num_warmup: int = 10,
    num_iters: int = 100,
) -> float:
    """Benchmark FP8 quantization latency (ms per call)."""
    device = torch.device(f"cuda:{current_platform.current_device()}")
    data = torch.randn(shape, dtype=torch.bfloat16, device=device)

    for _ in range(num_warmup):
        scaled_fp8_blockwise(data, weight_block_size=block_size)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        scaled_fp8_blockwise(data, weight_block_size=block_size)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_iters


def benchmark_broadcast(
    tensors: list[torch.Tensor],
    num_warmup: int = 5,
    num_iters: int = 20,
) -> float:
    """Benchmark NCCL broadcast latency (ms per call)."""
    for _ in range(num_warmup):
        for tensor in tensors:
            dist.broadcast(tensor, src=0)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        for tensor in tensors:
            dist.broadcast(tensor, src=0)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_iters


def main() -> None:
    parser = argparse.ArgumentParser(description="FP8 weight sync benchmark")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=[4096, 4096],
        help="Weight matrix shape (M N)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        nargs=2,
        default=[128, 128],
        help="FP8 block size (BLOCK_M BLOCK_N)",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Warmup iterations for quantization",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=100,
        help="Benchmark iterations for quantization",
    )
    args = parser.parse_args()

    shape = tuple(args.shape)
    block_size = list(args.block_size)
    scale_shape = (
        (shape[0] + block_size[0] - 1) // block_size[0],
        (shape[1] + block_size[1] - 1) // block_size[1],
    )

    # Always benchmark quantization (single-GPU friendly)
    ms_quant = benchmark_quantization(shape, block_size, args.num_warmup, args.num_iters)

    if not dist.is_initialized():
        print(f"Single-GPU benchmark (shape={shape}, block={block_size})")
        print(f"  Quantization: {ms_quant:.3f} ms")
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{current_platform.current_device()}")

    # BF16 broadcast benchmark: one tensor
    bf16_tensor = torch.randn(shape, dtype=torch.bfloat16, device=device)
    ms_bf16 = benchmark_broadcast(
        [bf16_tensor],
        num_warmup=args.num_warmup,
        num_iters=max(args.num_iters // 5, 10),
    )
    bytes_bf16 = bf16_tensor.numel() * 2

    # FP8 broadcast benchmark: rank 0 quantizes, all ranks participate in broadcast
    torch.manual_seed(42)
    data = torch.randn(shape, dtype=torch.bfloat16, device=device)
    if rank == 0:
        fp8_weight, scale = scaled_fp8_blockwise(data, weight_block_size=block_size)
    else:
        fp8_weight = torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
        scale = torch.empty(scale_shape, dtype=torch.float32, device=device)

    ms_fp8_broadcast = benchmark_broadcast(
        [fp8_weight, scale],
        num_warmup=args.num_warmup,
        num_iters=max(args.num_iters // 5, 10),
    )
    bytes_fp8 = fp8_weight.numel() * 1 + scale.numel() * 4

    total_ms_fp8 = ms_quant + ms_fp8_broadcast

    if rank == 0:
        print(
            f"Multi-GPU benchmark (shape={shape}, block={block_size}, world_size={world_size})"
        )
        print(f"  BF16 broadcast:       {ms_bf16:.3f} ms")
        print(f"    bandwidth:          {bytes_bf16 / ms_bf16 / 1e6:.2f} GB/s")
        print(f"    data volume:        {bytes_bf16 / 1e6:.2f} MB")
        print(f"  FP8 quantization:     {ms_quant:.3f} ms")
        print(f"  FP8 broadcast:        {ms_fp8_broadcast:.3f} ms")
        print(f"    bandwidth:          {bytes_fp8 / ms_fp8_broadcast / 1e6:.2f} GB/s")
        print(f"    data volume:        {bytes_fp8 / 1e6:.2f} MB")
        print(f"  FP8 total (q+b):      {total_ms_fp8:.3f} ms")
        print(f"  Speedup vs BF16:      {ms_bf16 / total_ms_fp8:.2f}x")
        print(f"  Comm. reduction:      {bytes_bf16 / bytes_fp8:.2f}x")


if __name__ == "__main__":
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    main()
    if dist.is_initialized():
        dist.destroy_process_group()
