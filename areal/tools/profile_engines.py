# SPDX-License-Identifier: Apache-2.0

"""Compare profiling results between Archon and FSDP/HuggingFace engines.

This tool runs profiling on both engines with the same configuration
and generates a side-by-side comparison report.

Usage:
    python -m areal.tools.profile_engines [OPTIONS]

Examples:
    # Default comparison
    python -m areal.tools.profile_engines

    # Forward only comparison
    python -m areal.tools.profile_engines --mode forward

    # With specific sequence lengths
    python -m areal.tools.profile_engines --seq-lens 128,256,512,128
"""

from __future__ import annotations

import argparse
import functools
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity

from areal.infra import current_platform


@dataclass
class EngineResult:
    """Container for engine profiling results."""

    name: str
    total_cuda_time_ms: float
    peak_memory_mb: float
    top_ops: list[tuple[str, float]]  # (op_name, cuda_time_ms)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Archon vs FSDP/HuggingFace engine performance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Forward + backward (default)
  %(prog)s --mode forward               # Forward only comparison
  %(prog)s --seq-lens 128,256,512,128   # Specific sequence lengths
  %(prog)s --batch-size 1 --total-len 16384  # Long sequence test
""",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model. Defaults to Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["forward", "backward", "both"],
        default="both",
        help="Profiling mode: forward, backward, or both (default: both)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of sequences in batch (default: 4)",
    )
    parser.add_argument(
        "--seq-lens",
        type=str,
        default=None,
        help="Comma-separated sequence lengths, e.g. '128,256,512,128'. "
        "If not specified, generates random lengths.",
    )
    parser.add_argument(
        "--total-len",
        type=int,
        default=10240,
        help="Total token count when seq-lens is not specified (default: 10240)",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=5,
        help="Number of profiling iterations (default: 5)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Data type (default: bf16)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for Chrome traces. Defaults to current directory.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for both engines",
    )
    parser.add_argument(
        "--top-ops",
        type=int,
        default=10,
        help="Number of top ops to show in comparison (default: 10)",
    )
    return parser.parse_args(argv)


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return dtype_map[dtype_str]


def profile_archon(
    model_path: str,
    dtype: torch.dtype,
    inputs_config: dict,
    mode: str,
    warmup_iters: int,
    profile_iters: int,
    gradient_checkpointing: bool,
    output_path: Path | None,
) -> EngineResult:
    """Profile Archon engine."""
    # Import functions for packed input creation
    from areal.tools.profile_archon import (
        create_packed_input,
        run_forward,
        run_forward_backward,
    )
    from areal.utils.testing_utils import load_archon_model

    print("\n" + "=" * 70)
    print("PROFILING ARCHON ENGINE")
    print("=" * 70)

    # Load model
    print("[Archon] Loading model...")
    model, _ = load_archon_model(model_path, dtype=dtype)

    if gradient_checkpointing:
        from areal.experimental.models.archon.activation_checkpoint import (
            ActivationCheckpointConfig,
            apply_activation_checkpointing,
        )

        ac_config = ActivationCheckpointConfig(mode="full")
        apply_activation_checkpointing(model, ac_config)

    device = torch.device(current_platform.device_type)
    inputs = create_packed_input(
        batch_size=inputs_config["batch_size"],
        seq_lens=inputs_config["seq_lens"],
        total_len=inputs_config["total_len"],
        vocab_size=inputs_config["vocab_size"],
        device=device,
    )

    # Select run function using partial to capture model and inputs
    if mode == "forward":
        run_fn = functools.partial(run_forward, model, inputs)
    else:
        run_fn = functools.partial(run_forward_backward, model, inputs)

    # Warmup
    print(f"[Archon] Warming up ({warmup_iters} iters)...")
    for _ in range(warmup_iters):
        run_fn()
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    # Profile
    print(f"[Archon] Profiling ({profile_iters} iters)...")
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        for _ in range(profile_iters):
            run_fn()
            torch.cuda.synchronize()

    # Extract results
    key_averages = prof.key_averages()
    total_cuda_time_us = sum(
        evt.device_time_total for evt in key_averages if evt.device_time_total > 0
    )
    total_cuda_time_ms = total_cuda_time_us / 1000 / profile_iters
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Get top ops
    sorted_events = sorted(
        key_averages, key=lambda x: x.device_time_total, reverse=True
    )
    top_ops = [
        (evt.key, evt.device_time_total / 1000 / profile_iters)
        for evt in sorted_events[:20]
    ]

    # Export trace
    if output_path:
        prof.export_chrome_trace(str(output_path))
        print(f"[Archon] Trace exported to: {output_path}")

    # Clear model from memory
    del model
    torch.cuda.empty_cache()

    return EngineResult(
        name="Archon",
        total_cuda_time_ms=total_cuda_time_ms,
        peak_memory_mb=peak_memory_mb,
        top_ops=top_ops,
    )


def profile_fsdp(
    model_path: str,
    dtype: torch.dtype,
    inputs_config: dict,
    mode: str,
    warmup_iters: int,
    profile_iters: int,
    gradient_checkpointing: bool,
    output_path: Path | None,
) -> EngineResult:
    """Profile FSDP/HuggingFace engine."""
    from areal.tools.profile_fsdp import (
        create_padded_input,
        load_hf_model,
        run_forward,
        run_forward_backward,
    )

    print("\n" + "=" * 70)
    print("PROFILING FSDP/HF ENGINE")
    print("=" * 70)

    # Load model
    print("[FSDP] Loading model...")
    model, _ = load_hf_model(
        model_path,
        dtype=dtype,
        attn_impl="sdpa",
        gradient_checkpointing=gradient_checkpointing,
    )

    # Create padded input
    device = torch.device(current_platform.device_type)
    inputs = create_padded_input(
        batch_size=inputs_config["batch_size"],
        seq_lens=inputs_config["seq_lens"],
        total_len=inputs_config["total_len"],
        vocab_size=inputs_config["vocab_size"],
        device=device,
    )

    # Select run function using partial to capture model and inputs
    if mode == "forward":
        run_fn = functools.partial(run_forward, model, inputs)
    else:
        run_fn = functools.partial(run_forward_backward, model, inputs)

    # Warmup
    print(f"[FSDP] Warming up ({warmup_iters} iters)...")
    for _ in range(warmup_iters):
        run_fn()
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    # Profile
    print(f"[FSDP] Profiling ({profile_iters} iters)...")
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        for _ in range(profile_iters):
            run_fn()
            torch.cuda.synchronize()

    # Extract results
    key_averages = prof.key_averages()
    total_cuda_time_us = sum(
        evt.device_time_total for evt in key_averages if evt.device_time_total > 0
    )
    total_cuda_time_ms = total_cuda_time_us / 1000 / profile_iters
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Get top ops
    sorted_events = sorted(
        key_averages, key=lambda x: x.device_time_total, reverse=True
    )
    top_ops = [
        (evt.key, evt.device_time_total / 1000 / profile_iters)
        for evt in sorted_events[:20]
    ]

    # Export trace
    if output_path:
        prof.export_chrome_trace(str(output_path))
        print(f"[FSDP] Trace exported to: {output_path}")

    # Clear model from memory
    del model
    torch.cuda.empty_cache()

    return EngineResult(
        name="FSDP/HF",
        total_cuda_time_ms=total_cuda_time_ms,
        peak_memory_mb=peak_memory_mb,
        top_ops=top_ops,
    )


def print_comparison(
    archon_result: EngineResult,
    fsdp_result: EngineResult,
    top_n: int = 10,
) -> None:
    """Print comparison report between two engines."""
    print("\n" + "=" * 80)
    print("COMPARISON REPORT: ARCHON vs FSDP/HF")
    print("=" * 80)

    # Summary comparison
    print("\n--- SUMMARY ---")
    print(f"{'Metric':<30} {'Archon':>15} {'FSDP/HF':>15} {'Diff':>15}")
    print("-" * 75)

    time_diff = archon_result.total_cuda_time_ms - fsdp_result.total_cuda_time_ms
    time_ratio = archon_result.total_cuda_time_ms / fsdp_result.total_cuda_time_ms
    print(
        f"{'CUDA Time (ms)':<30} {archon_result.total_cuda_time_ms:>15.2f} "
        f"{fsdp_result.total_cuda_time_ms:>15.2f} {time_diff:>+15.2f}"
    )
    print(f"{'  (ratio)':<30} {'':>15} {'':>15} {time_ratio:>14.2f}x")

    mem_diff = archon_result.peak_memory_mb - fsdp_result.peak_memory_mb
    mem_ratio = archon_result.peak_memory_mb / fsdp_result.peak_memory_mb
    print(
        f"{'Peak Memory (MB)':<30} {archon_result.peak_memory_mb:>15.2f} "
        f"{fsdp_result.peak_memory_mb:>15.2f} {mem_diff:>+15.2f}"
    )
    print(f"{'  (ratio)':<30} {'':>15} {'':>15} {mem_ratio:>14.2f}x")

    # Top ops comparison
    print(f"\n--- TOP {top_n} OPS BY CUDA TIME ---")
    print(
        f"{'Rank':<6} {'Archon Op':<35} {'Time(ms)':>10} {'FSDP/HF Op':<35} {'Time(ms)':>10}"
    )
    print("-" * 100)

    for i in range(min(top_n, len(archon_result.top_ops), len(fsdp_result.top_ops))):
        archon_op, archon_time = archon_result.top_ops[i]
        fsdp_op, fsdp_time = fsdp_result.top_ops[i]

        # Truncate op names
        archon_op_short = archon_op[:33] + ".." if len(archon_op) > 35 else archon_op
        fsdp_op_short = fsdp_op[:33] + ".." if len(fsdp_op) > 35 else fsdp_op

        print(
            f"{i + 1:<6} {archon_op_short:<35} {archon_time:>10.2f} "
            f"{fsdp_op_short:<35} {fsdp_time:>10.2f}"
        )

    # Performance verdict
    print("\n--- VERDICT ---")
    if time_ratio < 0.95:
        print(f"Archon is {(1 / time_ratio - 1) * 100:.1f}% FASTER than FSDP/HF")
    elif time_ratio > 1.05:
        print(f"Archon is {(time_ratio - 1) * 100:.1f}% SLOWER than FSDP/HF")
    else:
        print("Performance is roughly equivalent (within 5%)")

    if mem_ratio < 0.95:
        print(f"Archon uses {(1 - mem_ratio) * 100:.1f}% LESS memory than FSDP/HF")
    elif mem_ratio > 1.05:
        print(f"Archon uses {(mem_ratio - 1) * 100:.1f}% MORE memory than FSDP/HF")
    else:
        print("Memory usage is roughly equivalent (within 5%)")


def run_comparison(args: argparse.Namespace) -> None:
    """Run comparison profiling."""
    # Setup environment
    if "LOCAL_RANK" not in os.environ:
        current_platform.set_device(0)

    dtype = get_dtype(args.dtype)

    # Get model path
    from transformers import AutoConfig

    from areal.utils.testing_utils import MODEL_PATHS

    model_path = args.model_path or MODEL_PATHS.get(
        "qwen2", "Qwen/Qwen2.5-0.5B-Instruct"
    )
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Parse seq_lens if provided
    seq_lens = None
    if args.seq_lens:
        seq_lens = [int(x.strip()) for x in args.seq_lens.split(",")]
        args.batch_size = len(seq_lens)

    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Generate seq_lens if not provided
    if seq_lens is None:
        from areal.tools.profiling_utils.utils import generate_random_seq_lens

        seq_lens = generate_random_seq_lens(args.batch_size, args.total_len)

    inputs_config = {
        "batch_size": args.batch_size,
        "seq_lens": seq_lens,
        "total_len": args.total_len,
        "vocab_size": config.vocab_size,
    }

    mode_str = {
        "forward": "FORWARD ONLY",
        "backward": "BACKWARD ONLY",
        "both": "FORWARD + BACKWARD",
    }[args.mode]

    print("\n" + "=" * 80)
    print(f"ENGINE COMPARISON ({mode_str})")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Dtype: {args.dtype}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence lengths: {seq_lens}")
    print(f"Total tokens: {sum(seq_lens)}")
    print(f"Warmup iters: {args.warmup_iters}")
    print(f"Profile iters: {args.profile_iters}")
    print(
        f"Gradient checkpointing: {'enabled' if args.gradient_checkpointing else 'disabled'}"
    )

    # Setup output paths
    output_dir = Path(args.output_dir) if args.output_dir else Path(".")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archon_trace = output_dir / f"profile_archon_{args.mode}_{timestamp}.json"
    fsdp_trace = output_dir / f"profile_fsdp_{args.mode}_{timestamp}.json"

    # Reset seeds before each profile
    torch.manual_seed(42)
    random.seed(42)

    # Profile Archon
    archon_result = profile_archon(
        model_path=model_path,
        dtype=dtype,
        inputs_config=inputs_config,
        mode=args.mode,
        warmup_iters=args.warmup_iters,
        profile_iters=args.profile_iters,
        gradient_checkpointing=args.gradient_checkpointing,
        output_path=archon_trace,
    )

    # Reset seeds again
    torch.manual_seed(42)
    random.seed(42)

    # Profile FSDP
    fsdp_result = profile_fsdp(
        model_path=model_path,
        dtype=dtype,
        inputs_config=inputs_config,
        mode=args.mode,
        warmup_iters=args.warmup_iters,
        profile_iters=args.profile_iters,
        gradient_checkpointing=args.gradient_checkpointing,
        output_path=fsdp_trace,
    )

    # Print comparison
    print_comparison(archon_result, fsdp_result, top_n=args.top_ops)

    print("\n" + "=" * 80)
    print("TRACE FILES")
    print("=" * 80)
    print(f"Archon: {archon_trace}")
    print(f"FSDP:   {fsdp_trace}")
    print("View with: https://ui.perfetto.dev/")


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    run_comparison(args)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
