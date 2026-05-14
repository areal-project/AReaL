# SPDX-License-Identifier: Apache-2.0
"""
Standalone benchmark for the fused linear-cross-entropy kernel.

Designed to be run outside pytest to measure forward+backward latency and
peak memory for the materialised reference path and the fused Triton path.

Usage::

    # Qwen3 single-GPU full-vocab benchmark
    uv run python -m benchmark.kernels.bench_linear_cross_entropy \\
        --mode both --tokens 2048 --hidden 4096 --vocab 152064 \\
        --dtype bfloat16 --warmup 5 --iters 15 --check-correctness

    # Qwen3 TP=2 benchmark. The reference path materialises only local
    # [tokens, vocab/tp] logits and uses vocab-parallel reductions.
    uv run torchrun --nproc_per_node=2 --nnodes=1 \\
        --master-addr=localhost --master_port=29501 \\
        -m benchmark.kernels.bench_linear_cross_entropy \\
        --mode both --tp-size 2 --tokens 2048 --hidden 4096 --vocab 152064 \\
        --dtype bfloat16 --warmup 5 --iters 15 --check-correctness

    # Qwen3 TP=4 benchmark
    uv run torchrun --nproc_per_node=4 --nnodes=1 \\
        --master-addr=localhost --master_port=29501 \\
        -m benchmark.kernels.bench_linear_cross_entropy \\
        --mode both --tp-size 4 --tokens 2048 --hidden 4096 --vocab 152064 \\
        --dtype bfloat16 --warmup 5 --iters 15 --check-correctness
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys

import torch
import torch.distributed as dist

from areal.utils.functional import gather_logprobs_entropy


def _setup_distributed(tp_size: int):
    if tp_size == 1:
        return None
    if not dist.is_available():
        raise RuntimeError("torch.distributed is required when --tp-size > 1")
    if not dist.is_initialized():
        required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_PORT")
        missing = [k for k in required if k not in os.environ]
        if missing:
            raise RuntimeError(
                "--tp-size > 1 must be launched with torchrun; missing env vars: "
                + ", ".join(missing)
            )
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    if world_size != tp_size:
        raise RuntimeError(
            f"--tp-size={tp_size} must match torchrun world_size={world_size}"
        )
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", dist.get_rank())))
    return dist.group.WORLD


def _rank(tp_group):
    return dist.get_rank(tp_group) if tp_group is not None else 0


def _world_size(tp_group):
    return dist.get_world_size(tp_group) if tp_group is not None else 1


def _make_inputs(num_tokens, hidden_size, vocab_size, dtype, tp_group=None, seed=0):
    world_size = _world_size(tp_group)
    rank = _rank(tp_group)
    if vocab_size % world_size != 0:
        raise ValueError(
            f"vocab_size={vocab_size} must be divisible by tp_size={world_size}"
        )
    local_vocab_size = vocab_size // world_size

    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden = (
        torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda", generator=g)
        * 0.02
    )
    weight = (
        torch.randn(
            local_vocab_size,
            hidden_size,
            dtype=dtype,
            device="cuda",
            generator=g,
        )
        * 0.02
    )
    if tp_group is not None:
        weight = weight + (rank * 0.001)
    labels = torch.randint(0, vocab_size, (num_tokens,), device="cuda", generator=g)
    return hidden.contiguous(), weight.contiguous(), labels.contiguous()


def _ref_step(hidden, weight, labels, temperature=1.0, tp_group=None):
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    logits = h.float() @ w.float().t()
    if tp_group is not None:
        lp, ent = gather_logprobs_entropy(
            logits, labels, temperature=temperature, tp_group=tp_group
        )
    else:
        log_softmax = torch.nn.functional.log_softmax(logits / temperature, dim=-1)
        lp = log_softmax.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        probs = log_softmax.exp()
        ent = -(probs * log_softmax).sum(dim=-1)
    (lp.sum() + ent.sum()).backward()
    if tp_group is not None:
        dist.all_reduce(h.grad, op=dist.ReduceOp.SUM, group=tp_group)
    return lp.detach(), ent.detach(), h.grad.detach(), w.grad.detach()


def _fused_step(hidden, weight, labels, temperature=1.0, tp_group=None):
    from areal.models.kernel import linear_cross_entropy

    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    lp, ent = linear_cross_entropy(h, w, labels, temperature, "none", tp_group)
    (lp.sum() + ent.sum()).backward()
    return lp.detach(), ent.detach(), h.grad.detach(), w.grad.detach()


def _check_correctness(hidden, weight, labels, dtype, tp_group=None):
    ref_lp, ref_ent, ref_dh, ref_dw = _ref_step(
        hidden, weight, labels, tp_group=tp_group
    )
    fused_lp, fused_ent, fused_dh, fused_dw = _fused_step(
        hidden, weight, labels, tp_group=tp_group
    )

    if dtype == torch.float32:
        rtol, atol = 1e-4, 1e-4
    elif dtype == torch.bfloat16:
        rtol, atol = 3e-2, 3e-2
    else:
        rtol, atol = 2e-2, 2e-2

    torch.testing.assert_close(fused_lp.float(), ref_lp.float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(fused_ent.float(), ref_ent.float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(fused_dh.float(), ref_dh.float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(fused_dw.float(), ref_dw.float(), rtol=rtol, atol=atol)


def _measure(label, fn, hidden, weight, labels, warmup, iters, tp_group=None):
    nvtx = torch.cuda.nvtx
    times = []
    mems = []

    # Warmup
    nvtx.range_push(f"{label}/warmup")
    for _ in range(warmup):
        fn(hidden, weight, labels, tp_group=tp_group)
        gc.collect()
        torch.cuda.empty_cache()
    nvtx.range_pop()

    nvtx.range_push(f"{label}/measure")
    for i in range(iters):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        nvtx.range_push(f"{label}/iter{i}")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(hidden, weight, labels, tp_group=tp_group)
        end.record()
        torch.cuda.synchronize()
        nvtx.range_pop()
        times.append(start.elapsed_time(end))
        mems.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
    nvtx.range_pop()

    return times, mems


def _distributed_max(value, tp_group):
    if tp_group is None:
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=tp_group)
    return float(tensor.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--vocab", type=int, default=152064)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument(
        "--use-cuda-profiler-api",
        action="store_true",
        help="Wrap the measurement region with cudaProfilerStart/Stop.",
    )
    parser.add_argument("--mode", choices=["both", "ref", "fused"], default="both")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; aborting.", file=sys.stderr)
        sys.exit(1)

    tp_group = _setup_distributed(args.tp_size)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    hidden, weight, labels = _make_inputs(
        args.tokens, args.hidden, args.vocab, dtype, tp_group=tp_group
    )
    if _rank(tp_group) == 0:
        print(
            f"[bench] tokens={args.tokens} hidden={args.hidden} vocab={args.vocab} "
            f"tp={args.tp_size} dtype={args.dtype} warmup={args.warmup} "
            f"iters={args.iters}"
        )

    if args.check_correctness:
        _check_correctness(hidden, weight, labels, dtype, tp_group=tp_group)
        if _rank(tp_group) == 0:
            print("[bench] correctness check passed")

    if args.use_cuda_profiler_api:
        torch.cuda.cudart().cudaProfilerStart()

    results = {}
    if args.mode in ("both", "ref"):
        t, m = _measure(
            "reference",
            _ref_step,
            hidden,
            weight,
            labels,
            args.warmup,
            args.iters,
            tp_group=tp_group,
        )
        results["reference"] = (t, m)
    if args.mode in ("both", "fused"):
        t, m = _measure(
            "fused",
            _fused_step,
            hidden,
            weight,
            labels,
            args.warmup,
            args.iters,
            tp_group=tp_group,
        )
        results["fused"] = (t, m)

    if args.use_cuda_profiler_api:
        torch.cuda.cudart().cudaProfilerStop()

    summaries = {}
    for name, (t, m) in results.items():
        local_median = sorted(t)[len(t) // 2]
        local_peak = max(m)
        summaries[name] = (
            _distributed_max(local_median, tp_group),
            _distributed_max(local_peak, tp_group),
        )

    if _rank(tp_group) == 0:
        for name, (median, peak) in summaries.items():
            print(f"[bench] {name:9s}  median={median:7.2f}ms  peak_mem={peak:8.1f}MB")

        if "reference" in summaries and "fused" in summaries:
            ref_med, ref_peak = summaries["reference"]
            fused_med, fused_peak = summaries["fused"]
            speedup = ref_med / fused_med if fused_med > 0 else math.inf
            mem_ratio = fused_peak / ref_peak if ref_peak > 0 else math.inf
            print(
                f"[bench] speedup={speedup:.2f}x  fused_peak/ref_peak={mem_ratio:.2f}x"
            )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
