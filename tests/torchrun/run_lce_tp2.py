import argparse
import gc
import math
import os

import torch
import torch.distributed as dist

from areal.infra.platforms import current_platform
from areal.models.lce import linear_cross_entropy
from areal.utils.functional import gather_logprobs_entropy


def _setup_distributed_environment() -> None:
    if dist.is_initialized():
        return
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ["MASTER_PORT"]
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )
    current_platform.set_device(rank)


def _get_tp_group() -> dist.ProcessGroup:
    return dist.distributed_c10d._get_default_group()


def _make_tp_inputs(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    vocab_per_rank = vocab_size // world_size
    assert vocab_size % world_size == 0

    g = torch.Generator(device=device).manual_seed(seed)
    hidden = (
        torch.randn(num_tokens, hidden_size, dtype=dtype, device=device, generator=g)
        * 0.02
    )
    labels = torch.randint(0, vocab_size, (num_tokens,), device=device, generator=g)
    weight_full = (
        torch.randn(vocab_size, hidden_size, dtype=dtype, device=device, generator=g)
        * 0.02
    )
    weight_shard = weight_full[
        rank * vocab_per_rank : (rank + 1) * vocab_per_rank
    ].contiguous()
    return (
        hidden.contiguous(),
        labels.contiguous(),
        weight_full.contiguous(),
        weight_shard,
    )


def _run_full_reference(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    entropy_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    logits = hidden_ref.float() @ weight_ref.float().t()
    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    ref_lp = log_softmax.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    probs = log_softmax.exp()
    ref_h = -(probs * log_softmax).sum(dim=-1)
    (ref_lp.sum() + entropy_weight * ref_h.sum()).backward()
    return ref_lp, ref_h, hidden_ref.grad, weight_ref.grad


def _run_tp_materialized_step(
    hidden: torch.Tensor,
    weight_shard: torch.Tensor,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h = hidden.detach().clone().requires_grad_(True)
    w = weight_shard.detach().clone().requires_grad_(True)
    local_logits = h.float() @ w.float().t()
    lp, ent = gather_logprobs_entropy(local_logits, labels, tp_group=tp_group)
    (lp.sum() + ent.sum()).backward()
    return lp, ent, h.grad, w.grad


def _run_fused_step(
    hidden: torch.Tensor,
    weight_shard: torch.Tensor,
    labels: torch.Tensor,
    tp_group: dist.ProcessGroup,
    entropy_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h = hidden.detach().clone().requires_grad_(True)
    w = weight_shard.detach().clone().requires_grad_(True)
    lp, ent = linear_cross_entropy(h, w, labels, 1.0, "none", tp_group)
    (lp.sum() + entropy_weight * ent.sum()).backward()
    return lp, ent, h.grad, w.grad


def _test_tp2_correctness(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    dtype: torch.dtype,
) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2
    device = current_platform.current_device()
    tp_group = _get_tp_group()

    hidden, labels, weight_full, weight_shard = _make_tp_inputs(
        num_tokens, hidden_size, vocab_size, dtype, device, seed=42
    )
    vocab_per_rank = vocab_size // world_size

    ref_lp, ref_h, ref_dh, ref_dw = _run_full_reference(
        hidden, weight_full, labels, entropy_weight=0.5
    )
    fused_lp, fused_h, fused_dh, fused_dw = _run_fused_step(
        hidden, weight_shard, labels, tp_group, entropy_weight=0.5
    )

    if dtype == torch.float32:
        rtol, atol = 2e-4, 2e-4
    else:
        rtol, atol = 3e-2, 3e-2

    torch.testing.assert_close(fused_lp.float(), ref_lp.float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(fused_h.float(), ref_h.float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(fused_dh.float(), ref_dh.float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(
        fused_dw.float(),
        ref_dw[rank * vocab_per_rank : (rank + 1) * vocab_per_rank].float(),
        rtol=rtol,
        atol=atol,
    )

    if rank == 0:
        print(
            f"[PASS] tp2_correctness: T={num_tokens} H={hidden_size} "
            f"V={vocab_size} dtype={dtype}"
        )


def _time_step(fn) -> tuple[float, float]:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), torch.cuda.max_memory_allocated() / (1024 * 1024)


def _test_tp2_performance(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2
    device = current_platform.current_device()
    dtype = torch.bfloat16
    tp_group = _get_tp_group()

    hidden, labels, _, weight_shard = _make_tp_inputs(
        num_tokens, hidden_size, vocab_size, dtype, device, seed=0
    )

    for _ in range(2):
        _run_fused_step(hidden, weight_shard, labels, tp_group)
        gc.collect()
        torch.cuda.empty_cache()
    for _ in range(2):
        _run_tp_materialized_step(hidden, weight_shard, labels, tp_group)
        gc.collect()
        torch.cuda.empty_cache()

    fused_times = []
    fused_mems = []
    for _ in range(5):
        t, m = _time_step(
            lambda: _run_fused_step(hidden, weight_shard, labels, tp_group)
        )
        fused_times.append(t)
        fused_mems.append(m)

    ref_times = []
    ref_mems = []
    for _ in range(5):
        t, m = _time_step(
            lambda: _run_tp_materialized_step(hidden, weight_shard, labels, tp_group)
        )
        ref_times.append(t)
        ref_mems.append(m)

    ref_med = sorted(ref_times)[len(ref_times) // 2]
    fused_med = sorted(fused_times)[len(fused_times) // 2]
    ref_peak = max(ref_mems)
    fused_peak = max(fused_mems)
    speedup = ref_med / fused_med if fused_med > 0 else math.inf
    mem_ratio = fused_peak / ref_peak if ref_peak > 0 else math.inf

    if rank == 0:
        print(
            f"\n[LCE-TP2-Bench] tokens={num_tokens} hidden={hidden_size} "
            f"vocab={vocab_size} dtype={dtype}\n"
            f"  tp materialized: {ref_med:7.2f} ms / {ref_peak:7.1f} MB peak\n"
            f"  fused          : {fused_med:7.2f} ms / {fused_peak:7.1f} MB peak\n"
            f"  speedup        : {speedup:5.2f}x   memory_ratio: {mem_ratio:5.2f}x"
        )

    assert fused_med < ref_med * 1.5, (
        f"Fused TP=2 LCE is more than 1.5x slower than TP materialized reference "
        f"(fused={fused_med:.2f}ms ref={ref_med:.2f}ms)."
    )
    assert fused_peak < ref_peak * 1.2, (
        f"Fused TP=2 LCE peak memory exceeds TP materialized reference by >20% "
        f"(fused={fused_peak:.1f}MB ref={ref_peak:.1f}MB)."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_type", choices=["correctness", "performance"], required=True
    )
    parser.add_argument("--num_tokens", type=int, required=True)
    parser.add_argument("--hidden_size", type=int, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    _setup_distributed_environment()
    try:
        if args.test_type == "correctness":
            _test_tp2_correctness(
                args.num_tokens, args.hidden_size, args.vocab_size, dtype
            )
        else:
            _test_tp2_performance(args.num_tokens, args.hidden_size, args.vocab_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
