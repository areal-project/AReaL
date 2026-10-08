"""Paired Megatron actor benchmark with fixed, shared-prefix text trajectories.

Run once with and once without ``--tree`` under torchrun. This excludes rollout
and AWEX so actor execution can be compared on identical inputs and topology.
"""

import argparse
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    DistributedDataParallelConfig,
    MegatronEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.engine import MegatronEngine
from areal.utils import logging

logger = logging.getLogger("HartsActorBenchmark")


def fixed_trajectories(
    batch_size: int, prefix_tokens: int, suffix_tokens: int
) -> dict[str, torch.Tensor]:
    """Give every trajectory one common prefix and a distinct continuation."""
    rng = torch.Generator(device="cpu").manual_seed(1729)
    prefix = torch.randint(1000, 50000, (prefix_tokens,), generator=rng)
    suffix = torch.randint(1000, 50000, (batch_size, suffix_tokens), generator=rng)
    input_ids = torch.cat((prefix.expand(batch_size, -1), suffix), dim=1)
    loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    loss_mask[:, prefix_tokens:] = True
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
        "loss_mask": loss_mask,
    }


def _loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict[str, Any],
    **kwargs: Any,
) -> torch.Tensor:
    del entropy, input_data, kwargs
    return -logprobs.float().mean()


def _loss_weight_fn(input_data: dict[str, Any]) -> torch.Tensor:
    return input_data["loss_mask"].count_nonzero()


def _timed_method(
    engine: MegatronEngine, name: str, durations: dict[str, float]
) -> None:
    original: Callable[..., Any] = getattr(engine, name)

    def timed(*args: Any, **kwargs: Any) -> Any:
        torch.cuda.synchronize(engine.device)
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            torch.cuda.synchronize(engine.device)
            durations[name] = time.perf_counter() - start

    setattr(engine, name, timed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tree", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prefix-tokens", type=int, default=6000)
    parser.add_argument("--suffix-tokens", type=int, default=1000)
    parser.add_argument("--max-tokens-per-mb", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    allocation = ModelAllocation.from_str("megatron:(attn:d4p1t2c1|ffn:d1e8)")
    config = TrainEngineConfig(
        backend="megatron:(attn:d4p1t2c1|ffn:d1e8)",
        experiment_name="harts-actor-benchmark",
        trial_name="tree" if args.tree else "baseline",
        path=args.model_path,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=args.max_tokens_per_mb),
        pad_to_maximum=args.tree,
        enable_tree_training=args.tree,
        optimizer=OptimizerConfig(lr=1e-8, warmup_steps=0),
        megatron=MegatronEngineConfig(
            bridge_type="megatron-bridge",
            use_precision_aware_optimizer=True,
            enable_chunked_logits=True,
            lm_head_loss_chunk_size=1024,
            use_deterministic_algorithms=False,
            ddp=DistributedDataParallelConfig(overlap_grad_reduce=False),
        ),
    )
    steps = args.warmup + args.repeats
    ft_spec = FinetuneSpec(
        total_train_epochs=1,
        dataset_size=steps * args.batch_size,
        train_batch_size=args.batch_size,
    )
    data = fixed_trajectories(args.batch_size, args.prefix_tokens, args.suffix_tokens)
    engine = MegatronEngine(config)
    engine.create_process_group(parallel_strategy=allocation.parallel)
    try:
        engine.initialize(addr=None, ft_spec=ft_spec)
        engine.train()
        durations: dict[str, float] = {}
        for method in ("_prepare_mb_list", "forward_backward_batch", "optimizer_step"):
            _timed_method(engine, method, durations)

        rank = dist.get_rank()
        for step in range(steps):
            durations.clear()
            torch.cuda.synchronize(engine.device)
            start = time.perf_counter()
            engine.train_batch(data, _loss_fn, _loss_weight_fn)
            torch.cuda.synchronize(engine.device)
            total = time.perf_counter() - start
            timings = torch.tensor(
                [
                    total,
                    durations["_prepare_mb_list"],
                    durations["forward_backward_batch"],
                    durations["optimizer_step"],
                ],
                device=engine.device,
                dtype=torch.float64,
            )
            dist.all_reduce(timings, op=dist.ReduceOp.MAX)
            if rank == 0:
                label = "warmup" if step < args.warmup else "measured"
                total_s, prep_s, forward_backward_s, optimizer_s = (
                    timings.cpu().tolist()
                )
                logger.info(
                    "%s tree=%s step=%d total=%.3fs prep=%.3fs "
                    "forward_backward=%.3fs optimizer=%.3fs",
                    label,
                    args.tree,
                    step,
                    total_s,
                    prep_s,
                    forward_backward_s,
                    optimizer_s,
                )
    finally:
        engine.destroy()


if __name__ == "__main__":
    main()
