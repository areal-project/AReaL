# SPDX-License-Identifier: Apache-2.0
"""Short torchrun PPO measurements; reuse workers while changing logical inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from examples.profile.calibration_cases import (
    make_actor_batch,
    select_cases,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--cases", default="pilot")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens-per-mb", type=int, default=32768)
    parser.add_argument("--bridge", default="mbridge")
    args = parser.parse_args()

    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu

    from areal.api import FinetuneSpec
    from areal.api.alloc_mode import ModelAllocation
    from areal.api.cli_args import (
        DistributedDataParallelConfig,
        MegatronEngineConfig,
        MicroBatchSpec,
        OptimizerConfig,
        PPOActorConfig,
        RejectionSamplingConfig,
    )
    from areal.engine.megatron_engine import MegatronPPOActor
    from areal.utils import logging, seeding

    logger = logging.getLogger("ActorCalibration")
    cases = select_cases("actor", args.cases)
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != args.dp * args.tp:
        raise ValueError("torchrun world size must equal DP * TP")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    seeding.set_random_seed(args.seed, key="actor-calibration")
    config = PPOActorConfig(
        experiment_name="elastic-calibration",
        trial_name=args.output.name,
        path=args.model,
        backend=f"megatron:d{args.dp}t{args.tp}",
        dtype="bfloat16",
        grad_reduce_dtype="bfloat16",
        disable_dropout=True,
        gradient_checkpointing=True,
        mb_spec=MicroBatchSpec(n_mbs=1, max_tokens_per_mb=args.max_tokens_per_mb),
        optimizer=OptimizerConfig(
            lr=1.7e-5,
            weight_decay=0.017,
            lr_scheduler_type="constant",
            warmup_steps_proportion=0.0,
        ),
        megatron=MegatronEngineConfig(
            bridge_type=args.bridge,
            use_precision_aware_optimizer=True,
            main_grads_dtype="bfloat16",
            main_params_dtype="float32",
            exp_avg_dtype="float32",
            exp_avg_sq_dtype="float32",
            enable_chunked_logits=True,
            entropy_requires_grad=False,
            ddp=DistributedDataParallelConfig(
                grad_reduce_in_fp32=False,
                overlap_grad_reduce=True,
                use_distributed_optimizer=True,
            ),
        ),
        ppo_n_minibatches=1,
        recompute_logprob=True,
        use_decoupled_loss=True,
        rejection_sampling=RejectionSamplingConfig(metric="ratio", upper=5.0),
        reward_bias=-0.5,
        reward_scaling=10.0,
        reward_norm=None,
        adv_norm=None,
        kl_ctl=0.0,
        mask_no_eos_with_zero=False,
    )
    if rank == 0:
        write_json(
            args.output / "launch.json",
            {
                "arguments": vars(args),
                "config": asdict(config),
                "entrypoint_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "state_policy": "weights and Adam moments advance across cases; fresh inputs each step",
                "timing": "CUDA events, one synchronization at step boundaries; CPU preparation excluded",
            },
        )
    started = time.perf_counter()
    engine = MegatronPPOActor(config)
    engine.seed = seeding.get_seed()
    try:
        engine.create_process_group(ModelAllocation.from_str(config.backend).parallel)
        engine.initialize(None, FinetuneSpec(1, 100000, 1))
    except Exception as error:
        write_json(
            args.output / f"error-rank{rank}.json",
            {
                "stage": "initialization",
                "type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    torch.cuda.synchronize()
    startup_seconds = time.perf_counter() - started
    phase_events = []
    microbatches = []
    optimizer_records = []

    @contextmanager
    def phase(name):
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        host_start = time.perf_counter()
        begin.record()
        try:
            yield
        finally:
            end.record()
            phase_events.append((name, begin, end, time.perf_counter() - host_start))

    original_optimizer_step = engine.optimizer_step
    original_forward_backward = engine.forward_backward_batch
    original_prepare = engine._prepare_mb_list

    def optimizer_step(*a, **kw):
        with phase("optimizer"):
            result = original_optimizer_step(*a, **kw)
        optimizer_records.append(result)
        return result

    def forward_backward(*a, **kw):
        with phase(
            "model_forward" if kw.get("forward_only") else "model_forward_backward"
        ):
            return original_forward_backward(*a, **kw)

    def prepare(*a, **kw):
        result = original_prepare(*a, **kw)
        microbatches.append(
            {
                "groups": list(result.group_lens),
                "padded_lengths": list(result.padded_to_lengths),
                "padding_lengths": list(result.padding_lengths),
            }
        )
        return result

    engine.optimizer_step = optimizer_step
    engine.forward_backward_batch = forward_backward
    engine._prepare_mb_list = prepare
    path = args.output / f"steps-rank{rank}.jsonl"
    successful = False
    try:
        with path.open("x", buffering=1) as output:
            for case in cases:
                for iteration in range(args.warmup + args.repeat):
                    data = make_actor_batch(
                        case,
                        seed=args.seed,
                        dp_rank=mpu.get_data_parallel_rank(),
                        device=engine.device,
                    )
                    for item in data:
                        item["versions"].masked_fill_(
                            item["loss_mask"], engine.get_version()
                        )
                    input_hash = hashlib.sha256(
                        b"".join(
                            item["input_ids"].cpu().numpy().tobytes() for item in data
                        )
                    ).hexdigest()
                    torch.cuda.synchronize()
                    dist.barrier()
                    torch.cuda.reset_peak_memory_stats()
                    phase_events.clear()
                    microbatches.clear()
                    optimizer_records.clear()
                    start_wall = time.time()
                    start = time.perf_counter()
                    with phase("logp"):
                        logps = engine.compute_logp(data)
                    for item, logp in zip(data, logps):
                        item["prox_logp"] = logp
                        item["logprobs"] = torch.roll(logp, 1, -1) * item["loss_mask"]
                    with phase("advantage"):
                        advantages = engine.compute_advantages(data)
                    with phase("ppo_update"):
                        engine.ppo_update(advantages)
                    engine.step_lr_scheduler()
                    engine.set_version(engine.get_version() + 1)
                    torch.cuda.synchronize()
                    wall_seconds = time.perf_counter() - start
                    if not optimizer_records or any(
                        result["update_successful"] != 1
                        or not math.isfinite(result["grad_norm"])
                        or result["grad_norm"] <= 0
                        for result in optimizer_records
                    ):
                        raise RuntimeError(
                            f"Invalid optimizer update: {optimizer_records}"
                        )
                    memory = {
                        "peak_allocated": torch.cuda.max_memory_allocated(),
                        "peak_reserved": torch.cuda.max_memory_reserved(),
                        "device_used": torch.cuda.device_memory_used(),
                    }
                    stats = engine.export_stats()
                    timings = [
                        dict(
                            name=n, gpu_seconds=b.elapsed_time(e) / 1000, host_seconds=h
                        )
                        for n, b, e, h in phase_events
                    ]
                    row = dict(
                        case=asdict(case),
                        rank=rank,
                        dp_rank=mpu.get_data_parallel_rank(),
                        tp_rank=mpu.get_tensor_model_parallel_rank(),
                        global_batch=args.dp * len(case.lengths),
                        local_batch=len(case.lengths),
                        loss_tokens=sum(n // 8 for n in case.lengths),
                        input_sha256=input_hash,
                        iteration=iteration,
                        warmup=iteration < args.warmup,
                        started_at=start_wall,
                        wall_seconds=wall_seconds,
                        startup_seconds=startup_seconds,
                        phases=timings,
                        microbatches=list(microbatches),
                        memory=memory,
                        stats=stats,
                        optimizer_updates=list(optimizer_records),
                    )
                    output.write(json.dumps(row, default=str) + "\n")
                    if rank == 0:
                        logger.info(
                            "%s iteration=%d warmup=%s seconds=%.3f peak_GiB=%.2f",
                            case.name,
                            iteration,
                            row["warmup"],
                            wall_seconds,
                            memory["peak_allocated"] / 2**30,
                        )
        successful = True
        if rank == 0:
            write_json(
                args.output / "completed.json",
                {"status": "completed", "cases": len(cases)},
            )
    except Exception as error:
        write_json(
            args.output / f"error-rank{rank}.json",
            {
                "type": type(error).__name__,
                "error": str(error),
                "case": case.name,
                "iteration": iteration,
            },
        )
        raise
    finally:
        if successful:
            engine.destroy()


if __name__ == "__main__":
    main()
