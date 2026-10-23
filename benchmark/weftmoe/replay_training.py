"""Replay captured CPU microbatch inputs through the real AReaL Megatron actor.

Run with torchrun and the original RL YAML. No rollout, synthetic routes, or
model checkpoint is substituted. Report whole train_batch plus residency costs;
the first iteration includes compiler warmup and is not a steady-state sample.
"""

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.engine.megatron_engine import MegatronEngine
from areal.trainer.ppo.actor import grpo_loss_fn
from areal.utils import logging, seeding, stats_tracker

logger = logging.getLogger("PerfTracer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--dataset-size",
        type=int,
        required=True,
        help="Effective training dataset size for the original LR schedule",
    )
    parser.add_argument(
        "--resident",
        action="store_true",
        help="Skip phase-boundary release to isolate steady-state execution",
    )
    parser.add_argument(
        "--freeze-weights",
        action="store_true",
        help="Run the optimizer at LR=0 so both backends see the same model/routes",
    )
    args = parser.parse_args()
    config, _ = load_expr_config(["--config", str(args.config)], GRPOConfig)
    if args.freeze_weights:
        config.actor.optimizer.lr = 0.0
        config.actor.optimizer.min_lr_ratio = 0.0
    if os.environ.get("AREAL_MEGATRON_PROFILE_DIR"):
        from benchmark.weftmoe.training_profile import instrument_engine

        instrument_engine()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    seeding.set_random_seed(config.seed, "actor")
    engine = MegatronEngine(config.actor)
    strategy = ModelAllocation.from_str(config.actor.backend).parallel
    engine.create_process_group(parallel_strategy=strategy)
    engine.initialize(
        addr=None,
        ft_spec=FinetuneSpec(
            total_train_epochs=config.total_train_epochs,
            dataset_size=args.dataset_size,
            train_batch_size=config.train_dataset.batch_size,
        ),
    )
    engine.init_weight_residency_adapter()
    rank = dist.get_rank()
    from megatron.core import parallel_state as mpu

    batch_file = (
        args.batch_dir / f"batch.rank{mpu.get_tensor_model_parallel_src_rank()}.pt"
    )
    batch = torch.load(batch_file, weights_only=False, map_location="cpu")
    cfg = config.actor
    loss_fn = partial(
        grpo_loss_fn,
        **{
            key: getattr(cfg, key)
            for key in (
                "eps_clip",
                "eps_clip_higher",
                "c_clip",
                "rejection_sampling",
                "m2_threshold",
                "importance_sampling_level",
                "prox_logp_method",
                "use_sapo_loss",
                "sapo_tau_pos",
                "sapo_tau_neg",
                "use_cispo_loss",
                "use_decoupled_loss",
            )
        },
        current_version=0,
    )
    if not args.resident:
        engine.offload()
    for step in range(args.steps):
        dist.barrier()
        begin = time.perf_counter()
        if not args.resident:
            engine.onload()
        loaded = time.perf_counter()
        engine.train()
        result = engine.train_batch(
            batch,
            loss_fn=loss_fn,
            loss_weight_fn=lambda item: item["loss_mask"].count_nonzero(),
        )
        engine.step_lr_scheduler()
        torch.cuda.synchronize()
        trained = time.perf_counter()
        if not args.resident:
            engine.offload()
        ended = time.perf_counter()
        row = dict(
            step=step,
            rank=rank,
            onload_s=loaded - begin,
            train_s=trained - loaded,
            offload_s=ended - trained,
            total_s=ended - begin,
            **result,
        )
        row["device_free_bytes"], row["device_total_bytes"] = torch.cuda.mem_get_info()
        adapter = engine._weftmoe_adapter
        if adapter is not None and adapter.runtime is not None:
            row["weft_workspace"] = adapter.runtime.native.usage()
        logger.info("AREAL_REPLAY_JSON %s", json.dumps(row))
        stats_tracker.export_all(reset=True)
    engine.destroy()


if __name__ == "__main__":
    main()
