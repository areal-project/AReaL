# SPDX-License-Identifier: Apache-2.0
"""Private torchrun worker for :mod:`areal.tools.benchmark_fsdp`."""

import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from areal.tools.benchmark_fsdp import write_json


def make_batch(
    lengths: list[int],
    prompt_tokens: int,
    vocab_size: int,
    seed: int,
    rank: int,
    world_size: int,
) -> tuple[dict[str, Any], str]:
    """Generate identical global data, then take a deterministic strided DP shard."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(
        1, vocab_size, (len(lengths), max(lengths)), generator=generator
    )
    positions = torch.arange(max(lengths))[None, :]
    attention = positions < torch.tensor(lengths)[:, None]
    ids.masked_fill_(~attention, 0)
    targets = attention & (positions >= prompt_tokens)
    # Shift on padded rows, before packing: boundary logits must never predict
    # the first token of another packed sequence.
    batch = {
        "input_ids": ids,
        "attention_mask": attention,
        "input_token_loss_mask": targets,
        "loss_mask": torch.roll(targets, shifts=-1, dims=-1),
    }
    digest = hashlib.sha256()
    for value in batch.values():
        digest.update(value.numpy().tobytes())
    return {
        key: value[rank::world_size].contiguous() for key, value in batch.items()
    }, digest.hexdigest()


def provenance() -> dict[str, Any]:
    result: dict[str, Any] = {"packages": {}}
    for package in ("torch", "transformers", "flash-attn", "numpy", "areal"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    try:
        root = Path(__file__).resolve().parents[2]
        result["git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        result["git_status"] = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        result["git_revision"] = None
    try:
        result["nvidia_driver_versions"] = sorted(
            set(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ],
                    text=True,
                    timeout=5,
                    stderr=subprocess.DEVNULL,
                ).splitlines()
            )
        )
    except (OSError, subprocess.SubprocessError):
        result["nvidia_driver_versions"] = None
    return result


def run(config: dict[str, Any]) -> None:
    import torch
    import torch.distributed as dist
    from transformers import AutoConfig

    from areal.api import FinetuneSpec, ParallelStrategy
    from areal.api.cli_args import MicroBatchSpec, OptimizerConfig, TrainEngineConfig
    from areal.engine.fsdp_engine import FSDPEngine
    from areal.trainer.sft.lm_engine import compute_packed_sft_loss

    if not torch.cuda.is_available():
        raise ValueError("This benchmark requires CUDA and FlashAttention 2")
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world != config["world_size"]:
        raise ValueError("torchrun world size differs from the planned workload")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    initial_free_bytes, _ = torch.cuda.mem_get_info()
    model_config = AutoConfig.from_pretrained(config["resolved_model_path"])
    # Keep v1 deliberately narrow: these dense text families have the project's
    # existing packed/varlen FA2 path. Other architectures need separate validation.
    if model_config.model_type not in ("qwen2", "qwen3", "llama"):
        raise ValueError("Supported dense text model types: qwen2, qwen3, llama")
    if max(config["sequence_lengths"]) > model_config.max_position_embeddings:
        raise ValueError("Sequence length exceeds the model context limit")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    trial = config["trial"]
    actor = TrainEngineConfig(
        experiment_name="fsdp-benchmark",
        trial_name=trial["id"],
        path=config["resolved_model_path"],
        backend=f"fsdp:d{world}",
        init_from_scratch=config["init_from_scratch"],
        attn_impl="flash_attention_2",
        dtype="bfloat16",
        optimizer_dtype="float32",
        grad_reduce_dtype="float32",
        disable_dropout=True,
        gradient_checkpointing=trial["gradient_checkpointing"],
        mb_spec=MicroBatchSpec(n_mbs=1, max_tokens_per_mb=trial["max_tokens_per_mb"]),
        optimizer=OptimizerConfig(lr=1e-5, warmup_steps=0),
    )
    engine = FSDPEngine(actor)
    engine.create_process_group(ParallelStrategy(data_parallel_size=world))
    engine.initialize(
        addr=None,
        ft_spec=FinetuneSpec(
            total_train_epochs=1,
            dataset_size=config["global_batch_size"]
            * (config["warmup_steps"] + config["steps"]),
            train_batch_size=config["global_batch_size"],
        ),
    )
    engine.train()
    batch, digest = make_batch(
        config["sequence_lengths"],
        config["prompt_tokens"],
        model_config.vocab_size,
        config["seed"],
        rank,
        world,
    )
    input_tokens = int(batch["attention_mask"].count_nonzero())
    loss_tokens = int(batch["loss_mask"].count_nonzero())
    batch = {k: v.to(engine.device) for k, v in batch.items()}
    samples = []
    for step in range(config["warmup_steps"] + config["steps"]):
        # The CPU coordination boundary is outside the measured engine call.
        dist.barrier(group=engine.cpu_group)
        torch.cuda.synchronize(engine.device)
        torch.cuda.reset_peak_memory_stats(engine.device)
        started = time.perf_counter()
        stats = engine.train_batch(
            input_=batch,
            loss_fn=compute_packed_sft_loss,
            loss_weight_fn=lambda mb: mb["loss_mask"].count_nonzero(),
        )
        torch.cuda.synchronize(engine.device)
        sample = {
            "seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(engine.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(engine.device),
            "prepared_microbatches": int(stats["num_micro_batches"]),
            "grad_norm": stats["grad_norm"],
            "update_successful": stats["update_successful"],
        }
        if stats["update_successful"] != 1 or not math.isfinite(stats["grad_norm"]):
            raise RuntimeError(
                "Optimizer update was skipped due to non-finite gradients"
            )
        if step >= config["warmup_steps"]:
            samples.append(sample)
        # All ranks drain metric tensors/events outside the timed/memory region.
        engine.export_stats()
    props = torch.cuda.get_device_properties(engine.device)
    result = {
        "rank": rank,
        "input_tokens": input_tokens,
        "loss_tokens": loss_tokens,
        "global_batch_sha256": digest,
        "samples": samples,
        "effective_engine_config": asdict(actor),
        "provenance": provenance(),
        "device": {
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "uuid": str(getattr(props, "uuid", "unavailable")),
            "free_memory_bytes_before_initialization": initial_free_bytes,
            "cuda_runtime": torch.version.cuda,
            "initial_seed": config["seed"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    write_json(Path(config["trial_dir"]) / f"rank-{rank}.json", result)
    engine.destroy()


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text())
    try:
        run(config)
    except BaseException as exc:
        write_json(
            Path(config["trial_dir"])
            / f"failure-{os.environ.get('RANK', 'unknown')}.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
