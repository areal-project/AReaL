# SPDX-License-Identifier: Apache-2.0

"""Acceptance-only torchrun worker for the GPU-staged AdamW prototype.

This file intentionally lives under ``tests/`` and does not provide a production
entry point.  It exercises a real Megatron-Core engine with either the ordinary
or staged optimizer and writes per-rank residency/memory evidence as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    MegatronEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.engine import MegatronEngine
from areal.engine.megatron_utils.gpu_staged_optimizer import (
    GPUStagedAdamWConfig,
)
from areal.utils import seeding
from areal.utils.data import broadcast_tensor_container


def _memory(label: str) -> dict[str, Any]:
    free, total = torch.cuda.mem_get_info()
    return {
        "label": label,
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "max_allocated": torch.cuda.max_memory_allocated(),
        "driver_used": total - free,
    }


def _iter_inner_optimizers(optimizer: Any):
    if hasattr(optimizer, "chained_optimizers"):
        optimizers = optimizer.chained_optimizers
    else:
        optimizers = [optimizer]
    for wrapper in optimizers:
        inner = getattr(wrapper, "optimizer", None)
        if inner is not None:
            yield inner


def _residency_evidence(engine: MegatronEngine) -> list[dict[str, Any]]:
    evidence = []
    for inner in _iter_inner_optimizers(engine.optimizer):
        managed = bool(getattr(inner, "manages_cpu_residency", False))
        item: dict[str, Any] = {
            "class": type(inner).__name__,
            "managed": managed,
            "cuda_state_numel": sum(
                tensor.numel()
                for state in inner.state.values()
                for tensor in state.values()
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda
            ),
        }
        if managed:
            inner.drain()
            slabs = inner.cpu_slabs
            item.update(
                {
                    "residency": inner.residency,
                    "owned_numel": slabs.master.numel(),
                    "staging_state_numel": inner.gpu_staging_state_numel,
                    "slab": {
                        name: {
                            "device": str(tensor.device),
                            "dtype": str(tensor.dtype),
                            "pinned": tensor.is_pinned(),
                            "storage_ptr": tensor.untyped_storage().data_ptr(),
                        }
                        for name, tensor in (
                            ("master", slabs.master),
                            ("exp_avg", slabs.exp_avg),
                            ("exp_avg_sq", slabs.exp_avg_sq),
                        )
                    },
                }
            )
            for param, state in inner.state.items():
                for key in ("master_param", "exp_avg", "exp_avg_sq"):
                    tensor = state[key]
                    assert tensor.device.type == "cpu"
                    assert tensor.dtype is torch.float32
                    assert tensor.is_pinned()
                    slab = getattr(slabs, "master" if key == "master_param" else key)
                    assert tensor.untyped_storage().data_ptr() == (
                        slab.untyped_storage().data_ptr()
                    )
        evidence.append(item)
    return evidence


def _mock_input(batch_size: int, sequence_length: int, device: torch.device):
    lengths = torch.full((batch_size,), sequence_length, device=device, dtype=torch.int)
    input_ids = torch.randint(
        128, 1024, (batch_size, sequence_length), device=device, dtype=torch.long
    )
    attention_mask = torch.arange(sequence_length, device=device).unsqueeze(
        0
    ) < lengths.unsqueeze(1)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _loss_fn(logprobs: torch.Tensor, entropy: torch.Tensor, input_data: dict, **kwargs):
    del entropy, input_data, kwargs
    return logprobs.float().mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--max-tokens-per-mb", type=int, default=16)
    parser.add_argument("--bucket-size-mb", type=float, default=1.0)
    parser.add_argument("--bridge-type", default="mbridge")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.reset_peak_memory_stats()
    records = [_memory("before_engine")]
    config = TrainEngineConfig(
        backend=args.backend,
        experiment_name="gpu-staged-acceptance",
        trial_name="gpu-staged-acceptance",
        path=args.model_path,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=args.max_tokens_per_mb),
        optimizer=OptimizerConfig(),
        megatron=MegatronEngineConfig(bridge_type=args.bridge_type),
    )
    engine = MegatronEngine(config)
    if args.staged:
        engine.configure_gpu_staged_adamw(
            GPUStagedAdamWConfig(buffer_count=2, bucket_size_mb=args.bucket_size_mb)
        )
    original_create_optimizer = engine._create_optimizer

    def tracked_create_optimizer(self: MegatronEngine, ft_spec: FinetuneSpec) -> None:
        del self
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        records.append(_memory("optimizer_before"))
        original_create_optimizer(ft_spec)
        torch.cuda.synchronize()
        records.append(_memory("optimizer_after"))

    engine._create_optimizer = MethodType(tracked_create_optimizer, engine)
    allocation = ModelAllocation.from_str(args.backend)
    ft_spec = FinetuneSpec(
        total_train_epochs=1,
        dataset_size=max(args.batch_size, 4),
        train_batch_size=args.batch_size,
    )
    engine.create_process_group(parallel_strategy=allocation.parallel)
    records.append(_memory("before_initialize"))
    engine.initialize(addr=None, ft_spec=ft_spec)
    torch.cuda.synchronize()
    records.append(_memory("after_initialize"))
    initial_residency = _residency_evidence(engine)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_payload(results: list[dict[str, Any]], error: str | None = None) -> None:
        payload = {
            "rank": rank,
            "world_size": dist.get_world_size(),
            "backend": args.backend,
            "staged": args.staged,
            "device_name": torch.cuda.get_device_name(),
            "memory": records,
            "initial_residency": initial_residency,
            "steps": results,
            "error": error,
        }
        (output_dir / f"rank_{rank}.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n"
        )

    write_payload([])

    results = []
    if not args.init_only:
        seeding.set_random_seed(1234, key=f"acceptance-data-{rank}")
        input_ = _mock_input(args.batch_size, args.sequence_length, engine.device)
        input_ = broadcast_tensor_container(
            input_,
            src_rank=engine.current_data_parallel_head(),
            group=engine.context_and_model_parallel_group,
        )
        for step in range(args.steps):
            torch.cuda.reset_peak_memory_stats()
            result = engine.train_batch(
                input_=input_,
                loss_fn=_loss_fn,
                loss_weight_fn=lambda data: data["cu_seqlens"][-1],
            )
            records.append(_memory(f"step_{step}_returned"))
            residency = _residency_evidence(engine)
            torch.cuda.synchronize()
            records.append(_memory(f"step_{step}_drained"))
            results.append({"result": result, "residency": residency})

    dist.barrier()
    write_payload(results)
    engine.destroy()


if __name__ == "__main__":
    main()
