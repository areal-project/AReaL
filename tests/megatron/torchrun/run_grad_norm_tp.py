#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""torchrun worker: run one train step under a given TP degree and report the
global grad norm plus whether duplicated params were correctly demoted.

Launched by tests/megatron/test_grad_norm_tp_invariance.py with nproc == tp.
The parent compares grad norms across TP degrees: a correct implementation
keeps the global grad norm TP-invariant, while double-counting replicated
params (tensor_model_parallel left True) inflates it as TP grows.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    MegatronEngineConfig,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.engine import MegatronEngine
from areal.infra.platforms import current_platform

VOCAB_SIZE = 100
BATCH_SIZE = 4
SEQLEN = 16


def _local_model_path() -> str:
    local = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
    return local if os.path.isdir(local) else "Qwen/Qwen3-0.6B"


def _deterministic_input() -> dict:
    # Fully deterministic and identical regardless of TP degree (dp == 1).
    device = current_platform.device_type
    ids = (torch.arange(BATCH_SIZE * SEQLEN, dtype=torch.long) % VOCAB_SIZE).reshape(
        BATCH_SIZE, SEQLEN
    )
    return dict(
        input_ids=ids.to(device),
        attention_mask=torch.ones(BATCH_SIZE, SEQLEN, dtype=torch.bool, device=device),
    )


def _loss_fn(logprobs, entropy, input_data, **kwargs):
    return torch.mean(logprobs)


def _check_duplicated_params_demoted(engine) -> tuple[bool, int]:
    """On a real model, every param marked _is_duplicated must have had its
    tensor_model_parallel flag cleared. Returns (ok, num_duplicated)."""
    n_dup = 0
    for model in engine.model:
        for _, param in model.named_parameters():
            if getattr(param, "_is_duplicated", False):
                n_dup += 1
                if getattr(param, "tensor_model_parallel", False):
                    return False, n_dup
    return True, n_dup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    model_path = _local_model_path()
    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test",
        trial_name="test",
        path=model_path,
        optimizer=OptimizerConfig(gradient_clipping=1e9),  # clip off; keep raw norm
        megatron=MegatronEngineConfig(),
    )
    alloc_mode = ModelAllocation.from_str(f"fsdp:d1p1t{args.tp}")
    ft_spec = FinetuneSpec(
        total_train_epochs=1, dataset_size=128, train_batch_size=BATCH_SIZE
    )

    engine = MegatronEngine(config)
    engine.create_process_group(alloc_mode.parallel)
    engine.initialize(addr=None, ft_spec=ft_spec)
    try:
        structural_ok, n_dup = _check_duplicated_params_demoted(engine)

        engine.train()
        stats = engine.train_batch(
            _deterministic_input(),
            loss_fn=_loss_fn,
            loss_weight_fn=lambda x: torch.tensor(1.0, device=engine.device),
        )
        grad_norm = float(stats["grad_norm"])

        if dist.get_rank() == 0:
            with open(args.output, "w") as f:
                json.dump(
                    {
                        "tp": args.tp,
                        "grad_norm": grad_norm,
                        "structural_ok": structural_ok,
                        "num_duplicated": n_dup,
                    },
                    f,
                )
    finally:
        engine.destroy()


if __name__ == "__main__":
    main()
