#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""torchrun worker for staged Muon TP numerical parity."""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from emerging_optimizers.utils import fp32_matmul_precision
from megatron.core.optimizer.muon import TensorParallelMuon

from areal.engine.megatron_utils.gpu_staged_muon import (
    GPUStagedMuon,
    GPUStagedMuonConfig,
)


def _run_partition_case(partition_dim: int) -> None:
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    shape = (5, 7) if partition_dim == 0 else (7, 5)
    generator = torch.Generator(device=device).manual_seed(
        20260831 + 10 * partition_dim + rank
    )
    initial = torch.randn(
        shape, generator=generator, device=device, dtype=torch.bfloat16
    )
    staged_param = torch.nn.Parameter(initial.clone())
    baseline_param = torch.nn.Parameter(initial.float())
    for param in (staged_param, baseline_param):
        param.tensor_model_parallel = True
        param.partition_dim = partition_dim
        param.partition_stride = 1

    pg_collection = SimpleNamespace(tp=dist.group.WORLD, expt_tp=dist.group.WORLD)
    kwargs = {
        "lr": 0.025,
        "momentum_beta": 0.82,
        "use_nesterov": True,
        "weight_decay": 0.015,
        "use_decoupled_weight_decay": True,
        "fp32_matmul_prec": "highest",
        "num_ns_steps": 3,
        "pg_collection": pg_collection,
        "mode": "duplicated",
    }
    baseline = TensorParallelMuon([baseline_param], **kwargs)
    staged = GPUStagedMuon(
        [
            {
                "params": [staged_param],
                "lr": kwargs["lr"],
                "momentum": kwargs["momentum_beta"],
                "weight_decay": kwargs["weight_decay"],
            }
        ],
        staged_config=GPUStagedMuonConfig(
            buffer_count=1,
            slot_size_mb=staged_param.numel() * 4 / (1024 * 1024),
        ),
        orthogonalize=baseline.orthogonalize,
        matmul_precision=lambda: fp32_matmul_precision(
            baseline.fp32_matmul_prec
        ),
        nesterov=baseline.nesterov,
        weight_decay_method=baseline.weight_decay_method,
    )
    staged.bind_parallel_groups(tp=dist.group.WORLD, expt_tp=dist.group.WORLD)
    staged.bind_owned_params(staged.param_groups)

    for step in range(3):
        grad = torch.randn(
            shape, generator=generator, device=device, dtype=torch.bfloat16
        ).mul_(step + 1)
        staged_param.decoupled_grad = grad
        baseline_param.grad = grad.float()
        baseline.step()
        staged.step()
        staged.drain()

        staged_state = staged.state[staged_param]
        baseline_state = baseline.state[baseline_param]
        torch.testing.assert_close(
            staged_state["master_param"],
            baseline_param.detach().cpu(),
            rtol=3e-6,
            atol=3e-6,
        )
        torch.testing.assert_close(
            staged_state["momentum_buffer"],
            baseline_state["momentum_buffer"].cpu(),
            rtol=3e-6,
            atol=3e-6,
        )
        torch.testing.assert_close(
            staged_param,
            baseline_param.detach().bfloat16(),
            rtol=0.0,
            atol=0.0,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        for partition_dim in (0, 1):
            _run_partition_case(partition_dim)
        dist.barrier()
        if dist.get_rank() == 0:
            with open(args.output, "w") as file:
                file.write("Passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
