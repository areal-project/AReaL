#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""torchrun worker for staged Muon parallel numerical parity."""

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


def _make_parallel_groups(tp_size: int) -> tuple[dist.ProcessGroup, dist.ProcessGroup]:
    world_size = dist.get_world_size()
    if world_size % tp_size != 0:
        raise ValueError(
            f"world size {world_size} is not divisible by TP size {tp_size}"
        )

    rank = dist.get_rank()
    tp_group = None
    for dp_rank in range(world_size // tp_size):
        ranks = list(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            tp_group = group

    dp_group = None
    for tp_rank in range(tp_size):
        ranks = list(range(tp_rank, world_size, tp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            dp_group = group

    assert tp_group is not None
    assert dp_group is not None
    return tp_group, dp_group


def _make_singleton_groups() -> dist.ProcessGroup:
    rank = dist.get_rank()
    local_group = None
    for group_rank in range(dist.get_world_size()):
        group = dist.new_group([group_rank])
        if rank == group_rank:
            local_group = group
    assert local_group is not None
    return local_group


def _run_case(
    *,
    name: str,
    tp_group: dist.ProcessGroup,
    expt_tp_group: dist.ProcessGroup,
    dp_group: dist.ProcessGroup,
    shape: tuple[int, int],
    partition_dim: int,
    expert_tp: bool = False,
    split_qkv: bool = False,
) -> None:
    tp_rank = dist.get_rank(expt_tp_group if expert_tp else tp_group)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    generator = torch.Generator(device=device).manual_seed(
        20260831 + sum(ord(char) for char in name) + tp_rank
    )
    initial = torch.randn(
        shape, generator=generator, device=device, dtype=torch.bfloat16
    )
    staged_param = torch.nn.Parameter(initial.clone())
    baseline_param = torch.nn.Parameter(initial.float())
    for param in (staged_param, baseline_param):
        param.tensor_model_parallel = not expert_tp
        param.partition_dim = partition_dim
        param.partition_stride = 1
        param.expert_tp = expert_tp
        param.is_qkv = split_qkv

    pg_collection = SimpleNamespace(tp=tp_group, expt_tp=expt_tp_group)
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
        "split_qkv": split_qkv,
        "is_qkv_fn": lambda param: getattr(param, "is_qkv", False),
        "qkv_split_shapes": (4, 2, 2) if split_qkv else None,
    }
    param_group = {
        "params": [baseline_param],
        "is_expert_parallel": expert_tp,
    }
    baseline = TensorParallelMuon([param_group], **kwargs)
    staged = GPUStagedMuon(
        [
            {
                "params": [staged_param],
                "lr": kwargs["lr"],
                "momentum": kwargs["momentum_beta"],
                "weight_decay": kwargs["weight_decay"],
                "is_expert_parallel": expert_tp,
            }
        ],
        staged_config=GPUStagedMuonConfig(
            buffer_count=1,
            slot_size_mb=staged_param.numel() * 4 / (1024 * 1024),
        ),
        orthogonalize=baseline.orthogonalize,
        matmul_precision=lambda: fp32_matmul_precision(baseline.fp32_matmul_prec),
        nesterov=baseline.nesterov,
        weight_decay_method=baseline.weight_decay_method,
        native_optimizer=baseline,
    )
    staged.bind_parallel_groups(tp=tp_group, expt_tp=expt_tp_group)
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
        if dist.get_world_size(dp_group) > 1:
            replicas = [
                torch.empty_like(staged_param)
                for _ in range(dist.get_world_size(dp_group))
            ]
            dist.all_gather(replicas, staged_param.detach(), group=dp_group)
            for replica in replicas[1:]:
                torch.testing.assert_close(replica, replicas[0], rtol=0.0, atol=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    try:
        tp_group, dp_group = _make_parallel_groups(args.tp_size)
        singleton_group = _make_singleton_groups()
        for partition_dim, shape in ((0, (5, 7)), (1, (7, 5))):
            _run_case(
                name=f"dense-axis-{partition_dim}",
                tp_group=tp_group,
                expt_tp_group=tp_group,
                dp_group=dp_group,
                shape=shape,
                partition_dim=partition_dim,
            )
        _run_case(
            name="qkv-split",
            tp_group=tp_group,
            expt_tp_group=tp_group,
            dp_group=dp_group,
            shape=(16, 6),
            partition_dim=0,
            split_qkv=True,
        )
        _run_case(
            name="expert-tp",
            tp_group=singleton_group,
            expt_tp_group=tp_group,
            dp_group=dp_group,
            shape=(6, 7),
            partition_dim=0,
            expert_tp=True,
        )
        dist.barrier()
        if dist.get_rank() == 0:
            with open(args.output, "w") as file:
                file.write("Passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
