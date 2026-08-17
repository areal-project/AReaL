# SPDX-License-Identifier: Apache-2.0

"""Real NCCL coverage for staged Muon's empty-owner all-gather adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from areal.engine.megatron_utils.gpu_staged_muon import (
    _freeze_owner_schema,
    _make_staged_layerwise_class,
)


def _param(value: float) -> torch.nn.Parameter:
    return torch.nn.Parameter(
        torch.full((2, 2), value, dtype=torch.bfloat16, device="cuda")
    )


def _run_schema(
    dense: list[list[torch.Tensor]],
    expert: list[list[torch.Tensor]] | None,
) -> tuple[int, list[str]]:
    wrapper_cls = _make_staged_layerwise_class()
    wrapper = object.__new__(wrapper_cls)
    wrapper.pg_collection = SimpleNamespace(
        dp_cp=dist.group.WORLD,
        expt_dp=dist.group.WORLD,
    )
    wrapper.dp_cp_params_list = dense
    wrapper.expt_dp_params_list = expert
    wrapper._staged_owner_schema = {
        "dense": _freeze_owner_schema("dense", dense),
        "expert": _freeze_owner_schema("expert", expert),
    }
    original = dist.all_gather
    domains: list[str] = []

    def traced_all_gather(*args: Any, **kwargs: Any) -> Any:
        domains.append("collective")
        return original(*args, **kwargs)

    dist.all_gather = traced_all_gather
    try:
        wrapper.allgather_params()
    finally:
        dist.all_gather = original
    return len(domains), domains


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    scenario = os.environ["MUON_ALLGATHER_SCENARIO"]
    output_dir = Path(os.environ["ACCEPTANCE_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        params: list[torch.nn.Parameter] = []
        if scenario == "all_empty":
            if world_size != 2:
                raise RuntimeError("all_empty needs world size 2")
            dense: list[list[torch.Tensor]] = [[], []]
            expert = None
            expected_collectives = 0
        elif scenario == "first_empty":
            if world_size != 2:
                raise RuntimeError("first_empty needs world size 2")
            value = _param(11.0 if rank == 1 else -1.0)
            params = [value]
            dense = [[], [value]]
            expert = None
            expected_collectives = 1
        elif scenario == "middle_empty":
            if world_size != 3:
                raise RuntimeError("middle_empty needs world size 3")
            first = _param(10.0 if rank == 0 else -1.0)
            last = _param(20.0 if rank == 2 else -2.0)
            params = [first, last]
            dense = [[first], [], [last]]
            expert = None
            expected_collectives = 1
        elif scenario == "two_domains":
            if world_size != 4:
                raise RuntimeError("two_domains needs world size 4")
            dense_param = _param(10.0 if rank == 2 else -1.0)
            expert_param = _param(20.0 if rank == 1 else -2.0)
            params = [dense_param, expert_param]
            dense = [[], [], [dense_param], []]
            expert = [[], [expert_param], [], []]
            expected_collectives = 2
        else:
            raise RuntimeError(f"unknown all-gather scenario {scenario!r}")

        collective_count, _ = _run_schema(dense, expert)
        if collective_count != expected_collectives:
            raise AssertionError(
                f"expected {expected_collectives} collectives, got {collective_count}"
            )
        expected_values = {
            "first_empty": (11.0,),
            "middle_empty": (10.0, 20.0),
            "two_domains": (10.0, 20.0),
        }.get(scenario, ())
        for param, expected in zip(params, expected_values, strict=True):
            torch.testing.assert_close(
                param,
                torch.full_like(param, expected),
                rtol=0.0,
                atol=0.0,
            )
        health = torch.ones(1, device="cuda")
        dist.all_reduce(health, group=dist.group.WORLD)
        if health.item() != world_size:
            raise AssertionError("post-all-gather NCCL health probe failed")
        (output_dir / f"rank_{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "scenario": scenario,
                    "collective_count": collective_count,
                    "values": [float(param.flatten()[0].item()) for param in params],
                    "health": int(health.item()),
                },
                sort_keys=True,
            )
            + "\n"
        )
        dist.barrier(group=dist.group.WORLD)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
