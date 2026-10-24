# SPDX-License-Identifier: Apache-2.0

"""Real DCP 2→4→2 acceptance: CPU DTensor or CUDA FSDP2, no model download."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from areal.api.alloc_mode import ModelAllocation
from areal.engine.fsdp_utils.checkpoint import DCPState
from areal.utils.atomic import atomic_json, recovery_digest
from areal.v2.training_service.checkpoint import (
    inspect_full_checkpoint,
    load_full_checkpoint,
    save_full_checkpoint,
)
from areal.v2.training_service.dp_restore import DPStateRestorePlan


class Engine:
    def __init__(self, device, mesh):
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        self.device = device
        weights = (
            torch.arange(128, dtype=torch.float32, device=device).reshape(16, 8) / 128
        )
        self.model = torch.nn.Linear(8, 16, bias=False, device=device)
        self.model.weight.data.copy_(weights)
        if device.type == "cuda":
            from torch.distributed.fsdp import fully_shard

            fully_shard(self.model, mesh=mesh)
            self.parallel_helper = SimpleNamespace(
                _ps=ModelAllocation.from_str(f"fsdp:d{self.world_size}").parallel,
                world_mesh={
                    "tp": init_device_mesh(
                        "cuda", (self.world_size, 1), mesh_dim_names=("dp", "tp")
                    )["tp"]
                },
            )
        else:
            self.model.weight = torch.nn.Parameter(
                distribute_tensor(weights, mesh, [Shard(0)])
            )
            self.parallel_helper = SimpleNamespace(
                _ps=ModelAllocation.from_str(f"fsdp:d{self.world_size}").parallel
            )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.01, foreach=False
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.9
        )
        self.version = 0

    def get_version(self):
        return self.version

    def set_version(self, version):
        self.version = version

    def save(self, meta):
        dcp.save({"dcp": DCPState(self.model, self.optimizer)}, checkpoint_id=meta.path)

    def load(self, meta):
        dcp.load({"dcp": DCPState(self.model, self.optimizer)}, checkpoint_id=meta.path)

    def step(self):
        self.optimizer.zero_grad()
        if self.device.type == "cuda":
            self.model(
                torch.full((4, 8), float(self.version + 1), device=self.device)
            ).square().mean().backward()
        else:
            # Real sharded Adam update; CPU mode validates DCP/DTensor rather
            # than claiming FSDP execution on a machine without accelerators.
            self.model.weight.grad = torch.ones_like(self.model.weight) * (
                self.version + 1
            )
        self.optimizer.step()
        self.lr_scheduler.step()
        self.version += 1

    def snapshot(self):
        def full(value):
            if isinstance(value, DTensor):
                value = value.full_tensor()
            return (
                value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else value
            )

        return {
            "model": {
                name: full(param) for name, param in self.model.named_parameters()
            },
            "optimizer": {
                name: {
                    key: full(value)
                    for key, value in self.optimizer.state[param].items()
                }
                for name, param in self.model.named_parameters()
            },
            "scheduler": self.lr_scheduler.state_dict(),
            "lrs": [group["lr"] for group in self.optimizer.param_groups],
            "version": self.version,
        }


def compare(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            compare(actual[key], expected[key])
    else:
        assert actual == expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    if args.device == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device(
        "cpu" if args.device == "cpu" else f"cuda:{os.environ['LOCAL_RANK']}"
    )
    dist.init_process_group("gloo" if args.device == "cpu" else "nccl")
    try:
        mesh = init_device_mesh(args.device, (dist.get_world_size(),))
        engine = Engine(device, mesh)
        compatibility = {
            "actor": {"backend": "fsdp:d2"},
            "fixture": "native-dp-portability-v1",
        }
        identity = recovery_digest(compatibility)
        if args.stage > 1:
            source = args.root / f"step-{args.stage - 1:08d}-fixture"
            manifest = json.loads((source / "manifest.json").read_text())
            plan = asdict(
                DPStateRestorePlan(
                    identity,
                    source.name,
                    args.stage - 1,
                    manifest["world_size"],
                    engine.world_size,
                    "a" * 64,
                )
            )
            metadata = dict(
                path=str(source),
                backend="fsdp",
                identity_sha256=identity,
                version=args.stage - 1,
                restore_plan=plan,
            )
            inspected = inspect_full_checkpoint(engine, loading=True, **metadata)
            assert load_full_checkpoint(engine, **metadata) == inspected
            expected = torch.load(source / "reference.pt", weights_only=True)
            compare(engine.snapshot(), expected)
        engine.step()
        target = args.root / f"step-{args.stage:08d}-fixture"
        if engine.rank == 0:
            target.mkdir()
        dist.barrier()
        save_full_checkpoint(
            engine,
            path=str(target),
            backend="fsdp",
            identity_sha256=identity,
            version=args.stage,
        )
        snapshot = engine.snapshot()
        if engine.rank == 0:
            torch.save(snapshot, target / "reference.pt")
            atomic_json(
                target / "manifest.json",
                {
                    "format_version": 1,
                    "global_step": args.stage,
                    "world_size": engine.world_size,
                    "compatibility": compatibility,
                },
            )
            atomic_json(
                target / "actor-layout.json",
                {
                    "backend": f"fsdp:d{engine.world_size}",
                    "world_size": engine.world_size,
                    "gpu_pool": [0, 1, 2, 3],
                },
            )
        dist.barrier()
        # Once a target checkpoint exists, ordinary same-layout loading works.
        metadata = dict(
            path=str(target),
            backend="fsdp",
            identity_sha256=identity,
            version=args.stage,
        )
        inspected = inspect_full_checkpoint(engine, loading=True, **metadata)
        assert load_full_checkpoint(engine, **metadata) == inspected
        compare(engine.snapshot(), snapshot)
        if engine.rank == 0:
            atomic_json(
                args.root / f"stage-{args.stage}.json",
                {
                    "world_size": engine.world_size,
                    "version": engine.version,
                    "device": args.device,
                    "tensor_optimizer_restore": "passed",
                },
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
