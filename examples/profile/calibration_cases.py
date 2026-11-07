# SPDX-License-Identifier: Apache-2.0
"""Frozen shape catalog for component calibration, independent of GPU layout."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ActorCase:
    name: str
    lengths: tuple[int, ...]


@dataclass(frozen=True)
class RolloutCase:
    name: str
    prompt: int
    output: int
    concurrency: int


ACTOR_CASES = tuple(
    [ActorCase(f"length-{n}", (n,)) for n in (1024, 4096, 16384, 32768)]
    + [ActorCase(f"batch-{b}", (4096,) * b) for b in (2, 4, 8)]
    + [ActorCase("mixed", (1024, 1024, 2048, 12288))]
)
ROLLOUT_CASES = tuple(
    [
        RolloutCase(f"prefill-{p}-c{c}", p, 32, c)
        for p in (512, 4096, 16384, 32768)
        for c in (1, 32)
    ]
    + [
        RolloutCase(f"decode-{p}-c{c}", p, 2048, c)
        for p in (1024, 16384)
        for c in (1, 8, 32, 64)
    ]
)


def select_cases(component: str, selection: str):
    catalog = ACTOR_CASES if component == "actor" else ROLLOUT_CASES
    pilot = {
        "actor": ("length-1024", "length-4096", "batch-4", "length-16384"),
        "rollout": ("prefill-4096-c1", "prefill-16384-c32", "decode-1024-c32"),
    }
    names = pilot[component] if selection == "pilot" else selection.split(",")
    if selection == "all":
        return catalog
    available = {case.name: case for case in catalog}
    if not set(names) <= available.keys():
        raise ValueError(f"Unknown cases: {set(names) - available.keys()}")
    return tuple(available[name] for name in names)


def case_seed(seed: int, name: str, dp_rank: int = 0, index: int = 0) -> int:
    identity = json.dumps([seed, name, dp_rank, index]).encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def make_actor_batch(case: ActorCase, *, seed: int, dp_rank: int, device="cpu"):
    """One independently rewarded trajectory per sequence, with raw token masks."""
    import torch

    data = []
    for i, length in enumerate(case.lengths):
        generator = torch.Generator().manual_seed(
            case_seed(seed, case.name, dp_rank, i)
        )
        ids = torch.randint(1000, 50000, (1, length), generator=generator)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[:, length - length // 8 :] = True
        item = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(mask),
            "loss_mask": mask,
            "rewards": torch.tensor([float(i % 2)], dtype=torch.float32),
            "versions": torch.where(mask, 0, -1),
        }
        data.append({key: value.to(device) for key, value in item.items()})
    return data


def catalog_manifest() -> dict:
    return {
        "schema_version": 1,
        "actor": [asdict(case) for case in ACTOR_CASES],
        "rollout": [asdict(case) for case in ROLLOUT_CASES],
        "actor_loss_fraction": "last 1/8 of each sequence",
        "actor_grouping": "one sequence per independently rewarded trajectory",
    }
