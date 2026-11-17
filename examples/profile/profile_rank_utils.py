# SPDX-License-Identifier: Apache-2.0

"""Rank selection helpers for profile scripts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import yaml

from areal.api.alloc_mode import ModelAllocation

_PP_RANK0_ALIASES = frozenset({"pp_rank0", "pp-rank0", "auto_pp", "auto-pp"})


def actor_backend_from_config(config_path: Path) -> str:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    actor_config = config.get("actor") or {}
    backend = actor_config.get("backend") or config.get("allocation_mode")
    if not backend:
        raise ValueError(
            f"Could not find actor.backend or allocation_mode in {config_path}"
        )
    return str(backend)


def pp_rank0_ranks_from_backend(backend: str) -> str:
    allocation = ModelAllocation.from_str(backend)
    pp_size = allocation.parallel.pipeline_parallel_size
    world_size = allocation.parallel.world_size
    if world_size % pp_size != 0:
        raise ValueError(
            f"World size {world_size} must be divisible by pp size {pp_size}"
        )
    pp_stage_stride = world_size // pp_size
    return ",".join(str(pp_rank * pp_stage_stride) for pp_rank in range(pp_size))


def pp_rank0_ranks_from_config(config_path: Path) -> str:
    return pp_rank0_ranks_from_backend(actor_backend_from_config(config_path))


def resolve_profile_ranks(profile_ranks: str, config_path: Path) -> str:
    if profile_ranks not in _PP_RANK0_ALIASES:
        return profile_ranks
    return pp_rank0_ranks_from_config(config_path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve profile rank selection.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile-ranks", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    print(resolve_profile_ranks(args.profile_ranks, args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
