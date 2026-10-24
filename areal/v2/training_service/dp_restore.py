# SPDX-License-Identifier: Apache-2.0

"""Explicit FSDP DP-only sidecar mapping; native DCP owns tensor resharding."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areal.utils.atomic import recovery_digest

from .runtime_state import reseed_runtime_state


@dataclass(frozen=True)
class DPStateRestorePlan:
    """Bind a new rank layout to one immutable run and source checkpoint."""

    identity_sha256: str
    checkpoint_generation: str
    policy_version: int
    source_world_size: int
    target_world_size: int
    transition_id: str
    rng_policy: str = "fsdp-dp-reseed-v1"

    def __post_init__(self) -> None:
        if any(
            not re.fullmatch(r"[a-f0-9]{64}", value)
            for value in (self.identity_sha256, self.transition_id)
        ):
            raise ValueError(
                "DP restore requires complete run and transition identities"
            )
        if any(
            type(value) is not int or value < 1
            for value in (
                self.policy_version,
                self.source_world_size,
                self.target_world_size,
            )
        ):
            raise ValueError("DP restore requires positive versions and world sizes")
        if self.source_world_size == self.target_world_size:
            raise ValueError(
                "Use ordinary full checkpoint loading for the same DP size"
            )
        if (
            not re.fullmatch(r"step-[A-Za-z0-9-]+", self.checkpoint_generation)
            or self.rng_policy != "fsdp-dp-reseed-v1"
        ):
            raise ValueError("Unsupported DP checkpoint generation or RNG policy")


def mapped_runtime_state(
    engine: Any, *, path: Path, identity_sha256: str, version: int, plan: dict[str, Any]
) -> dict[str, Any]:
    """Validate every source rank before any target rank enters collective load."""
    approved = DPStateRestorePlan(**plan)
    if (
        approved.identity_sha256,
        approved.policy_version,
        approved.target_world_size,
        approved.checkpoint_generation,
    ) != (
        identity_sha256,
        version,
        engine.world_size,
        path.name,
    ) or not 0 <= engine.rank < engine.world_size:
        raise ValueError("DP restore plan does not match this checkpoint/target worker")
    manifest = json.loads((path / "manifest.json").read_text())
    if (
        manifest["format_version"],
        manifest["global_step"],
        manifest["world_size"],
        recovery_digest(manifest["compatibility"]),
    ) != (1, version, approved.source_world_size, identity_sha256):
        raise ValueError("DP restore source manifest does not match the approved run")
    from areal.api.alloc_mode import ModelAllocation

    layout_path = path / "actor-layout.json"
    source_backend = (
        json.loads(layout_path.read_text())["backend"]
        if layout_path.exists()
        else manifest["compatibility"]["actor"]["backend"]
    )
    source = ModelAllocation.from_str(source_backend)
    target = engine.parallel_helper._ps
    if (
        source.backend != "fsdp"
        or source.parallel.world_size != approved.source_world_size
        or target.world_size != approved.target_world_size
        or any(
            getattr(strategy, name) != 1
            for strategy in (source.parallel, target)
            for name in ("tp_size", "pp_size", "cp_size", "ep_size", "etp_size")
        )
    ):
        raise ValueError("DP restore only supports pure FSDP data parallel layouts")
    states = []
    for rank in range(approved.source_world_size):
        state = json.loads((path / f"runtime-{rank}.json").read_text())
        if (
            state["format_version"],
            state["rank"],
            state["world_size"],
            state["policy_version"],
            state["backend"],
            state["identity_sha256"],
        ) != (1, rank, approved.source_world_size, version, "fsdp", identity_sha256):
            raise ValueError("DP restore source rank identity mismatch")
        if (engine.device.type == "cuda") != (state["cuda_rng"] is not None):
            raise ValueError("DP restore source/target device types differ")
        states.append(state)
    common = states[0]
    for state in states[1:]:
        for key in ("lr_scheduler", "optimizer_group_lrs"):
            if recovery_digest(state[key]) != recovery_digest(common[key]):
                raise ValueError("DP restore source ranks disagree on scheduler/LRs")
    material = f"{approved.rng_policy}:{identity_sha256}:{approved.transition_id}:{path.name}:{engine.world_size}:{engine.rank}"
    seed = (
        int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
        % (2**31 - 1)
        + 1
    )
    return reseed_runtime_state(engine, common, seed)
