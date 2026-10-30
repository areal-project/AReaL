# SPDX-License-Identifier: Apache-2.0

"""Batch-boundary host-memory observations for routing regressions."""

from pathlib import Path
from typing import Any

import torch

from areal.engine.r3.packed import PackedRouting


def routing_memory_snapshot(
    routes: list[torch.Tensor | PackedRouting],
) -> dict[str, int]:
    """Count unique backing storages, not the repeated microbatch views."""
    storages = {}
    for route in routes:
        values = route.values if isinstance(route, PackedRouting) else route
        storage = values.untyped_storage()
        storages[(str(values.device), storage.data_ptr())] = storage.nbytes()
    metrics = {"cpu_routing_storage_bytes": sum(storages.values())}
    try:
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("Rss", "Pss", "Private_Dirty"):
                metrics[f"host_{key.lower()}_bytes"] = int(value.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return metrics


def log_routing_memory(logger: Any, metrics: dict[str, int]) -> None:
    logger.info("R3 CPU batch memory before forward: %s", metrics)
