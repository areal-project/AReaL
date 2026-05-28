# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import aiohttp

from areal.utils import logging

logger = logging.getLogger("TrainDataProxy")


@dataclass
class WorkerInfo:
    addr: str
    rank: int = 0
    world_size: int = 1
    dp_rank: int = 0
    dp_size: int = 1
    is_dp_head: bool = True
    local_rank: int = 0


@dataclass
class WorkerTopology:
    workers: list[WorkerInfo] = field(default_factory=list)
    dp_heads: list[int] = field(default_factory=list)
    dp_size: int = 1
    dp_groups: list[list[int]] = field(default_factory=list)
    pp_size: int = 1
    tp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1


async def discover_topology(
    worker_addrs: list[str],
    timeout: float = 10.0,
) -> WorkerTopology:
    workers: list[WorkerInfo] = []
    meta: dict[str, int] = {}

    async def _fetch(session: aiohttp.ClientSession, addr: str) -> dict:
        async with session.get(f"{addr}/topology") as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(
                    f"Failed to discover topology from {addr}: "
                    f"HTTP {resp.status}: {text}"
                )
            return await resp.json(content_type=None)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout)
    ) as session:
        tasks = [_fetch(session, addr) for addr in worker_addrs]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    for i, resp in enumerate(responses):
        addr = worker_addrs[i]
        if isinstance(resp, BaseException):
            raise RuntimeError(f"Failed to discover topology from {addr}: {resp}")
        data = resp
        if not meta:
            meta = {
                "pp_size": int(data.get("pp_size", 1)),
                "tp_size": int(data.get("tp_size", 1)),
                "cp_size": int(data.get("cp_size", 1)),
                "ep_size": int(data.get("ep_size", 1)),
            }
        workers.append(
            WorkerInfo(
                addr=addr,
                rank=data.get("rank", 0),
                world_size=data.get("world_size", 1),
                dp_rank=data.get("dp_rank", 0),
                dp_size=data.get("dp_size", 1),
                is_dp_head=data.get("is_dp_head", True),
                local_rank=data.get("local_rank", 0),
            )
        )

    dp_heads = [i for i, w in enumerate(workers) if w.is_dp_head]
    dp_size = workers[0].dp_size if workers else 1
    dp_groups: list[list[int]] = [[] for _ in range(max(dp_size, 1))]
    for i, w in enumerate(workers):
        if w.dp_rank >= len(dp_groups):
            dp_groups.extend([] for _ in range(w.dp_rank - len(dp_groups) + 1))
        dp_groups[w.dp_rank].append(i)

    return WorkerTopology(
        workers=workers,
        dp_heads=dp_heads,
        dp_size=dp_size,
        dp_groups=dp_groups,
        pp_size=meta.get("pp_size", 1),
        tp_size=meta.get("tp_size", 1),
        cp_size=meta.get("cp_size", 1),
        ep_size=meta.get("ep_size", 1),
    )
