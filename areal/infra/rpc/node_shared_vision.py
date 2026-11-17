# SPDX-License-Identifier: Apache-2.0

"""Node-shared, immutable CPU vision inputs for the mixed RL worker."""

from __future__ import annotations

import math
import os
import socket
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class _VisionRef:
    index: int


def _extract_vision(data: Any) -> tuple[Any, list[torch.Tensor]]:
    tensors: list[torch.Tensor] = []
    indices: dict[int, int] = {}

    def visit(value, in_vision=False):
        if isinstance(value, torch.Tensor):
            if (
                in_vision
                and value.device.type == "cpu"
                and value.is_floating_point()
                and value.layout == torch.strided
                and value.numel() > 0
                and not value.requires_grad
            ):
                index = indices.get(id(value))
                if index is None:
                    index = len(tensors)
                    indices[id(value)] = index
                    tensors.append(value)
                return _VisionRef(index)
            return value
        if isinstance(value, dict):
            return {
                key: visit(
                    item,
                    in_vision
                    or (isinstance(key, str) and key.startswith("multi_modal_input")),
                )
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list)):
            items = [visit(item, in_vision) for item in value]
            if isinstance(value, list):
                return items
            return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
        return value

    return visit(data), tensors


def _restore_vision(data: Any, tensors: list[torch.Tensor]) -> Any:
    if isinstance(data, _VisionRef):
        return tensors[data.index]
    if isinstance(data, dict):
        return {key: _restore_vision(value, tensors) for key, value in data.items()}
    if isinstance(data, (tuple, list)):
        items = [_restore_vision(value, tensors) for value in data]
        if isinstance(data, list):
            return items
        return type(data)(*items) if hasattr(data, "_fields") else tuple(items)
    return data


class NodeSharedVisionTransport:
    """Broadcast pixels once per node and map one backing store on local ranks.

    Token tensors use the existing broadcast. Vision tensors are immutable CPU
    inputs; model forward copies the current microbatch to the accelerator.
    Communication stays within the supplied parent group, so disjoint DP
    groups progress independently without creating extra process groups.
    Calls within each parent group must follow the same order. Local ranks
    must share the mapping directory.
    """

    def __init__(self, broadcast_container: Callable, *, node_name: str | None = None):
        self.broadcast_container = broadcast_container
        self.node_name = (
            node_name or os.environ.get("SLURMD_NODENAME") or socket.gethostname()
        )
        self._layouts: dict[tuple[Any, int], tuple[list[int], list[int], int]] = {}

    def _node_layout(self, group, src_rank):
        key = (group, src_rank)
        if key not in self._layouts:
            ranks = (
                dist.get_process_group_ranks(group)
                if group is not None
                else list(range(dist.get_world_size()))
            )
            nodes = [None] * len(ranks)
            dist.all_gather_object(nodes, self.node_name, group=group)
            ranks_by_node: dict[str, list[int]] = {}
            for rank, node in zip(ranks, nodes, strict=True):
                ranks_by_node.setdefault(node, []).append(rank)
            leaders = [
                src_rank if src_rank in members else members[0]
                for members in ranks_by_node.values()
            ]
            local_ranks = ranks_by_node[self.node_name]
            leader = src_rank if src_rank in local_ranks else local_ranks[0]
            self._layouts[key] = (ranks, leaders, leader)
        return self._layouts[key]

    def __call__(self, data, src_rank=0, group=None, *, preserve_tensor_aliases=False):
        if dist.get_backend(group) != "gloo":
            return self.broadcast_container(
                data,
                src_rank=src_rank,
                group=group,
                preserve_tensor_aliases=preserve_tensor_aliases,
            )
        tensors = []
        skeleton = data
        if dist.get_rank() == src_rank and preserve_tensor_aliases:
            skeleton, tensors = _extract_vision(data)
        specs = [[(tuple(t.shape), t.dtype) for t in tensors]]
        dist.broadcast_object_list(specs, src=src_rank, group=group)
        if not specs[0]:
            return self.broadcast_container(
                data,
                src_rank=src_rank,
                group=group,
                preserve_tensor_aliases=preserve_tensor_aliases,
            )

        skeleton = self.broadcast_container(
            skeleton,
            src_rank=src_rank,
            group=group,
            preserve_tensor_aliases=preserve_tensor_aliases,
        )
        ranks, leaders, leader = self._node_layout(group, src_rank)
        local_tensors = []
        for index, (shape, dtype) in enumerate(specs[0]):
            path = None
            mapped = None
            try:
                if dist.get_rank() == leader:
                    fd, path = tempfile.mkstemp(
                        prefix=f"areal-vision-{os.environ.get('SLURM_JOB_ID', 'local')}-",
                        dir=os.environ.get("AREAL_VISION_SHM_DIR", "/dev/shm"),
                    )
                    try:
                        os.ftruncate(
                            fd,
                            math.prod(shape)
                            * torch.empty((), dtype=dtype, device="cpu").element_size(),
                        )
                    finally:
                        os.close(fd)
                names = [None] * len(ranks)
                dist.all_gather_object(names, path, group=group)
                mapped = torch.from_file(
                    names[ranks.index(leader)],
                    shared=True,
                    size=math.prod(shape),
                    dtype=dtype,
                    device="cpu",
                ).reshape(shape)
                # Open every local mapping, then unlink before populating the
                # file. Unlinked buffers are owned by live mappings and are
                # reclaimed when their last consumer exits.
                dist.barrier(group=group)
                if path is not None:
                    os.unlink(path)
                    path = None
                if dist.get_rank() == src_rank:
                    mapped.copy_(tensors[index])
                    # Only one rank per destination node receives the pixels.
                    # Reuse the parent group: dynamic subgroups would couple DP
                    # replicas through process-group initialization state.
                    sends = [
                        dist.isend(mapped, dst=destination, group=group)
                        for destination in leaders
                        if destination != src_rank
                    ]
                    for request in sends:
                        request.wait()
                elif dist.get_rank() == leader:
                    dist.recv(mapped, src=src_rank, group=group)
                # Every local rank has opened the mapping and the leader has
                # finished receiving before it becomes visible to consumers.
                dist.barrier(group=group)
                local_tensors.append(mapped)
            finally:
                # Before all mappings open, only a sparse, unpopulated file
                # exists. Normal failures still remove that name here.
                if path is not None:
                    os.unlink(path)
        return _restore_vision(skeleton, local_tensors)
