# SPDX-License-Identifier: Apache-2.0

"""Node-shared, immutable CPU vision inputs for the mixed RL worker."""

from __future__ import annotations

import math
import mmap
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


_MAPPING_HEADER_SIZE = 64


def _check_mapping_errors(error, ranks, group, phase):
    errors = [None] * len(ranks)
    dist.all_gather_object(errors, error, group=group)
    failures = [
        f"rank {rank}: {message}"
        for rank, message in zip(ranks, errors, strict=True)
        if message is not None
    ]
    if failures:
        raise RuntimeError(f"Vision mapping {phase} failed: " + "; ".join(failures))


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

    @staticmethod
    def _open_vision_mapping(path, marker, shape, dtype):
        # Never create a receiver file: private container mounts must fail rather
        # than silently map zero pixels. Validate the file through the same fd used
        # for mmap, avoiding a path check followed by a second, creating open.
        fd = os.open(path, os.O_RDWR)
        try:
            size = (
                math.prod(shape)
                * torch.empty((), dtype=dtype, device="cpu").element_size()
            )
            if (
                os.fstat(fd).st_size != _MAPPING_HEADER_SIZE + size
                or os.pread(fd, _MAPPING_HEADER_SIZE, 0) != marker
            ):
                raise ValueError("Vision mapping is not the leader's backing store")
            storage = mmap.mmap(
                fd, _MAPPING_HEADER_SIZE + size, access=mmap.ACCESS_WRITE
            )
        finally:
            os.close(fd)
        try:
            # frombuffer retains the mmap owner until the last tensor view dies.
            return torch.frombuffer(
                storage,
                dtype=dtype,
                count=math.prod(shape),
                offset=_MAPPING_HEADER_SIZE,
            ).reshape(shape)
        except Exception:
            storage.close()
            raise

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
                error = None
                marker = None
                try:
                    if dist.get_rank() == leader:
                        fd, path = tempfile.mkstemp(
                            prefix=f"areal-vision-{os.environ.get('SLURM_JOB_ID', 'local')}-",
                            dir=os.environ.get("AREAL_VISION_SHM_DIR", "/dev/shm"),
                        )
                        try:
                            marker = os.urandom(_MAPPING_HEADER_SIZE)
                            os.ftruncate(
                                fd,
                                _MAPPING_HEADER_SIZE
                                + math.prod(shape)
                                * torch.empty(
                                    (), dtype=dtype, device="cpu"
                                ).element_size(),
                            )
                            if os.write(fd, marker) != len(marker):
                                raise OSError("Incomplete vision mapping header write")
                        finally:
                            os.close(fd)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                _check_mapping_errors(error, ranks, group, "creation")
                names = [None] * len(ranks)
                dist.all_gather_object(names, (path, marker), group=group)
                error = None
                try:
                    local_path, local_marker = names[ranks.index(leader)]
                    mapped = self._open_vision_mapping(
                        local_path, local_marker, shape, dtype
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                # Every rank must report open/validation failures before any
                # rank unlinks, copies pixels or enters point-to-point calls.
                _check_mapping_errors(error, ranks, group, "opening")
                error = None
                try:
                    if path is not None:
                        os.unlink(path)
                        path = None
                    if dist.get_rank() == src_rank:
                        mapped.copy_(tensors[index])
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                _check_mapping_errors(error, ranks, group, "preparation")
                if dist.get_rank() == src_rank:
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
                    try:
                        os.unlink(path)
                    except OSError:
                        # Do not mask the collective error with a local cleanup
                        # error (e.g. the mapping directory became inaccessible).
                        pass
        return _restore_vision(skeleton, local_tensors)
