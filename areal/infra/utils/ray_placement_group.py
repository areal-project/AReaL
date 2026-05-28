# SPDX-License-Identifier: Apache-2.0

import abc
from dataclasses import dataclass, field
from typing import Any

import ray
import ray.exceptions
import torch
from ray.util.placement_group import (
    PlacementGroup,
    placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from areal.api.cli_args import SchedulingSpec
from areal.infra.utils.ray import create_resource_spec
from areal.utils import logging

logger = logging.getLogger("RayPlacementGroup")

MAIN_WORKER_GPU_FRAC_FOR_COLOCATION = 0.9


def ray_resource_type():
    # npu before cuda because mindspeed patches cuda.is_available
    from areal.infra.platforms import is_npu_available

    if is_npu_available:
        return "NPU"

    if torch.cuda.is_available():
        return "GPU"

    return "CPU"


def _create_bundle_specs_split(
    n_gpus_per_node: int, cpu: int, gpu: int, mem: int
) -> list[dict]:
    if gpu <= 0:
        # no GPU bundles; single CPU-only bundle
        return [_bundle_spec(cpu, 0, mem)]
    if n_gpus_per_node <= 0:
        raise ValueError(f"n_gpus_per_node must be > 0, got {n_gpus_per_node}")

    n_full_nodes, remainder_gpu = divmod(gpu, n_gpus_per_node)
    total_nodes = n_full_nodes + (1 if remainder_gpu > 0 else 0)

    # Split CPU/mem proportionally to GPUs
    gpu_per_node = [n_gpus_per_node] * n_full_nodes
    if remainder_gpu > 0:
        gpu_per_node.append(remainder_gpu)

    bundles = []
    cpu_left, mem_left = cpu, mem
    for i, g in enumerate(gpu_per_node):
        if i == total_nodes - 1:
            c, m = cpu_left, mem_left
        else:
            ratio = g / gpu
            c = int(round(cpu * ratio))
            m = int(round(mem * ratio))
            cpu_left -= c
            mem_left -= m

        bundles.append(_bundle_spec(c, g, m))

    return bundles


def _bundle_spec(cpu: int, gpu: int, mem: int) -> dict:
    """
    Create a bundle dict for a given cpu, gpu, mem requirement
    """

    device = ray_resource_type()
    if device == "CPU" and gpu > 0:
        raise ValueError(
            f"Current detected device is CPU but specified number of GPUs is {gpu}"
        )
    if device == "CPU":
        return {"CPU": cpu, "memory": mem * 1024**3}
    return {"CPU": cpu, device: float(gpu), "memory": mem * 1024**3}


def _actor_resource_spec(cpu: int, gpu: int, mem: int) -> dict:
    """
    Create a dictionary for passing into ray actor options specifying resource requirements
    """
    device = ray_resource_type()
    if device == "CPU" and gpu > 0:
        raise ValueError(
            f"Current detected device is CPU but specified number of GPUs is {gpu}"
        )

    return create_resource_spec(device, cpu, gpu, mem * 1024**3)


def _create_placement_group(role: str, bundles: list[dict], timeout) -> PlacementGroup:
    """
    Helper to create and wait for a placement group
    """
    pg = placement_group(bundles=bundles, strategy="PACK")
    try:
        ray.get(pg.ready(), timeout=timeout)
    except ray.exceptions.GetTimeoutError:
        logger.error(
            f"Ray placement group timeout for role {role}\n"
            f"ray.nodes(): {ray.nodes()}"
            f"bundles: {bundles}"
        )
        raise
    return pg


@dataclass
class RayPlacementStrategy(abc.ABC):
    _placement_groups: list[PlacementGroup] = field(default_factory=list)

    @abc.abstractmethod
    def create_placement_group(
        self,
        role: str,
        schedulings: list[SchedulingSpec],
        n_gpus_per_node: int,
        timeout: int,
    ) -> list[PlacementGroup]: ...

    @abc.abstractmethod
    def actor_resources(
        self, spec: SchedulingSpec, gpu_multiplier=1
    ) -> tuple[dict, PlacementGroupSchedulingStrategy]: ...


@dataclass
class SharedRayPlacementStrategy(RayPlacementStrategy):
    # primarily for training, where multiple training workers share 1 placement group
    _bundles: list[dict] = field(default_factory=list)
    _current_bundle_idx: int = 0

    def create_placement_group(
        self,
        role: str,
        schedulings: list[SchedulingSpec],
        n_gpus_per_node: int,
        timeout=30,
    ) -> list[PlacementGroup]:
        if len(self._placement_groups) > 0:
            raise RuntimeError(
                "SharedRayPlacementStrategy should only have a single placement group, cannot create another placement group"
            )
        bundles = [_bundle_spec(spec.cpu, spec.gpu, spec.mem) for spec in schedulings]
        pg = _create_placement_group(role, bundles, timeout)
        self._bundles = bundles
        self._placement_groups.append(pg)
        return self._placement_groups

    def actor_resources(
        self, spec: SchedulingSpec, gpu_multiplier=MAIN_WORKER_GPU_FRAC_FOR_COLOCATION
    ) -> tuple[dict, PlacementGroupSchedulingStrategy]:
        options = _actor_resource_spec(spec.cpu, spec.gpu * gpu_multiplier, spec.mem)
        placement_group = self._placement_groups[0]

        # Build scheduling strategy with optional bundle index
        strategy_kwargs: dict[str, Any] = {
            "placement_group": placement_group,
            "placement_group_capture_child_tasks": True,
            "placement_group_bundle_index": self._current_bundle_idx,
        }

        self._current_bundle_idx += 1

        return options, PlacementGroupSchedulingStrategy(**strategy_kwargs)


@dataclass
class SeparatedRayPlacementStrategy(RayPlacementStrategy):
    # primarily for rollout, where 1 rollout takes 1 placement group
    _current_placement_group_idx: int = 0

    def _create_bundles(self, spec: SchedulingSpec, n_gpus_per_node: int):
        return [_bundle_spec(spec.cpu, spec.gpu, spec.mem)]

    def _get_resource_spec(self, spec: SchedulingSpec):
        return _actor_resource_spec(spec.cpu, spec.gpu, spec.mem)

    def create_placement_group(
        self,
        role: str,
        schedulings: list[SchedulingSpec],
        n_gpus_per_node: int,
        timeout=30,
    ) -> list[PlacementGroup]:
        for spec in schedulings:
            bundles = self._create_bundles(spec, n_gpus_per_node)
            pg = _create_placement_group(role, bundles, timeout)
            self._placement_groups.append(pg)
        return self._placement_groups

    def actor_resources(
        self, spec: SchedulingSpec, gpu_multiplier=1
    ) -> tuple[dict, PlacementGroupSchedulingStrategy]:
        if gpu_multiplier != 1 and spec.gpu > 1:
            raise RuntimeError(
                "Colocation is not supported in Ray for multi-GPU instances."
            )
        if self._current_placement_group_idx >= len(self._placement_groups):
            raise RuntimeError("Placement groups are full")

        options = self._get_resource_spec(spec)

        placement_group = self._placement_groups[self._current_placement_group_idx]
        self._current_placement_group_idx += 1

        strategy_kwargs: dict[str, Any] = {
            "placement_group": placement_group,
            "placement_group_capture_child_tasks": True,
            "placement_group_bundle_index": 0,
        }

        return options, PlacementGroupSchedulingStrategy(**strategy_kwargs)


@dataclass
class DeferredDeviceRayPlacementStrategy(SeparatedRayPlacementStrategy):
    # primarily for rollout where the launch_server procedure will take accelerators, so we only create the PG and pass it on

    def _create_bundles(self, spec: SchedulingSpec, n_gpus_per_node: int):
        bundles = _create_bundle_specs_split(
            n_gpus_per_node, spec.cpu, spec.gpu, spec.mem
        )
        return bundles

    def _get_resource_spec(self, spec: SchedulingSpec):
        return _actor_resource_spec(0, 0, 0)
