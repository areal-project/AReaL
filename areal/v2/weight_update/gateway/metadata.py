# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace
from itertools import combinations


def merge_parameter_metadata(metadata: list[dict]) -> list[dict]:
    """Reconstruct complete replicas from rank-local AWEX parameter reports.

    Extents, rather than TP coordinates, identify shards: FSDP DP shards can
    share a TP coordinate. Identical extents are replicas, not additive shards.
    """
    from awex.meta.weight_meta import ParameterReplicaMeta

    from areal.infra.rpc.serialization import deserialize_value, serialize_value

    if not metadata:
        raise ValueError("Missing parameter metadata")
    by_name = defaultdict(list)
    for parameter in deserialize_value(metadata):
        by_name[parameter.name].append(parameter)
    merged = []
    for name, parameters in sorted(by_name.items()):
        reference = parameters[0]
        shape = tuple(reference.global_shape)
        if math.prod(shape) != reference.global_numel:
            raise ValueError(f"Invalid global shape for {name}")
        extents = defaultdict(list)
        ranks = set()
        for parameter in parameters:
            if (
                tuple(parameter.global_shape) != shape
                or parameter.global_numel != reference.global_numel
                or parameter.dtype != reference.dtype
            ):
                raise ValueError(f"Inconsistent parameter metadata for {name}")
            for shard in parameter.shards:
                if shard.global_rank in ranks:
                    raise ValueError(f"Duplicate rank metadata for {name}")
                ranks.add(shard.global_rank)
                offset, size = tuple(shard.global_offset), tuple(shard.shape)
                if (
                    shard.name != name
                    or shard.dtype != reference.dtype
                    or len(offset) != len(shape)
                    or len(size) != len(shape)
                    or math.prod(size) != shard.numel
                    or any(
                        start < 0 or width < 0 or start + width > bound
                        for start, width, bound in zip(offset, size, shape)
                    )
                ):
                    raise ValueError(f"Invalid shard extent for {name}")
                extents[offset, size].append(shard)
        ordered = sorted(extents)
        if not ordered or len({len(shards) for shards in extents.values()}) != 1:
            raise ValueError(f"Incomplete replicas for {name}")
        if sum(math.prod(size) for _, size in ordered) != reference.global_numel:
            raise ValueError(f"Incomplete shard coverage for {name}")
        for (left, left_size), (right, right_size) in combinations(ordered, 2):
            if all(
                max(a, b) < min(a + aw, b + bw)
                for a, aw, b, bw in zip(left, left_size, right, right_size)
            ):
                raise ValueError(f"Overlapping shard extents for {name}")
        for shards in extents.values():
            shards.sort(key=lambda shard: shard.global_rank)
        replicas = [
            ParameterReplicaMeta(shards=[extents[key][i] for key in ordered])
            for i in range(len(extents[ordered[0]]))
        ]
        merged.append(
            replace(
                reference,
                shards=[shard for replica in replicas for shard in replica.shards],
                replicas=replicas,
            )
        )
    return serialize_value(merged)


def merge_inference_metadata(responses: list[dict]) -> list[dict]:
    """Keep one complete engine layout; AWEX replicates it across engines."""
    engines = []
    for response in responses:
        metadata = response.get("result", response.get("meta", response))
        engines.append(
            merge_parameter_metadata(
                metadata if isinstance(metadata, list) else [metadata]
            )
        )
    if not engines or not engines[0]:
        raise ValueError("Missing inference parameter metadata")
    if any(engine != engines[0] for engine in engines[1:]):
        raise ValueError("Inference engines disagree on parameter metadata")
    return engines[0]
