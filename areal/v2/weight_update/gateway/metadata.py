# SPDX-License-Identifier: Apache-2.0
"""Canonical AWEX replicas from rank-local serialized parameter metadata."""

from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy
from itertools import combinations


def _data(value: dict) -> dict:
    return value.get("data", value)


def merge_parameter_metadata(entries: list[dict]) -> list[dict]:
    """Union complementary shards; retain identical full tensors as replicas.

    AWEX indexes by parameter name and sends every shard of the chosen replica.
    Equal geometry on different ranks therefore belongs to different replicas.
    """
    grouped = defaultdict(list)
    for entry in entries:
        grouped[_data(entry)["name"]].append(entry)
    result = []
    for name, parameters in sorted(grouped.items()):
        parameter = deepcopy(parameters[0])
        target = _data(parameter)
        global_shape = tuple(target["global_shape"])
        tiles = defaultdict(dict)
        for item in parameters:
            data = _data(item)
            if any(
                data[field] != target[field]
                for field in ("global_shape", "global_numel", "dtype")
            ):
                raise ValueError(f"Inconsistent AWEX parameter metadata: {name}")
            for shard in data["shards"]:
                local = _data(shard)
                offset, shape = tuple(local["global_offset"]), tuple(local["shape"])
                if math.prod(shape) == 0:
                    continue
                if (
                    len(shape) != len(global_shape)
                    or len(offset) != len(shape)
                    or any(
                        start < 0 or size < 0 or start + size > total
                        for start, size, total in zip(offset, shape, global_shape)
                    )
                ):
                    raise ValueError(f"Invalid AWEX shard geometry: {name}")
                identity = (local.get("engine_rank", 0), local["global_rank"])
                tiles[offset, shape][identity] = deepcopy(shard)
        geometries = sorted(tiles)
        if (
            not geometries
            or sum(math.prod(shape) for _, shape in geometries)
            != target["global_numel"]
        ):
            raise ValueError(f"Incomplete AWEX parameter coverage: {name}")
        for (a, sa), (b, sb) in combinations(geometries, 2):
            if all(
                max(x, y) < min(x + sx, y + sy) for x, sx, y, sy in zip(a, sa, b, sb)
            ):
                raise ValueError(f"Overlapping AWEX parameter shards: {name}")
        groups = [
            [by_rank[key] for key in sorted(by_rank)]
            for _, by_rank in sorted(tiles.items())
        ]
        copies = len(groups[0])
        if any(len(group) != copies for group in groups):
            raise ValueError(f"Incomplete AWEX parameter replica: {name}")
        replica_template = target["replicas"][0]
        replicas = []
        for index in range(copies):
            replica = deepcopy(replica_template)
            _data(replica)["shards"] = [group[index] for group in groups]
            replicas.append(replica)
        target["shards"] = [shard for group in groups for shard in group]
        target["replicas"] = replicas
        result.append(parameter)
    return result


def inference_metadata_template(responses: list[dict]) -> list[dict]:
    """Keep one identical per-engine template; AWEX performs engine fanout."""
    templates = []
    for response in responses:
        entries = response.get("result", response.get("meta", response))
        templates.append(
            merge_parameter_metadata(
                entries if isinstance(entries, list) else [entries]
            )
        )
    if any(template != templates[0] for template in templates[1:]):
        raise ValueError("AWEX inference engines report different shard layouts")
    return templates[0]
