# SPDX-License-Identifier: Apache-2.0
"""Serve mapper artifacts to SGLang's retention layer by policy-version pair.

SGLang imports this module by dotted path (``--kv-mapper-module``) and calls
``create_mapper_bridge(registry, model_layout)``. The registry is a directory of artifacts named
``v<source>-v<target>``; ``resolve`` returns a handle when that directory holds a loadable artifact
whose layout matches the server's KV pool, and ``translate`` maps per-layer ``[tokens, kv_heads,
head_dim]`` tensors in the pool's layout. Version numbers are AReaL's policy versions; the
checkpoint digests inside the artifact are recorded but cannot be checked here, because the server
does not know which checkpoint each version came from.
"""

import pathlib
from dataclasses import dataclass

import torch

from areal.experimental.kvmap.apply import DenseCache, translate_cache
from areal.experimental.kvmap.artifact import META_FILE, MapperArtifact
from areal.experimental.kvmap.fast import StackedMapper


@dataclass(frozen=True, kw_only=True)
class MapperHandle:
    source_version: int
    target_version: int
    artifact: MapperArtifact


class MapperBridge:
    def __init__(self, *, registry: pathlib.Path, model_layout: dict):
        self.registry = registry
        self.model_layout = model_layout
        self._handles: dict[tuple[int, int], MapperHandle] = {}
        self._stacked: dict[tuple[int, int, str], StackedMapper | None] = {}

    def artifact_directory(
        self, source_version: int, target_version: int
    ) -> pathlib.Path:
        return self.registry / f"v{source_version}-v{target_version}"

    def resolve(self, source_version: int, target_version: int) -> MapperHandle | None:
        """Return a cached handle, load one if the artifact is complete on disk, or None."""
        key = (source_version, target_version)
        if key in self._handles:
            return self._handles[key]
        directory = self.artifact_directory(source_version, target_version)
        if not (directory / META_FILE).is_file():
            return None
        artifact = MapperArtifact.load(directory)
        self._assert_layout(artifact)
        handle = MapperHandle(
            source_version=source_version,
            target_version=target_version,
            artifact=artifact,
        )
        self._handles[key] = handle
        return handle

    def translate(
        self,
        handle: MapperHandle,
        *,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        positions: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Map pool-layout tensors: each ``[tokens, kv_heads, head_dim]`` in, same shapes out.

        ``positions`` may concatenate several requests, so one call can serve every request that
        resumes after a weight update. Same-layer artifacts take the stacked fast path.
        """
        stacked = self._stacked_mapper(handle, keys[0].device)
        if stacked is not None:
            new_keys, new_values = stacked.translate(keys=torch.stack(keys), values=torch.stack(values), positions=positions)
            return list(new_keys.unbind(0)), list(new_values.unbind(0))
        source = DenseCache(
            keys=tuple(k.permute(1, 0, 2).unsqueeze(0) for k in keys),
            values=tuple(v.permute(1, 0, 2).unsqueeze(0) for v in values),
        )
        target = translate_cache(
            source=source,
            positions=positions.unsqueeze(0),
            artifact=handle.artifact,
            output_dtype=keys[0].dtype,
        )
        return [k[0].permute(1, 0, 2).contiguous() for k in target.keys], [
            v[0].permute(1, 0, 2).contiguous() for v in target.values
        ]

    def _stacked_mapper(self, handle: MapperHandle, device: torch.device) -> StackedMapper | None:
        """Return the prepared fast mapper for this pair and device, or None when the artifact is not same-layer."""
        key = (handle.source_version, handle.target_version, str(device))
        if key not in self._stacked:
            try:
                self._stacked[key] = StackedMapper(handle.artifact, device=device)
            except ValueError:
                self._stacked[key] = None
        return self._stacked[key]

    def _assert_layout(self, artifact: MapperArtifact) -> None:
        expected = {
            "num_layers": artifact.target.num_layers,
            "num_kv_heads": artifact.target.num_kv_heads,
            "head_dim": artifact.target.head_dim,
        }
        for field, value in expected.items():
            actual = self.model_layout.get(field)
            if actual != value:
                raise ValueError(
                    f"artifact target {field} is {value} but the server KV pool has {actual}; the registry does not belong to this model"
                )
        theta = self.model_layout.get("rope_theta")
        if theta is not None and float(theta) != artifact.target.rope_theta:
            raise ValueError(
                f"artifact rope_theta {artifact.target.rope_theta} differs from the server model's {theta}"
            )


def create_mapper_bridge(*, registry: str, model_layout: dict) -> MapperBridge:
    """Entry point SGLang calls once per scheduler process."""
    path = pathlib.Path(registry)
    if not path.is_dir():
        raise FileNotFoundError(f"kv mapper registry {registry} is not a directory")
    return MapperBridge(registry=path, model_layout=model_layout)
