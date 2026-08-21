# SPDX-License-Identifier: Apache-2.0

"""MCore 0.17 checkpoint capabilities for CPU-resident staged optimizers.

This module is the concentrated compatibility boundary for the private
Megatron-Core distributed-optimizer checkpoint contract.  Upstream can replace
it with managed-optimizer hooks around sharded-state template construction and
optimizer-state load finalization.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from areal.engine.megatron_utils.checkpoint_snapshot import (
    SnapshotCapacityReport,
    preflight_snapshot_requirements,
)
from areal.engine.megatron_utils.optimizer_chain import (
    get_megatron_optimizer_chain_children,
    iter_megatron_optimizer_leaf_paths,
    iter_megatron_optimizer_leaves,
)

_SUPPORTED_MEGATRON_CORE_VERSION = "0.17.0"
_PARAM_GROUP_OWNERSHIP_KEYS = (
    "wd_mult",
    "lr_mult",
    "is_expert_parallel",
    "is_decoupled_lr",
)
_OUTER_OPTIMIZER_KEYS = frozenset({"param_groups"})
_OUTER_STATE_KEYS = frozenset(
    {
        "optimizer",
        "grad_scaler",
        "param_state",
        "param_state_sharding_type",
        "managed_checkpoint_identity",
    }
)
_MANAGED_IDENTITY_KEY = "managed_checkpoint_identity"
_MANAGED_IDENTITY_VERSION = 1


def _check_mcore_checkpoint_compatibility(
    *, require_distributed_adamw_schema: bool = True
) -> None:
    version = importlib.metadata.version("megatron-core")
    if version != _SUPPORTED_MEGATRON_CORE_VERSION:
        raise RuntimeError(
            "GPU-staged optimizer checkpoint compatibility supports "
            f"megatron-core {_SUPPORTED_MEGATRON_CORE_VERSION}, found {version}"
        )

    from megatron.core.optimizer import distrib_optimizer as distrib_optimizer_module
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    signature = inspect.signature(DistributedOptimizer.sharded_state_dict)
    required = {"model_sharded_state_dict", "is_loading", "metadata"}
    if not required.issubset(signature.parameters):
        raise RuntimeError(
            "Megatron DistributedOptimizer.sharded_state_dict no longer exposes "
            f"the required checkpoint parameters: {sorted(required)}"
        )
    if (
        require_distributed_adamw_schema
        and not distrib_optimizer_module.HAVE_APEX_OR_TE
    ):
        raise RuntimeError(
            "GPU-staged AdamW checkpoint schema requires the MCore 0.17 "
            "Apex/Transformer Engine param-group step representation"
        )


def iter_managed_optimizers(optimizer: Any | None) -> Iterator[Any]:
    """Yield unique CPU-residency owners below arbitrary MCore chains."""
    seen: set[int] = set()
    for leaf in iter_megatron_optimizer_leaves(optimizer):
        base = getattr(leaf, "optimizer", leaf)
        if base is None or getattr(base, "manages_cpu_residency", False) is not True:
            continue
        identity = id(base)
        if identity in seen:
            continue
        seen.add(identity)
        yield base


def configure_managed_checkpoint_snapshots(
    optimizer: Any | None,
    identities: Mapping[tuple[int, ...], Mapping[str, Any]],
    *,
    parent: str | None,
    transaction: ManagedCheckpointLoadTransaction | None = None,
) -> None:
    """Attach stable tree identities without creating rollback artifacts."""
    for path, leaf in iter_megatron_optimizer_leaf_paths(optimizer):
        base = getattr(leaf, "optimizer", leaf)
        if base is None or getattr(base, "manages_cpu_residency", False) is not True:
            continue
        configure = getattr(base, "configure_checkpoint_snapshot", None)
        if not callable(configure):
            # Compatibility for managed test doubles or an upstream residency
            # capability which owns a different rollback implementation.
            continue
        try:
            identity = identities[path]
        except KeyError as error:
            raise RuntimeError(
                f"managed optimizer leaf path {path} lacks rollback identity"
            ) from error
        kwargs: dict[str, Any] = {
            "parent": parent,
            "leaf_identity": dict(identity),
        }
        if transaction is not None and transaction.replacement_generation is not None:
            if not _accepts_keyword(configure, "replacement_generation"):
                raise RuntimeError(
                    "managed optimizer snapshot configuration is not replacement-aware"
                )
            kwargs.update(
                replacement_generation=transaction.replacement_generation,
                attempt_token=transaction.attempt_token,
            )
        try:
            configure(**kwargs)
        except BaseException:
            if (
                transaction is not None
                and transaction.replacement_generation is not None
                and getattr(base, "checkpoint_snapshot_attempt_token", None)
                is transaction.attempt_token
                and not any(
                    configured is base for configured in transaction.snapshot_configured
                )
            ):
                transaction.snapshot_configured.append(base)
            raise
        if (
            transaction is not None
            and transaction.replacement_generation is not None
            and not any(
                configured is base for configured in transaction.snapshot_configured
            )
        ):
            transaction.snapshot_configured.append(base)


def preflight_managed_checkpoint_snapshots(
    transaction: ManagedCheckpointLoadTransaction,
) -> tuple[SnapshotCapacityReport, ...]:
    """Aggregate every local leaf's disk requirement before any slab mutation."""
    requirements = []
    for inner in transaction.leaves:
        token_aware, current = _leaf_checkpoint_attempt_token(inner)
        if (
            token_aware
            and current is not None
            and current is not transaction.attempt_token
        ):
            raise RuntimeError(
                "managed checkpoint leaf is already active under another begin attempt"
            )
        retry_build_cleanup = getattr(
            inner, "retry_checkpoint_snapshot_build_cleanup", None
        )
        if callable(retry_build_cleanup):
            retry_build_cleanup()
        requirement = getattr(inner, "checkpoint_snapshot_requirement", None)
        if not callable(requirement):
            # Structural test doubles and older managed capabilities may own
            # their rollback journal themselves.  The manager configures and
            # version-checks every GPUStagedAdamW leaf before reaching here.
            continue
        requirements.append(requirement())
    return preflight_snapshot_requirements(tuple(requirements))


def _ownership_tuple(group: Mapping[str, Any], group_index: int) -> tuple[Any, ...]:
    ownership: list[Any] = []
    for key in _PARAM_GROUP_OWNERSHIP_KEYS:
        aliases = (key, f"pre_{key}")
        present = [alias for alias in aliases if alias in group]
        if len(present) != 1:
            raise ValueError(
                f"optimizer param-group {group_index} must contain exactly one of "
                f"{aliases}, found {present}"
            )
        value = group[present[0]]
        if key in {"is_expert_parallel", "is_decoupled_lr"}:
            if type(value) is not bool:
                raise TypeError(
                    f"optimizer param-group {group_index} field {present[0]} "
                    "must be bool"
                )
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                f"optimizer param-group {group_index} field {present[0]} "
                "must be a finite non-negative number"
            )
        ownership.append(value)
    return tuple(ownership)


def _normalized_group_keys(group: Mapping[str, Any]) -> set[str]:
    keys = set(group)
    for key in _PARAM_GROUP_OWNERSHIP_KEYS:
        prefixed = f"pre_{key}"
        if prefixed in keys:
            keys.remove(prefixed)
            keys.add(key)
    return keys


def validate_managed_adamw_param_group(
    checkpoint_group: Mapping[str, Any],
    expected_group: Mapping[str, Any],
    *,
    location: str,
    ignore_params: bool,
) -> None:
    """Validate the one AdamW metadata contract used before and after DCP.

    MCore's outer state may spell ownership fields with a ``pre_`` prefix,
    while torch Optimizer state uses the canonical spelling.  Normalizing that
    representation here keeps the metadata-only preflight and the slab-backed
    inner load on exactly the same value-domain rules.
    """
    if not isinstance(checkpoint_group, Mapping):
        raise TypeError(f"{location} must be a mapping")
    if not isinstance(expected_group, Mapping):
        raise TypeError(f"{location} runtime metadata must be a mapping")

    checkpoint_keys = _normalized_group_keys(checkpoint_group)
    expected_keys = _normalized_group_keys(expected_group)
    if ignore_params:
        checkpoint_keys.discard("params")
        expected_keys.discard("params")
    if checkpoint_keys != expected_keys:
        missing = sorted(expected_keys - checkpoint_keys)
        extra = sorted(checkpoint_keys - expected_keys)
        raise ValueError(
            f"{location} metadata mismatch: missing={missing}, unexpected={extra}"
        )

    normalized: dict[str, Any] = {}
    for key, value in checkpoint_group.items():
        canonical = key.removeprefix("pre_") if key.startswith("pre_") else key
        if canonical in normalized:
            raise ValueError(f"{location} contains duplicate field {canonical!r}")
        normalized[canonical] = value
    expected = {
        (key.removeprefix("pre_") if key.startswith("pre_") else key): value
        for key, value in expected_group.items()
    }

    def finite_number(name: str) -> float:
        value = normalized[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{location} field {name} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{location} field {name} must be finite")
        return result

    for name in ("lr", "initial_lr", "max_lr", "min_lr", "weight_decay"):
        if name in normalized and finite_number(name) < 0.0:
            raise ValueError(f"{location} field {name} must be non-negative")
    if "eps" in normalized and finite_number("eps") <= 0.0:
        raise ValueError(f"{location} field eps must be positive")
    for name in ("lr_mult", "wd_mult"):
        if name in normalized and finite_number(name) < 0.0:
            raise ValueError(f"{location} field {name} must be non-negative")

    if "betas" in normalized:
        betas = normalized["betas"]
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise TypeError(f"{location} field betas must be a pair")
        for beta in betas:
            if (
                isinstance(beta, bool)
                or not isinstance(beta, (int, float))
                or not math.isfinite(float(beta))
                or not 0.0 <= float(beta) < 1.0
            ):
                raise ValueError(
                    f"{location} field betas must contain finite values in [0, 1)"
                )

    if "step" in normalized:
        step = normalized["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise TypeError(f"{location} field step must be a non-negative int")

    for name in (
        "is_expert_parallel",
        "is_decoupled_lr",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
    ):
        if name in normalized and (
            normalized[name] is not None and type(normalized[name]) is not bool
        ):
            raise TypeError(f"{location} field {name} must be bool or None")

    mutable = {
        "params",
        "lr",
        "initial_lr",
        "max_lr",
        "min_lr",
        "betas",
        "eps",
        "weight_decay",
        "step",
    }
    for name, value in normalized.items():
        if name in mutable:
            continue
        if name not in expected or value != expected[name]:
            raise ValueError(f"{location} field {name} does not match runtime metadata")


def _validate_group_metadata(
    checkpoint_group: Mapping[str, Any],
    expected_group: Mapping[str, Any],
    group_index: int,
    leaf_path: tuple[int, ...],
) -> None:
    location = f"managed optimizer leaf path {leaf_path} param-group {group_index}"
    validate_managed_adamw_param_group(
        checkpoint_group,
        expected_group,
        location=location,
        ignore_params=True,
    )


def _indexed_chain_states(
    state: Any, child_count: int, path: tuple[int, ...]
) -> tuple[Any, ...]:
    if isinstance(state, Mapping):
        expected_indices = set(range(child_count))
        actual_indices = set(state)
        if actual_indices != expected_indices:
            raise ValueError(
                f"optimizer checkpoint tree path {path} indices mismatch: "
                f"missing={sorted(expected_indices - actual_indices)}, "
                f"unexpected={sorted(actual_indices - expected_indices, key=repr)}"
            )
        return tuple(state[index] for index in range(child_count))
    if isinstance(state, Sequence) and not isinstance(state, (str, bytes, bytearray)):
        if len(state) != child_count:
            raise ValueError(
                f"optimizer checkpoint tree path {path} expected {child_count} "
                f"children, found {len(state)}"
            )
        return tuple(state)
    raise TypeError(
        f"optimizer checkpoint tree path {path} must be an indexed mapping or sequence"
    )


def _managed_outer_leaf_states(
    optimizer: Any, state_dict: Any
) -> tuple[tuple[tuple[int, ...], Any, Any], ...]:
    paired: list[tuple[tuple[int, ...], Any, Any]] = []
    active: set[int] = set()
    seen: dict[int, tuple[int, ...]] = {}

    def pair(node: Any, state: Any, path: tuple[int, ...]) -> None:
        identity = id(node)
        if identity in active:
            raise ValueError(
                f"optimizer checkpoint tree contains a cycle at path {path}"
            )
        if identity in seen:
            raise ValueError(
                f"optimizer checkpoint tree shares path {path} with {seen[identity]}"
            )
        seen[identity] = path
        children = get_megatron_optimizer_chain_children(node)
        if children is None:
            paired.append((path, node, state))
            return
        active.add(identity)
        try:
            if len(children) == 1:
                pair(children[0], state, (*path, 0))
                return
            child_states = _indexed_chain_states(state, len(children), path)
            for child_index, (child, child_state) in enumerate(
                zip(children, child_states)
            ):
                pair(child, child_state, (*path, child_index))
        finally:
            active.remove(identity)

    pair(optimizer, state_dict, ())
    return tuple(paired)


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _uses_custom_muon_checkpoint_schema(optimizer: Any | None) -> bool:
    """Require the explicit capability tag before consulting custom hooks."""
    checkpoint_format = getattr(optimizer, "managed_checkpoint_format", None)
    return (
        isinstance(checkpoint_format, str) and checkpoint_format == "muon_dp_reshard_v2"
    )


def _iter_leaf_buffers(buffers: Any) -> Iterator[Any]:
    if isinstance(buffers, (list, tuple)):
        for value in buffers:
            yield from _iter_leaf_buffers(value)
        return
    yield buffers


def build_managed_optimizer_identities(
    optimizer: Any | None,
    model_parameter_names: Mapping[Any, str],
) -> dict[tuple[int, ...], dict[str, Any]]:
    """Build DP-reshard-stable identities from tree path and model buffers."""
    custom_builder = getattr(optimizer, "managed_checkpoint_identities", None)
    if _uses_custom_muon_checkpoint_schema(optimizer):
        if not callable(custom_builder):
            raise RuntimeError("staged Muon checkpoint identity hook is unavailable")
        identities = custom_builder(model_parameter_names)
        if not isinstance(identities, dict):
            raise TypeError("managed checkpoint identity builder must return a dict")
        return identities
    identities: dict[tuple[int, ...], dict[str, Any]] = {}
    for path, leaf in iter_megatron_optimizer_leaf_paths(optimizer):
        base = getattr(leaf, "optimizer", leaf)
        if base is None or getattr(base, "manages_cpu_residency", False) is not True:
            continue
        group_ranges = getattr(leaf, "opt_group_ranges", None)
        buffers = getattr(leaf, "buffers", None)
        if group_ranges is None or buffers is None:
            continue
        buffer_schema: list[dict[str, Any]] = []
        for buffer_index, buffer in enumerate(_iter_leaf_buffers(buffers)):
            param_index_map = getattr(buffer, "param_index_map", None)
            if not isinstance(param_index_map, Mapping):
                raise RuntimeError(
                    f"managed optimizer leaf path {path} buffer {buffer_index} "
                    "lacks param_index_map"
                )
            parameters: list[dict[str, Any]] = []
            for param in param_index_map:
                try:
                    name = model_parameter_names[param]
                except KeyError as exc:
                    raise RuntimeError(
                        f"managed optimizer leaf path {path} contains a buffer "
                        "parameter without a stable model name"
                    ) from exc
                parameters.append(
                    {
                        "name": name,
                        "shape": list(param.shape),
                        "dtype": str(param.dtype),
                    }
                )
            buffer_schema.append(
                {
                    "buffer_index": buffer_index,
                    "param_dtype": str(getattr(buffer, "param_dtype", "unknown")),
                    "grad_dtype": str(getattr(buffer, "grad_dtype", "unknown")),
                    "numel_unpadded": int(buffer.numel_unpadded),
                    "parameters": sorted(parameters, key=lambda item: item["name"]),
                }
            )
        if not buffer_schema:
            raise RuntimeError(f"managed optimizer leaf path {path} has no buffers")

        group_schema = []
        for group_index, group_range in enumerate(group_ranges):
            group = group_range["orig_group"]
            group_schema.append(
                {
                    "ownership": _ownership_tuple(group, group_index),
                    "fields": sorted(_normalized_group_keys(group) - {"params"}),
                }
            )
        identities[path] = {
            "version": _MANAGED_IDENTITY_VERSION,
            "tree_path": list(path),
            "data_parallel_group_index": int(leaf.data_parallel_group_idx),
            "buffer_signature": _stable_digest(buffer_schema),
            "group_schema_signature": _stable_digest(group_schema),
        }
    return identities


def _optimizer_tree_map(
    optimizer: Any,
    leaf_fn: Any,
    *,
    path: tuple[int, ...] = (),
    sharded_prefix: str = "",
) -> Any:
    children = get_megatron_optimizer_chain_children(optimizer)
    if children is None:
        return leaf_fn(optimizer, path, sharded_prefix)
    if len(children) == 1:
        return _optimizer_tree_map(
            children[0], leaf_fn, path=(*path, 0), sharded_prefix=sharded_prefix
        )
    return {
        child_index: _optimizer_tree_map(
            child,
            leaf_fn,
            path=(*path, child_index),
            sharded_prefix=f"{sharded_prefix}chained_{child_index}.",
        )
        for child_index, child in enumerate(children)
    }


def _managed_sharded_object(
    leaf: Any, path: tuple[int, ...], prefix: str, name: str, value: Any
) -> Any:
    from megatron.core.dist_checkpointing.mapping import ShardedObject

    data_parallel_group = leaf.data_parallel_group
    return ShardedObject(
        f"{prefix}optimizer.distributed.dp_group_idx_"
        f"{leaf.data_parallel_group_idx}.{name}",
        value,
        (1,),
        (0,),
        replica_id=(
            leaf.distributed_optimizer_instance_id,
            0,
            data_parallel_group.rank(),
        ),
    )


def build_managed_optimizer_outer_template(
    optimizer: Any,
    identities: Mapping[tuple[int, ...], Mapping[str, Any]],
) -> Any:
    """Build outer-only DCP templates without invoking sharded_state_dict()."""
    custom_builder = getattr(optimizer, "managed_checkpoint_outer_template", None)
    if _uses_custom_muon_checkpoint_schema(optimizer):
        if not callable(custom_builder):
            raise RuntimeError("staged Muon checkpoint outer hook is unavailable")
        _check_mcore_checkpoint_compatibility(require_distributed_adamw_schema=False)
        return custom_builder()
    _check_mcore_checkpoint_compatibility()

    def build_leaf(leaf: Any, path: tuple[int, ...], prefix: str) -> dict[str, Any]:
        base = getattr(leaf, "optimizer", leaf)
        if (
            base is None
            or getattr(base, "manages_cpu_residency", False) is not True
            or getattr(leaf, "opt_group_ranges", None) is None
        ):
            return {}
        try:
            identity = dict(identities[path])
        except KeyError as exc:
            raise RuntimeError(
                f"managed optimizer leaf path {path} lacks checkpoint identity"
            ) from exc
        return {
            "optimizer": _managed_sharded_object(leaf, path, prefix, "optimizer", None),
            _MANAGED_IDENTITY_KEY: _managed_sharded_object(
                leaf, path, prefix, _MANAGED_IDENTITY_KEY, identity
            ),
        }

    return _optimizer_tree_map(optimizer, build_leaf)


def is_managed_optimizer_tensor_checkpoint_key(key: Any) -> bool:
    """Recognize only MCore 0.17 tensor/bucket keys omitted by preflight."""
    value = str(key)
    if value.startswith(
        ("optimizer.gpu_staged_muon.v2.", "optimizer.gpu_staged_muon.rank_")
    ):
        return value.endswith(
            (".master_param", ".momentum_buffer", ".exp_avg", ".exp_avg_sq")
        )
    if ".gbuf_idx_" in value and ".bucket_idx_" in value:
        return value.endswith((".param", ".exp_avg", ".exp_avg_sq"))
    return value.endswith(
        (
            ".per_bucket_numel",
            ".per_bucket_numel_unpadded",
        )
    ) or any(
        marker in value
        for marker in (".per_bucket_numel/", ".per_bucket_numel_unpadded/")
    )


def _is_managed_optimizer_state_tensor_key(key: Any) -> bool:
    value = str(key)
    if value.startswith("optimizer.gpu_staged_muon.v2."):
        return value.endswith(
            (".master_param", ".momentum_buffer", ".exp_avg", ".exp_avg_sq")
        )
    return (
        "optimizer.distributed." in value
        and ".gbuf_idx_" in value
        and ".bucket_idx_" in value
        and value.endswith((".param", ".exp_avg", ".exp_avg_sq"))
    )


def _is_managed_optimizer_tensor_namespace_key(key: Any) -> bool:
    """Recognize every tensor inside a managed optimizer state namespace."""
    value = str(key)
    return value.startswith(
        ("optimizer.gpu_staged_muon.v2.", "optimizer.gpu_staged_muon.rank_")
    ) or (
        "optimizer.distributed." in value
        and ".gbuf_idx_" in value
        and ".bucket_idx_" in value
    )


def build_managed_optimizer_tensor_manifest(
    sharded_optimizer_state: Any,
) -> dict[str, tuple[tuple[int, ...], str]]:
    """Describe managed optimizer tensors without reading or writing their data."""
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    manifest: dict[str, tuple[tuple[int, ...], str]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, ShardedTensor):
            if not _is_managed_optimizer_state_tensor_key(value.key):
                return
            descriptor = (tuple(value.global_shape), str(value.dtype))
            previous = manifest.setdefault(value.key, descriptor)
            if previous != descriptor:
                raise ValueError(
                    f"managed optimizer tensor {value.key!r} has inconsistent "
                    f"template metadata: {previous!r} versus {descriptor!r}"
                )
            return
        if isinstance(value, Mapping):
            for child in value.values():
                visit(child)
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    # A rank-local owner shard can legitimately be empty.  Every checkpoint
    # participant must reach the manifest gather; only the merged WORLD union
    # is required to contain optimizer tensors.
    visit(sharded_optimizer_state)
    return manifest


def merge_managed_optimizer_tensor_manifests(
    manifests: Sequence[Mapping[str, tuple[tuple[int, ...], str]]],
) -> dict[str, tuple[tuple[int, ...], str]]:
    """Merge rank-local manifests while rejecting conflicting global metadata."""
    merged: dict[str, tuple[tuple[int, ...], str]] = {}
    owners: dict[str, int] = {}
    for rank, manifest in enumerate(manifests):
        for key, descriptor in manifest.items():
            key = str(key)
            normalized = (tuple(descriptor[0]), str(descriptor[1]))
            previous = merged.setdefault(key, normalized)
            if previous != normalized:
                raise ValueError(
                    f"managed optimizer tensor {key!r} differs across checkpoint "
                    f"participants: {previous!r} versus rank {rank} {normalized!r}"
                )
            if key.startswith("optimizer.gpu_staged_muon.v2."):
                previous_owner = owners.setdefault(key, rank)
                if previous_owner != rank:
                    raise ValueError(
                        "Muon checkpoint logical state has duplicate owners: "
                        f"key={key!r}, ranks=({previous_owner}, {rank})"
                    )
    if not merged:
        raise ValueError("managed optimizer tensor manifest union is empty")
    return merged


def validate_managed_optimizer_source_tensor_metadata(
    checkpoint_path: str,
    expected_manifest: Mapping[str, tuple[tuple[int, ...], str]],
) -> None:
    """Fail before DCP can cast or copy corrupt source tensors into CPU slabs.

    Save-time DP chunks may differ from the current DP ownership.  Therefore
    the contract fixes the global one-dimensional tensor, FP32 dtype, and exact
    no-gap/no-overlap coverage, but deliberately does not compare chunk offsets
    with the current rank-local template.
    """
    import torch
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    try:
        metadata = FileSystemReader(checkpoint_path).read_metadata()
    except BaseException as error:
        raise RuntimeError(
            "managed optimizer requires readable torch_dist checkpoint metadata"
        ) from error

    entries = metadata.state_dict_metadata
    actual_keys = {
        str(key)
        for key, value in entries.items()
        if isinstance(value, TensorStorageMetadata)
        and _is_managed_optimizer_tensor_namespace_key(key)
    }
    expected_keys = set(expected_manifest)
    if actual_keys != expected_keys:
        raise KeyError(
            "managed optimizer source tensor key mismatch: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )

    partitions: dict[str, tuple[tuple[int, int], ...]] = {}
    bucket_states: dict[str, set[str]] = {}

    def metadata_integer(value: Any, *, location: str, minimum: int = 0) -> int:
        if type(value) is not int or value < minimum:
            raise ValueError(
                f"{location} must be an integer >= {minimum}, got {value!r}"
            )
        return value

    for key in sorted(expected_keys):
        entry = entries.get(key)
        if not isinstance(entry, TensorStorageMetadata):
            raise TypeError(
                f"managed optimizer source tensor {key!r} lacks tensor metadata"
            )
        expected_shape, expected_dtype = expected_manifest[key]
        source_dtype = entry.properties.dtype
        if source_dtype is not torch.float32:
            raise TypeError(
                f"managed optimizer source tensor {key!r} dtype must be "
                f"torch.float32, found {source_dtype}"
            )
        if str(source_dtype) != expected_dtype:
            raise TypeError(
                f"managed optimizer source tensor {key!r} dtype mismatch: "
                f"expected {expected_dtype}, found {source_dtype}"
            )
        if entry.properties.layout is not torch.strided:
            raise ValueError(
                f"managed optimizer source tensor {key!r} must use strided layout"
            )
        shape = tuple(
            metadata_integer(
                dimension,
                location=(
                    f"managed optimizer source tensor {key!r} global shape[{index}]"
                ),
            )
            for index, dimension in enumerate(entry.size)
        )
        normalized_expected_shape = tuple(
            metadata_integer(
                dimension,
                location=(f"managed optimizer expected tensor {key!r} shape[{index}]"),
            )
            for index, dimension in enumerate(expected_shape)
        )
        if shape != normalized_expected_shape or len(shape) != 1:
            raise ValueError(
                f"managed optimizer source tensor {key!r} global shape mismatch: "
                f"expected {normalized_expected_shape}, found {shape}"
            )

        chunks: list[tuple[int, int]] = []
        for chunk_index, chunk in enumerate(entry.chunks):
            offsets = tuple(chunk.offsets)
            sizes = tuple(chunk.sizes)
            if len(offsets) != 1 or len(sizes) != 1:
                raise ValueError(
                    f"managed optimizer source tensor {key!r} chunk {chunk_index} "
                    "must be one-dimensional"
                )
            offset = metadata_integer(
                offsets[0],
                location=(
                    f"managed optimizer source tensor {key!r} chunk "
                    f"{chunk_index} offset"
                ),
            )
            size = metadata_integer(
                sizes[0],
                location=(
                    f"managed optimizer source tensor {key!r} chunk {chunk_index} size"
                ),
                minimum=1,
            )
            if offset + size > shape[0]:
                raise ValueError(
                    f"managed optimizer source tensor {key!r} chunk {chunk_index} "
                    f"is out of bounds: offset={offset}, size={size}, shape={shape}"
                )
            chunks.append((offset, size))
        chunks.sort()
        if key.startswith("optimizer.gpu_staged_muon.v2.") and chunks != [
            (0, shape[0])
        ]:
            raise ValueError(
                f"Muon checkpoint tensor {key!r} must have exactly one complete "
                f"payload, found chunks={chunks!r}"
            )
        cursor = 0
        for offset, size in chunks:
            if offset != cursor:
                relation = "overlap" if offset < cursor else "gap"
                raise ValueError(
                    f"managed optimizer source tensor {key!r} chunk coverage has "
                    f"a {relation} at offset {offset}; expected {cursor}"
                )
            cursor += size
        if cursor != shape[0]:
            raise ValueError(
                f"managed optimizer source tensor {key!r} chunk coverage ends at "
                f"{cursor}, expected {shape[0]}"
            )
        partition = tuple(chunks)
        prefix, state_name = key.rsplit(".", 1)
        partitions[key] = partition
        bucket_states.setdefault(prefix, set()).add(state_name)

    for prefix, states in bucket_states.items():
        expected_states = (
            {"master_param", "momentum_buffer"}
            if "optimizer.gpu_staged_muon." in prefix and "momentum_buffer" in states
            else (
                {"master_param", "exp_avg", "exp_avg_sq"}
                if "optimizer.gpu_staged_muon." in prefix
                else {"param", "exp_avg", "exp_avg_sq"}
            )
        )
        if states != expected_states:
            raise KeyError(
                f"managed optimizer bucket {prefix!r} state mismatch: "
                f"missing={sorted(expected_states - states)}, "
                f"unexpected={sorted(states - expected_states)}"
            )
        state_partitions = [
            partitions[f"{prefix}.{name}"] for name in sorted(expected_states)
        ]
        if state_partitions[1:] != state_partitions[:-1]:
            raise ValueError(
                f"managed optimizer bucket {prefix!r} state tensors have different "
                "source chunk partitions"
            )


def attach_managed_optimizer_identities(
    optimizer: Any,
    sharded_state: Any,
    identities: Mapping[tuple[int, ...], Mapping[str, Any]],
) -> None:
    """Attach leaf identity beside its MCore outer state for save/load."""
    if _uses_custom_muon_checkpoint_schema(optimizer):
        return

    def attach(
        node: Any,
        state: Any,
        path: tuple[int, ...],
        prefix: str,
    ) -> None:
        children = get_megatron_optimizer_chain_children(node)
        if children is None:
            base = getattr(node, "optimizer", node)
            if (
                base is None
                or getattr(base, "manages_cpu_residency", False) is not True
                or getattr(node, "opt_group_ranges", None) is None
            ):
                return
            if not isinstance(state, dict):
                raise TypeError(
                    f"managed optimizer sharded leaf path {path} must be a dict"
                )
            state[_MANAGED_IDENTITY_KEY] = _managed_sharded_object(
                node,
                path,
                prefix,
                _MANAGED_IDENTITY_KEY,
                dict(identities[path]),
            )
            return
        if len(children) == 1:
            attach(children[0], state, (*path, 0), prefix)
            return
        child_states = _indexed_chain_states(state, len(children), path)
        for child_index, (child, child_state) in enumerate(zip(children, child_states)):
            attach(
                child,
                child_state,
                (*path, child_index),
                f"{prefix}chained_{child_index}.",
            )

    attach(optimizer, sharded_state, (), "")


def validate_managed_optimizer_outer_state(
    optimizer: Any | None,
    state_dict: Any,
    identities: Mapping[tuple[int, ...], Mapping[str, Any]],
) -> int:
    """Validate raw MCore groups before its ownership-tuple normalization.

    MCore 0.17 maps checkpoint groups into a dict keyed by four ownership
    fields.  A duplicate checkpoint key is therefore silently last-wins unless
    this validation runs before ``DistributedOptimizer.load_state_dict``.
    Tensor state is validated separately by the slab-backed load template and
    ``GPUStagedAdamW.prepare_checkpoint_load``.
    """
    if optimizer is None:
        return 0
    custom_validator = getattr(
        optimizer, "validate_managed_checkpoint_outer_state", None
    )
    if _uses_custom_muon_checkpoint_schema(optimizer):
        if not callable(custom_validator):
            raise RuntimeError("staged Muon checkpoint validator is unavailable")
        _check_mcore_checkpoint_compatibility(require_distributed_adamw_schema=False)
        custom_validator(state_dict)
        return len(tuple(iter_managed_optimizers(optimizer)))
    validated = 0
    compatibility_checked = False
    for leaf_path, leaf, leaf_state in _managed_outer_leaf_states(
        optimizer, state_dict
    ):
        base = getattr(leaf, "optimizer", leaf)
        if base is None or getattr(base, "manages_cpu_residency", False) is not True:
            continue
        group_ranges = getattr(leaf, "opt_group_ranges", None)
        if group_ranges is None:
            # Managed test doubles and non-MCore wrappers have no raw MCore
            # ownership schema and remain covered by their own strict loader.
            continue
        if not compatibility_checked:
            _check_mcore_checkpoint_compatibility()
            compatibility_checked = True
        if not isinstance(leaf_state, Mapping):
            raise TypeError(
                f"managed optimizer leaf path {leaf_path} outer state must be a mapping"
            )
        unexpected_outer = set(leaf_state) - _OUTER_STATE_KEYS
        if unexpected_outer:
            raise ValueError(
                f"managed optimizer leaf path {leaf_path} has unexpected outer fields: "
                f"{sorted(unexpected_outer)}"
            )
        if _MANAGED_IDENTITY_KEY not in leaf_state:
            raise KeyError(
                f"managed optimizer leaf path {leaf_path} checkpoint lacks stable identity; "
                "pre-identity staged checkpoints are unsupported"
            )
        try:
            expected_identity = identities[leaf_path]
        except KeyError as exc:
            raise RuntimeError(
                f"managed optimizer leaf path {leaf_path} has no runtime identity"
            ) from exc
        if leaf_state[_MANAGED_IDENTITY_KEY] != expected_identity:
            raise ValueError(
                f"managed optimizer leaf path {leaf_path} identity mismatch: "
                f"expected={expected_identity}, "
                f"checkpoint={leaf_state[_MANAGED_IDENTITY_KEY]}"
            )
        optimizer_state = leaf_state.get("optimizer")
        if not isinstance(optimizer_state, Mapping):
            raise TypeError(
                f"managed optimizer leaf path {leaf_path} optimizer state must be a mapping"
            )
        if set(optimizer_state) != _OUTER_OPTIMIZER_KEYS:
            raise ValueError(
                f"managed optimizer leaf path {leaf_path} optimizer fields mismatch: "
                f"expected={sorted(_OUTER_OPTIMIZER_KEYS)}, "
                f"actual={sorted(optimizer_state)}"
            )
        checkpoint_groups = optimizer_state["param_groups"]
        if not isinstance(checkpoint_groups, (list, tuple)):
            raise TypeError(
                f"managed optimizer leaf path {leaf_path} param_groups must be a list or tuple"
            )
        expected_groups = tuple(
            group_range["orig_group"] for group_range in group_ranges
        )
        runtime_groups = tuple(base.param_groups)
        if len(expected_groups) != len(runtime_groups) or any(
            expected is not runtime
            for expected, runtime in zip(expected_groups, runtime_groups)
        ):
            raise RuntimeError(
                f"managed optimizer leaf path {leaf_path} runtime ownership metadata "
                "does not match opt_group_ranges"
            )
        if len(checkpoint_groups) != len(expected_groups):
            raise ValueError(
                f"managed optimizer leaf path {leaf_path} param-group count mismatch: "
                f"expected={len(expected_groups)}, actual={len(checkpoint_groups)}"
            )

        expected_by_ownership: dict[tuple[Any, ...], tuple[int, Mapping[str, Any]]] = {}
        for group_index, group in enumerate(expected_groups):
            if not isinstance(group, Mapping):
                raise TypeError(
                    f"managed optimizer leaf path {leaf_path} expected param-group "
                    f"{group_index} is not a mapping"
                )
            ownership = _ownership_tuple(group, group_index)
            if ownership in expected_by_ownership:
                other_index = expected_by_ownership[ownership][0]
                raise RuntimeError(
                    f"runtime optimizer has duplicate ownership {ownership} in "
                    f"param-groups {other_index} and {group_index}"
                )
            expected_by_ownership[ownership] = (group_index, group)

        checkpoint_by_ownership: dict[
            tuple[Any, ...], tuple[int, Mapping[str, Any]]
        ] = {}
        for group_index, group in enumerate(checkpoint_groups):
            if not isinstance(group, Mapping):
                raise TypeError(
                    f"managed optimizer leaf path {leaf_path} checkpoint param-group "
                    f"{group_index} is not a mapping"
                )
            ownership = _ownership_tuple(group, group_index)
            if ownership in checkpoint_by_ownership:
                other_index = checkpoint_by_ownership[ownership][0]
                raise ValueError(
                    f"checkpoint has duplicate param-group ownership {ownership} at "
                    f"indices {other_index} and {group_index}"
                )
            checkpoint_by_ownership[ownership] = (group_index, group)

        expected_ownership = set(expected_by_ownership)
        checkpoint_ownership = set(checkpoint_by_ownership)
        if checkpoint_ownership != expected_ownership:
            raise ValueError(
                f"managed optimizer leaf path {leaf_path} ownership mismatch: "
                f"missing={sorted(expected_ownership - checkpoint_ownership, key=repr)}, "
                f"unexpected={sorted(checkpoint_ownership - expected_ownership, key=repr)}"
            )
        for ownership, (group_index, group) in checkpoint_by_ownership.items():
            expected_group = expected_by_ownership[ownership][1]
            _validate_group_metadata(group, expected_group, group_index, leaf_path)
        validated += 1
    return validated


def has_managed_mcore_outer_schema(optimizer: Any | None) -> bool:
    """Return whether a managed leaf exposes MCore DP ownership metadata."""
    if optimizer is None:
        return False
    if _uses_custom_muon_checkpoint_schema(optimizer):
        return True
    for leaf in iter_megatron_optimizer_leaves(optimizer):
        base = getattr(leaf, "optimizer", leaf)
        if (
            base is not None
            and getattr(base, "manages_cpu_residency", False) is True
            and getattr(leaf, "opt_group_ranges", None) is not None
        ):
            return True
    return False


def managed_optimizer_outer_template(sharded_optimizer_state: Any) -> Any:
    """Retain only opaque MCore optimizer metadata for a DCP preflight load.

    The returned tree contains no ``param_state`` sharded tensors, so loading
    it cannot write slab-backed views.  ChainedOptimizer uses integer-indexed
    mappings recursively; a DistributedOptimizer leaf owns an ``optimizer``
    ShardedObject containing the raw param-group metadata.
    """
    if not isinstance(sharded_optimizer_state, Mapping):
        raise TypeError("managed optimizer sharded state must be a mapping")
    if "optimizer" in sharded_optimizer_state:
        return {"optimizer": sharded_optimizer_state["optimizer"]}
    if not sharded_optimizer_state:
        raise ValueError("managed optimizer sharded state tree is empty")
    if any(not isinstance(index, int) for index in sharded_optimizer_state):
        raise TypeError(
            "managed chained optimizer sharded state keys must be integer indices"
        )
    expected = set(range(len(sharded_optimizer_state)))
    if set(sharded_optimizer_state) != expected:
        raise ValueError(
            "managed chained optimizer sharded state indices mismatch: "
            f"expected={sorted(expected)}, actual={sorted(sharded_optimizer_state)}"
        )
    return {
        index: managed_optimizer_outer_template(sharded_optimizer_state[index])
        for index in range(len(sharded_optimizer_state))
    }


class ManagedCheckpointTransactionPhase(Enum):
    """Coordinator-visible lifecycle for one synchronous managed load."""

    LOAD_ACTIVE = auto()
    ROLLBACK_PENDING = auto()
    POISONED = auto()
    RELOAD_REQUIRED = auto()
    COMMIT_DECIDED = auto()
    CLEANUP_PENDING = auto()
    CLEAN = auto()


@dataclass
class ManagedCheckpointCommitToken:
    """One shared irreversible decision bit observed by every local leaf."""

    decided: bool = False


@dataclass(eq=False)
class ManagedCheckpointAttemptToken:
    """Unique process-local authority for one coordinated begin attempt."""


@dataclass(eq=False)
class ManagedCheckpointReloadGeneration:
    """Manager-owned authority for one poison-to-replacement recovery cycle."""

    active_attempt: ManagedCheckpointAttemptToken | None = None


@dataclass(frozen=True)
class ManagedCheckpointBeginAuthority:
    """A leaf that proved it acquired one exact begin-attempt token."""

    leaf: Any
    attempt_token: Any
    token_aware: bool
    recovery_action_token: object = field(default_factory=object)


@dataclass
class ManagedCheckpointCleanupEntry:
    """One leaf's token-bound post-commit cleanup progress."""

    leaf: Any
    commit_token: ManagedCheckpointCommitToken
    action_token: object = field(default_factory=object)
    decision_pending: bool = True
    cleanup_pending: bool = True
    receipt_release_started: bool = False
    diagnostics: list[str] = field(default_factory=list)


@dataclass
class ManagedCheckpointCleanupJournal:
    """Post-commit transitions and references that can only be discarded."""

    entries: list[ManagedCheckpointCleanupEntry]


@dataclass
class ManagedCheckpointLoadTransaction:
    """Operation-local journal shared by all managed optimizer leaves."""

    leaves: tuple[Any, ...]
    attempt_token: ManagedCheckpointAttemptToken = field(
        default_factory=ManagedCheckpointAttemptToken
    )
    begun: list[ManagedCheckpointBeginAuthority] = field(default_factory=list)
    recovery_authorities: list[ManagedCheckpointBeginAuthority] = field(
        default_factory=list
    )
    recovery_completed: list[Any] = field(default_factory=list)
    recovery_owner: bool = False
    reload_generation: ManagedCheckpointReloadGeneration | None = None
    replacement_generation: ManagedCheckpointReloadGeneration | None = None
    replacement_attempt: bool = False
    snapshot_configured: list[Any] = field(default_factory=list)
    prepared: bool = False
    commit_prepared: bool = False
    committed: bool = False
    poisoned: bool = False
    phase: ManagedCheckpointTransactionPhase = (
        ManagedCheckpointTransactionPhase.LOAD_ACTIVE
    )
    commit_token: ManagedCheckpointCommitToken | None = field(
        default_factory=ManagedCheckpointCommitToken
    )
    prepared_cleanup_journal: ManagedCheckpointCleanupJournal | None = None
    cleanup_journal: ManagedCheckpointCleanupJournal | None = None
    cleanup_receipt_releases: list[ManagedCheckpointCleanupEntry] = field(
        default_factory=list
    )
    rollback_diagnostics: list[str] = field(default_factory=list)
    post_commit_error: BaseException | None = None

    @property
    def cleanup_pending(self) -> list[Any]:
        journal_entries = (
            [] if self.cleanup_journal is None else self.cleanup_journal.entries
        )
        pending = [
            entry.leaf
            for entry in journal_entries
            if entry.decision_pending or entry.cleanup_pending
        ]
        pending.extend(entry.leaf for entry in self.cleanup_receipt_releases)
        return pending


class ManagedCheckpointPhaseError(RuntimeError):
    """Uniform error reported by every rank after one failed phase vote."""

    def __init__(
        self,
        phase: str,
        failures: Sequence[Mapping[str, Any]],
        local_error: BaseException | None = None,
    ) -> None:
        summary = "; ".join(
            f"global_rank={failure['global_rank']} "
            f"{failure['error_type']}: {failure['message']}"
            for failure in failures
        )
        super().__init__(f"managed checkpoint phase {phase!r} failed: {summary}")
        self.phase = phase
        self.failures = tuple(dict(failure) for failure in failures)
        self.local_error = local_error


def vote_managed_checkpoint_phase(
    process_group: Any,
    phase: str,
    local_error: BaseException | None,
    *,
    details: Mapping[str, Any] | None = None,
    require_consistent_details: bool = False,
) -> ManagedCheckpointPhaseError | None:
    """Collect one explicit status from every checkpoint participant rank."""
    import torch.distributed as dist

    if process_group is None:
        raise RuntimeError(
            f"managed checkpoint phase {phase!r} requires an explicit process group"
        )
    if not dist.is_initialized():
        raise RuntimeError(
            f"managed checkpoint phase {phase!r} requires initialized distributed state"
        )
    status = {
        "ok": local_error is None,
        "phase": phase,
        "global_rank": dist.get_rank(),
        "group_rank": dist.get_rank(process_group),
        "error_type": type(local_error).__name__ if local_error else "",
        "message": str(local_error)[:1024] if local_error else "",
        "details": dict(details or {}),
    }
    statuses: list[dict[str, Any] | None] = [
        None for _ in range(dist.get_world_size(process_group))
    ]
    try:
        dist.all_gather_object(statuses, status, group=process_group)
    except BaseException as vote_error:
        raise RuntimeError(
            f"managed checkpoint status vote failed during phase {phase!r}"
        ) from vote_error
    failures = [item for item in statuses if item is not None and not item["ok"]]
    if not failures and require_consistent_details:
        reference = statuses[0]["details"] if statuses[0] is not None else {}
        inconsistent = [
            item
            for item in statuses
            if item is not None and item["details"] != reference
        ]
        if inconsistent:
            failures = [
                {
                    **item,
                    "error_type": "CheckpointRequestMismatch",
                    "message": (
                        f"request details {item['details']!r} do not match "
                        f"group-rank 0 details {reference!r}"
                    ),
                }
                for item in inconsistent
            ]
    if not failures:
        return None
    return ManagedCheckpointPhaseError(phase, failures, local_error)


def _add_rollback_note(error: BaseException, rollback_error: BaseException) -> None:
    error.add_note(
        f"GPU-staged optimizer checkpoint rollback failed: {rollback_error!r}"
    )


def _mark_transaction_poisoned(
    transaction: ManagedCheckpointLoadTransaction,
    error: BaseException,
    *,
    leaves: Sequence[Any] | None = None,
) -> None:
    if transaction.committed or transaction.phase in (
        ManagedCheckpointTransactionPhase.COMMIT_DECIDED,
        ManagedCheckpointTransactionPhase.CLEANUP_PENDING,
    ):
        transaction.post_commit_error = error
        return
    transaction.poisoned = True
    # A retained recovery transaction already owns the exact leaf attempt and
    # action identities required for retry.  A control-plane failure must keep
    # that authority intact instead of publishing a second poison generation.
    if transaction.recovery_owner:
        if transaction.phase is not ManagedCheckpointTransactionPhase.RELOAD_REQUIRED:
            transaction.phase = ManagedCheckpointTransactionPhase.POISONED
        return
    transaction.phase = ManagedCheckpointTransactionPhase.POISONED
    poison_leaves = transaction.leaves if leaves is None else leaves
    seen: list[Any] = []
    for inner in poison_leaves:
        if any(previous is inner for previous in seen):
            continue
        seen.append(inner)
        marker = getattr(inner, "mark_checkpoint_poisoned", None)
        if callable(marker):
            try:
                marker(error)
            except BaseException as marker_error:
                _add_rollback_note(error, marker_error)


def poison_managed_checkpoint_transaction(
    transaction: ManagedCheckpointLoadTransaction, error: BaseException
) -> None:
    """Fail-close every local managed leaf after a distributed rollback fault."""
    _mark_transaction_poisoned(transaction, error)


def prepare_managed_checkpoint_save(
    optimizer: Any | None, *, async_save: bool
) -> tuple[Any, ...]:
    """Drain managed optimizers before exposing their CPU-authoritative state."""
    managed = tuple(iter_managed_optimizers(optimizer))
    if not managed:
        return ()
    _check_mcore_checkpoint_compatibility(require_distributed_adamw_schema=False)
    # The manager installs a mutation fence before asking MCore to build an
    # async request.  Re-entering this helper while that fence exists would
    # wait on the request currently being constructed, so async setup owns its
    # one initial drain explicitly.
    if async_save:
        unsupported = [
            inner
            for inner in managed
            if getattr(inner, "supports_managed_async_checkpoint", True) is not True
        ]
        if unsupported:
            raise RuntimeError(
                "managed asynchronous checkpoint is not supported for staged Muon"
            )
        return managed
    for inner in managed:
        inner.prepare_checkpoint_save()
    return managed


def begin_managed_async_checkpoint_save(
    optimizer: Any | None,
    *,
    checkpoint_id: str,
    path: str,
    control_group: Any,
    wait_fn: Callable[[], None],
    identities: Mapping[tuple[int, ...], Mapping[str, Any]],
) -> tuple[Any, ...]:
    """Install every leaf fence before MCore observes a source slab view."""
    _check_mcore_checkpoint_compatibility(require_distributed_adamw_schema=False)
    begun: list[Any] = []
    try:
        for tree_path, leaf in iter_megatron_optimizer_leaf_paths(optimizer):
            base = getattr(leaf, "optimizer", leaf)
            if (
                base is None
                or getattr(base, "manages_cpu_residency", False) is not True
            ):
                continue
            begin = getattr(base, "begin_async_checkpoint_save", None)
            if not callable(begin):
                raise RuntimeError(
                    "managed optimizer lacks begin_async_checkpoint_save()"
                )
            try:
                identity = identities[tree_path]
            except KeyError as error:
                raise RuntimeError(
                    f"managed optimizer leaf path {tree_path} lacks async identity"
                ) from error
            begin(
                checkpoint_id=checkpoint_id,
                path=path,
                control_group=control_group,
                wait_fn=wait_fn,
                leaf_identity=dict(identity),
            )
            begun.append(base)
    except BaseException as error:
        for base in begun:
            fail = getattr(base, "fail_async_checkpoint_save", None)
            if callable(fail):
                fail(error)
        raise
    return tuple(begun)


def bind_managed_async_checkpoint_request(
    leaves: Sequence[Any], request: Any, call_idx: int
) -> None:
    for leaf in leaves:
        leaf.bind_async_checkpoint_request(request, call_idx)


def complete_managed_async_checkpoint_save(leaves: Sequence[Any]) -> None:
    """Validate source generations before releasing their mutation fences."""
    errors: list[BaseException] = []
    for leaf in leaves:
        try:
            leaf.complete_async_checkpoint_save()
        except BaseException as error:
            errors.append(error)
    if errors:
        primary = errors[0]
        for error in errors[1:]:
            primary.add_note(f"another managed async source failed: {error!r}")
        raise primary


def fail_managed_async_checkpoint_save(
    leaves: Sequence[Any], error: BaseException
) -> None:
    for leaf in leaves:
        try:
            leaf.fail_async_checkpoint_save(error)
        except BaseException as leaf_error:
            error.add_note(f"failed to poison managed async leaf: {leaf_error!r}")


_MISSING_ATTEMPT_AUTHORITY = object()


def _leaf_checkpoint_attempt_token(leaf: Any) -> tuple[bool, Any | None]:
    """Read a token-aware leaf's authority exactly once.

    Structural test doubles predating the managed protocol remain supported,
    but production leaves expose this property and are always token checked.
    """
    static_value = inspect.getattr_static(
        leaf, "checkpoint_load_attempt_token", _MISSING_ATTEMPT_AUTHORITY
    )
    if static_value is _MISSING_ATTEMPT_AUTHORITY:
        return False, None
    return True, getattr(leaf, "checkpoint_load_attempt_token")


def _accepts_keyword(callback: Callable[..., Any], name: str) -> bool:
    parameters = inspect.signature(callback).parameters
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _validate_begin_authority(
    authority: ManagedCheckpointBeginAuthority,
    *,
    allow_released: bool = False,
) -> bool:
    """Verify leaf identity and token before using rollback authority.

    ``False`` means this exact authority was already released, which makes a
    repeated abort idempotent.  A different live token always belongs to a
    different transaction and must never be touched.
    """
    if not authority.token_aware:
        return True
    token_aware, current = _leaf_checkpoint_attempt_token(authority.leaf)
    if not token_aware:
        raise RuntimeError("managed checkpoint leaf lost its attempt-token capability")
    if current is None and allow_released:
        return False
    if current is not authority.attempt_token:
        raise RuntimeError(
            "managed checkpoint leaf is owned by a different begin attempt"
        )
    return True


def _invoke_attempt_callback(
    callback: Callable[..., Any],
    authority: ManagedCheckpointBeginAuthority,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if authority.token_aware:
        if not _accepts_keyword(callback, "attempt_token"):
            raise RuntimeError(
                "token-aware managed optimizer callback lacks attempt_token"
            )
        kwargs["attempt_token"] = authority.attempt_token
    return callback(*args, **kwargs)


def create_managed_checkpoint_load_transaction(
    optimizer: Any | None,
) -> ManagedCheckpointLoadTransaction:
    """Create an unbegun transaction for a distributed phase coordinator."""
    transaction = ManagedCheckpointLoadTransaction(
        tuple(iter_managed_optimizers(optimizer))
    )
    reload_generations: list[Any] = []
    recovery_required = False
    recovery_lifecycles: list[str] = []
    for leaf in transaction.leaves:
        token_aware, current = _leaf_checkpoint_attempt_token(leaf)
        if token_aware and current is not None:
            transaction.recovery_authorities.append(
                ManagedCheckpointBeginAuthority(leaf, current, True)
            )
        lifecycle = getattr(leaf, "checkpoint_lifecycle", "CLEAN")
        if lifecycle in {"RECOVERY_PENDING", "POISONED", "RELOAD_REQUIRED"}:
            recovery_required = True
            recovery_lifecycles.append(lifecycle)
        generation = getattr(leaf, "checkpoint_reload_generation", None)
        if generation is not None and not any(
            generation is previous for previous in reload_generations
        ):
            reload_generations.append(generation)
    if len(reload_generations) > 1:
        raise RuntimeError(
            "managed checkpoint leaves belong to different reload generations"
        )
    if reload_generations:
        transaction.reload_generation = reload_generations[0]
    elif recovery_required:
        transaction.reload_generation = ManagedCheckpointReloadGeneration()
    if recovery_required:
        transaction.recovery_owner = True
        if recovery_lifecycles and set(recovery_lifecycles) == {"RELOAD_REQUIRED"}:
            transaction.phase = ManagedCheckpointTransactionPhase.RELOAD_REQUIRED
    return transaction


def begin_managed_checkpoint_replacement(
    transaction: ManagedCheckpointLoadTransaction,
) -> None:
    """Rotate one attempt token while retaining the manager reload generation."""
    if transaction.phase is not ManagedCheckpointTransactionPhase.RELOAD_REQUIRED:
        raise RuntimeError("managed replacement load requires RELOAD_REQUIRED state")
    if transaction.reload_generation is None:
        raise RuntimeError("managed replacement load is missing its reload generation")
    if transaction.recovery_authorities or transaction.recovery_completed:
        raise RuntimeError(
            "managed replacement load started before recovery acknowledgement"
        )
    if transaction.begun or transaction.snapshot_configured:
        raise RuntimeError("managed replacement attempt still owns leaf authority")
    if transaction.reload_generation.active_attempt is not None:
        raise RuntimeError("managed reload generation already has an active attempt")
    transaction.attempt_token = ManagedCheckpointAttemptToken()
    transaction.reload_generation.active_attempt = transaction.attempt_token
    transaction.commit_token = ManagedCheckpointCommitToken()
    transaction.replacement_generation = transaction.reload_generation
    transaction.replacement_attempt = True
    transaction.recovery_owner = False
    transaction.prepared = False
    transaction.commit_prepared = False
    transaction.prepared_cleanup_journal = None
    transaction.rollback_diagnostics.clear()
    transaction.phase = ManagedCheckpointTransactionPhase.LOAD_ACTIVE


def cancel_managed_checkpoint_replacement_configuration(
    transaction: ManagedCheckpointLoadTransaction,
) -> None:
    """Release only pre-begin snapshot authority from the current replacement."""
    if transaction.replacement_generation is None:
        return
    errors: list[BaseException] = []
    pending: list[Any] = []
    for leaf in reversed(transaction.snapshot_configured):
        cancel = getattr(leaf, "cancel_checkpoint_snapshot_configuration", None)
        if not callable(cancel):
            errors.append(
                RuntimeError(
                    "managed optimizer lacks replacement snapshot cancellation"
                )
            )
            pending.append(leaf)
            continue
        try:
            cancel(
                transaction.replacement_generation,
                transaction.attempt_token,
            )
        except BaseException as error:
            errors.append(error)
            pending.append(leaf)
    transaction.snapshot_configured[:] = reversed(pending)
    if errors:
        transaction.recovery_owner = True
        transaction.replacement_attempt = False
        transaction.phase = ManagedCheckpointTransactionPhase.POISONED
        primary = errors[0]
        for error in errors[1:]:
            primary.add_note(
                f"additional replacement snapshot cancellation failure: {error!r}"
            )
        raise primary
    transaction.snapshot_configured.clear()
    if transaction.reload_generation is not None:
        transaction.reload_generation.active_attempt = None
    transaction.replacement_generation = None
    transaction.replacement_attempt = False
    transaction.recovery_owner = True
    transaction.phase = ManagedCheckpointTransactionPhase.RELOAD_REQUIRED


def validate_managed_checkpoint_load_request(
    optimizer: Any | None, *, with_model: bool, with_optimizer: bool
) -> None:
    """Reject staged-Muon partial loads before model or optimizer mutation."""
    managed = tuple(iter_managed_optimizers(optimizer))
    has_fixed_muon = any(
        getattr(inner, "managed_checkpoint_format", None) == "muon_dp_reshard_v2"
        for inner in managed
    )
    if not has_fixed_muon:
        return
    if with_model and not with_optimizer:
        raise RuntimeError(
            "staged Muon model-only checkpoint load is not supported; "
            "load model and optimizer together"
        )
    if with_optimizer and not with_model:
        raise RuntimeError(
            "staged Muon optimizer-only checkpoint load is not supported; "
            "load model and optimizer together"
        )


def begin_managed_checkpoint_load(
    optimizer: Any | None,
) -> ManagedCheckpointLoadTransaction:
    """Create disk rollback snapshots before DCP can mutate slab-backed views."""
    transaction = create_managed_checkpoint_load_transaction(optimizer)
    try:
        if transaction.phase is ManagedCheckpointTransactionPhase.RELOAD_REQUIRED:
            begin_managed_checkpoint_replacement(transaction)
            for leaf in transaction.leaves:
                authorize = getattr(leaf, "authorize_checkpoint_replacement", None)
                if not callable(authorize):
                    raise RuntimeError(
                        "managed optimizer lacks replacement snapshot authorization"
                    )
                authorize(
                    transaction.replacement_generation,
                    transaction.attempt_token,
                )
                transaction.snapshot_configured.append(leaf)
        preflight_managed_checkpoint_snapshots(transaction)
        apply_begin_managed_checkpoint_load(transaction)
    except BaseException as error:
        abort_managed_checkpoint_load(transaction, error, poison=False)
        raise
    return transaction


def apply_begin_managed_checkpoint_load(
    transaction: ManagedCheckpointLoadTransaction,
) -> None:
    """Begin every local leaf while leaving distributed rollback to its coordinator."""
    managed = transaction.leaves
    if not managed:
        return
    _check_mcore_checkpoint_compatibility(require_distributed_adamw_schema=False)
    for inner in managed:
        begin = getattr(inner, "begin_checkpoint_load", None)
        if not callable(begin):
            raise RuntimeError("managed optimizer lacks begin_checkpoint_load()")
        token_aware, current = _leaf_checkpoint_attempt_token(inner)
        if token_aware and current is not None:
            raise RuntimeError(
                "managed checkpoint leaf is already active under another begin attempt"
            )
        authority = ManagedCheckpointBeginAuthority(
            inner, transaction.attempt_token, token_aware
        )
        try:
            begin_kwargs: dict[str, Any] = {}
            if transaction.replacement_generation is not None and token_aware:
                if not _accepts_keyword(begin, "replacement_generation"):
                    raise RuntimeError(
                        "managed optimizer begin is not replacement-aware"
                    )
                begin_kwargs["replacement_generation"] = (
                    transaction.replacement_generation
                )
            _invoke_attempt_callback(begin, authority, **begin_kwargs)
        except BaseException:
            # A leaf may publish its token before a later snapshot action
            # fails.  Only that positive proof grants this transaction abort
            # authority; a pre-existing different token is never journaled.
            if token_aware:
                _, acquired = _leaf_checkpoint_attempt_token(inner)
                if acquired is transaction.attempt_token:
                    transaction.begun.append(authority)
            raise
        if token_aware:
            _, acquired = _leaf_checkpoint_attempt_token(inner)
            if acquired is not transaction.attempt_token:
                raise RuntimeError(
                    "managed optimizer begin returned without publishing its "
                    "attempt authority"
                )
        transaction.begun.append(authority)


def prepare_managed_checkpoint_recovery(
    transaction: ManagedCheckpointLoadTransaction,
    *,
    retain_authority: bool = False,
) -> None:
    """Best-effort retained rollback retry before a rank-global full reload."""
    begun_leaves = {id(authority.leaf) for authority in transaction.begun}
    prebegin_configuration_errors: list[BaseException] = []
    pending_configuration: list[Any] = []
    if transaction.replacement_generation is not None:
        for leaf in transaction.snapshot_configured:
            if id(leaf) in begun_leaves:
                pending_configuration.append(leaf)
                continue
            cancel = getattr(leaf, "cancel_checkpoint_snapshot_configuration", None)
            if not callable(cancel):
                error = RuntimeError(
                    "managed optimizer lacks replacement snapshot cancellation"
                )
                prebegin_configuration_errors.append(error)
                pending_configuration.append(leaf)
                continue
            try:
                cancel(
                    transaction.replacement_generation,
                    transaction.attempt_token,
                )
            except BaseException as error:
                prebegin_configuration_errors.append(error)
                pending_configuration.append(leaf)
    transaction.snapshot_configured[:] = pending_configuration
    if prebegin_configuration_errors:
        primary = prebegin_configuration_errors[0]
        for error in prebegin_configuration_errors[1:]:
            primary.add_note(
                f"additional replacement snapshot recovery failure: {error!r}"
            )
        raise primary
    recovery_errors: list[BaseException] = []
    for inner in transaction.leaves:
        if any(completed is inner for completed in transaction.recovery_completed):
            continue
        recover = getattr(inner, "prepare_checkpoint_recovery", None)
        if not callable(recover):
            recovery_errors.append(
                RuntimeError("managed optimizer lacks prepare_checkpoint_recovery()")
            )
            continue
        try:
            authority = next(
                (
                    candidate
                    for candidate in transaction.recovery_authorities
                    if candidate.leaf is inner
                ),
                None,
            )
            if authority is None:
                recovery_kwargs = {}
                if _accepts_keyword(recover, "reload_generation"):
                    recovery_kwargs["reload_generation"] = transaction.reload_generation
                recover(**recovery_kwargs)
            else:
                leaf_generation = getattr(inner, "checkpoint_reload_generation", None)
                if (
                    leaf_generation is not None
                    and leaf_generation is not transaction.reload_generation
                ):
                    raise RuntimeError(
                        "managed checkpoint recovery generation mismatch"
                    )
                receipt_matches = getattr(
                    inner, "checkpoint_recovery_receipt_matches", None
                )
                receipt_kwargs: dict[str, Any] = {}
                if callable(receipt_matches) and _accepts_keyword(
                    receipt_matches, "reload_generation"
                ):
                    receipt_kwargs["reload_generation"] = transaction.reload_generation
                if callable(receipt_matches) and receipt_matches(
                    authority.attempt_token,
                    authority.recovery_action_token,
                    **receipt_kwargs,
                ):
                    confirm = getattr(inner, "confirm_checkpoint_recovery", None)
                    if callable(confirm):
                        confirm_kwargs: dict[str, Any] = {}
                        if _accepts_keyword(confirm, "reload_generation"):
                            confirm_kwargs["reload_generation"] = (
                                transaction.reload_generation
                            )
                        confirm(
                            authority.attempt_token,
                            authority.recovery_action_token,
                            **confirm_kwargs,
                        )
                else:
                    _validate_begin_authority(authority)
                    binder = getattr(inner, "bind_checkpoint_recovery_action", None)
                    if callable(binder):
                        binder(
                            authority.attempt_token,
                            authority.recovery_action_token,
                        )
                    recovery_kwargs = {}
                    if _accepts_keyword(recover, "recovery_action_token"):
                        recovery_kwargs["recovery_action_token"] = (
                            authority.recovery_action_token
                        )
                    if _accepts_keyword(recover, "reload_generation"):
                        recovery_kwargs["reload_generation"] = (
                            transaction.reload_generation
                        )
                    _invoke_attempt_callback(recover, authority, **recovery_kwargs)
                    confirm = getattr(inner, "confirm_checkpoint_recovery", None)
                    if callable(confirm):
                        confirm_kwargs = {}
                        if _accepts_keyword(confirm, "reload_generation"):
                            confirm_kwargs["reload_generation"] = (
                                transaction.reload_generation
                            )
                        confirm(
                            authority.attempt_token,
                            authority.recovery_action_token,
                            **confirm_kwargs,
                        )
        except BaseException as recovery_error:
            recovery_errors.append(recovery_error)
        else:
            transaction.recovery_completed.append(inner)
    if recovery_errors:
        primary = recovery_errors[0]
        for recovery_error in recovery_errors[1:]:
            primary.add_note(
                f"additional managed checkpoint recovery failure: {recovery_error!r}"
            )
        raise primary
    configuration_errors: list[BaseException] = []
    pending_configuration: list[Any] = []
    if transaction.replacement_generation is not None:
        for leaf in reversed(transaction.snapshot_configured):
            cancel = getattr(leaf, "cancel_checkpoint_snapshot_configuration", None)
            if not callable(cancel):
                error = RuntimeError(
                    "managed optimizer lacks replacement snapshot cancellation"
                )
                configuration_errors.append(error)
                pending_configuration.append(leaf)
                continue
            try:
                cancel(
                    transaction.replacement_generation,
                    transaction.attempt_token,
                )
            except BaseException as error:
                configuration_errors.append(error)
                pending_configuration.append(leaf)
    transaction.snapshot_configured[:] = reversed(pending_configuration)
    if configuration_errors:
        primary = configuration_errors[0]
        for error in configuration_errors[1:]:
            primary.add_note(
                f"additional replacement snapshot recovery failure: {error!r}"
            )
        raise primary
    transaction.snapshot_configured.clear()
    if transaction.reload_generation is not None:
        transaction.reload_generation.active_attempt = None
    transaction.replacement_generation = None
    transaction.replacement_attempt = False
    transaction.recovery_owner = True
    transaction.begun.clear()
    transaction.prepared = False
    transaction.commit_prepared = False
    transaction.prepared_cleanup_journal = None
    transaction.rollback_diagnostics.clear()
    transaction.phase = ManagedCheckpointTransactionPhase.RELOAD_REQUIRED
    if not retain_authority:
        acknowledge_managed_checkpoint_recovery(transaction)


def acknowledge_managed_checkpoint_recovery(
    transaction: ManagedCheckpointLoadTransaction,
) -> None:
    """Release recovery identities only after every checkpoint rank voted success."""
    if transaction.phase is not ManagedCheckpointTransactionPhase.RELOAD_REQUIRED:
        raise RuntimeError("managed checkpoint recovery is not ready to acknowledge")
    if transaction.reload_generation is None:
        raise RuntimeError("managed checkpoint recovery lacks a reload generation")
    transaction.recovery_authorities.clear()
    transaction.recovery_completed.clear()
    transaction.poisoned = False


def prepare_managed_checkpoint_load(
    transaction: ManagedCheckpointLoadTransaction,
) -> None:
    """Validate all leaves without releasing any rollback snapshot."""
    if transaction.committed:
        raise RuntimeError("managed checkpoint transaction is already committed")
    for authority in transaction.begun:
        _validate_begin_authority(authority)
        inner = authority.leaf
        prepare = getattr(inner, "prepare_checkpoint_load", None)
        if not callable(prepare):
            raise RuntimeError(
                "managed optimizer lacks two-phase prepare_checkpoint_load()"
            )
        prepare()
    transaction.prepared = True


def prepare_managed_checkpoint_commit(
    transaction: ManagedCheckpointLoadTransaction,
) -> None:
    """Run every fallible leaf commit check while all snapshots remain live."""
    if transaction.committed:
        raise RuntimeError("managed checkpoint already has a global commit decision")
    if not transaction.prepared:
        raise RuntimeError("managed checkpoint transaction was not prepared")
    begun_leaves = []
    for authority in transaction.begun:
        _validate_begin_authority(authority)
        begun_leaves.append(authority.leaf)
    preparers = [
        getattr(inner, "prepare_checkpoint_commit", None) for inner in begun_leaves
    ]
    if any(not callable(prepare) for prepare in preparers):
        raise RuntimeError("managed optimizer lacks prepare_checkpoint_commit()")
    deciders = [
        getattr(inner, "decide_checkpoint_commit", None) for inner in begun_leaves
    ]
    if any(not callable(decider) for decider in deciders):
        raise RuntimeError("managed optimizer lacks decide_checkpoint_commit()")
    if transaction.commit_token is None:
        raise RuntimeError("managed checkpoint commit token is missing")
    for prepare in preparers:
        prepare(transaction.commit_token)
    transaction.prepared_cleanup_journal = ManagedCheckpointCleanupJournal(
        [
            ManagedCheckpointCleanupEntry(inner, transaction.commit_token)
            for inner in begun_leaves
        ]
    )
    transaction.commit_prepared = True


def decide_managed_checkpoint_commit(
    transaction: ManagedCheckpointLoadTransaction,
) -> None:
    """Make the sole non-failing commit decision without invoking leaf code."""
    if transaction.committed:
        return
    if not transaction.commit_prepared:
        raise RuntimeError("managed checkpoint commit was not prepared")
    if transaction.prepared_cleanup_journal is None:
        raise RuntimeError("managed checkpoint cleanup journal was not prepared")
    if transaction.commit_token is None:
        raise RuntimeError("managed checkpoint commit token is missing")
    transaction.phase = ManagedCheckpointTransactionPhase.COMMIT_DECIDED
    transaction.committed = True
    transaction.commit_token.decided = True
    transaction.cleanup_journal = transaction.prepared_cleanup_journal
    transaction.prepared_cleanup_journal = None
    transaction.begun.clear()


def retry_managed_checkpoint_cleanup(
    transaction: ManagedCheckpointLoadTransaction,
) -> None:
    """Retry only snapshot cleanups still pending after a commit decision."""
    if not transaction.committed:
        raise RuntimeError("cannot clean snapshots before the global commit decision")
    cleanup_errors: list[BaseException] = []
    if transaction.cleanup_journal is None and not transaction.cleanup_receipt_releases:
        if transaction.phase is ManagedCheckpointTransactionPhase.CLEAN:
            transaction.post_commit_error = None
            transaction.commit_token = None
            return
        raise RuntimeError("committed checkpoint is missing its cleanup journal")
    still_pending: list[ManagedCheckpointCleanupEntry] = []

    def record_failure(
        entry: ManagedCheckpointCleanupEntry, error: BaseException
    ) -> None:
        entry.diagnostics[:] = [repr(error)]
        recorder = getattr(entry.leaf, "record_checkpoint_cleanup_error", None)
        if not callable(recorder):
            return
        try:
            recorder(error)
        except BaseException as diagnostic_error:
            entry.diagnostics[:] = [
                f"{error!r}; cleanup diagnostic failed: {diagnostic_error!r}"
            ]

    cleanup_entries = (
        []
        if transaction.cleanup_journal is None
        else transaction.cleanup_journal.entries
    )
    for entry in cleanup_entries:
        inner = entry.leaf
        if entry.decision_pending:
            decider = getattr(inner, "decide_checkpoint_commit", None)
            if not callable(decider):
                cleanup_error = RuntimeError(
                    "managed optimizer lacks decide_checkpoint_commit()"
                )
                cleanup_errors.append(cleanup_error)
                record_failure(entry, cleanup_error)
                still_pending.append(entry)
                continue
            try:
                decider()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
                record_failure(entry, cleanup_error)
                still_pending.append(entry)
                continue
            entry.decision_pending = False
            entry.diagnostics.clear()
        if not entry.cleanup_pending:
            continue
        binder = getattr(inner, "bind_checkpoint_cleanup_action", None)
        if callable(binder):
            try:
                binder(entry.commit_token, entry.action_token)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
                record_failure(entry, cleanup_error)
                still_pending.append(entry)
                continue
        discard = getattr(inner, "discard_checkpoint_snapshot", None)
        if not callable(discard):
            cleanup_error = RuntimeError(
                "managed optimizer lacks discard_checkpoint_snapshot()"
            )
            cleanup_errors.append(cleanup_error)
            record_failure(entry, cleanup_error)
            still_pending.append(entry)
            continue
        try:
            discard()
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
            record_failure(entry, cleanup_error)
            still_pending.append(entry)
            continue
        acknowledge = getattr(inner, "acknowledge_checkpoint_cleanup", None)
        if callable(acknowledge):
            try:
                acknowledge(entry.commit_token, entry.action_token)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
                record_failure(entry, cleanup_error)
                still_pending.append(entry)
                continue
        confirm = getattr(inner, "confirm_checkpoint_cleanup", None)
        if callable(confirm):
            try:
                confirm(entry.commit_token, entry.action_token)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
                record_failure(entry, cleanup_error)
                still_pending.append(entry)
                continue
        entry.cleanup_pending = False
        if callable(
            getattr(inner, "release_checkpoint_cleanup_receipt", None)
        ) and not any(
            pending is entry for pending in transaction.cleanup_receipt_releases
        ):
            transaction.cleanup_receipt_releases.append(entry)
    if transaction.cleanup_journal is not None:
        transaction.cleanup_journal.entries = still_pending
    if cleanup_errors:
        transaction.phase = ManagedCheckpointTransactionPhase.CLEANUP_PENDING
        primary = cleanup_errors[0]
        transaction.post_commit_error = primary
        for cleanup_error in cleanup_errors[1:]:
            primary.add_note(
                "additional managed checkpoint snapshot cleanup failure: "
                f"{cleanup_error!r}"
            )
        raise primary
    transaction.cleanup_journal = None
    release_errors: list[BaseException] = []
    pending_releases: list[ManagedCheckpointCleanupEntry] = []
    for entry in transaction.cleanup_receipt_releases:
        inner = entry.leaf
        status = getattr(inner, "checkpoint_cleanup_receipt_status", None)
        release = getattr(inner, "release_checkpoint_cleanup_receipt", None)
        if not callable(status) or not callable(release):
            release_error = RuntimeError(
                "managed optimizer cleanup receipt lacks terminal release protocol"
            )
            release_errors.append(release_error)
            record_failure(entry, release_error)
            pending_releases.append(entry)
            continue
        try:
            receipt_status = status(entry.commit_token, entry.action_token)
            if receipt_status == "ABSENT" and entry.receipt_release_started:
                entry.diagnostics.clear()
                continue
            if receipt_status != "MATCH":
                raise RuntimeError(
                    "managed optimizer cleanup receipt disappeared before release"
                )
            entry.receipt_release_started = True
            release(entry.commit_token, entry.action_token)
            if status(entry.commit_token, entry.action_token) != "ABSENT":
                raise RuntimeError(
                    "managed optimizer cleanup receipt release had no effect"
                )
        except BaseException as release_error:
            release_errors.append(release_error)
            record_failure(entry, release_error)
            pending_releases.append(entry)
            continue
        entry.diagnostics.clear()
    transaction.cleanup_receipt_releases = pending_releases
    if release_errors:
        transaction.phase = ManagedCheckpointTransactionPhase.CLEANUP_PENDING
        primary = release_errors[0]
        transaction.post_commit_error = primary
        for release_error in release_errors[1:]:
            primary.add_note(
                "additional managed checkpoint cleanup receipt release failure: "
                f"{release_error!r}"
            )
        raise primary
    transaction.post_commit_error = None
    transaction.commit_token = None
    if transaction.reload_generation is not None:
        transaction.reload_generation.active_attempt = None
    transaction.reload_generation = None
    transaction.replacement_generation = None
    transaction.replacement_attempt = False
    transaction.snapshot_configured.clear()
    transaction.recovery_authorities.clear()
    transaction.recovery_completed.clear()
    transaction.recovery_owner = False
    transaction.phase = ManagedCheckpointTransactionPhase.CLEAN


def commit_managed_checkpoint_load(
    transaction: ManagedCheckpointLoadTransaction,
) -> None:
    """Prepare commit, decide globally, then clean snapshots idempotently."""
    if transaction.committed:
        retry_managed_checkpoint_cleanup(transaction)
        return
    prepare_managed_checkpoint_commit(transaction)
    decide_managed_checkpoint_commit(transaction)
    retry_managed_checkpoint_cleanup(transaction)


def complete_managed_checkpoint_load(
    transaction: ManagedCheckpointLoadTransaction,
) -> None:
    """Backward-compatible prepare plus global commit."""
    prepare_managed_checkpoint_load(transaction)
    commit_managed_checkpoint_load(transaction)


def abort_managed_checkpoint_load(
    transaction: ManagedCheckpointLoadTransaction,
    error: BaseException,
    *,
    poison: bool = True,
) -> None:
    """Best-effort rollback without replacing the original load exception."""
    rollback_errors: list[BaseException] = []
    if (
        not transaction.committed
        and transaction.phase is ManagedCheckpointTransactionPhase.CLEAN
        and not transaction.begun
    ):
        return
    if transaction.committed or transaction.phase in (
        ManagedCheckpointTransactionPhase.COMMIT_DECIDED,
        ManagedCheckpointTransactionPhase.CLEANUP_PENDING,
        ManagedCheckpointTransactionPhase.CLEAN,
    ):
        raise RuntimeError("cannot abort after the global commit decision")
    transaction.phase = ManagedCheckpointTransactionPhase.ROLLBACK_PENDING
    acquired = list(transaction.begun)
    pending: list[ManagedCheckpointBeginAuthority] = []
    for leaf_index in reversed(range(len(acquired))):
        authority = acquired[leaf_index]
        inner = authority.leaf
        try:
            if not _validate_begin_authority(authority, allow_released=True):
                continue
            abort = getattr(inner, "abort_checkpoint_load", None)
            if not callable(abort):
                raise RuntimeError("managed optimizer lacks abort_checkpoint_load()")
            abort_kwargs = (
                {"poison": poison} if _accepts_keyword(abort, "poison") else {}
            )
            if transaction.replacement_generation is not None:
                if not _accepts_keyword(abort, "replacement_generation"):
                    raise RuntimeError(
                        "managed optimizer abort is not replacement-aware"
                    )
                abort_kwargs["replacement_generation"] = (
                    transaction.replacement_generation
                )
            _invoke_attempt_callback(abort, authority, error, **abort_kwargs)
            if authority.token_aware:
                _, current = _leaf_checkpoint_attempt_token(inner)
                if current is authority.attempt_token and not poison:
                    raise RuntimeError(
                        "managed optimizer abort retained completed attempt authority"
                    )
        except BaseException as rollback_error:
            rollback_errors.append(rollback_error)
            if authority.token_aware:
                pending.append(authority)
            transaction.rollback_diagnostics.append(
                f"leaf_index={leaf_index} {type(rollback_error).__name__}: "
                f"{rollback_error}"
            )
    for rollback_error in rollback_errors:
        _add_rollback_note(error, rollback_error)
    transaction.begun[:] = reversed(pending)
    if transaction.replacement_generation is not None:
        pending_configuration: list[Any] = []
        for leaf in reversed(transaction.snapshot_configured):
            cancel = getattr(leaf, "cancel_checkpoint_snapshot_configuration", None)
            if not callable(cancel):
                cleanup_error = RuntimeError(
                    "managed optimizer lacks replacement snapshot cancellation"
                )
                rollback_errors.append(cleanup_error)
                _add_rollback_note(error, cleanup_error)
                pending_configuration.append(leaf)
                continue
            try:
                cancel(
                    transaction.replacement_generation,
                    transaction.attempt_token,
                )
            except BaseException as cleanup_error:
                rollback_errors.append(cleanup_error)
                _add_rollback_note(error, cleanup_error)
                pending_configuration.append(leaf)
        transaction.snapshot_configured[:] = reversed(pending_configuration)
    transaction.prepared = False
    transaction.commit_prepared = False
    transaction.prepared_cleanup_journal = None
    if poison or rollback_errors or transaction.poisoned:
        poison_targets: list[Any] = []
        for authority in acquired:
            if not authority.token_aware:
                poison_targets.append(authority.leaf)
                continue
            try:
                _, current = _leaf_checkpoint_attempt_token(authority.leaf)
            except BaseException as authority_error:
                rollback_errors.append(authority_error)
                _add_rollback_note(error, authority_error)
                continue
            if current is None or current is authority.attempt_token:
                poison_targets.append(authority.leaf)
        _mark_transaction_poisoned(
            transaction,
            error,
            leaves=poison_targets,
        )
        if transaction.replacement_attempt:
            transaction.recovery_owner = True
            transaction.replacement_attempt = False
    else:
        if transaction.replacement_attempt:
            if transaction.reload_generation is not None:
                transaction.reload_generation.active_attempt = None
            transaction.phase = ManagedCheckpointTransactionPhase.RELOAD_REQUIRED
            transaction.recovery_owner = True
            transaction.replacement_attempt = False
            transaction.replacement_generation = None
            transaction.snapshot_configured.clear()
        else:
            transaction.phase = ManagedCheckpointTransactionPhase.CLEAN


def reset_managed_optimizer_from_model(optimizer: Any | None) -> int:
    """Initialize CPU masters from loaded model shards when optimizer is absent."""
    transaction = begin_managed_checkpoint_load(optimizer)
    if not transaction.leaves:
        return 0
    try:
        for inner in transaction.leaves:
            apply_reset = getattr(inner, "apply_model_checkpoint_reset", None)
            if callable(apply_reset):
                apply_reset()
            else:
                inner.reset_from_model_params()
        for inner in transaction.leaves:
            finalize_reset = getattr(inner, "finalize_model_checkpoint_reset", None)
            if callable(finalize_reset):
                finalize_reset()
        prepare_managed_checkpoint_load(transaction)
        commit_managed_checkpoint_load(transaction)
    except BaseException as error:
        if not transaction.committed:
            abort_managed_checkpoint_load(transaction, error, poison=True)
        raise
    return len(transaction.leaves)


def apply_managed_optimizer_reset_from_model(
    transaction: ManagedCheckpointLoadTransaction,
) -> int:
    """Apply model-only reset inside a manager-owned global transaction."""
    for inner in transaction.leaves:
        inner.apply_model_checkpoint_reset()
    for inner in transaction.leaves:
        inner.finalize_model_checkpoint_reset()
    return len(transaction.leaves)
