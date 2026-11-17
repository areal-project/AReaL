# SPDX-License-Identifier: Apache-2.0

"""Full Megatron state transfer through immutable shared-filesystem snapshots.

Rank files contain raw tensor bytes. Native metadata and factories define how
those bytes map to a new TP/DP layout. Call at an idle training boundary.
"""

from __future__ import annotations

import base64
import bisect
import copy
import gc
import hashlib
import json
import math
import mmap
import os
import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from areal.engine.megatron_utils.optimizer_chain import (
    get_managed_base_optimizer,
    iter_megatron_optimizer_leaves,
)

_FORMAT = "rank-raw-files-v1"
_METADATA = {
    "distrib_optim_sharding_type": "fully_reshardable",
    "distrib_optim_fully_reshardable_mem_efficient": True,
}


def validate_engine(engine: Any) -> None:
    """Check the optimizer representation before changing its residency."""
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    leaves = list(iter_megatron_optimizer_leaves(engine.optimizer))
    if not leaves or engine.checkpointer is None:
        raise NotImplementedError(
            "Raw state transfer requires a Megatron distributed optimizer"
        )
    for leaf in leaves:
        if (
            type(leaf) is not DistributedOptimizer
            or get_managed_base_optimizer(leaf) is not None
            or type(leaf.optimizer).__name__ != "FusedAdam"
            or getattr(leaf.config, "use_precision_aware_optimizer", False)
        ):
            raise NotImplementedError(
                "Raw state transfer requires ordinary DistributedOptimizer/FusedAdam"
            )
        if any(
            leaf._get_state_key_dtype(key) != torch.float32
            for key in leaf.optimizer_state_keys
        ):
            raise NotImplementedError(
                "Raw state transfer requires FP32 optimizer moments"
            )


def _initialize_lazy_adam(engine: Any) -> None:
    for leaf in iter_megatron_optimizer_leaves(engine.optimizer):
        if leaf.optimizer.state:
            continue
        if any(int(group.get("step", 0)) for group in leaf.optimizer.param_groups):
            raise ValueError("Optimizer steps are present without optimizer tensors")
        leaf.load_state_dict(leaf.state_dict())
        for parameter in leaf.model_param_group_index_map:
            states = leaf._get_main_param_and_optimizer_states(parameter)
            for key in leaf.optimizer_state_keys:
                states[key].zero_()


@contextmanager
def _meta_buffers(engine: Any):
    """Provide native loading templates without gathering optimizer tensors."""
    saved = []
    try:
        for leaf in iter_megatron_optimizer_leaves(engine.optimizer):
            previous = leaf.__dict__.get("get_parameter_state_dp_zero")

            def empty_state(*, _leaf=leaf, empty_data=False, **kwargs):
                if not empty_data:
                    raise RuntimeError(
                        "State-transfer templates require empty_data=True"
                    )
                return {
                    "buckets_coalesced": True,
                    **{
                        index: {
                            dtype: {
                                "numel_unpadded": _leaf.buffers[index].numel_unpadded,
                                **{
                                    key: torch.empty(
                                        _leaf.buffers[index].numel_unpadded,
                                        device="meta",
                                        dtype=torch.float32,
                                    )
                                    for key in ("param",) + _leaf.optimizer_state_keys
                                },
                            }
                            for dtype in mapping
                        }
                        for (index, mapping) in enumerate(_leaf.gbuf_ranges)
                    },
                }

            leaf.get_parameter_state_dp_zero = empty_state
            saved.append((leaf, previous))
        yield
    finally:
        for leaf, previous in saved:
            if previous is None:
                del leaf.get_parameter_state_dp_zero
            else:
                leaf.get_parameter_state_dp_zero = previous


def _model_state(engine: Any) -> dict:
    return {
        "model"
        if len(engine.model) == 1
        else f"model{index}": model.module.sharded_state_dict()
        for (index, model) in enumerate(engine.model)
    }


def _state_template(engine: Any, tree: dict) -> dict:
    with _meta_buffers(engine):
        tree["optimizer"] = engine.optimizer.sharded_state_dict(
            tree, is_loading=True, metadata=_METADATA
        )
    tree["lr_scheduler"] = engine.lr_scheduler.state_dict()
    return tree


def _gather_results(engine: Any, result=None, error=None) -> list:
    results = [None] * torch.distributed.get_world_size(engine.cpu_group)
    torch.distributed.all_gather_object(
        results, {"result": result, "error": error}, group=engine.cpu_group
    )
    failures = [row["error"] for row in results if row["error"] is not None]
    if failures:
        raise RuntimeError("Raw state transfer failed: " + "; ".join(failures))
    return [row["result"] for row in results]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _metadata_digest(value):
    import numpy as np

    def stable(item):
        if isinstance(item, torch.Tensor):
            return {
                "tensor": hashlib.sha256(
                    item.detach()
                    .cpu()
                    .contiguous()
                    .reshape(-1)
                    .view(torch.uint8)
                    .numpy()
                ).hexdigest(),
                "dtype": str(item.dtype),
                "shape": list(item.shape),
            }
        if isinstance(item, np.ndarray):
            return {
                "array": hashlib.sha256(item.tobytes()).hexdigest(),
                "dtype": str(item.dtype),
                "shape": list(item.shape),
            }
        if isinstance(item, dict):
            return {str(key): stable(child) for (key, child) in item.items()}
        if isinstance(item, (list, tuple)):
            return [stable(child) for child in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    return hashlib.sha256(
        json.dumps(stable(value), sort_keys=True).encode()
    ).hexdigest()


def _split_flat_range(shape, first, last):
    shape = tuple(shape)
    if not 0 <= first <= last <= math.prod(shape):
        raise ValueError("Invalid raw state metadata")
    if first == last:
        return
    if not shape:
        yield ((), (), 0)
        return
    stride = math.prod(shape[1:])
    if first % stride == 0 and last % stride == 0:
        yield (
            (first // stride,) + (0,) * (len(shape) - 1),
            ((last - first) // stride,) + shape[1:],
            first,
        )
        return
    (begin, end) = (first // stride, (last - 1) // stride)
    if begin == end:
        for offset, extent, flat in _split_flat_range(
            shape[1:], first - begin * stride, last - begin * stride
        ):
            yield ((begin,) + offset, (1,) + extent, begin * stride + flat)
        return
    for offset, extent, flat in _split_flat_range(
        shape[1:], first - begin * stride, stride
    ):
        yield ((begin,) + offset, (1,) + extent, begin * stride + flat)
    if end > begin + 1:
        yield (
            (begin + 1,) + (0,) * (len(shape) - 1),
            (end - begin - 1,) + shape[1:],
            (begin + 1) * stride,
        )
    for offset, extent, flat in _split_flat_range(shape[1:], 0, last - end * stride):
        yield ((end,) + offset, (1,) + extent, end * stride + flat)


def _intersection(source, target):
    first = tuple(max(a, b) for (a, b) in zip(source["offset"], target["offset"]))
    last = tuple(
        min(a + n, b + m)
        for (a, n, b, m) in zip(
            source["offset"], source["shape"], target["offset"], target["shape"]
        )
    )
    if any(a >= b for (a, b) in zip(first, last)):
        return None
    return (
        tuple(slice(a - b, c - b) for (a, c, b) in zip(first, last, source["offset"])),
        tuple(slice(a - b, c - b) for (a, c, b) in zip(first, last, target["offset"])),
        math.prod(b - a for (a, b) in zip(first, last)),
    )


def _validate_records(records):
    groups = {}
    for entry in records:
        shape = tuple(entry["global_shape"])
        if len(shape) != len(entry["offset"]) or len(shape) != len(entry["shape"]):
            raise ValueError("Invalid raw state metadata")
        if not all(
            0 <= a < a + n <= total
            for (a, n, total) in zip(entry["offset"], entry["shape"], shape)
        ):
            raise ValueError("Invalid raw state metadata")
        groups.setdefault(entry["key"], []).append(entry)
    for key, entries in groups.items():
        shape = entries[0]["global_shape"]
        if not all(
            e["global_shape"] == shape and e["dtype"] == entries[0]["dtype"]
            for e in entries
        ):
            raise ValueError("Invalid raw state metadata")
        if sum(math.prod(e["shape"]) for e in entries) != math.prod(shape):
            raise ValueError((key, "coverage"))
        for index, entry in enumerate(entries):
            for other in entries[:index]:
                if _intersection(entry, other) is not None:
                    raise ValueError((key, "overlap"))
    return {
        "keys": len(groups),
        "elements": sum(
            math.prod(entries[0]["global_shape"]) for entries in groups.values()
        ),
    }


def _encode(value):
    return base64.b64encode(pickle.dumps(value, protocol=5)).decode()


def _decode(value):
    return pickle.loads(base64.b64decode(value))


def _identity(value):
    return (
        value.key,
        tuple(value.global_shape),
        tuple(value.global_offset),
        tuple((1,) * value.prepend_axis_num + value.local_shape),
    )


def _clean_common(value):
    if isinstance(value, dict):
        if value.get("param_state", 1) is None:
            del value["param_state"]
        for child in value.values():
            _clean_common(child)
    elif isinstance(value, list):
        for child in value:
            _clean_common(child)


def _save_rank(engine, directory, snapshot_id, tree):
    from megatron.core import parallel_state as mpu
    from megatron.core.dist_checkpointing.dict_utils import merge, nested_values
    from megatron.core.dist_checkpointing.mapping import (
        ShardedObject,
        ShardedTensor,
        is_main_replica,
    )
    from megatron.core.dist_checkpointing.serialization import (
        extract_sharded_base,
        load_preprocess,
    )

    torch.cuda.synchronize()
    tree = _state_template(engine, tree)
    (processed, nonpersistent, factories) = load_preprocess(tree)
    (sharded, common) = extract_sharded_base(processed)
    merge(common, nonpersistent)
    _clean_common(common)
    values = list(nested_values(sharded))
    models = [
        v
        for v in values
        if isinstance(v, ShardedTensor) and v.data.device.type != "meta"
    ]
    optimizers = [
        v
        for v in values
        if isinstance(v, ShardedTensor) and v.data.device.type == "meta"
    ]
    parameters = {id(p): p for model in engine.model for p in model.parameters()}
    ordered = sorted(parameters.values(), key=lambda p: p.data_ptr())
    starts = [p.data_ptr() for p in ordered]
    owned = {}
    for leaf in iter_megatron_optimizer_leaves(engine.optimizer):
        for parameter in leaf.model_param_group_index_map:
            if id(parameter) in owned:
                raise ValueError("Invalid raw state metadata")
            owned[id(parameter)] = leaf
    directory = Path(directory)
    rank = torch.distributed.get_rank()
    final = directory / f"rank-{rank}.bin"
    pending = directory / f"rank-{rank}.bin.partial"
    records = []
    seen_models = set()
    seen_optimizer = set()

    def write_tensor(stream, key, global_shape, offset, shape, data):
        cpu = data.detach().reshape(-1).cpu().contiguous()
        raw = cpu.view(torch.uint8).numpy()
        padding = -stream.tell() % 64
        if padding:
            stream.write(b"\x00" * padding)
        first = stream.tell()
        stream.write(memoryview(raw))
        records.append(
            {
                "key": key,
                "global_shape": list(global_shape),
                "offset": list(offset),
                "shape": list(shape),
                "dtype": str(data.dtype),
                "file": final.name,
                "byte_offset": first,
                "bytes": data.numel() * data.element_size(),
            }
        )

    with pending.open("xb", buffering=16 * 1024 * 1024) as stream:
        for value in models:
            if value.flattened_range is not None:
                raise ValueError("Invalid raw state metadata")
            extent = (1,) * value.prepend_axis_num + value.local_shape
            if not math.prod(extent):
                continue
            if (
                is_main_replica(value.replica_id)
                and _identity(value) not in seen_models
            ):
                seen_models.add(_identity(value))
                write_tensor(
                    stream,
                    value.key,
                    value.global_shape,
                    value.global_offset,
                    extent,
                    value.data,
                )
            position = bisect.bisect_right(starts, value.data.data_ptr()) - 1
            if position < 0:
                raise ValueError(value.key)
            parameter = ordered[position]
            if not (
                parameter.dtype == value.data.dtype
                and parameter.is_contiguous()
                and value.data.is_contiguous()
            ):
                raise ValueError(value.key)
            byte_start = value.data.data_ptr() - parameter.data_ptr()
            if not (byte_start >= 0 and byte_start % parameter.element_size() == 0):
                raise ValueError(value.key)
            begin = byte_start // parameter.element_size()
            if begin + value.data.numel() > parameter.numel():
                raise ValueError(value.key)
            leaf = owned.get(id(parameter))
            if leaf is None:
                continue
            replica = tuple(value.replica_id)
            if len(replica) != 3:
                raise ValueError(replica)
            if not is_main_replica(
                (*replica[:2], leaf.distributed_optimizer_instance_id)
            ):
                continue
            interval = leaf._get_model_param_range_map(parameter)["param"]
            first = max(begin, interval.start)
            last = min(begin + value.data.numel(), interval.end)
            if first >= last:
                continue
            states = leaf._get_main_param_and_optimizer_states(parameter)
            matches = [
                v
                for v in optimizers
                if v.key.endswith("." + value.key)
                and _identity(v)[1:] == _identity(value)[1:]
            ]
            if _identity(value) in seen_optimizer:
                continue
            if not matches:
                raise ValueError(("Missing native optimizer metadata", value.key))
            seen_optimizer.add(_identity(value))
            keys = {}
            for state in matches:
                kind = state.key[: -len(value.key) - 1].rsplit(".", 1)[-1]
                if kind in keys and keys[kind] != state.key:
                    raise ValueError((value.key, kind))
                keys[kind] = state.key
            if set(keys) != {"param", *leaf.optimizer_state_keys}:
                raise ValueError((value.key, keys))
            for offset, shape, flat_start in _split_flat_range(
                extent, first - begin, last - begin
            ):
                count = math.prod(shape)
                source_first = begin + flat_start - interval.start
                global_offset = tuple(
                    a + b for (a, b) in zip(value.global_offset, offset)
                )
                for kind in ("param",) + leaf.optimizer_state_keys:
                    part = states[kind].reshape(-1)[source_first : source_first + count]
                    if part.numel() != count:
                        raise ValueError("Invalid raw state metadata")
                    write_tensor(
                        stream,
                        keys[kind],
                        value.global_shape,
                        global_offset,
                        shape,
                        part,
                    )
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(final)
    objects = {
        v.unique_key: _encode(v.data)
        for v in values
        if isinstance(v, ShardedObject) and is_main_replica(v.replica_id)
    }
    rng_id = f"{mpu.get_pipeline_model_parallel_rank()}/{mpu.get_tensor_model_parallel_rank()}/{mpu.get_data_parallel_rank()}"
    rng = _encode(engine.checkpointer.get_rng_state(use_dist_ckpt=False))
    result = {
        "rank": rank,
        "snapshot_id": snapshot_id,
        "records": records,
        "common": _encode(common),
        "objects": objects,
        "rng_id": rng_id,
        "rng": rng,
        "inherited_rng": getattr(engine, "_state_transfer_rng_bank", {}),
        "version": engine.get_version(),
        "file_bytes": final.stat().st_size,
        "common_sha256": _metadata_digest(common),
        "expected_keys": sorted(
            {
                v.key
                for v in values
                if isinstance(v, ShardedTensor)
                and math.prod((1,) * v.prepend_axis_num + v.local_shape)
            }
        ),
    }
    path = directory / f"rank-{rank}.json"
    _write_json(path, result)
    del (
        tree,
        processed,
        nonpersistent,
        factories,
        sharded,
        common,
        values,
        models,
        optimizers,
    )
    gc.collect()


class _FileReader:
    def __init__(self, directory, manifest):
        self.directory = directory
        self.by_key = {}
        self.handles = {}
        self.maps = {}
        self.objects = {}
        for rank in manifest["ranks"]:
            self.objects.update(rank["objects"])
            for entry in rank["records"]:
                self.by_key.setdefault(entry["key"], []).append(entry)

    def load(self, value):
        from megatron.core.dist_checkpointing.mapping import (
            ShardedObject,
            ShardedTensor,
        )

        if isinstance(value, ShardedObject):
            return _decode(self.objects[value.unique_key])
        if not (isinstance(value, ShardedTensor) and value.flattened_range is None):
            raise ValueError("Invalid raw state metadata")
        extent = (1,) * value.prepend_axis_num + value.local_shape
        target = {"offset": value.global_offset, "shape": extent}
        output = torch.empty(extent, dtype=value.dtype, device="cpu")
        value.data = None
        if not output.numel():
            return output.reshape(value.local_shape)
        covered = 0
        for entry in self.by_key[value.key]:
            if tuple(entry["global_shape"]) != value.global_shape:
                raise ValueError("Invalid raw state metadata")
            if entry["dtype"] != str(value.dtype):
                raise ValueError("Invalid raw state metadata")
            parts = _intersection(entry, target)
            if parts is None:
                continue
            filename = entry["file"]
            if Path(filename).name != filename:
                raise ValueError("Invalid raw state metadata")
            if filename not in self.maps:
                handle = (self.directory / filename).open("rb")
                self.handles[filename] = handle
                self.maps[filename] = mmap.mmap(
                    handle.fileno(), 0, access=mmap.ACCESS_COPY
                )
            source = torch.frombuffer(
                self.maps[filename],
                dtype=value.dtype,
                count=math.prod(entry["shape"]),
                offset=entry["byte_offset"],
            ).reshape(entry["shape"])
            (source_slice, target_slice, count) = parts
            try:
                output[target_slice].copy_(source[source_slice])
            finally:
                del source
            covered += count
        if covered != output.numel():
            raise ValueError((value.key, covered, output.numel()))
        return output.reshape(value.local_shape)

    def close(self):
        for mapping in self.maps.values():
            mapping.close()
        for handle in self.handles.values():
            handle.close()
        self.maps.clear()
        self.handles.clear()


def _restore_rank(engine, directory, manifest, tree):
    from megatron.core import parallel_state as mpu
    from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace, merge
    from megatron.core.dist_checkpointing.mapping import apply_factory_merges
    from megatron.core.dist_checkpointing.serialization import (
        extract_sharded_base,
        load_preprocess,
    )

    directory = Path(directory)
    tree = _state_template(engine, tree)
    lazy = _detach_parameter_states(tree["optimizer"])
    (processed, nonpersistent, factories) = load_preprocess(tree)
    (sharded, _) = extract_sharded_base(processed)
    reader = _FileReader(directory, manifest)
    try:
        loaded = dict_list_map_inplace(reader.load, sharded)
        common = _decode(manifest["ranks"][0]["common"])
        merge(common, nonpersistent)
        merge(common, loaded)
        loaded = apply_factory_merges(common, factories)
        for path, metadata in lazy:
            node = loaded
            for key in path:
                node = node[key]
            node["param_state"] = _LazyParameterState(metadata, reader.load)
        for index, model in enumerate(engine.model):
            model.load_state_dict(
                loaded["model" if len(engine.model) == 1 else f"model{index}"]
            )
        engine.optimizer.load_state_dict(loaded["optimizer"])
        engine.lr_scheduler.load_state_dict(loaded["lr_scheduler"])
        engine.set_version(manifest["version"])
        bank = {}
        for rank in manifest["ranks"]:
            bank.update(rank["inherited_rng"])
        for rank in manifest["ranks"]:
            bank[rank["rng_id"]] = rank["rng"]
        rng_id = f"{mpu.get_pipeline_model_parallel_rank()}/{mpu.get_tensor_model_parallel_rank()}/{mpu.get_data_parallel_rank()}"
        policy = "target-initialized"
        if rng_id in bank:
            engine.checkpointer.load_rng_states(_decode(bank[rng_id]))
            policy = "restored-logical-rank"
        engine._state_transfer_rng_bank = bank
        torch.cuda.synchronize()
        del tree, processed, nonpersistent, factories, sharded, common, loaded, lazy
        gc.collect()
        return {"version": manifest["version"], "rng_policy": policy}
    finally:
        reader.close()


class _LazyParameterState:
    """Materialize only parameters requested by the native DP-local loader."""

    def __init__(self, metadata, load_tensor):
        self.metadata = metadata
        self.load_tensor = load_tensor

    def __getitem__(self, index):
        from megatron.core.dist_checkpointing.dict_utils import (
            dict_list_map_inplace,
            merge,
        )
        from megatron.core.dist_checkpointing.mapping import apply_factory_merges
        from megatron.core.dist_checkpointing.serialization import (
            extract_sharded_base,
            load_preprocess,
        )

        (template, nonpersistent, factories) = load_preprocess(
            copy.deepcopy(self.metadata[index])
        )
        (sharded, common) = extract_sharded_base(template)
        merge(common, nonpersistent)
        merge(common, dict_list_map_inplace(self.load_tensor, sharded))
        return apply_factory_merges(common, factories)


def _detach_parameter_states(node, path=("optimizer",)):
    if "param_state" in node:
        return [(path, node.pop("param_state"))]
    return [
        entry
        for (key, child) in node.items()
        if isinstance(child, dict)
        for entry in _detach_parameter_states(child, path + (key,))
    ]


def _publish(directory: Path, world_size: int, snapshot_id: str) -> dict:
    ranks = [
        json.loads((directory / f"rank-{rank}.json").read_text())
        for rank in range(world_size)
    ]
    for index, rank in enumerate(ranks):
        if rank["rank"] != index or rank["snapshot_id"] != snapshot_id:
            raise ValueError("Rank snapshot identity mismatch")
        filename = f"rank-{index}.bin"
        if (directory / filename).stat().st_size != rank["file_bytes"]:
            raise ValueError(f"Rank file size mismatch: {filename}")
        rank["files"] = {filename: rank["file_bytes"]}
    records = [entry for rank in ranks for entry in rank["records"]]
    if {entry["key"] for entry in records} != {
        key for rank in ranks for key in rank["expected_keys"]
    }:
        raise ValueError("Snapshot tensor keys are incomplete")
    coverage = _validate_records(records)
    if (
        len({rank["common_sha256"] for rank in ranks}) != 1
        or len({rank["version"] for rank in ranks}) != 1
    ):
        raise ValueError("Rank optimizer metadata or versions differ")
    manifest = {
        "format": _FORMAT,
        "snapshot_id": snapshot_id,
        "ranks": ranks,
        "version": ranks[0]["version"],
        "coverage": coverage,
    }
    pending = directory / "complete.json.partial"
    _write_json(pending, manifest)
    pending.replace(directory / "complete.json")
    _fsync_directory(directory)
    return {"version": manifest["version"], "snapshot_id": snapshot_id}


def save_state(engine: Any, directory: str) -> dict:
    """Collectively save a complete, immutable snapshot before returning."""
    directory = Path(directory)
    (header, error) = (None, None)
    rank = torch.distributed.get_rank(engine.cpu_group)
    if rank == 0:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "complete.json").exists():
                raise FileExistsError("Snapshot is already complete")
            header = {"snapshot_id": uuid4().hex, "format": _FORMAT}
            _write_json(directory / "snapshot.json", header)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    header = _gather_results(engine, header, error)[0]
    (tree, error) = (None, None)
    try:
        if json.loads((directory / "snapshot.json").read_text()) != header:
            raise ValueError("Snapshot directory is not shared by all ranks")
        if any(
            (directory / name).exists()
            for name in (
                f"rank-{rank}.bin",
                f"rank-{rank}.bin.partial",
                f"rank-{rank}.json",
            )
        ):
            raise FileExistsError("Rank snapshot files already exist")
        engine._ensure_weight_residency().release_grad_memory()
        torch.cuda.empty_cache()
        _initialize_lazy_adam(engine)
        tree = _model_state(engine)
    except Exception as exc:
        error = f"rank {rank}: {type(exc).__name__}: {exc}"
    _gather_results(engine, error=error)
    error = None
    try:
        _save_rank(engine, directory, header["snapshot_id"], tree)
    except Exception as exc:
        error = f"rank {rank}: {type(exc).__name__}: {exc}"
    _gather_results(engine, error=error)
    (receipt, error) = (None, None)
    if rank == 0:
        try:
            receipt = _publish(
                directory,
                torch.distributed.get_world_size(engine.cpu_group),
                header["snapshot_id"],
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    return _gather_results(engine, receipt, error)[0]


def restore_state(engine: Any, directory: str) -> dict:
    """Collectively restore a completed snapshot into the initialized layout."""
    directory = Path(directory)
    (manifest, tree, error, digest) = (None, None, None, None)
    try:
        data = (directory / "complete.json").read_bytes()
        manifest = json.loads(data)
        if manifest["format"] != _FORMAT:
            raise ValueError("Unsupported raw snapshot format")
        for rank in manifest["ranks"]:
            for filename, size in rank["files"].items():
                if (
                    Path(filename).name != filename
                    or (directory / filename).stat().st_size != size
                ):
                    raise ValueError(f"Invalid snapshot rank file: {filename}")
        engine._ensure_weight_residency().release_grad_memory()
        torch.cuda.empty_cache()
        _initialize_lazy_adam(engine)
        tree = _model_state(engine)
        digest = hashlib.sha256(data).hexdigest()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    digests = _gather_results(engine, digest, error)
    if len(set(digests)) != 1:
        raise ValueError("Workers read different raw snapshot manifests")
    (receipt, error) = (None, None)
    try:
        receipt = _restore_rank(engine, directory, manifest, tree)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _gather_results(engine, error=error)
    return receipt
