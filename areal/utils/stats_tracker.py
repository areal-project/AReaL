# SPDX-License-Identifier: Apache-2.0

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from threading import Lock

import torch
import torch.distributed as dist

from areal.infra.platforms import current_platform
from areal.utils import logging
from areal.utils.seqpack import flat2d

logger = logging.getLogger("StatsTracker")


class ReduceType(Enum):
    AVG_MIN_MAX = auto()
    AVG = auto()
    SUM = auto()
    MIN = auto()
    MAX = auto()
    SCALAR = auto()


@dataclass(frozen=True)
class _StatMetadata:
    reduce_type: ReduceType
    denominator: str | None
    has_reduce_group: bool


MOE_AUX_LOSSES = {}


class DistributedStatsTracker:
    def __init__(self, name: str = ""):
        self.lock = Lock()
        self.scope_stack = []
        if name:
            self.scope_stack.append(name.strip("/"))
        self.denominators = {}  # key -> denominator key
        self.reduce_types = {}  # key -> ReduceType
        # Per-key override of the reduce_group. If set, its value takes
        # precedence over the `reduce_group` passed to `export`. This is used,
        # e.g., to make CP-local SFT stats (loss/entropy/vocab_*) reduce across
        # DP + CP so the reported numbers are CP-invariant (#1242 follow-up).
        self.reduce_groups = {}  # key -> dist.ProcessGroup

        self.stats = defaultdict(list)

    def scope(self, name):
        """Context manager for hierarchical scoping"""
        with self.lock:
            return self.Scope(self, name)

    def scope_func_wrapper(self, name):
        """Decorator for hierarchical scoping"""

        def decorator(func):
            def wrapper(*args, **kwargs):
                with self.scope(name):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    class Scope:
        def __init__(self, tracker, name):
            self.tracker = tracker
            self.name = name.strip("/")

        def __enter__(self):
            self.tracker.scope_stack.append(self.name)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.tracker.scope_stack.pop()

    def _get_full_key(self, key):
        """Combine scope stack with current key"""
        if not self.scope_stack:
            return key
        return "/".join(self.scope_stack + [key])

    @contextmanager
    def disable_scope(self):
        tmp = self.scope_stack
        self.scope_stack = []
        yield
        self.scope_stack = tmp

    @contextmanager
    def record_timing(self, key):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            with self.lock:
                # NOTE: timing records are fixed under the "timeperf" scope
                full_key = f"timeperf/{key}"
                self._set_reduce_type(full_key, ReduceType.SCALAR)
                self.stats[full_key].append(time.perf_counter() - start_time)

    def denominator(self, *, reduce_group=None, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
                    raise ValueError(
                        f"`{key}` must be a pytorch bool tensor: {value.dtype}"
                    )
                if value.numel() == 0:
                    raise ValueError(f"`{key}` must be non-empty")
                full_key = self._get_full_key(key)
                self._set_reduce_type(full_key, ReduceType.SUM)
                self.stats[full_key].append(value.detach().clone())
                if reduce_group is not None:
                    self.reduce_groups[full_key] = reduce_group

    def scalar(self, *, reduce_group=None, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                full_key = self._get_full_key(key)
                self._set_reduce_type(full_key, ReduceType.SCALAR)
                self.stats[full_key].append(float(value))
                if reduce_group is not None:
                    self.reduce_groups[full_key] = reduce_group

    def stat(
        self,
        denominator: str,
        reduce_type: ReduceType | None = None,
        *,
        reduce_group=None,
        **kwargs,
    ):
        """Record multiple values from a dictionary.

        If `reduce_group` is provided, it overrides the `reduce_group` passed
        to `export` for these specific keys. This enables recording stats that
        must be reduced across a different topology than the default (e.g.,
        loss/vocab_* under CP-local loss need DP+CP reduce while n_seqs stays
        DP-only).
        """
        with self.lock:
            for key, value in kwargs.items():
                if not isinstance(value, torch.Tensor) or value.dtype != torch.float:
                    raise ValueError(
                        f"`{key}` should be a pytorch float tensor: {value.dtype}"
                    )
                if value.numel() == 0:
                    raise ValueError(f"`{key}` should be non-empty")
                if reduce_type == ReduceType.SCALAR:
                    raise ValueError("Cannot use the scalar reduce type for a tensor")
                full_key = self._get_full_key(key)

                denorm = self._get_full_key(denominator)
                if denorm not in self.stats or not self.stats[denorm]:
                    raise ValueError(f"Denominator `{denorm}` does not exist")
                for x, y in zip(self.stats[denorm], self.stats[full_key] + [value]):
                    assert x.shape == y.shape, (x.shape, y.shape)
                self.denominators[full_key] = denorm

                if reduce_type is None:
                    reduce_type = ReduceType.AVG_MIN_MAX
                self._set_reduce_type(full_key, reduce_type)
                self.stats[full_key].append(value.detach().clone())
                if reduce_group is not None:
                    self.reduce_groups[full_key] = reduce_group

    def _set_reduce_type(self, key, reduce_type):
        if not isinstance(reduce_type, ReduceType):
            raise ValueError("reduce_type must be a ReduceType enum")
        self.reduce_types[key] = reduce_type

    def _effective_reduce_group(
        self,
        key,
        default_reduce_group,
        sync_metadata: dict[str, _StatMetadata] | None = None,
        key_sync_group=None,
    ):
        """Return the per-key reduce_group override if any, else the default."""
        if sync_metadata is not None:
            metadata = sync_metadata.get(key)
            if metadata is not None and metadata.has_reduce_group:
                if key_sync_group is not None:
                    return key_sync_group
                if key in self.reduce_groups:
                    return self.reduce_groups[key]
                raise RuntimeError(
                    f"Key `{key}` has a per-key reduce_group on another rank, "
                    "but no key_sync_group was provided for this rank."
                )
        if key in self.reduce_groups:
            return self.reduce_groups[key]
        return default_reduce_group

    def _device_for_placeholder_tensor(self, group=None):
        if group is not None:
            try:
                backend = str(dist.get_backend(group)).lower()
                platform_backend = current_platform.communication_backend
                if backend == platform_backend:
                    return current_platform.device_type
            except (AttributeError, RuntimeError, ValueError):
                pass
        try:
            if current_platform.is_initialized():
                return current_platform.device_type
        except AttributeError:
            pass
        return "cpu"

    def _placeholder_scalar(
        self,
        fill: float = 0.0,
        like: torch.Tensor | None = None,
        group=None,
    ):
        """Scalar identity tensor for reductions (0.0 for SUM/AVG, +/-inf for
        MIN/MAX). Placed on ``like``'s device if given, else on the device
        matching ``group``."""
        device = (
            like.device
            if like is not None
            else self._device_for_placeholder_tensor(group)
        )
        return torch.tensor(
            fill,
            dtype=torch.float32,
            device=device,
        )

    def _local_metadata(self, keys: list[str]) -> dict[str, _StatMetadata]:
        return {
            key: _StatMetadata(
                reduce_type=self.reduce_types.get(key, ReduceType.SCALAR),
                denominator=self.denominators.get(key),
                has_reduce_group=key in self.reduce_groups,
            )
            for key in keys
        }

    @staticmethod
    def _merge_metadata(
        all_metadata: list[dict[str, _StatMetadata]],
    ) -> dict[str, _StatMetadata]:
        metadata: dict[str, _StatMetadata] = {}
        for rank_metadata in all_metadata:
            for key, key_metadata in rank_metadata.items():
                if key not in metadata:
                    metadata[key] = key_metadata
                elif metadata[key] != key_metadata:
                    raise ValueError(
                        f"Inconsistent stats metadata for key `{key}`: "
                        f"{metadata[key]} vs {key_metadata}"
                    )
        return metadata

    def export(
        self,
        key=None,
        reduce_group=None,
        key_sync_group=None,
        reset=True,
    ) -> dict[str, float]:
        """Get aggregated statistics"""
        with self.lock:
            if key is not None:
                full_key = self._get_full_key(key)
                result = self._aggregate(full_key, reduce_group)
                if reset:
                    if full_key in self.denominators:
                        self.denominators.pop(full_key)
                    if full_key in self.reduce_types:
                        self.reduce_types.pop(full_key)
                    if full_key in self.reduce_groups:
                        self.reduce_groups.pop(full_key)
                    self.stats.pop(full_key)
                return result

            # synchronize keys across ranks
            keys = list(self.stats.keys())
            sync_metadata = self._local_metadata(keys)
            if reduce_group is not None:
                all_metadata = [None for _ in range(dist.get_world_size(reduce_group))]
                dist.all_gather_object(
                    all_metadata,
                    self._local_metadata(keys),
                    group=reduce_group,
                )
                sync_metadata = self._merge_metadata(all_metadata)

            # Per-key reduce groups may be wider than the default export group
            # (for example DP+CP vs DP). Only those override keys need the
            # wider key/metadata sync; default keys keep the original
            # reduce_group alignment.
            if key_sync_group is not None:
                override_keys = list(self.reduce_groups.keys())
                all_override_metadata = [
                    None for _ in range(dist.get_world_size(key_sync_group))
                ]
                dist.all_gather_object(
                    all_override_metadata,
                    self._local_metadata(override_keys),
                    group=key_sync_group,
                )
                override_metadata = self._merge_metadata(all_override_metadata)
                for override_key, override_value in override_metadata.items():
                    if (
                        override_key in sync_metadata
                        and sync_metadata[override_key] != override_value
                    ):
                        raise ValueError(
                            f"Inconsistent stats metadata for key `{override_key}`: "
                            f"{sync_metadata[override_key]} vs {override_value}"
                        )
                    sync_metadata[override_key] = override_value

            if reduce_group is not None or key_sync_group is not None:
                # Should ensure that the orders are the same
                keys = sorted(sync_metadata)
            results = {}
            for key in keys:
                results.update(
                    self._aggregate(
                        key,
                        reduce_group,
                        sync_metadata=sync_metadata,
                        key_sync_group=key_sync_group,
                    )
                )
            if reset:
                self.denominators = {}
                self.reduce_types = {}
                self.reduce_groups = {}
                self.stats = defaultdict(list)
            results = {
                k: v.cpu().item() if torch.is_tensor(v) else v
                for k, v in results.items()
            }
            return results

    def _aggregate(
        self,
        key,
        reduce_group,
        sync_metadata: dict[str, _StatMetadata] | None = None,
        key_sync_group=None,
    ):
        metadata = sync_metadata.get(key) if sync_metadata is not None else None
        reduce_type = self.reduce_types.get(
            key,
            metadata.reduce_type if metadata is not None else ReduceType.SCALAR,
        )

        result = {}
        if reduce_type == ReduceType.AVG_MIN_MAX:
            result["/".join([key, "avg"])] = self._avg_of(
                key, reduce_group, sync_metadata, key_sync_group
            )
            result["/".join([key, "min"])] = self._min_of(
                key, reduce_group, sync_metadata, key_sync_group
            )
            result["/".join([key, "max"])] = self._max_of(
                key, reduce_group, sync_metadata, key_sync_group
            )
        elif reduce_type == ReduceType.AVG:
            result[key] = self._avg_of(key, reduce_group, sync_metadata, key_sync_group)
        elif reduce_type == ReduceType.SUM:
            result[key] = self._sum_of(key, reduce_group, sync_metadata, key_sync_group)
        elif reduce_type == ReduceType.MIN:
            result[key] = self._min_of(key, reduce_group, sync_metadata, key_sync_group)
        elif reduce_type == ReduceType.MAX:
            result[key] = self._max_of(key, reduce_group, sync_metadata, key_sync_group)
        elif reduce_type == ReduceType.SCALAR:
            effective_group = self._effective_reduce_group(
                key, reduce_group, sync_metadata, key_sync_group
            )
            # `.get`: a rank may learn about this key via key/metadata sync
            # without holding any local values for it.
            stats = self.stats.get(key, [])
            value = self._placeholder_scalar(
                fill=float(sum(stats)), group=effective_group
            )
            cnt = self._placeholder_scalar(
                fill=float(len(stats)), group=effective_group
            )

            if effective_group is not None:
                dist.all_reduce(value, group=effective_group)
                dist.all_reduce(cnt, group=effective_group)
            result[key] = float(value / cnt) if float(cnt) > 0 else 0.0
            result[key + "__count"] = int(cnt)
        else:
            raise ValueError(f"Unknown reduce type: {reduce_type}")

        keys_to_pop = [k for k, v in result.items() if v is None]
        for k in keys_to_pop:
            result.pop(k)
        return result

    def _sum_of(
        self,
        key,
        reduce_group,
        sync_metadata: dict[str, _StatMetadata] | None = None,
        key_sync_group=None,
    ):
        values = self.stats.get(key, [])
        effective_group = self._effective_reduce_group(
            key, reduce_group, sync_metadata, key_sync_group
        )
        if not values:
            x = self._placeholder_scalar(group=effective_group)
            if effective_group is not None:
                dist.all_reduce(x, group=effective_group)
            return float(x)

        metadata = sync_metadata.get(key) if sync_metadata is not None else None
        denominator = self.denominators.get(
            key,
            metadata.denominator if metadata is not None else None,
        )
        if denominator is None:
            x = sum([x.sum() for x in values], self._placeholder_scalar(like=values[0]))
            if effective_group is not None:
                dist.all_reduce(x, group=effective_group)
        else:
            if denominator not in self.stats:
                raise ValueError(
                    f"Denominator `{denominator}` not set for key `{key}`."
                )
            xs = []
            for v, d in zip(values, self.stats[denominator]):
                xs.append(torch.where(d, v, 0.0).sum())
            x = sum(xs, self._placeholder_scalar(like=values[0]))
            if effective_group is not None:
                dist.all_reduce(x, group=effective_group)
        return float(x)

    def _denominator_of(
        self,
        key,
        sync_metadata: dict[str, _StatMetadata] | None = None,
    ):
        if key in self.denominators:
            return self.denominators[key]
        metadata = sync_metadata.get(key) if sync_metadata is not None else None
        if metadata is not None and metadata.denominator is not None:
            return metadata.denominator
        raise ValueError(f"Denominator not set for key `{key}`.")

    def _avg_of(
        self,
        key,
        reduce_group,
        sync_metadata: dict[str, _StatMetadata] | None = None,
        key_sync_group=None,
    ):
        values = self.stats.get(key, [])
        if not values:
            effective_group = self._effective_reduce_group(
                key, reduce_group, sync_metadata, key_sync_group
            )
            x = self._placeholder_scalar(group=effective_group)
            d = self._placeholder_scalar(group=effective_group)
            if effective_group is not None:
                dist.all_reduce(x, group=effective_group)
                dist.all_reduce(d, group=effective_group)
            if d == 0:
                return None
            return x / d

        denominator = self._denominator_of(key, sync_metadata)
        if denominator not in self.stats:
            raise ValueError(f"Denominator `{denominator}` not set for key `{key}`.")
        xs = []
        ds = []
        for v, d in zip(values, self.stats[denominator]):
            xs.append(torch.where(d, v, 0.0).sum())
            ds.append(d.sum())
        x = sum(xs, self._placeholder_scalar(like=values[0]))
        d = sum(ds, self._placeholder_scalar(like=values[0]))
        effective_group = self._effective_reduce_group(
            key, reduce_group, sync_metadata, key_sync_group
        )
        if effective_group is not None:
            dist.all_reduce(x, group=effective_group)
            dist.all_reduce(d, group=effective_group)
        if d == 0:
            return None
        return x / d

    def _min_of(
        self,
        key,
        reduce_group,
        sync_metadata: dict[str, _StatMetadata] | None = None,
        key_sync_group=None,
    ):
        values = self.stats.get(key, [])
        if not values:
            effective_group = self._effective_reduce_group(
                key, reduce_group, sync_metadata, key_sync_group
            )
            x = self._placeholder_scalar(fill=float("inf"), group=effective_group)
            if effective_group is not None:
                dist.all_reduce(x, group=effective_group, op=dist.ReduceOp.MIN)
            if torch.isinf(x):
                return None
            return float(x)

        denominator = self._denominator_of(key, sync_metadata)
        if denominator not in self.stats:
            raise ValueError(f"Denominator `{denominator}` not set for key `{key}`.")
        xs = []
        for v, d in zip(values, self.stats[denominator]):
            xs.append(torch.where(d, v, float("inf")).min())
        x = torch.stack(xs).min()
        effective_group = self._effective_reduce_group(
            key, reduce_group, sync_metadata, key_sync_group
        )
        if effective_group is not None:
            dist.all_reduce(x, group=effective_group, op=dist.ReduceOp.MIN)
        if torch.isinf(x):
            return None
        return float(x)

    def _max_of(
        self,
        key,
        reduce_group,
        sync_metadata: dict[str, _StatMetadata] | None = None,
        key_sync_group=None,
    ):
        values = self.stats.get(key, [])
        if not values:
            effective_group = self._effective_reduce_group(
                key, reduce_group, sync_metadata, key_sync_group
            )
            x = self._placeholder_scalar(fill=-float("inf"), group=effective_group)
            if effective_group is not None:
                dist.all_reduce(x, group=effective_group, op=dist.ReduceOp.MAX)
            if torch.isinf(x):
                return None
            return float(x)

        denominator = self._denominator_of(key, sync_metadata)
        if denominator not in self.stats:
            raise ValueError(f"Denominator `{denominator}` not set for key `{key}`.")
        xs = []
        for v, d in zip(values, self.stats[denominator]):
            xs.append(torch.where(d, v, -float("inf")).max())
        x = torch.stack(xs).max()
        effective_group = self._effective_reduce_group(
            key, reduce_group, sync_metadata, key_sync_group
        )
        if effective_group is not None:
            dist.all_reduce(x, group=effective_group, op=dist.ReduceOp.MAX)
        if torch.isinf(x):
            return None
        return float(x)


DEFAULT_TRACKER = DistributedStatsTracker()
stat = DEFAULT_TRACKER.stat
denominator = DEFAULT_TRACKER.denominator
export = DEFAULT_TRACKER.export
scope = DEFAULT_TRACKER.scope
scalar = DEFAULT_TRACKER.scalar
record_timing = DEFAULT_TRACKER.record_timing
scope_func_wrapper = DEFAULT_TRACKER.scope_func_wrapper

TRACKERS = {"": DEFAULT_TRACKER}
LOCK = Lock()


def get(name: str = ""):
    global TRACKERS, LOCK
    with LOCK:
        if name not in TRACKERS:
            TRACKERS[name] = DistributedStatsTracker(name)
        return TRACKERS[name]


def export_all(reduce_group=None, key_sync_group=None, reset=True) -> dict[str, float]:
    stat = {}
    duplicate_keys = set()
    tracker_keys = list(TRACKERS.keys())
    sync_group = key_sync_group if key_sync_group is not None else reduce_group
    if sync_group is not None:
        all_trackers = [None for _ in range(dist.get_world_size(sync_group))]
        dist.all_gather_object(all_trackers, list(TRACKERS.keys()), group=sync_group)
        tracker_keys = sorted(list(set(flat2d(all_trackers))))
    for tracker_key in tracker_keys:
        tracker = get(tracker_key)
        x = tracker.export(
            reduce_group=reduce_group,
            key_sync_group=key_sync_group,
            reset=reset,
        )
        for k in x.keys():
            if k in stat:
                duplicate_keys.add(k)
        stat.update(x)
    if duplicate_keys:
        logger.warning(f"Duplicate stat keys detected: {list(duplicate_keys)}")
    return stat
