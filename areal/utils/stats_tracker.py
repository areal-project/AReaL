# SPDX-License-Identifier: Apache-2.0

import time
from collections import defaultdict
from contextlib import contextmanager
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

    def _effective_reduce_group(self, key, default_reduce_group):
        """Return the per-key reduce_group override if any, else the default."""
        return self.reduce_groups.get(key, default_reduce_group)

    def export(self, key=None, reduce_group=None, reset=True) -> dict[str, float]:
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
            if reduce_group is not None:
                all_keys = [None for _ in range(dist.get_world_size(reduce_group))]
                dist.all_gather_object(all_keys, keys, group=reduce_group)
                # Should ensure that the orders are the same
                keys = sorted(list(set(flat2d(all_keys))))
            results = {}
            for key in keys:
                results.update(self._aggregate(key, reduce_group))
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

    def _aggregate(self, key, reduce_group):
        reduce_type = self.reduce_types.get(key, ReduceType.SCALAR)

        result = {}
        if reduce_type == ReduceType.AVG_MIN_MAX:
            result["/".join([key, "avg"])] = self._avg_of(key, reduce_group)
            result["/".join([key, "min"])] = self._min_of(key, reduce_group)
            result["/".join([key, "max"])] = self._max_of(key, reduce_group)
        elif reduce_type == ReduceType.AVG:
            result[key] = self._avg_of(key, reduce_group)
        elif reduce_type == ReduceType.SUM:
            result[key] = self._sum_of(key, reduce_group)
        elif reduce_type == ReduceType.MIN:
            result[key] = self._min_of(key, reduce_group)
        elif reduce_type == ReduceType.MAX:
            result[key] = self._max_of(key, reduce_group)
        elif reduce_type == ReduceType.SCALAR:
            if current_platform.is_initialized():
                device = current_platform.device_type
            else:
                device = "cpu"
            value = torch.tensor(
                sum(self.stats[key]), dtype=torch.float32, device=device
            )
            cnt = torch.tensor(len(self.stats[key]), dtype=torch.float32, device=device)
            effective_group = self._effective_reduce_group(key, reduce_group)
            if effective_group is not None:
                dist.all_reduce(value, group=effective_group)
                dist.all_reduce(cnt, group=effective_group)
            result[key] = float(value / cnt)
            result[key + "__count"] = int(cnt)
        else:
            raise ValueError(f"Unknown reduce type: {reduce_type}")

        keys_to_pop = [k for k, v in result.items() if v is None]
        for k in keys_to_pop:
            result.pop(k)
        return result

    def _sum_of(self, key, reduce_group):
        values = self.stats[key]
        effective_group = self._effective_reduce_group(key, reduce_group)
        if key not in self.denominators:
            x = sum([x.sum() for x in values])
            if effective_group is not None:
                dist.all_reduce(x, group=effective_group)
        else:
            denominator = self.denominators[key]
            if denominator not in self.stats:
                raise ValueError(
                    f"Denominator `{denominator}` not set for key `{key}`."
                )
            xs = []
            for v, d in zip(values, self.stats[denominator]):
                xs.append(torch.where(d, v, 0.0).sum())
            x = sum(xs)
            if effective_group is not None:
                dist.all_reduce(x, group=effective_group)
        return float(x)

    def _avg_of(self, key, reduce_group):
        values = self.stats[key]
        denominator = self.denominators[key]
        if denominator not in self.stats:
            raise ValueError(f"Denominator `{denominator}` not set for key `{key}`.")
        xs = []
        ds = []
        for v, d in zip(values, self.stats[denominator]):
            xs.append(torch.where(d, v, 0.0).sum())
            ds.append(d.sum())
        x = sum(xs)
        d = sum(ds)
        effective_group = self._effective_reduce_group(key, reduce_group)
        if effective_group is not None:
            dist.all_reduce(x, group=effective_group)
            dist.all_reduce(d, group=effective_group)
        if d == 0:
            return None
        return x / d

    def _min_of(self, key, reduce_group):
        values = self.stats[key]
        denominator = self.denominators[key]
        if denominator not in self.stats:
            raise ValueError(f"Denominator `{denominator}` not set for key `{key}`.")
        xs = []
        for v, d in zip(values, self.stats[denominator]):
            xs.append(torch.where(d, v, float("inf")).min())
        x = min(xs)
        effective_group = self._effective_reduce_group(key, reduce_group)
        if effective_group is not None:
            dist.all_reduce(x, group=effective_group, op=dist.ReduceOp.MIN)
        if torch.isinf(x):
            return None
        return float(x)

    def _max_of(self, key, reduce_group):
        values = self.stats[key]
        denominator = self.denominators[key]
        if denominator not in self.stats:
            raise ValueError(f"Denominator `{denominator}` not set for key `{key}`.")
        xs = []
        for v, d in zip(values, self.stats[denominator]):
            xs.append(torch.where(d, v, -float("inf")).max())
        x = max(xs)
        effective_group = self._effective_reduce_group(key, reduce_group)
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


def export_all(reduce_group=None, reset=True) -> dict[str, float]:
    stat = {}
    duplicate_keys = set()
    tracker_keys = list(TRACKERS.keys())
    if reduce_group is not None:
        all_trackers = [None for _ in range(dist.get_world_size(reduce_group))]
        dist.all_gather_object(all_trackers, list(TRACKERS.keys()), group=reduce_group)
        tracker_keys = sorted(list(set(flat2d(all_trackers))))
    for tracker_key in tracker_keys:
        tracker = get(tracker_key)
        x = tracker.export(reduce_group=reduce_group, reset=reset)
        for k in x.keys():
            if k in stat:
                duplicate_keys.add(k)
        stat.update(x)
    if duplicate_keys:
        logger.warning(f"Duplicate stat keys detected: {list(duplicate_keys)}")
    return stat
