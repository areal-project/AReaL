import math
from collections import defaultdict
from collections.abc import Sized

import numpy as np
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler


class ATGiGPOSampler(AbstractCurriculumSampler):
    """Proportional multi-source sampler.

    Each training step draws samples from every data source in proportion to
    its size (number of rows).  Each source maintains its own shuffled index
    pool; when a pool is exhausted the source's epoch counter increments and
    the pool is reshuffled.
    """

    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        self.batch_size: int = data_config.get("train_batch_size", 64)

        # ---- build per-source index lists ----
        self.task2indices: dict[str, list[int]] = defaultdict(list)
        if hasattr(data_source, "dataframe") and "data_source" in data_source.dataframe.column_names:
            ds_col = data_source.dataframe["data_source"]
            for i, task in enumerate(ds_col):
                self.task2indices[str(task)].append(i)
        else:
            for i in range(len(data_source)):
                item = data_source[i]
                task = item.get("data_source", "unknown") if isinstance(item, dict) else "unknown"
                self.task2indices[task].append(i)

        self.task_types: list[str] = sorted(self.task2indices.keys())
        self.dataset_sizes: dict[str, int] = {t: len(self.task2indices[t]) for t in self.task_types}

        # ---- fixed proportions from dataset sizes ----
        total = sum(self.dataset_sizes.values())
        self.proportions: dict[str, float] = {t: self.dataset_sizes[t] / max(total, 1) for t in self.task_types}

        # ---- per-source counters ----
        self.epoch_counts: dict[str, int] = {t: 0 for t in self.task_types}
        self.step_counts: dict[str, int] = {t: 0 for t in self.task_types}
        self._global_step: int = 0

        # ---- per-source shuffled index pools ----
        self._rng = np.random.default_rng(seed=42)
        self._index_pools: dict[str, list[int]] = {}
        for t in self.task_types:
            pool = list(self.task2indices[t])
            self._rng.shuffle(pool)
            self._index_pools[t] = pool
        self._pool_cursors: dict[str, int] = {t: 0 for t in self.task_types}

    # ------------------------------------------------------------------
    # Required by AbstractCurriculumSampler
    # ------------------------------------------------------------------
    def update(self, batch: DataProto) -> None:
        self._global_step += 1

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _draw_from_pool(self, task: str, n: int) -> list[int]:
        """Draw *n* indices from *task*'s pool, cycling (new epoch) when exhausted."""
        pool = self._index_pools[task]
        cursor = self._pool_cursors[task]
        result: list[int] = []
        remaining = n
        while remaining > 0:
            available = len(pool) - cursor
            if available <= 0:
                self._rng.shuffle(pool)
                cursor = 0
                self.epoch_counts[task] += 1
            take = min(remaining, len(pool) - cursor)
            result.extend(pool[cursor : cursor + take])
            cursor += take
            remaining -= take
        self._pool_cursors[task] = cursor
        self.step_counts[task] += 1
        return result

    def __iter__(self):
        # Allocate counts proportionally, rounding via largest-remainder
        raw = {t: self.proportions[t] * self.batch_size for t in self.task_types}
        counts = {t: int(v) for t, v in raw.items()}
        remainder = self.batch_size - sum(counts.values())
        if remainder > 0:
            by_frac = sorted(self.task_types, key=lambda t: raw[t] - counts[t], reverse=True)
            for t in by_frac[:remainder]:
                counts[t] += 1

        indices: list[int] = []
        for task in self.task_types:
            if counts[task] > 0:
                indices.extend(self._draw_from_pool(task, counts[task]))

        self._rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.batch_size

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def get_metrics(self) -> dict:
        metrics: dict[str, float] = {}
        for t in self.task_types:
            metrics[f"at_gigpo/{t}/proportion"] = self.proportions[t]
            metrics[f"at_gigpo/{t}/epoch_count"] = float(self.epoch_counts[t])
            metrics[f"at_gigpo/{t}/step_count"] = float(self.step_counts[t])
            metrics[f"at_gigpo/{t}/pool_cursor"] = float(self._pool_cursors[t])
        metrics["at_gigpo/global_step"] = float(self._global_step)
        return metrics

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "epoch_counts": dict(self.epoch_counts),
            "step_counts": dict(self.step_counts),
            "_global_step": self._global_step,
            "_pool_cursors": dict(self._pool_cursors),
            "_index_pools": {t: list(p) for t, p in self._index_pools.items()},
            "_rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        for t in self.task_types:
            if t in state.get("epoch_counts", {}):
                self.epoch_counts[t] = state["epoch_counts"][t]
            if t in state.get("step_counts", {}):
                self.step_counts[t] = state["step_counts"][t]
            if t in state.get("_pool_cursors", {}):
                self._pool_cursors[t] = state["_pool_cursors"][t]
            if t in state.get("_index_pools", {}):
                self._index_pools[t] = state["_index_pools"][t]
        self._global_step = state.get("_global_step", self._global_step)
        if "_rng_state" in state:
            self._rng.bit_generator.state = state["_rng_state"]
