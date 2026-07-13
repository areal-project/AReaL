"""CPU-only fake TrainEngine for training-service integration tests."""

from __future__ import annotations

from typing import Any

from areal.api import TrainEngine


def _sum_numbers(value: Any) -> float:
    if isinstance(value, dict):
        return sum(_sum_numbers(v) for v in value.values())
    if isinstance(value, list):
        return sum(_sum_numbers(v) for v in value)
    if isinstance(value, tuple):
        return sum(_sum_numbers(v) for v in value)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


class FakeTrainEngine(TrainEngine):
    """Minimal concrete TrainEngine used by integration tests."""

    def __init__(self, *args: Any, **kwargs: Any):
        self._initialized = False
        self._version = 0
        self._train_mode = True
        self._offloaded = False
        self._zero_grad_calls = 0
        self._optimizer_step_calls = 0
        self._lr_step_calls = 0
        self._last_saved_meta: Any = None
        self._last_loaded_meta: Any = None
        self._init_kwargs = dict(kwargs)

    def create_process_group(self, parallel_strategy=None):
        return None

    def initialize(self, *args, **kwargs):
        self._initialized = True
        return None

    @property
    def data_parallel_group(self):
        return None

    @property
    def data_parallel_rank(self) -> int:
        return 0

    @property
    def data_parallel_world_size(self) -> int:
        return 1

    def current_data_parallel_head(self) -> int:
        return 0

    def is_data_parallel_head(self) -> bool:
        return True

    @property
    def context_and_model_parallel_group(self):
        return None

    @property
    def cpu_group(self):
        return None

    @property
    def initialized(self) -> bool:
        return self._initialized

    def train(self, mode: bool = True):
        self._train_mode = mode
        return None

    def update_weights(self, meta):
        return None

    def connect_engine(self, engine, meta):
        return None

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
    ) -> list[dict[str, Any]]:
        return data

    def prepare_batch(
        self,
        dataloader,
        workflow,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn=None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ) -> list[dict[str, Any]]:
        return []

    def set_version(self, version: int):
        self._version = int(version)

    def get_version(self) -> int:
        return self._version

    def save(self, meta):
        self._last_saved_meta = meta

    def load(self, meta):
        self._last_loaded_meta = meta

    def optimizer_zero_grad(self):
        self._zero_grad_calls += 1

    def optimizer_step(self):
        self._optimizer_step_calls += 1
        return {
            "update_successful": 1.0,
            "grad_norm": float(self._optimizer_step_calls),
            "lr": 1e-3,
        }

    def lr_scheduler_step(self):
        self._lr_step_calls += 1

    def forward_backward_batch(
        self,
        mb_list,
        process_output_fn,
        forward_only: bool = False,
    ) -> None:
        return None

    def train_batch(
        self,
        input_: dict[str, Any],
        loss_fn=None,
        loss_weight_fn=None,
    ) -> dict[str, float]:
        return {
            "total": _sum_numbers(input_),
            "version": float(self._version),
            "train_mode": float(self._train_mode),
        }

    def eval_batch(
        self,
        input_: dict[str, Any],
        loss_fn=None,
        loss_weight_fn=None,
    ) -> float:
        return _sum_numbers(input_) + self._version

    def forward_batch(
        self,
        input_: dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn=None,
    ) -> dict[str, Any]:
        return {
            "total": _sum_numbers(input_),
            "version": self._version,
            "train_mode": self._train_mode,
            "output_seqlens": output_seqlens,
        }

    def train_lm(self, input_, **kwargs):
        _ = kwargs
        return self.train_batch(input_)

    def evaluate_lm(self, input_, **kwargs):
        _ = kwargs
        return self.eval_batch(input_)

    def export_stats(self) -> dict[str, float]:
        return {
            "version": float(self._version),
            "train_mode": float(self._train_mode),
            "offloaded": float(self._offloaded),
            "zero_grad_calls": float(self._zero_grad_calls),
            "optimizer_step_calls": float(self._optimizer_step_calls),
            "lr_step_calls": float(self._lr_step_calls),
            "saved_meta_size": float(len(str(self._last_saved_meta))),
            "loaded_meta_size": float(len(str(self._last_loaded_meta))),
            "world_size": float(self._init_kwargs.get("world_size", -1)),
        }

    def onload(self) -> None:
        self._offloaded = False

    def offload(self) -> None:
        self._offloaded = True

    def get_device_stats(self):
        return {"device": "cpu"}
