# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch

from areal.api import TrainEngine
from areal.experimental.training_service.controller.controller import (
    GatewayTrainController,
)
from areal.infra import TrainController
from areal.infra.rpc.serialization import serialize_value
from areal.utils import logging, stats_tracker
from areal.utils.data import batched_call
from areal.utils.perf_tracer import trace_perf

logger = logging.getLogger("RWEngine")


def _rw_valid_pairs(x: dict[str, Any]) -> torch.Tensor:
    seqlens = x["cu_seqlens"][1:] - x["cu_seqlens"][:-1]
    return seqlens.view(-1, 2).ne(0).all(dim=1)


def _rw_loss_weight(x: dict[str, Any]) -> torch.Tensor:
    return _rw_valid_pairs(x).count_nonzero().float()


def _log_empty_rw_stats(device: torch.device) -> None:
    n_pairs = torch.zeros(1, dtype=torch.bool, device=device)
    stats_tracker.denominator(n_pairs=n_pairs)
    stats_tracker.stat(
        correct_ratio=torch.zeros(1, dtype=torch.float32, device=device),
        pos_score=torch.zeros(1, dtype=torch.float32, device=device),
        neg_score=torch.zeros(1, dtype=torch.float32, device=device),
        loss=torch.zeros(1, dtype=torch.float32, device=device),
        denominator="n_pairs",
    )


class RWEngine:
    def __init__(self, engine: TrainEngine):
        self.engine = engine

    @trace_perf("rw_engine.train_rw", category="compute")
    @stats_tracker.scope_func_wrapper("rw")
    def train_rw(self, data: list[dict[str, Any]]) -> None:
        batched_call(self._train_rw, data, unpack=False)

    def _train_rw(self, data: dict[str, Any]) -> None:
        """Train on a batch (reward model)."""
        if _rw_loss_weight(data) == 0:
            _log_empty_rw_stats(data["cu_seqlens"].device)
        self.engine.train()
        stats = self.engine.train_batch(
            input_=data,
            loss_fn=compute_rw_loss,
            loss_weight_fn=_rw_loss_weight,
        )
        stats_tracker.scalar(**stats)

    @trace_perf("rw_engine.evaluate_rw", category="compute")
    @stats_tracker.scope_func_wrapper("rw-eval")
    def evaluate_rw(self, data: list[dict[str, Any]]) -> None:
        batched_call(self._evaluate_rw, data, unpack=False)

    def _evaluate_rw(self, data: dict[str, Any]) -> None:
        if _rw_loss_weight(data) == 0:
            _log_empty_rw_stats(data["cu_seqlens"].device)
        self.engine.eval()
        self.engine.eval_batch(
            input_=data,
            loss_fn=compute_rw_loss,
            loss_weight_fn=_rw_loss_weight,
        )


class RWController(TrainController):
    def train_rw(self, *args, **kwargs):
        self._custom_function_call(
            "train_rw", *args, rpc_meta={"broadcast": True}, **kwargs
        )

    def evaluate_rw(self, *args, **kwargs):
        # rw_modeling_collate_fn produces 2 sequences (chosen + rejected) per
        # example; group_size=2 keeps each pair on the same DP rank.
        args, kwargs = self._pad_eval_dispatch_args(args, kwargs, group_size=2)
        self._custom_function_call(
            "evaluate_rw",
            *args,
            group_size=2,
            rpc_meta={"broadcast": True},
            **kwargs,
        )


class RWControllerV2(GatewayTrainController):
    def train_rw(self, *args, **kwargs):
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        self._gateway_post_result("/rw/train", payload)

    def evaluate_rw(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("group_size", 2)
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        self._gateway_post_result("/rw/evaluate", payload)


def compute_rw_loss(scores: torch.Tensor, input_: dict[str, Any]) -> torch.Tensor:
    device = scores.device
    cu_seqlens = input_["cu_seqlens"]
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu()
    valid_pairs = _rw_valid_pairs(input_)
    if not valid_pairs.any():
        _log_empty_rw_stats(device)
        return torch.zeros((), dtype=torch.float32, device=device)

    valid_pair_mask = valid_pairs.to(device=device)
    terminal_indices = seqlens.cumsum(0).to(device=device) - 1

    assert scores.shape[0] == seqlens.sum(), (scores.shape, seqlens.sum())
    scores = scores[terminal_indices].view(-1, 2)[valid_pair_mask].float()

    loss = -(torch.nn.functional.logsigmoid(scores[:, 0] - scores[:, 1]))
    logging_loss = loss.detach()
    loss = loss.mean()

    # Logging.
    with torch.no_grad():
        stats_tracker.denominator(
            n_pairs=torch.ones(scores.shape[0], dtype=torch.bool, device=device),
        )
        stats_tracker.stat(
            correct_ratio=(scores[:, 0] > scores[:, 1]).detach().float(),
            pos_score=scores[:, 0].detach().float(),
            neg_score=scores[:, 1].detach().float(),
            loss=logging_loss.float(),
            denominator="n_pairs",
        )
    return loss
