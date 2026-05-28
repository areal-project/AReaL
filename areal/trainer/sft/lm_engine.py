# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch

from areal.api import TrainEngine
from areal.experimental.training_service.controller.controller import (
    GatewayTrainController,
)
from areal.infra import TrainController
from areal.infra.rpc.serialization import serialize_value
from areal.utils import stats_tracker
from areal.utils.data import batched_call
from areal.utils.perf_tracer import trace_perf


class LMEngine:
    def __init__(self, engine: TrainEngine):
        self.engine = engine

    @trace_perf("lm_engine.train_lm", category="compute")
    @stats_tracker.scope_func_wrapper("sft")
    def train_lm(self, data: list[dict[str, Any]]) -> None:
        batched_call(self._train_lm, data, unpack=False)

    def _train_lm(self, data: dict[str, Any]) -> None:
        self.engine.train()
        data["loss_mask"] = torch.roll(data["loss_mask"].bool(), shifts=-1, dims=-1)
        stats = self.engine.train_batch(
            input_=data,
            loss_fn=compute_packed_sft_loss,
            loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
        )
        stats_tracker.scalar(**stats)

    @trace_perf("lm_engine.evaluate_lm", category="compute")
    @stats_tracker.scope_func_wrapper("sft-eval")
    def evaluate_lm(self, data: list[dict[str, Any]]) -> None:
        batched_call(self._evaluate_lm, data, unpack=False)

    def _evaluate_lm(self, data: dict[str, Any]) -> None:
        self.engine.eval()
        data["loss_mask"] = torch.roll(data["loss_mask"].bool(), shifts=-1, dims=-1)
        self.engine.eval_batch(
            input_=data,
            loss_fn=compute_packed_sft_loss,
            loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
        )


class LMController(TrainController):
    def train_lm(self, *args, **kwargs):
        self._custom_function_call(
            "train_lm", *args, rpc_meta={"broadcast": True}, **kwargs
        )

    def evaluate_lm(self, *args, **kwargs):
        args, kwargs = self._pad_eval_dispatch_args(args, kwargs, group_size=1)
        self._custom_function_call(
            "evaluate_lm", *args, rpc_meta={"broadcast": True}, **kwargs
        )


class LMControllerV2(GatewayTrainController):
    def train_lm(self, *args, **kwargs):
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        self._gateway_post_result("/sft/train", payload)

    def evaluate_lm(self, *args, **kwargs):
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        self._gateway_post_result("/sft/evaluate", payload)


def compute_packed_sft_loss(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_: dict[str, Any],
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute SFT loss from logprobs."""
    del entropy  # SFT does not use entropy
    cu_seqlens: torch.Tensor = input_["cu_seqlens"]
    loss_mask = input_["loss_mask"].bool()

    logprobs = torch.where(loss_mask, logprobs, 0)

    device = logprobs.device
    loss = -logprobs.sum() / (1e-5 + loss_mask.count_nonzero())
    with torch.no_grad():
        batch_size = cu_seqlens.shape[0] - 1
        seqlogp = torch.zeros(batch_size, dtype=torch.float64, device=device)
        n_seqs = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for i in range(batch_size):
            m = loss_mask[cu_seqlens[i] : cu_seqlens[i + 1]]
            logp = logprobs[cu_seqlens[i] : cu_seqlens[i + 1]]
            valid_tokens = int(m.count_nonzero().item())
            if valid_tokens == 0:
                # This is a padded dummy sequence created in `padded_mb_input`.
                # When Ulysses SP is enabled, padded inputs are passed into the loss function.
                # So we skip it.
                continue

            n_seqs[i] = True
            seqlogp[i] = torch.where(m, logp.detach(), 0.0).sum() / valid_tokens

    ## Logging stats
    stats_tracker.denominator(
        n_seqs=n_seqs,
        n_tokens=torch.ones(logprobs.shape[0], dtype=torch.bool, device=device),
        n_valid_tokens=loss_mask,
        prompt_tokens=loss_mask.logical_not(),
    )
    stats_tracker.stat(ppl=(-seqlogp).exp().float(), denominator="n_seqs")
    stats_tracker.stat(loss=-logprobs.detach(), denominator="n_valid_tokens")

    if vocab_min_logits is not None and vocab_max_logits is not None:
        stats_tracker.stat(
            vocab_min_logits=vocab_min_logits,
            vocab_max_logits=vocab_max_logits,
            denominator="n_tokens",
        )

    return loss
