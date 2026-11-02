# SPDX-License-Identifier: Apache-2.0
"""PointerHead SFT: one packed question row per sequence."""

from __future__ import annotations

import functools
import math
from typing import Any

import torch
import torch.nn.functional as F

from areal.api import TrainEngine
from areal.dataset.decision import QUESTION_TYPE_IDS, ROLE_DECIDE, ROLE_OPT
from areal.trainer.sft.lm_engine import LMEngine
from areal.utils import stats_tracker

_SCORE = QUESTION_TYPE_IDS["score"]


class DecisionEngine(LMEngine):
    def __init__(self, engine: TrainEngine):
        super().__init__(engine)
        c = engine.config
        self._loss_fn = functools.partial(
            compute_packed_decision_loss,
            ord_w=c.ord_w,
            label_smoothing=c.label_smoothing,
            brier_w=c.brier_w,
            focal_gamma=c.focal_gamma,
        )

    def _train_lm(self, data: dict[str, Any]) -> None:
        self.engine.train()
        stats = self.engine.train_batch(
            input_=data,
            loss_fn=self._loss_fn,
            loss_weight_fn=_decision_loss_weight,
        )
        stats_tracker.scalar(**stats)

    def _evaluate_lm(self, data: dict[str, Any]) -> None:
        self.engine.eval()
        self.engine.eval_batch(
            input_=data,
            loss_fn=self._loss_fn,
            loss_weight_fn=_decision_loss_weight,
        )


def _decision_loss_weight(x: dict[str, Any]) -> torch.Tensor:
    return (x["token_roles"] == ROLE_DECIDE).count_nonzero().float()


def question_loss(
    z: torch.Tensor,
    label: torch.Tensor | int,
    qtype: torch.Tensor | int,
    soft_target: torch.Tensor | None,
    ord_w: float,
    label_smoothing: float = 0.0,
    brier_w: float = 0.0,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """Per-question hard/soft CE, plus optional Brier, focal, or ordinal RPS."""
    options = (label_smoothing, brier_w, focal_gamma)
    if (
        not all(math.isfinite(v) and v >= 0 for v in options)
        or label_smoothing > 1
        or sum(v > 0 for v in options) > 1
    ):
        raise ValueError(
            "choose at most one finite, nonnegative loss modifier; "
            "smoothing must be <= 1"
        )
    z = z.float().reshape(-1)
    if (
        soft_target is not None
        and soft_target.numel() == z.numel()
        and bool((soft_target >= 0).all())
    ):
        t = soft_target.to(device=z.device, dtype=z.dtype).reshape(-1)
        return -(t * F.log_softmax(z, dim=-1)).sum()

    y = torch.as_tensor(label, dtype=torch.long).reshape(-1).to(device=z.device)[:1]
    loss = F.cross_entropy(z[None], y, label_smoothing=label_smoothing)
    if brier_w:
        target = F.one_hot(y[0], z.numel()).to(dtype=z.dtype)
        loss = loss + brier_w * (F.softmax(z, dim=-1) - target).square().sum()
    if focal_gamma:
        loss = (1 - torch.exp(-loss)).pow(focal_gamma) * loss
    qtype_id = int(torch.as_tensor(qtype).reshape(-1)[0].item())
    if qtype_id == _SCORE and ord_w > 0 and z.numel() > 1:
        p = F.softmax(z, dim=-1)
        observed_cdf = (
            torch.arange(p.numel() - 1, device=z.device) >= int(y[0].item())
        ).to(dtype=p.dtype)
        loss = loss + ord_w * (p.cumsum(-1)[:-1] - observed_cdf).square().mean()
    return loss


def compute_packed_decision_loss(
    proj_states: torch.Tensor,
    input_: dict[str, Any],
    ord_w: float = 0.0,
    label_smoothing: float = 0.0,
    brier_w: float = 0.0,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """Mean question loss over packed rows. Skip dummy rows with no ``<decide>``."""
    proj_states = proj_states.float()
    q_all, k_all = proj_states.chunk(2, dim=-1)
    scale = 1.0 / math.sqrt(q_all.size(-1))
    cu = input_["cu_seqlens"]
    roles = input_["token_roles"]
    labels = input_["choice_label"]
    qtypes = input_["question_type"]
    soft = input_["soft_target"]

    losses: list[torch.Tensor] = []
    accs: list[torch.Tensor] = []
    for i in range(cu.shape[0] - 1):
        start, end = cu[i], cu[i + 1]
        row = roles[start:end]
        decide = (row == ROLE_DECIDE).nonzero(as_tuple=False).squeeze(-1)
        opts = (row == ROLE_OPT).nonzero(as_tuple=False).squeeze(-1)
        if decide.numel() != 1 or opts.numel() == 0:
            continue
        di = start + decide.reshape(())
        oi = start + opts
        z = (k_all[oi] @ q_all[di]) * scale
        y = labels[di]
        losses.append(
            question_loss(
                z, y, qtypes[di], soft[oi], ord_w, label_smoothing, brier_w, focal_gamma
            )
        )
        accs.append((z.argmax(dim=-1) == y.to(device=z.device)).float())

    device = proj_states.device
    if losses:
        n = torch.ones(len(losses), dtype=torch.bool, device=device)
        loss_t = torch.stack(losses).detach().float()
        acc_t = torch.stack(accs).detach().float()
    else:
        n = torch.zeros(1, dtype=torch.bool, device=device)
        loss_t = acc_t = torch.zeros(1, dtype=torch.float32, device=device)
    stats_tracker.denominator(n_questions=n)
    stats_tracker.stat(
        decision_loss=loss_t, decision_acc=acc_t, denominator="n_questions"
    )
    if not losses:
        return proj_states.sum() * 0.0
    return torch.stack(losses).mean()
