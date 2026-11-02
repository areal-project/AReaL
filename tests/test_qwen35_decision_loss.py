# SPDX-License-Identifier: Apache-2.0
"""CPU tests for packed PointerHead decision loss and DecisionEngine."""

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from areal.dataset.decision import QUESTION_TYPE_IDS, ROLE_DECIDE, ROLE_OPT
from areal.trainer.sft.decision import (
    DecisionEngine,
    compute_packed_decision_loss,
    question_loss,
)
from areal.utils.stats_tracker import DistributedStatsTracker

_CHOICE = QUESTION_TYPE_IDS["choice"]
_SCORE = QUESTION_TYPE_IDS["score"]


def _proj_from_logits(z: torch.Tensor, dp: int = 8) -> torch.Tensor:
    """Build ``[K + 1, 2 * dp]`` states whose PointerHead scores recover ``z``."""
    k = z.numel()
    scale = 1.0 / math.sqrt(dp)
    q_all = torch.zeros(k + 1, dp, dtype=torch.float32)
    k_all = torch.zeros(k + 1, dp, dtype=torch.float32)
    q_all[-1, 0] = 1.0
    k_all[:-1, 0] = z.float() / scale
    return torch.cat([q_all, k_all], dim=-1)


def _question_tensors(
    z: torch.Tensor,
    label: int,
    qtype: int,
    soft: list[float] | None = None,
    dp: int = 8,
) -> dict[str, torch.Tensor]:
    k = z.numel()
    roles = torch.tensor([ROLE_OPT] * k + [ROLE_DECIDE], dtype=torch.long)
    choice_label = torch.full((k + 1,), -1, dtype=torch.long)
    choice_label[-1] = label
    question_type = torch.full((k + 1,), -1, dtype=torch.long)
    question_type[-1] = qtype
    if soft is None:
        soft_target = torch.full((k + 1,), -1.0, dtype=torch.float32)
    else:
        soft_target = torch.tensor(list(soft) + [-1.0], dtype=torch.float32)
    return {
        "proj": _proj_from_logits(z, dp=dp),
        "token_roles": roles,
        "choice_label": choice_label,
        "question_type": question_type,
        "soft_target": soft_target,
    }


def _pack(questions: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, dict]:
    proj = torch.cat([q["proj"] for q in questions], dim=0)
    lengths = [q["token_roles"].numel() for q in questions]
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(lengths, dtype=torch.int32), dim=0)
    input_ = {
        "cu_seqlens": cu,
        "token_roles": torch.cat([q["token_roles"] for q in questions]),
        "choice_label": torch.cat([q["choice_label"] for q in questions]),
        "question_type": torch.cat([q["question_type"] for q in questions]),
        "soft_target": torch.cat([q["soft_target"] for q in questions]),
    }
    return proj, input_


def _dummy_row(n: int = 3) -> dict[str, torch.Tensor]:
    return {
        "proj": torch.zeros(n, 16, dtype=torch.float32),
        "token_roles": torch.zeros(n, dtype=torch.long),
        "choice_label": torch.full((n,), -1, dtype=torch.long),
        "question_type": torch.full((n,), -1, dtype=torch.long),
        "soft_target": torch.full((n,), -1.0, dtype=torch.float32),
    }


class _FakeEngine:
    def __init__(self, **config):
        defaults = dict(
            is_decision_model=False,
            ord_w=0.0,
            label_smoothing=0.0,
            brier_w=0.0,
            focal_gamma=0.0,
        )
        defaults.update(config)
        self.config = SimpleNamespace(**defaults)
        self.calls = []

    def train(self):
        return None

    def eval(self):
        return None

    def train_batch(self, input_, loss_fn, loss_weight_fn):
        self.calls.append(("train", input_, loss_fn, loss_weight_fn))
        return {"grad_norm": 0.0}

    def eval_batch(self, input_, loss_fn, loss_weight_fn):
        self.calls.append(("eval", input_, loss_fn, loss_weight_fn))
        return None


def test_hard_ce_matches_cross_entropy():
    z = torch.tensor([0.4, -0.7, 1.2])
    actual = question_loss(z, 1, _CHOICE, None, 0.0)
    expected = F.cross_entropy(z[None], torch.tensor([1]))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "options",
    [{}, {"label_smoothing": 0.05}, {"brier_w": 0.5}, {"focal_gamma": 1.0}],
)
def test_training_losses_match_definitions(options):
    z = torch.tensor([0.4, -0.7, 1.2], requires_grad=True)
    ce = F.cross_entropy(z[None], torch.tensor([1]))
    target = F.one_hot(torch.tensor(1), 3).float()
    expected = ce
    if "label_smoothing" in options:
        epsilon = options["label_smoothing"]
        expected = -(((1 - epsilon) * target + epsilon / 3) * z.log_softmax(-1)).sum()
    elif "brier_w" in options:
        expected = ce + options["brier_w"] * (z.softmax(-1) - target).square().sum()
    elif "focal_gamma" in options:
        expected = (1 - z.softmax(-1)[1]) ** options["focal_gamma"] * ce
    actual = question_loss(z, 1, _CHOICE, None, 0.0, **options)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(z.grad).all()


def test_soft_target_ce_and_ignores_modifiers():
    z = torch.tensor([2.0, 0.0, -2.0])
    t = torch.tensor([1 / 3, 1 / 3, 1 / 3])
    expected = -(t * F.log_softmax(z, -1)).sum()
    base = question_loss(z, 0, _CHOICE, t, 0.0)
    torch.testing.assert_close(base, expected)
    for options in (
        {"label_smoothing": 0.05},
        {"brier_w": 0.5},
        {"focal_gamma": 1.0},
    ):
        torch.testing.assert_close(
            question_loss(z, 0, _CHOICE, t, 0.0, **options), base
        )


def test_ordinal_rps_on_score_questions():
    z = torch.tensor([0.2, 0.1, 0.7]).log()
    base = F.cross_entropy(z[None], torch.tensor([1]))
    p = z.softmax(-1)
    observed = (torch.arange(2) >= 1).to(p.dtype)
    expected = base + 0.5 * (p.cumsum(-1)[:-1] - observed).square().mean()
    actual = question_loss(z, 1, _SCORE, None, 0.5)
    torch.testing.assert_close(actual, expected)
    choice = question_loss(z, 1, _CHOICE, None, 0.5)
    torch.testing.assert_close(choice, base)


def test_score_loss_is_proper_at_true_distribution():
    logits = torch.tensor([0.2, 0.8]).log().requires_grad_()
    loss = sum(
        p * question_loss(logits, y, _SCORE, None, 0.5)
        for y, p in enumerate([0.2, 0.8])
    )
    loss.backward()
    assert logits.grad.abs().max().item() < 1e-6


@pytest.mark.parametrize(
    "bad",
    [
        {"label_smoothing": -0.1},
        {"label_smoothing": 1.1},
        {"brier_w": float("nan")},
        {"focal_gamma": -1},
        {"brier_w": 0.5, "focal_gamma": 1.0},
    ],
)
def test_loss_options_fail_before_training(bad):
    with pytest.raises(ValueError):
        question_loss(torch.zeros(3), 0, _CHOICE, None, 0, **bad)


def test_packed_loss_matches_mean_question_loss():
    z0 = torch.tensor([0.4, -0.7, 1.2])
    z1 = torch.tensor([-0.3, 0.9])
    q0 = _question_tensors(z0, label=2, qtype=_CHOICE)
    q1 = _question_tensors(z1, label=0, qtype=_SCORE)
    proj, input_ = _pack([q0, _dummy_row(), q1])
    tracker = DistributedStatsTracker()
    with patch("areal.trainer.sft.decision.stats_tracker", tracker):
        loss = compute_packed_decision_loss(proj, input_, ord_w=0.5)
    expected = torch.stack(
        [
            question_loss(z0, 2, _CHOICE, None, 0.5),
            question_loss(z1, 0, _SCORE, None, 0.5),
        ]
    ).mean()
    torch.testing.assert_close(loss, expected)
    stats = tracker.export(reset=True)
    assert stats["n_questions"] == 2.0
    torch.testing.assert_close(
        torch.tensor(stats["decision_loss/avg"]), expected, rtol=1e-5, atol=1e-6
    )
    assert stats["decision_acc/avg"] == 0.5


def test_packed_soft_target_and_weight_fn():
    z = torch.tensor([2.0, 0.0, -2.0])
    t = [1 / 3, 1 / 3, 1 / 3]
    q = _question_tensors(z, label=0, qtype=_CHOICE, soft=t)
    proj, input_ = _pack([q])
    proj = proj.clone().requires_grad_(True)
    tracker = DistributedStatsTracker()
    with patch("areal.trainer.sft.decision.stats_tracker", tracker):
        loss = compute_packed_decision_loss(proj, input_, brier_w=0.5)
    expected = question_loss(z, 0, _CHOICE, torch.tensor(t), 0.0, brier_w=0.5)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(proj.grad).all()
    dummy_proj, dummy_in = _pack([_dummy_row(2)])
    tracker2 = DistributedStatsTracker()
    with patch("areal.trainer.sft.decision.stats_tracker", tracker2):
        empty = compute_packed_decision_loss(dummy_proj, dummy_in)
    assert float(empty) == 0.0


def test_synthetic_optimization_decreases_loss():
    z = torch.tensor([0.1, 0.0, -0.2])
    q = _question_tensors(z, label=0, qtype=_CHOICE)
    proj = q["proj"].clone().requires_grad_(True)
    _, input_ = _pack([q])
    opt = torch.optim.SGD([proj], lr=0.5)
    tracker = DistributedStatsTracker()
    losses = []
    for _ in range(8):
        opt.zero_grad()
        with patch("areal.trainer.sft.decision.stats_tracker", tracker):
            loss = compute_packed_decision_loss(proj, input_)
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
        tracker.export(reset=True)
    assert losses[-1] < losses[0]


def test_decision_engine_binds_config_and_skips_sft_mask_roll():
    engine = _FakeEngine(is_decision_model=True, ord_w=0.25, label_smoothing=0.05)
    dec = DecisionEngine(engine)
    data = {"token_roles": torch.tensor([1, 2]), "loss_mask": torch.ones(1, 2)}
    dec._train_lm(data)
    dec._evaluate_lm(data)
    assert data["loss_mask"].equal(torch.ones(1, 2))
    assert len(engine.calls) == 2
    for _, _, loss_fn, weight_fn in engine.calls:
        assert loss_fn.func is compute_packed_decision_loss
        assert loss_fn.keywords["ord_w"] == 0.25
        assert loss_fn.keywords["label_smoothing"] == 0.05
        assert weight_fn(data) == 1
