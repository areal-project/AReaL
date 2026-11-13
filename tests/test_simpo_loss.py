# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from areal.api.cli_args import DPOEngineConfig
from areal.trainer.dpo.dpo_engine import compute_dpo_loss
from areal.utils.functional import dpo_preference_loss


def test_simpo_preference_loss_math():
    logits = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32)
    beta = 2.0
    gamma = 0.5

    loss = dpo_preference_loss(logits, beta=beta, loss_type="simpo", simpo_gamma=gamma)
    # Expected: -log_sigmoid(beta * logits - gamma)
    expected = -torch.nn.functional.logsigmoid(beta * logits - gamma)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_simpo_margin_effect():
    logits = torch.tensor([0.0], dtype=torch.float32)
    beta = 1.0
    gamma = 0.5

    loss = dpo_preference_loss(logits, beta=beta, loss_type="simpo", simpo_gamma=gamma)
    # -log(sigmoid(-0.5)) = log(1 + exp(0.5))
    expected = math.log(1.0 + math.exp(0.5))
    assert math.isclose(loss.item(), expected, rel_tol=1e-5)

    # Larger gamma enforces higher penalty when margin is not met
    gamma_large = 2.0
    loss_large = dpo_preference_loss(
        logits, beta=beta, loss_type="simpo", simpo_gamma=gamma_large
    )
    assert loss_large.item() > loss.item()


def test_compute_dpo_loss_simpo_forward_backward():
    # Construct 2 pairs (4 sequences)
    # Pair 0: chosen (len 3), rejected (len 2)
    # Pair 1: chosen (len 2), rejected (len 4)
    cu_seqlens = torch.tensor([0, 3, 5, 7, 11], dtype=torch.long)
    tot_tokens = 11

    logprobs = torch.nn.Parameter(
        torch.randn(tot_tokens, dtype=torch.float32, requires_grad=True)
    )
    ref_logprobs = torch.zeros(tot_tokens, dtype=torch.float32)
    loss_mask = torch.ones(tot_tokens, dtype=torch.bool)

    input_data = {
        "cu_seqlens": cu_seqlens,
        "loss_mask": loss_mask,
        "ref_logprobs": ref_logprobs,
    }

    loss = compute_dpo_loss(
        logprobs=logprobs,
        entropy=None,
        input_=input_data,
        beta=0.5,
        loss_type="simpo",
        simpo_gamma=0.5,
    )

    assert torch.isfinite(loss)
    assert loss.ndim == 0  # scalar

    loss.backward()
    assert logprobs.grad is not None
    assert torch.isfinite(logprobs.grad).all()


def test_simpo_length_normalization_distinguishes_verbosity():
    # Sequence with equal total log-prob but different completion lengths
    # Chosen: 4 tokens with logp = -1.0 each -> sum = -4.0, avg = -1.0
    # Rejected: 2 tokens with logp = -2.0 each -> sum = -4.0, avg = -2.0
    cu_seqlens = torch.tensor([0, 4, 6], dtype=torch.long)
    tot_tokens = 6
    logprobs = torch.tensor([-1.0, -1.0, -1.0, -1.0, -2.0, -2.0], dtype=torch.float32)
    ref_logprobs = torch.zeros(tot_tokens, dtype=torch.float32)
    loss_mask = torch.ones(tot_tokens, dtype=torch.bool)

    input_data = {
        "cu_seqlens": cu_seqlens,
        "loss_mask": loss_mask,
        "ref_logprobs": ref_logprobs,
    }

    # Under standard DPO without length normalization, chosen_sum - rejected_sum = (-4) - (-4) = 0
    loss_standard = compute_dpo_loss(
        logprobs=logprobs,
        entropy=None,
        input_=input_data,
        beta=1.0,
        loss_type="sigmoid",
    )
    # Under SimPO, chosen_avg = -1.0, rejected_avg = -2.0 -> chosen is preferred!
    loss_simpo = compute_dpo_loss(
        logprobs=logprobs,
        entropy=None,
        input_=input_data,
        beta=1.0,
        loss_type="simpo",
        simpo_gamma=0.0,
    )

    # logits = (-1.0) - (-2.0) = 1.0 > 0. loss = -log_sigmoid(1.0) < -log_sigmoid(0.0)
    assert loss_simpo.item() < loss_standard.item()


def test_dpo_engine_config_simpo_validation():
    # Valid SimPO config
    cfg = DPOEngineConfig(loss_type="simpo", simpo_gamma=0.8)
    assert cfg.loss_type == "simpo"
    assert cfg.simpo_gamma == 0.8

    # Negative gamma should fail
    with pytest.raises(ValueError, match="simpo_gamma must be non-negative"):
        DPOEngineConfig(loss_type="simpo", simpo_gamma=-0.1)

    # Invalid loss type should fail
    with pytest.raises(ValueError, match="Unsupported DPO loss_type"):
        DPOEngineConfig(loss_type="unknown")


def test_compute_dpo_loss_simpo_without_ref_logprobs_matches_with_ref():
    cu_seqlens = torch.tensor([0, 4, 6], dtype=torch.long)
    logprobs = torch.tensor([-1.0, -1.0, -1.0, -1.0, -2.0, -2.0])
    loss_mask = torch.ones(6, dtype=torch.bool)
    input_no_ref = {"cu_seqlens": cu_seqlens, "loss_mask": loss_mask}
    input_with_ref = {**input_no_ref, "ref_logprobs": torch.full((6,), -3.0)}

    loss_no_ref = compute_dpo_loss(
        logprobs, None, input_no_ref, beta=1.0, loss_type="simpo", simpo_gamma=0.0
    )
    loss_with_ref = compute_dpo_loss(
        logprobs, None, input_with_ref, beta=1.0, loss_type="simpo", simpo_gamma=0.0
    )

    torch.testing.assert_close(loss_no_ref, loss_with_ref, rtol=0, atol=0)


class _RecordingDPOActor:
    def __init__(self):
        self.batches = []
        self.cleared = []

    def train_dpo(self, batch):
        self.batches.append(batch)
        raise _StopTraining

    def evaluate_dpo(self, batch):
        self.batches.append(batch)

    def clear_batches(self, *args):
        self.cleared.append(args)


class _StopTraining(Exception):
    pass


def _build_simpo_trainer(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from areal.trainer import dpo_trainer
    from areal.trainer.dpo_trainer import DPOTrainer

    monkeypatch.setattr(
        dpo_trainer.stats_tracker, "record_timing", lambda *a, **k: nullcontext()
    )
    monkeypatch.setattr(
        dpo_trainer.perf_tracer, "trace_scope", lambda *a, **k: nullcontext()
    )

    trainer = DPOTrainer.__new__(DPOTrainer)
    trainer.config = SimpleNamespace(total_train_epochs=1, total_train_steps=None)
    trainer.recover_info = None
    trainer.train_dataloader = [[{"input_ids": 0}]]
    trainer.saver = SimpleNamespace(maybe_wait_for_staging=lambda: None)
    trainer.actor = _RecordingDPOActor()
    trainer.ref = None
    trainer._load_bcast_from = lambda data_generator: next(data_generator)
    trainer._evaluate_before_train = lambda: False
    return trainer


def test_dpo_trainer_simpo_train_step_runs_without_ref(monkeypatch):
    trainer = _build_simpo_trainer(monkeypatch)

    with pytest.raises(_StopTraining):
        trainer.train()

    assert trainer.actor.batches == [[{"input_ids": 0}]]


def test_dpo_trainer_simpo_evaluate_runs_without_ref(monkeypatch):
    from types import SimpleNamespace

    from areal.trainer import dpo_trainer

    trainer = _build_simpo_trainer(monkeypatch)
    trainer.valid_dataloader = [[{"input_ids": 1}]]
    trainer.actor.cpu_group = None
    monkeypatch.setattr(dpo_trainer.dist, "barrier", lambda group=None: None)
    monkeypatch.setattr(
        dpo_trainer, "current_platform", SimpleNamespace(synchronize=lambda: None)
    )
    monkeypatch.setattr(dpo_trainer, "is_single_controller", lambda: True)

    trainer._evaluate_fn()

    assert trainer.actor.batches == [[{"input_ids": 1}]]
    assert trainer.actor.cleared == [([{"input_ids": 1}], None)]
