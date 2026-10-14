# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch

from areal.api.cli_args import LossTermConfig, PPOActorConfig
from areal.api.loss import get_named_loss_names
from areal.trainer.mopd.loss import mopd_loss_fn
from areal.trainer.ppo.actor import grpo_loss_fn
from areal.trainer.ppo.loss import ActorLoss
from areal.trainer.ppo.loss_functions import mopd_loss
from areal.trainer.ppo.loss_reduction import prepare_policy_gradient_batch
from areal.utils.data import pack_tensor_dict


@pytest.fixture(autouse=True)
def isolate_plugin_diagnostics():
    with patch("areal.trainer.ppo.loss_functions.stats_tracker"):
        yield


def quadratic_loss(*, outputs, batch, mask, factor=1.0):
    return outputs["logprobs"].square() * factor


def _data():
    mask = torch.tensor(
        [[1, 1, 1, 1, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0]],
        dtype=torch.bool,
    )
    return dict(
        input_ids=torch.arange(20).reshape(4, 5),
        attention_mask=torch.ones_like(mask),
        loss_mask=mask,
        turn_ids=torch.tensor([[-1, 0, 0, 1, 1]] * 4),
        advantages=torch.arange(20).reshape(4, 5).float() / 10 - 0.5,
        logprobs=torch.full((4, 5), -0.2),
        prox_logp=torch.full((4, 5), -0.1),
    )


def _config(
    mode="token_mean",
    implementation="tests.test_actor_loss_plugins.quadratic_loss",
    **kw,
):
    return PPOActorConfig(
        loss_aggregation=mode,
        loss_aggregation_divisor=4 if mode == "constant" else None,
        losses={"objective": LossTermConfig(implementation=implementation, **kw)},
    )


def _slice(data, start, end):
    return {
        k: v[start:end] if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }


def _evaluate(step, data, x):
    with patch("areal.trainer.ppo.loss.stats_tracker"):
        numerator = step(x, torch.zeros_like(x), data)
    return {
        name: value / step.loss_weight(data)[name].clamp_min(1e-12)
        for name, value in numerator.items()
    }


@pytest.mark.parametrize(
    "mode", ["token_mean", "seq_mean", "constant", "prompt_mean", "turn_mean"]
)
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_named_reductions_match_independent_oracle(mode, packed, device):
    data = {k: v.to(device) for k, v in _data().items()}
    prepared = ActorLoss(_config(mode)).prepare(data, [2, 2])
    step = prepared.for_steps([data])[0]
    values = torch.arange(20, device=device).reshape(4, 5).float() / 10
    mask = data["loss_mask"]
    square = values.square()
    if mode == "token_mean":
        expected = square[mask].mean()
    elif mode == "seq_mean":
        expected = torch.stack([square[i][mask[i]].mean() for i in range(3)]).mean()
    elif mode == "constant":
        expected = square[mask].sum() / 12
    elif mode == "prompt_mean":
        expected = (square[:2][mask[:2]].mean() + square[2:][mask[2:]].mean()) / 2
    else:
        expected = torch.stack(
            [
                square[0, :2].mean(),
                square[0, 2:4].mean(),
                square[1, :2].mean(),
                square[2, :2].mean(),
                square[2, 2],
            ]
        ).mean()
    if packed:
        data = pack_tensor_dict(data)
        values = values.flatten()
    values.requires_grad_()
    actual = _evaluate(step, data, values)["objective"]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert torch.isfinite(torch.autograd.grad(actual, values)[0]).all()
    assert get_named_loss_names(step.loss_weight) == ("objective",)


@pytest.mark.parametrize(
    "mode", ["token_mean", "seq_mean", "constant", "prompt_mean", "turn_mean"]
)
def test_named_microbatch_split_preserves_objective_and_gradient(mode):
    data = _data()
    step = ActorLoss(_config(mode)).prepare(data, [2, 2]).for_steps([data])[0]
    x = torch.linspace(-0.5, 0.8, 20).reshape(4, 5).requires_grad_()
    expected = _evaluate(step, data, x)["objective"]
    numerator = 0
    with patch("areal.trainer.ppo.loss.stats_tracker"):
        for start, end in [(0, 1), (1, 3), (3, 4)]:
            numerator = (
                numerator
                + step(x[start:end], x[start:end] * 0, _slice(data, start, end))[
                    "objective"
                ]
            )
    actual = numerator / step.loss_weight(data)["objective"]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        torch.autograd.grad(actual, x, retain_graph=True)[0],
        torch.autograd.grad(expected, x)[0],
        rtol=1e-6,
        atol=1e-7,
    )


def test_prompt_terms_cross_optimizer_steps_with_independent_empty_support():
    data = _data()
    data["other"] = data["loss_mask"].clone()
    data["other"][:2] = False
    data["empty"] = torch.zeros_like(data["loss_mask"])
    spec = dict(
        implementation="tests.test_actor_loss_plugins.quadratic_loss",
        reduction="prompt_mean",
    )
    config = PPOActorConfig(
        losses={
            "all": LossTermConfig(**spec),
            "other": LossTermConfig(**spec, mask_key="other", coefficient=2),
            "empty": LossTermConfig(**spec, mask_key="empty"),
        }
    )
    prepared = ActorLoss(config).prepare(data, [2, 2])
    batches = [_slice(data, i, i + 1) for i in range(4)]
    steps = prepared.for_steps(batches)
    x = torch.arange(20).reshape(4, 5).float().requires_grad_()
    means = [
        _evaluate(step, mb, x[i : i + 1])
        for i, (step, mb) in enumerate(zip(steps, batches))
    ]
    mask = data["loss_mask"]
    expected = (x[:2].square()[mask[:2]].mean() + x[2:].square()[mask[2:]].mean()) / 2
    torch.testing.assert_close(
        sum(v["all"] for v in means) / 4, expected, rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        sum(v["other"] for v in means) / 4,
        2 * x[2].square()[mask[2]].mean(),
        rtol=1e-6,
        atol=1e-6,
    )
    assert all(v["empty"].item() == 0 for v in means)
    assert means[3]["all"].requires_grad


@pytest.mark.parametrize("mode", ["token_mean", "seq_mean", "prompt_mean", "constant"])
@pytest.mark.parametrize("surrogate", ["ppo", "sapo", "cispo"])
def test_named_policy_matches_legacy_loss_and_gradient(mode, surrogate):
    data = _data()
    config = _config(mode, "policy")
    if surrogate == "sapo":
        config.use_sapo_loss = True
    if surrogate == "cispo":
        config.use_cispo_loss = True
        config.eps_clip_higher = 0.3
    legacy = prepare_policy_gradient_batch(
        data, mode=mode, divisor=config.loss_aggregation_divisor, group_sizes=[2, 2]
    ).for_steps([data])[0]
    step = ActorLoss(config).prepare(data, [2, 2]).for_steps([data])[0]
    x = torch.linspace(-0.4, 0.1, 20).reshape(4, 5).requires_grad_()
    with patch("areal.trainer.ppo.actor.stats_tracker"):
        expected = grpo_loss_fn(
            x,
            x * 0,
            data,
            eps_clip=0.2,
            eps_clip_higher=config.eps_clip_higher,
            c_clip=None,
            loss_step=legacy,
            use_sapo_loss=config.use_sapo_loss,
            use_cispo_loss=config.use_cispo_loss,
        )
    actual = _evaluate(step, data, x)["objective"]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        torch.autograd.grad(actual, x, retain_graph=True)[0],
        torch.autograd.grad(expected, x)[0],
        rtol=1e-6,
        atol=1e-7,
    )


def test_named_mopd_uses_immutable_behavior_logprobs():
    data = _data()
    data.update(
        mopd_behavior_logprobs=torch.full((4, 5), -2.0),
        mopd_teacher_logp_sum=torch.full((4, 5), -0.3),
        mopd_teacher_weight_sum=torch.ones(4, 5),
    )
    x = torch.zeros(4, 5, requires_grad=True)
    actual = mopd_loss(
        outputs={"logprobs": x},
        batch=data,
        mask=data["loss_mask"],
        importance_ratio_cap=2.0,
    )
    expected, _ = mopd_loss_fn(
        x,
        data["mopd_behavior_logprobs"],
        data["mopd_teacher_logp_sum"],
        data["mopd_teacher_weight_sum"],
        data["loss_mask"],
        importance_ratio_cap=2.0,
        reduction="none",
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    del data["mopd_behavior_logprobs"]
    with pytest.raises(ValueError, match="immutable"):
        mopd_loss(outputs={"logprobs": x}, batch=data, mask=data["loss_mask"])


@pytest.mark.parametrize(
    "kw",
    [
        {"coefficient": -1},
        {"coefficient": float("nan")},
        {"reduction": "unknown"},
        {"kwargs": {"a": object()}},
    ],
)
def test_loss_config_rejects_invalid_specs(kw):
    with pytest.raises(ValueError):
        LossTermConfig(**kw)


def test_disabled_loss_does_not_import_and_invalid_runtime_kwargs_fail():
    config = _config()
    config.losses["disabled"] = LossTermConfig(
        implementation="missing.module", enabled=False
    )
    assert len(ActorLoss(config).terms) == 1
    with pytest.raises(ValueError, match="cannot override"):
        ActorLoss(_config(kwargs={"mask": 1}))


def test_explicit_reductions_override_unused_default_validation():
    config = PPOActorConfig(
        loss_aggregation="constant",
        losses={"policy": LossTermConfig(reduction="token_mean")},
    )
    assert ActorLoss(config).terms[0].reduction == "token_mean"
    config = PPOActorConfig(
        loss_aggregation="seq_mean",
        enable_tree_training=True,
        losses={"policy": LossTermConfig(reduction="token_mean")},
    )
    assert ActorLoss(config).terms[0].reduction == "token_mean"


def test_constant_term_accepts_shared_divisor_with_token_default():
    config = PPOActorConfig(
        loss_aggregation_divisor=8,
        losses={"policy": LossTermConfig(reduction="constant")},
    )
    assert ActorLoss(config).terms[0].reduction == "constant"


def test_named_config_omegaconf_round_trip():
    from omegaconf import OmegaConf

    config = _config("turn_mean")
    structured = OmegaConf.structured(config)
    restored = OmegaConf.to_object(structured.losses)
    assert restored["objective"] == config.losses["objective"]
    assert ActorLoss(structured).terms[0].reduction == "turn_mean"


@pytest.mark.parametrize("r3_enabled", [False, True])
@pytest.mark.parametrize("fail_training", [False, True])
def test_actor_named_update_preserves_schedule_and_r3(r3_enabled, fail_training):
    from types import SimpleNamespace

    from areal.trainer.ppo.actor import PPOActor

    visits = []

    def train_batch(batch, loss_fn, loss_weight_fn):
        assert get_named_loss_names(loss_weight_fn) == ("objective",)
        assert "routed_experts" not in batch
        assert "r3_routing_valid" not in batch
        assert "_r3_row_indices" not in batch
        if r3_enabled:
            indices = batch["input_ids"][:, 0] // 5
            torch.testing.assert_close(
                engine._r3_pending_routed_experts, routes[indices], rtol=0, atol=0
            )
            torch.testing.assert_close(
                engine._r3_pending_valid, validity[indices], rtol=0, atol=0
            )
        if fail_training:
            raise RuntimeError("injected training failure")
        x = torch.ones_like(batch["advantages"], requires_grad=True)
        numerator = loss_fn(x, x * 0, batch)["objective"]
        (numerator / loss_weight_fn(batch)["objective"].clamp_min(1)).backward()
        assert torch.isfinite(x.grad).all()
        visits.extend(batch["input_ids"][:, 0].tolist())
        return {}

    engine = SimpleNamespace(
        train=lambda: None,
        train_batch=train_batch,
        _r3_enabled=r3_enabled,
        _r3_pending_routed_experts=None,
        _r3_pending_valid=None,
        get_version=lambda: 0,
        data_parallel_group=None,
        device=torch.device("cpu"),
    )
    cfg = _config("turn_mean")
    cfg.ppo_n_minibatches = 2
    actor = PPOActor(cfg, engine)
    data = _data()
    routes = torch.arange(4 * 5 * 2 * 2).reshape(4, 5, 2, 2)
    validity = torch.tensor([True, False, True, True])
    if r3_enabled:
        data.update(routed_experts=routes, r3_routing_valid=validity)
    data.update(
        rewards=torch.ones(4),
        kl_rewards=torch.zeros(4, 5),
        tot_rewards=torch.zeros(4, 5),
    )
    with (
        patch("areal.trainer.ppo.actor.stats_tracker"),
        patch("areal.trainer.ppo.loss.stats_tracker"),
        pytest.raises(RuntimeError, match="injected training failure")
        if fail_training
        else nullcontext(),
    ):
        actor._ppo_update(data)
    assert engine._r3_pending_routed_experts is None
    assert engine._r3_pending_valid is None
    if fail_training:
        return
    from areal.utils.data import split_training_batch_into_microbatches

    expected = [
        row
        for mb in split_training_batch_into_microbatches(_data(), n_mbs=2)
        for row in mb["input_ids"][:, 0].tolist()
    ]
    assert visits == expected


@pytest.mark.parametrize("action", ["mask", "clamp"])
def test_named_policy_rejection_preserves_original_support(action):
    from areal.api.cli_args import RejectionSamplingConfig

    data = _data()
    data["prox_logp"][0, 0] = 3.0
    config = _config("token_mean", "policy")
    config.rejection_sampling = RejectionSamplingConfig(action=action, upper=2.0)
    step = ActorLoss(config).prepare(data, [2, 2]).for_steps([data])[0]
    x = torch.zeros(4, 5, requires_grad=True)
    with patch("areal.trainer.ppo.actor.stats_tracker"):
        expected = grpo_loss_fn(
            x,
            x * 0,
            data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            rejection_sampling=config.rejection_sampling,
        )
    actual = _evaluate(step, data, x)["objective"]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        torch.autograd.grad(actual, x, retain_graph=True)[0],
        torch.autograd.grad(expected, x)[0],
        rtol=1e-6,
        atol=1e-7,
    )
    assert step.loss_weight(data)["objective"] == 9


def test_entropy_requires_gradient_and_has_expected_coefficient():
    from areal.trainer.ppo.loss_functions import entropy_loss

    data = _data()
    x = torch.ones(4, 5, requires_grad=True)
    with pytest.raises(ValueError, match="differentiable entropy"):
        entropy_loss(
            outputs={"logprobs": x, "entropy": x.detach()},
            batch=data,
            mask=data["loss_mask"],
        )
    cfg = _config("token_mean", "entropy", coefficient=0.2)
    step = ActorLoss(cfg).prepare(data, [2, 2]).for_steps([data])[0]
    with patch("areal.trainer.ppo.loss.stats_tracker"):
        loss = step(x, x * 2, data)["objective"] / step.loss_weight(data)["objective"]
    grad = torch.autograd.grad(loss, x)[0]
    torch.testing.assert_close(grad, -0.4 * data["loss_mask"] / 9, rtol=1e-6, atol=1e-7)


def invalid_loss(*, outputs, batch, mask, kind):
    if kind == "shape":
        return outputs["logprobs"].sum()
    if kind == "dtype":
        return torch.zeros_like(mask, dtype=torch.long)
    if kind == "nan":
        return outputs["logprobs"] * float("nan")
    return outputs["logprobs"].detach()


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "detached"])
def test_invalid_plugin_outputs_are_rejected(kind):
    data = _data()
    cfg = _config(
        implementation="tests.test_actor_loss_plugins.invalid_loss",
        kwargs={"kind": kind},
    )
    step = ActorLoss(cfg).prepare(data, [2, 2]).for_steps([data])[0]
    x = torch.ones(4, 5, requires_grad=True)
    with pytest.raises((ValueError, RuntimeError)):
        _evaluate(step, data, x)


@pytest.mark.parametrize("case", ["missing", "negative", "decreasing", "float"])
def test_turn_reduction_rejects_invalid_ids(case):
    data = _data()
    if case == "missing":
        del data["turn_ids"]
    elif case == "negative":
        data["turn_ids"][0, 1] = -1
    elif case == "decreasing":
        data["turn_ids"][0, 1:4] = torch.tensor([1, 0, 1])
    else:
        data["turn_ids"] = data["turn_ids"].float()
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        ActorLoss(_config("turn_mean")).prepare(data, [2, 2])


def test_named_mopd_replaces_implicit_composition_and_checks_preparation():
    from types import SimpleNamespace

    from areal.api.cli_args import MOPDLossConfig
    from areal.trainer.ppo.actor import PPOActor

    config = PPOActorConfig(
        losses={"distillation": LossTermConfig(implementation="mopd", coefficient=0.7)}
    )
    actor = PPOActor(config, SimpleNamespace())
    actor.configure_mopd_loss(
        MOPDLossConfig(
            rl_coefficient=0, distillation_coefficient=0.2, importance_ratio_cap=3
        )
    )
    data = _data()
    data.update(
        mopd_behavior_logprobs=torch.full((4, 5), -0.2),
        mopd_teacher_logp_sum=torch.full((4, 5), -0.4),
        mopd_teacher_weight_sum=torch.ones(4, 5),
    )
    step = actor.actor_loss.prepare(data, [2, 2]).for_steps([data])[0]
    x = torch.zeros(4, 5, requires_grad=True)
    expected, _ = mopd_loss_fn(
        x,
        data["mopd_behavior_logprobs"],
        data["mopd_teacher_logp_sum"],
        data["mopd_teacher_weight_sum"],
        data["loss_mask"],
    )
    actual = _evaluate(step, data, x)["distillation"]
    torch.testing.assert_close(actual, 0.7 * expected, rtol=1e-6, atol=1e-7)
    bad = PPOActor(_config(implementation="policy"), SimpleNamespace())
    with pytest.raises(ValueError, match="teacher targets"):
        bad.configure_mopd_loss(
            MOPDLossConfig(rl_coefficient=1, distillation_coefficient=1)
        )


def test_named_policy_non_float32_inputs_log_float32_diagnostics():
    from areal.trainer.ppo.loss_functions import policy_loss
    from areal.utils.stats_tracker import DistributedStatsTracker

    data = _data()
    for key in ("advantages", "logprobs", "prox_logp"):
        data[key] = data[key].double()
    x = torch.zeros(4, 5, dtype=torch.float64, requires_grad=True)
    tracker = DistributedStatsTracker()
    with patch("areal.trainer.ppo.loss_functions.stats_tracker", tracker):
        loss = policy_loss(
            outputs={"logprobs": x, "entropy": x * 0},
            batch=data,
            mask=data["loss_mask"],
        )
    assert torch.isfinite(loss).all()
    assert "approx_kl/avg" in tracker.export()


def test_custom_support_is_prediction_aligned_and_not_shifted_again():
    from types import SimpleNamespace

    from areal.trainer.ppo.actor import PPOActor

    cfg = _config(mask_key="selected")
    cfg.kl_ctl = 0
    cfg.mask_no_eos_with_zero = False
    cfg.recompute_logprob = False
    actor = PPOActor(cfg, SimpleNamespace())
    data = _data()
    data["loss_mask"] = torch.tensor([[0, 1, 1, 1, 0]] * 4, dtype=torch.bool)
    data["rewards"] = torch.tensor([0.0, 1.0, 0.0, 1.0])
    data["selected"] = torch.zeros(4, 5, dtype=torch.bool)
    data["selected"][:, 1] = True
    prepared = actor._compute_advantages(data)
    # _compute_advantages updates its input in place.
    if prepared is not None:
        data = prepared
    actor.actor_loss.prepare(data, [2, 2])
    support = actor.actor_loss.terms[0].view(data)["loss_mask"]
    torch.testing.assert_close(support, data["selected"], rtol=0, atol=0)
