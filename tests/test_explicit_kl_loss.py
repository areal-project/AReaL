# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
import torch

from areal.api.cli_args import (
    DatasetSourceConfig,
    InferenceEngineConfig,
    MOPDConfig,
    MOPDLossConfig,
    MOPDTeacherEngineConfig,
    MOPDTeacherSpec,
    NormConfig,
    PPOActorConfig,
    PPOConfig,
    SchedulingSpec,
    SchedulingStrategy,
    TrainDatasetConfig,
)
from areal.trainer.mopd.execution import MOPDExecutionPlan
from areal.trainer.ppo.actor import PPOActor, grpo_loss_fn
from areal.trainer.ppo.loss_reduction import (
    PG_TOKEN_WEIGHTS,
    prepare_policy_gradient_batch,
)
from areal.trainer.rl_trainer import _requires_reference_model
from areal.utils.data import TrajBatchMeta
from areal.utils.stats_tracker import DistributedStatsTracker

MEGATRON_BACKEND = "megatron:(attn:d1p1t2c2|ffn:d1p1e4)"
STUDENT = [[-1.0, -2.0, -0.5, -3.0]]
REF = [[-0.2, -3.5, -0.5, -1.5]]
TEACHER = [[-0.4, -1.0, -0.8, -0.9]]


def _tensor(values, *, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype)


def _rl_input_data(student, ref=REF, *, dtype=torch.float64, **overrides):
    detached = _tensor(student, dtype=dtype)
    data = {
        "logprobs": detached.clone(),
        "prox_logp": detached.clone(),
        "advantages": torch.zeros_like(detached),
        "loss_mask": torch.ones_like(detached, dtype=torch.bool),
    }
    if ref is not None:
        data["ref_logp"] = _tensor(ref, dtype=dtype)
    data.update(overrides)
    return data


def _loss_and_grad(student, input_data, *, dtype=torch.float64, **kwargs):
    logprobs = _tensor(student, dtype=dtype).requires_grad_()
    with patch("areal.trainer.ppo.actor.stats_tracker", MagicMock()):
        loss = grpo_loss_fn(
            logprobs=logprobs,
            entropy=torch.zeros_like(logprobs),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            **kwargs,
        )
    (grad,) = torch.autograd.grad(loss, logprobs)
    return loss.detach(), grad


def _mopd_config(**overrides) -> MOPDConfig:
    colocate = SchedulingStrategy(type="colocation", target="actor", fork=True)
    kwargs = {
        "teachers": {"t1": MOPDTeacherSpec(path="checkpoints/t1")},
        "teacher_groups": {"g1": {"t1": 1.0}},
        "teacher_engine": MOPDTeacherEngineConfig(
            backend=MEGATRON_BACKEND,
            optimizer=None,
            disable_dropout=True,
            scheduling_strategy=colocate,
        ),
    }
    kwargs.update(overrides)
    return MOPDConfig(**kwargs)


def _ppo_config(*, mopd: MOPDConfig | None = None, **overrides) -> PPOConfig:
    kwargs = {
        "experiment_name": "kl-loss-test",
        "trial_name": "trial0",
        **overrides,
    }
    if mopd is not None:
        colocate = SchedulingStrategy(type="colocation", target="actor", fork=True)
        kwargs.setdefault(
            "actor",
            PPOActorConfig(
                backend=MEGATRON_BACKEND,
                weight_update_mode="awex",
                scheduling_spec=(SchedulingSpec(port_count=3),),
            ),
        )
        kwargs.setdefault(
            "rollout",
            InferenceEngineConfig(
                backend="sglang:d4",
                scheduling_strategy=colocate,
            ),
        )
        kwargs["mopd"] = mopd
        kwargs.setdefault(
            "train_dataset",
            TrainDatasetConfig(
                sources=[
                    DatasetSourceConfig(
                        path="datasets/train",
                        type="rl",
                        teacher_group="g1",
                    )
                ]
            ),
        )
    return PPOConfig(**kwargs)


@pytest.mark.parametrize("kl_loss_type", ["low_var_kl", "k3"])
def test_explicit_kl_loss_k3_value_and_gradient(kl_loss_type):
    """low_var_kl/k3 computes Schulman non-negative k3 loss and exact gradient."""
    coef = 0.25
    data = _rl_input_data(STUDENT, REF)
    loss, grad = _loss_and_grad(
        STUDENT,
        data,
        use_kl_loss=True,
        kl_loss_coef=coef,
        kl_loss_type=kl_loss_type,
    )

    student = _tensor(STUDENT)
    ref = _tensor(REF)
    log_ratio = ref - student
    expected_token_kl = torch.exp(log_ratio) - log_ratio - 1.0
    expected_loss = coef * expected_token_kl.mean()
    expected_grad = (coef / student.numel()) * (1.0 - torch.exp(log_ratio))

    assert (expected_token_kl >= 0).all()
    torch.testing.assert_close(loss, expected_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad, expected_grad, rtol=1e-12, atol=1e-12)
    assert torch.all(torch.sign(grad) == torch.sign(student - ref))


@pytest.mark.parametrize("kl_loss_type", ["k2", "mse"])
def test_explicit_kl_loss_k2_value_and_gradient(kl_loss_type):
    """k2/mse computes 0.5*(logp - ref)^2 with reverse-KL analytical gradient."""
    coef = 0.3
    data = _rl_input_data(STUDENT, REF)
    loss, grad = _loss_and_grad(
        STUDENT,
        data,
        use_kl_loss=True,
        kl_loss_coef=coef,
        kl_loss_type=kl_loss_type,
    )

    student = _tensor(STUDENT)
    ref = _tensor(REF)
    expected_loss = coef * (0.5 * (student - ref).square()).mean()
    expected_grad = (coef / student.numel()) * (student - ref)

    torch.testing.assert_close(loss, expected_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad, expected_grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kl_loss_type", ["low_var_kl", "k2"])
def test_explicit_kl_loss_zero_when_policy_matches_reference(kl_loss_type):
    data = _rl_input_data(STUDENT, STUDENT)
    loss, grad = _loss_and_grad(
        STUDENT,
        data,
        use_kl_loss=True,
        kl_loss_coef=0.5,
        kl_loss_type=kl_loss_type,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0, dtype=torch.float64))
    torch.testing.assert_close(grad, torch.zeros_like(grad))


@pytest.mark.parametrize("kl_loss_type", ["low_var_kl", "k2"])
def test_explicit_kl_loss_sanitizes_masked_nonfinite_entries(kl_loss_type):
    """Masked NaN/Inf positions cannot poison explicit KL loss or gradients."""
    student = [[-1.0, float("nan"), -0.5, float("-inf")]]
    ref = [[-0.2, float("inf"), -0.7, float("nan")]]
    mask = torch.tensor([[True, False, True, False]])
    data = _rl_input_data(student, ref, loss_mask=mask)

    loss, grad = _loss_and_grad(
        student,
        data,
        use_kl_loss=True,
        kl_loss_coef=0.4,
        kl_loss_type=kl_loss_type,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(grad).all()
    torch.testing.assert_close(grad[~mask], torch.zeros(2, dtype=torch.float64))

    clean_data = _rl_input_data(
        [[-1.0, 0.0, -0.5, 0.0]], [[-0.2, 0.0, -0.7, 0.0]], loss_mask=mask
    )
    clean_loss, clean_grad = _loss_and_grad(
        [[-1.0, 0.0, -0.5, 0.0]],
        clean_data,
        use_kl_loss=True,
        kl_loss_coef=0.4,
        kl_loss_type=kl_loss_type,
    )
    torch.testing.assert_close(loss, clean_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad[mask], clean_grad[mask], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kl_loss_type", ["low_var_kl", "k2"])
def test_explicit_kl_loss_clamps_extreme_log_ratios(kl_loss_type):
    student = [[-100.0, 50.0]]
    ref = [[100.0, -50.0]]
    data = _rl_input_data(student, ref)

    loss, grad = _loss_and_grad(
        student,
        data,
        use_kl_loss=True,
        kl_loss_coef=0.1,
        kl_loss_type=kl_loss_type,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(grad).all()


@pytest.mark.parametrize("mode", ["token_mean", "seq_mean", "prompt_mean", "constant"])
@pytest.mark.parametrize("kl_loss_type", ["low_var_kl", "k2"])
def test_explicit_kl_loss_respects_loss_aggregation_and_packed_parity(
    mode, kl_loss_type
):
    """Explicit KL loss follows pg_reduction across modes and padded/packed layouts."""
    student = [[-1.0, -2.0, -0.5, 0.0], [-0.4, -1.5, -2.5, -0.2], [-1.2, 0.0, 0.0, 0.0]]
    ref = [[-0.2, -2.5, -0.4, 0.0], [-0.8, -1.0, -1.5, -0.6], [-0.2, 0.0, 0.0, 0.0]]
    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, True],
            [True, False, False, False],
        ]
    )
    data = _rl_input_data(student, ref, dtype=torch.float32, loss_mask=mask)
    prepared = prepare_policy_gradient_batch(
        data,
        mode=mode,
        group_sizes=[2, 1],
        divisor=4.0 if mode == "constant" else None,
    )
    loss_step = prepared.for_steps([data])[0]
    padded_loss, padded_grad = _loss_and_grad(
        student,
        data,
        dtype=torch.float32,
        use_kl_loss=True,
        kl_loss_coef=0.2,
        kl_loss_type=kl_loss_type,
        loss_step=loss_step,
    )

    student_t = _tensor(student, dtype=torch.float32)
    ref_t = _tensor(ref, dtype=torch.float32)
    if kl_loss_type == "low_var_kl":
        ratio = torch.clamp(ref_t - student_t, min=-20.0, max=20.0)
        token_kl = torch.clamp(torch.exp(ratio) - ratio - 1.0, min=-10.0, max=10.0)
    else:
        ratio = torch.clamp(student_t - ref_t, min=-20.0, max=20.0)
        token_kl = 0.5 * ratio.square()
    expected = 0.2 * loss_step.bind(data).aggregate(token_kl, mask)
    torch.testing.assert_close(padded_loss, expected, rtol=1e-5, atol=1e-6)

    packed_data = {
        "logprobs": data["logprobs"][mask],
        "prox_logp": data["prox_logp"][mask],
        "advantages": data["advantages"][mask],
        "ref_logp": data["ref_logp"][mask],
        "loss_mask": mask[mask],
        "cu_seqlens": torch.tensor([0, 3, 7, 8], dtype=torch.int32),
    }
    if PG_TOKEN_WEIGHTS in data:
        packed_data[PG_TOKEN_WEIGHTS] = data[PG_TOKEN_WEIGHTS][mask]
    packed_loss, packed_grad = _loss_and_grad(
        _tensor(student, dtype=torch.float32)[mask],
        packed_data,
        dtype=torch.float32,
        use_kl_loss=True,
        kl_loss_coef=0.2,
        kl_loss_type=kl_loss_type,
        loss_step=loss_step,
    )
    torch.testing.assert_close(packed_loss, padded_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(packed_grad, padded_grad[mask], rtol=1e-5, atol=1e-6)


def test_explicit_kl_loss_keeps_original_denominator_with_version_mask():
    """policy_version_mask filters numerator tokens without shrinking denominator."""
    student = [[-1.0, -2.0, -0.5, -3.0]]
    ref = [[-0.2, -3.5, -0.5, -1.5]]
    version_mask = torch.tensor([[True, False, True, False]])
    data = _rl_input_data(student, ref, policy_version_mask=version_mask)

    loss, grad = _loss_and_grad(
        student,
        data,
        use_kl_loss=True,
        kl_loss_coef=0.4,
        kl_loss_type="k2",
    )

    student_t, ref_t = _tensor(student), _tensor(ref)
    expected_token_kl = torch.where(
        version_mask, 0.5 * (student_t - ref_t).square(), 0.0
    )
    expected_loss = 0.4 * expected_token_kl.sum() / 4
    expected_grad = torch.where(version_mask, 0.4 * (student_t - ref_t) / 4, 0.0)
    torch.testing.assert_close(loss, expected_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad, expected_grad, rtol=1e-12, atol=1e-12)


def test_explicit_kl_loss_composes_with_joint_opd():
    """Explicit reference KL adds cleanly to Joint OPD (teacher_logp + RL)."""
    student_t = _tensor(STUDENT)
    ref_t = _tensor(REF)
    teacher_t = _tensor(TEACHER)
    data = _rl_input_data(
        STUDENT,
        REF,
        teacher_logp=teacher_t.clone(),
        rl_loss_weight=1.0,
        distill_loss_weight=0.5,
    )

    base_loss, base_grad = _loss_and_grad(STUDENT, data, use_kl_loss=False)
    kl_loss, kl_grad = _loss_and_grad(
        STUDENT,
        data,
        use_kl_loss=True,
        kl_loss_coef=0.2,
        kl_loss_type="low_var_kl",
    )

    log_ratio = ref_t - student_t
    expected_kl = 0.2 * (torch.exp(log_ratio) - log_ratio - 1.0).mean()
    expected_kl_grad = (0.2 / student_t.numel()) * (1.0 - torch.exp(log_ratio))
    torch.testing.assert_close(kl_loss, base_loss + expected_kl, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        kl_grad, base_grad + expected_kl_grad, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("rl_coefficient", [1.0, 0.0])
def test_explicit_kl_loss_composes_with_joint_and_pure_mopd(rl_coefficient):
    """Both Joint MOPD and Pure MOPD early-return paths add explicit reference KL loss."""
    mopd_cfg = MOPDLossConfig(
        rl_coefficient=rl_coefficient,
        distillation_coefficient=0.5,
    )
    student_t = _tensor(STUDENT)
    ref_t = _tensor(REF)
    data = _rl_input_data(
        STUDENT,
        REF,
        mopd_teacher_logp_sum=_tensor(TEACHER),
        mopd_teacher_weight_sum=torch.ones_like(student_t),
        mopd_behavior_logprobs=student_t.clone(),
    )

    base_loss, base_grad = _loss_and_grad(
        STUDENT,
        data,
        use_kl_loss=False,
        mopd_loss_config=mopd_cfg,
    )
    total_loss, total_grad = _loss_and_grad(
        STUDENT,
        data,
        use_kl_loss=True,
        kl_loss_coef=0.1,
        kl_loss_type="low_var_kl",
        mopd_loss_config=mopd_cfg,
    )

    log_ratio = ref_t - student_t
    expected_kl = 0.1 * (torch.exp(log_ratio) - log_ratio - 1.0).mean()
    expected_kl_grad = (0.1 / student_t.numel()) * (1.0 - torch.exp(log_ratio))
    torch.testing.assert_close(
        total_loss, base_loss + expected_kl, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        total_grad, base_grad + expected_kl_grad, rtol=1e-12, atol=1e-12
    )


def test_explicit_kl_loss_validates_inputs():
    with pytest.raises(ValueError, match="requires 'ref_logp'"):
        _loss_and_grad(STUDENT, _rl_input_data(STUDENT, None), use_kl_loss=True)

    with pytest.raises(ValueError, match="ref_logp shape"):
        _loss_and_grad(
            STUDENT,
            _rl_input_data(STUDENT, [[-0.1, -0.2]]),
            use_kl_loss=True,
        )

    with pytest.raises(ValueError, match="Unsupported kl_loss_type"):
        _loss_and_grad(
            STUDENT,
            _rl_input_data(STUDENT, REF),
            use_kl_loss=True,
            kl_loss_type="k1",
        )


def test_ppo_actor_and_ppo_config_explicit_kl_validation():
    default_actor = PPOActorConfig()
    assert default_actor.use_kl_loss is False
    assert default_actor.kl_loss_coef == 0.0
    assert default_actor.kl_loss_type == "low_var_kl"

    with pytest.raises(ValueError, match="cannot be combined with actor.kl_ctl"):
        PPOActorConfig(use_kl_loss=True, kl_loss_coef=0.001)

    with pytest.raises(ValueError, match="finite non-negative"):
        PPOActorConfig(kl_ctl=0.0, use_kl_loss=True, kl_loss_coef=-0.01)

    with pytest.raises(ValueError, match="finite non-negative"):
        PPOActorConfig(kl_ctl=0.0, use_kl_loss=True, kl_loss_coef=float("nan"))

    with pytest.raises(ValueError, match="finite non-negative"):
        PPOActorConfig(kl_ctl=0.0, use_kl_loss=True, kl_loss_coef=float("inf"))

    with pytest.raises(ValueError, match="finite non-negative"):
        PPOActorConfig(kl_ctl=0.0, use_kl_loss=True, kl_loss_coef=True)

    with pytest.raises(ValueError, match="kl_loss_type"):
        PPOActorConfig(kl_loss_type="k1")

    with patch("areal.api.cli_args.logger.warning") as mock_warn:
        cfg = PPOActorConfig(kl_ctl=0.0, use_kl_loss=True, kl_loss_coef=0.0)
        assert cfg.use_kl_loss is True
        mock_warn.assert_called_once()
        assert "kl_loss_coef=0.0" in mock_warn.call_args.args[0]

    with pytest.raises(ValueError, match="requires a reference model"):
        _ppo_config(
            actor=PPOActorConfig(kl_ctl=0.0, use_kl_loss=True, kl_loss_coef=0.001)
        )

    valid_ppo = _ppo_config(
        actor=PPOActorConfig(kl_ctl=0.0, use_kl_loss=True, kl_loss_coef=0.001),
        ref=PPOActorConfig(),
    )
    assert _requires_reference_model(valid_ppo) is True


def test_requires_reference_model_and_mopd_plan():
    no_ref = _ppo_config(actor=PPOActorConfig(kl_ctl=0.1, use_kl_loss=False), ref=None)
    assert _requires_reference_model(no_ref) is False

    kl_ctl_only = _ppo_config(
        actor=PPOActorConfig(kl_ctl=0.001, use_kl_loss=False), ref=PPOActorConfig()
    )
    assert _requires_reference_model(kl_ctl_only) is True

    kl_loss_only = _ppo_config(
        actor=PPOActorConfig(kl_ctl=0.0, use_kl_loss=True, kl_loss_coef=0.001),
        ref=PPOActorConfig(),
    )
    assert _requires_reference_model(kl_loss_only) is True

    pure_mopd_default = _ppo_config(
        mopd=_mopd_config(
            loss=MOPDLossConfig(rl_coefficient=0.0, distillation_coefficient=1.0)
        ),
        ref=PPOActorConfig(backend=MEGATRON_BACKEND),
    )
    plan_default = MOPDExecutionPlan.from_config(pure_mopd_default)
    assert plan_default is not None
    assert plan_default.requires_ref is False

    pure_mopd_with_kl_loss = _ppo_config(
        mopd=_mopd_config(
            loss=MOPDLossConfig(rl_coefficient=0.0, distillation_coefficient=1.0)
        ),
        actor=PPOActorConfig(
            backend=MEGATRON_BACKEND,
            weight_update_mode="awex",
            scheduling_spec=(SchedulingSpec(port_count=3),),
            kl_ctl=0.0,
            use_kl_loss=True,
            kl_loss_coef=0.01,
        ),
        ref=PPOActorConfig(backend=MEGATRON_BACKEND),
    )
    plan_kl_loss = MOPDExecutionPlan.from_config(pure_mopd_with_kl_loss)
    assert plan_kl_loss is not None
    assert plan_kl_loss.requires_ref is True


def test_advantage_pipeline_with_kl_ctl_zero_preserves_ref_logp():
    """Disabling kl_ctl leaves outcome advantages length-invariant and ref_logp unmutated."""
    seq_len = 9
    ref_logp = torch.full((2, seq_len), -0.4)
    ref_copy = ref_logp.clone()
    data = {
        "input_ids": torch.ones(2, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(2, seq_len, dtype=torch.bool),
        "loss_mask": torch.tensor(
            [[0, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1]]
        ),
        "logprobs": torch.full((2, seq_len), -0.8),
        "ref_logp": ref_logp,
        "rewards": torch.tensor([1.0, 0.0]),
        "is_truncated": torch.zeros(2, dtype=torch.bool),
    }
    meta = TrajBatchMeta(
        rollout_groups=[[0, 1]],
        traj_group_sizes=[2],
        logical_group_sizes=[2],
    )
    cfg = PPOActorConfig(
        kl_ctl=0.0,
        use_kl_loss=True,
        kl_loss_coef=0.001,
        reward_norm=NormConfig(mean_level="group", std_level="group", group_size=2),
        adv_norm=None,
    )
    out = PPOActor(cfg, engine=None)._compute_advantages(data, meta=meta)

    torch.testing.assert_close(ref_logp, ref_copy, rtol=0, atol=0)
    torch.testing.assert_close(out["ref_logp"], ref_copy, rtol=0, atol=0)
    assert out["kl_rewards"].dtype == torch.float32
    torch.testing.assert_close(
        out["kl_rewards"], torch.zeros_like(out["kl_rewards"]), rtol=0, atol=0
    )
    active = out["loss_mask"].bool()
    assert abs(out["advantages"][active].mean().item()) < 1e-6


@pytest.mark.parametrize("pure_mopd", [False, True])
def test_grpo_loss_fn_logs_explicit_kl_stats(pure_mopd):
    """stats_tracker records explicit_kl_loss and kl_loss_coef on both RL and Pure MOPD paths."""
    tracker = DistributedStatsTracker()
    student_t = _tensor(STUDENT, dtype=torch.float32)
    ref_t = _tensor(REF, dtype=torch.float32)
    data = _rl_input_data(STUDENT, REF, dtype=torch.float32)
    kwargs = {}
    if pure_mopd:
        data.update(
            mopd_teacher_logp_sum=_tensor(TEACHER, dtype=torch.float32),
            mopd_teacher_weight_sum=torch.ones_like(student_t),
            mopd_behavior_logprobs=student_t.clone(),
        )
        kwargs["mopd_loss_config"] = MOPDLossConfig(
            rl_coefficient=0.0, distillation_coefficient=1.0
        )

    with patch("areal.trainer.ppo.actor.stats_tracker", tracker):
        grpo_loss_fn(
            logprobs=student_t.clone().requires_grad_(),
            entropy=torch.zeros_like(student_t),
            input_data=data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            use_kl_loss=True,
            kl_loss_coef=0.02,
            kl_loss_type="low_var_kl",
            **kwargs,
        )

    stats = tracker.export(reset=True)
    log_ratio = ref_t - student_t
    expected_avg = (torch.exp(log_ratio) - log_ratio - 1.0).mean().item()
    assert stats["explicit_kl_loss/avg"] == pytest.approx(expected_avg, rel=1e-5)
    assert stats["kl_loss_coef"] == pytest.approx(0.02, rel=1e-6)
