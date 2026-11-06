# SPDX-License-Identifier: Apache-2.0

"""Plugin contracts, configuration transport, and logical reward normalization."""

import copy
from dataclasses import asdict, replace

import pytest
import torch
from omegaconf import OmegaConf

from areal.api.cli_args import (
    AdvantageConfig,
    AdvantageFunctionConfig,
    NormConfig,
    PPOActorConfig,
)
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.trainer.ppo.actor import PPOActor
from areal.trainer.ppo.advantage import (
    AdvantageInputs,
    OutcomeResult,
    compute_advantages,
    validate_advantage_result,
)
from areal.trainer.ppo.advantage_algorithms import token_gae
from areal.utils.data import RolloutGroup, TrajBatchMeta


def scale_outcome(inputs, *, scale=2.0, references=None):
    return OutcomeResult(
        inputs.data["rewards"] * scale,
        None if references is None else inputs.data["rewards"].new_tensor(references),
    )


def offset_process(inputs, *, offset):
    return inputs.advantages + offset * inputs.loss_mask


def offset_compute(inputs, *, offset):
    result = compute_advantages(replace(inputs, config=PPOActorConfig(kl_ctl=0)))
    return replace(result, advantages=result.advantages + offset * result.loss_mask)


def invalid_estimator(inputs, *, field, kind):
    result = token_gae(inputs)
    tensor = getattr(result, field)
    value = {
        "shape": lambda: tensor[:, :-1],
        "dtype": lambda: tensor.long(),
        "type": lambda: tensor.tolist(),
        "device": lambda: tensor.to("meta"),
        "nan": lambda: torch.full_like(tensor, float("nan")),
    }[kind]()
    return replace(result, **{field: value})


def invalid_process(inputs):
    return inputs.advantages.long()


def invalid_outcome(inputs):
    return OutcomeResult(inputs.data["rewards"].long())


def length_lambda(context):
    return 1.0 / context["timestep_lengths"].clamp_min(1).float()


def _spec(name, **kwargs):
    return AdvantageFunctionConfig(implementation=f"{__name__}.{name}", kwargs=kwargs)


def _batch(rewards=(1.0, 2.0)):
    n = len(rewards)
    return {
        "input_ids": torch.ones(n, 5, dtype=torch.long),
        "attention_mask": torch.ones(n, 5, dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 1, 1, 1]]).expand(n, -1).clone(),
        "turn_ids": torch.tensor([[-1, 0, 0, 1, 1]]).expand(n, -1).clone(),
        "logprobs": torch.zeros(n, 5),
        "rewards": torch.tensor(rewards),
        "is_truncated": torch.zeros(n, dtype=torch.bool),
    }


def _actor(**kwargs):
    return PPOActor(PPOActorConfig(kl_ctl=0, **kwargs), engine=None)


def test_stage_plugins_compose_in_order_and_preserve_returns():
    """Outcome scaling precedes GAE; process shaping does not change returns."""
    actor = _actor(
        advantage=AdvantageConfig(
            outcome_reward_shaping=_spec("scale_outcome", scale=2.0),
            process_adv_shaping=_spec("offset_process", offset=3.0),
        )
    )
    result = actor._compute_advantages(_batch())
    baseline = _actor()._compute_advantages(_batch((2.0, 4.0)))
    torch.testing.assert_close(result["returns"], baseline["returns"], rtol=0, atol=0)
    torch.testing.assert_close(
        result["advantages"],
        baseline["advantages"] + 3 * baseline["loss_mask"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        result["rewards"], torch.tensor([2.0, 4.0]), rtol=0, atol=0
    )


def test_full_compute_plugin_receives_kwargs():
    """A full override returns the same validated result contract."""
    result = _actor(
        advantage=AdvantageConfig(compute=_spec("offset_compute", offset=3.0))
    )._compute_advantages(_batch())
    baseline = _actor()._compute_advantages(_batch())
    torch.testing.assert_close(
        result["advantages"],
        baseline["advantages"] + 3 * baseline["loss_mask"],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "stage", ["estimator", "outcome_reward_shaping", "process_adv_shaping"]
)
def test_full_compute_rejects_stage_override(stage):
    """Stage settings must not silently disappear under a full override."""
    with pytest.raises(ValueError, match="cannot be combined"):
        _actor(
            advantage=AdvantageConfig(
                compute=_spec("offset_compute", offset=1),
                **{stage: AdvantageFunctionConfig()},
            )
        )


@pytest.mark.parametrize("unit,override", [("token", "turn"), ("turn", "token")])
def test_explicit_estimator_uses_its_own_lambda_timestep_unit(unit, override):
    """Estimator overrides also change the lengths seen by lambda functions."""
    cfg = dict(gae_lambda=f"{__name__}.length_lambda")
    actual = _actor(
        gae_timestep_unit=unit,
        advantage=AdvantageConfig(
            estimator=AdvantageFunctionConfig(implementation=f"{override}_gae")
        ),
        **cfg,
    )._compute_advantages(_batch())
    expected = _actor(gae_timestep_unit=override, **cfg)._compute_advantages(_batch())
    for key in ("advantages", "returns", "tot_rewards"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_explicit_process_selection_and_kwargs_override_legacy_values():
    """Configured GVPO kwargs win; missing kwargs inherit call-local settings."""
    data = _batch((-1.0, 0.0))
    data["token_rewards"] = -torch.ones(2, 5)
    actor = _actor(
        token_rewards_as_adv=True,
        advantage=AdvantageConfig(
            process_adv_shaping=AdvantageFunctionConfig(
                implementation="gvpo", kwargs={"negative_scale": 0.5}
            )
        ),
    )
    result = actor._compute_advantages(
        copy.deepcopy(data),
        advantage_shaping_mode="additive",
        gvpo_negative_scale=10,
        gvpo_zero_penalty=0.7,
    )
    baseline = _actor(token_rewards_as_adv=True)._compute_advantages(
        data,
        advantage_shaping_mode="gvpo",
        gvpo_negative_scale=0.5,
        gvpo_zero_penalty=0.7,
    )
    torch.testing.assert_close(
        result["advantages"], baseline["advantages"], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "mode,kwargs", [("gvpo", {"negative_scale": -1}), ("process_weighted", {})]
)
def test_process_settings_validated_without_process_rewards(mode, kwargs):
    """Absent PRM tensors must not conceal an invalid selected composition."""
    with pytest.raises(ValueError):
        _actor(
            token_rewards_as_adv=True,
            mask_no_eos_with_zero=True,
            advantage=AdvantageConfig(
                process_adv_shaping=AdvantageFunctionConfig(
                    implementation=mode, kwargs=kwargs
                )
            ),
        )._compute_advantages(_batch())


@pytest.mark.parametrize("field", ["advantages", "returns", "total_rewards"])
@pytest.mark.parametrize("kind", ["shape", "dtype", "type", "device", "nan"])
def test_invalid_estimator_outputs_rejected(field, kind):
    """Plugin tensors must be aligned, floating-point and finite."""
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _actor(
            advantage=AdvantageConfig(
                estimator=_spec("invalid_estimator", field=field, kind=kind)
            )
        )._compute_advantages(_batch())


@pytest.mark.parametrize(
    "stage,fn",
    [
        ("outcome_reward_shaping", "invalid_outcome"),
        ("process_adv_shaping", "invalid_process"),
    ],
)
def test_invalid_stage_dtype_rejected(stage, fn):
    """Stages cannot return integer learning targets."""
    with pytest.raises(TypeError, match="floating-point"):
        _actor(advantage=AdvantageConfig(**{stage: _spec(fn)}))._compute_advantages(
            _batch()
        )


@pytest.mark.parametrize(
    "name", ["input_ids", "values", "advantages", "original_rewards"]
)
def test_full_compute_extras_cannot_overwrite_rollout_fields(name):
    """Custom result extras may update only documented derived fields."""
    data = _batch()
    result = compute_advantages(AdvantageInputs(data, PPOActorConfig(kl_ctl=0)))
    with pytest.raises(ValueError, match="Unsupported advantage extra"):
        validate_advantage_result(
            replace(result, extras={name: torch.zeros(2, 5)}), data
        )


def test_config_survives_omegaconf_and_rpc_roundtrip():
    """Only paths and serializable kwargs cross the worker boundary."""
    config = PPOActorConfig(
        advantage=AdvantageConfig(
            process_adv_shaping=_spec("offset_process", offset=2.0)
        )
    )
    restored = deserialize_value(serialize_value(config))
    assert asdict(restored.advantage) == asdict(config.advantage)
    plain = OmegaConf.to_container(OmegaConf.structured(config.advantage), resolve=True)
    assert asdict(AdvantageConfig(**plain)) == asdict(config.advantage)
    result = PPOActor(restored, engine=None)._compute_advantages(_batch())
    assert torch.isfinite(result["advantages"]).all()


@pytest.mark.parametrize(
    "spec",
    [
        dict(implementation=""),
        dict(implementation=1),
        dict(kwargs=[]),
        dict(kwargs={"bad": object()}),
    ],
)
def test_invalid_function_config_rejected(spec):
    """Function references must contain a nonempty path and serializable kwargs."""
    with pytest.raises((TypeError, ValueError)):
        AdvantageFunctionConfig(**spec)


def test_functions_resolved_once_per_actor(monkeypatch):
    """Batch computation does not repeatedly import configured plugins."""
    actor = _actor(
        advantage=AdvantageConfig(process_adv_shaping=_spec("offset_process", offset=1))
    )

    def fail_import(*args):
        raise AssertionError("unexpected repeated import")

    monkeypatch.setattr("areal.trainer.ppo.advantage.import_from_string", fail_import)
    actor._compute_advantages(_batch())
    actor._compute_advantages(_batch())


@pytest.mark.parametrize("explicit", [False, True])
def test_shaped_logical_rewards_use_shaped_reference_units(explicit):
    """Multiple tensor rows must not reweight logical rollout statistics."""
    meta = TrajBatchMeta(
        1, [3], [5], [RolloutGroup((2, 1), (1.0, 3.0) if explicit else ())]
    )
    references = [2.0, 6.0] if explicit else None
    actor = _actor(
        reward_norm=NormConfig(
            group_size=2,
            mean_level="group",
            std_level="group",
            std_unbiased=False,
            eps=0,
        ),
        advantage=AdvantageConfig(
            outcome_reward_shaping=_spec("scale_outcome", references=references)
        ),
    )
    result = actor._compute_advantages(_batch((1.0, 1.0, 3.0)), meta)
    torch.testing.assert_close(
        result["advantages"][:, 0], torch.tensor([-1.0, -1.0, 1.0]), rtol=0, atol=0
    )


def test_custom_shaping_rejects_untransformed_explicit_references():
    """Raw references cannot silently normalize rewards in different units."""
    meta = TrajBatchMeta(1, [3], [5], [RolloutGroup((2, 1), (1.0, 3.0))])
    actor = _actor(
        reward_norm=NormConfig(group_size=2, mean_level="group"),
        advantage=AdvantageConfig(outcome_reward_shaping=_spec("scale_outcome")),
    )
    with pytest.raises(RuntimeError, match="requires normalization_rewards"):
        actor._compute_advantages(_batch((1.0, 1.0, 3.0)), meta)


def test_shaping_without_references_still_rejects_unequal_logical_rows():
    """The existing logical reward consistency check must survive extraction."""
    meta = TrajBatchMeta(1, [3], [5], [RolloutGroup((2, 1))])
    actor = _actor(
        reward_norm=NormConfig(group_size=2, mean_level="group"),
        advantage=AdvantageConfig(outcome_reward_shaping=_spec("scale_outcome")),
    )
    with pytest.raises(RuntimeError, match="Split rollout row rewards differ"):
        actor._compute_advantages(_batch((1.0, 2.0, 3.0)), meta)


@pytest.mark.parametrize("decoupled", [False, True])
def test_optional_logprob_none_remains_supported(decoupled):
    """Missing optional logprobs must not become invalid custom result extras."""
    data = _batch()
    data.update(ref_logp=None, prox_logp=None)
    result = _actor(use_decoupled_loss=decoupled)._compute_advantages(data)
    assert torch.isfinite(result["advantages"]).all()


@pytest.mark.parametrize("name", ["loss_mask", "is_truncated", "input_token_loss_mask"])
def test_plugin_masks_must_be_binary(name):
    """Fractional masks would disagree with downstream boolean support tests."""
    data = _batch()
    result = compute_advantages(AdvantageInputs(data, PPOActorConfig(kl_ctl=0)))
    value = torch.full_like(
        data["rewards"] if name == "is_truncated" else data["loss_mask"],
        0.5,
        dtype=torch.float32,
    )
    result = (
        replace(result, loss_mask=value)
        if name == "loss_mask"
        else replace(result, extras={name: value})
    )
    with pytest.raises(RuntimeError, match="must be binary"):
        validate_advantage_result(result, data)


@pytest.mark.parametrize("references", [[2.0], [float("nan"), 6.0]])
def test_invalid_shaped_reference_rejected(references):
    """Reference vectors must match logical rollouts and contain finite scores."""
    meta = TrajBatchMeta(1, [3], [5], [RolloutGroup((2, 1), (1.0, 3.0))])
    actor = _actor(
        reward_norm=NormConfig(group_size=2, mean_level="group"),
        advantage=AdvantageConfig(
            outcome_reward_shaping=_spec("scale_outcome", references=references)
        ),
    )
    with pytest.raises((ValueError, RuntimeError)):
        actor._compute_advantages(_batch((1.0, 1.0, 3.0)), meta)


def test_shaped_references_allow_distinct_rows_in_one_logical_rollout():
    """Explicit shaped references retain per-row rewards without reweighting them."""
    meta = TrajBatchMeta(1, [3], [5], [RolloutGroup((2, 1), (1.0, 3.0))])
    actor = _actor(
        reward_norm=NormConfig(
            group_size=2,
            mean_level="group",
            std_level="group",
            std_unbiased=False,
            eps=0,
        ),
        advantage=AdvantageConfig(
            outcome_reward_shaping=_spec("scale_outcome", references=[2.0, 6.0])
        ),
    )
    result = actor._compute_advantages(_batch((1.0, 2.0, 3.0)), meta)
    torch.testing.assert_close(
        result["advantages"][:, 0], torch.tensor([-1.0, 0.0, 1.0]), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "stage,key",
    [
        ("estimator", "unknown"),
        ("process_adv_shaping", "unknown"),
        ("outcome_reward_shaping", "unknown"),
        ("compute", "unknown"),
    ],
)
def test_unknown_builtin_rejected_at_initialization(stage, key):
    """Typos fail before rollout or training starts."""
    with pytest.raises(ValueError, match="Unknown"):
        _actor(
            advantage=AdvantageConfig(
                **{stage: AdvantageFunctionConfig(implementation=key)}
            )
        )


def test_default_pipeline_preserves_integer_outcome_rewards():
    """Legacy custom workflows may supply integer task rewards."""
    data = _batch()
    data["rewards"] = data["rewards"].long()
    actual = _actor()._compute_advantages(data)
    expected = _actor()._compute_advantages(_batch())
    assert actual["rewards"].dtype == torch.int64
    torch.testing.assert_close(
        actual["advantages"], expected["advantages"], rtol=0, atol=0
    )


def test_shaped_references_require_complete_group_metadata():
    """Manually incomplete metadata cannot change distributed normalization paths."""
    meta = TrajBatchMeta(1, [2], [5])
    actor = _actor(
        advantage=AdvantageConfig(
            outcome_reward_shaping=_spec("scale_outcome", references=[2.0, 4.0])
        )
    )
    with pytest.raises(ValueError, match="complete trajectory metadata"):
        actor._compute_advantages(_batch(), meta)
