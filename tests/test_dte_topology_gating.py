"""Tests for the separation-only DTE configuration boundary."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_HELPER_PATH = Path(__file__).parents[1] / "areal/utils/dte.py"


def _load_helpers():
    spec = importlib.util.spec_from_file_location("areal_utils_dte", _HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _config(
    *,
    enabled=False,
    delta_method="adamw",
    anchor_interval=20,
    topology="separation",
    actor_backend="megatron:d1",
    rollout_backend="sglang:d1",
    actor_version="v2",
    rollout_version="v2",
    weight_update_mode="awex",
    use_lora=False,
):
    actor_spec = SimpleNamespace(env_vars={})
    rollout_spec = SimpleNamespace(env_vars={})
    return SimpleNamespace(
        actor=SimpleNamespace(
            backend=actor_backend,
            _version=actor_version,
            weight_update_mode=weight_update_mode,
            enable_delta_weight_update=enabled,
            weight_update_delta_method=delta_method,
            weight_update_anchor_interval=anchor_interval,
            use_lora=use_lora,
            scheduling_spec=(actor_spec,),
        ),
        rollout=SimpleNamespace(
            backend=rollout_backend,
            _version=rollout_version,
            scheduling_strategy=SimpleNamespace(type=topology),
            scheduling_spec=(rollout_spec,),
        ),
    )


def test_disabled_delta_weight_update_does_not_change_environment():
    config = _config()

    exported = _load_helpers().apply_dte_config_envvars(config, environ={})

    assert exported == {}
    assert config.actor.scheduling_spec[0].env_vars == {}
    assert config.rollout.scheduling_spec[0].env_vars == {}


def test_delta_weight_transfer_exports_only_separation_adamw_switches():
    config = _config(enabled=True)

    exported = _load_helpers().apply_dte_config_envvars(config, environ={})

    assert exported == {
        "DTE_SEPARATION_WEIGHT_UPDATE": "1",
        "DTE_DELTA_TRANSFER": "1",
        "DTE_DELTA_ANCHOR_INTERVAL": "20",
        "DTE_STREAMING_RECONSTRUCT": "1",
    }
    assert config.actor.scheduling_spec[0].env_vars == exported
    assert config.rollout.scheduling_spec[0].env_vars == exported


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"topology": "colocation"}, "only with.*separation"),
        (
            {"delta_method": "snapshot"},
            "weight_update_delta_method must be 'adamw'",
        ),
        (
            {"anchor_interval": -1},
            "weight_update_anchor_interval must be non-negative",
        ),
        ({"actor_backend": "fsdp:d1"}, "requires actor.backend='megatron"),
        ({"rollout_backend": "vllm:d1"}, "requires rollout.backend='sglang"),
        ({"actor_version": "v1"}, "requires actor._version='v2'"),
        ({"rollout_version": "v1"}, "requires rollout._version='v2'"),
        ({"weight_update_mode": "xccl"}, "requires actor.weight_update_mode='awex'"),
        ({"use_lora": True}, "does not support actor.use_lora=True"),
    ],
)
def test_dte_rejects_out_of_scope_modes(kwargs, match):
    kwargs = dict(kwargs)
    topology = kwargs.pop("topology", "separation")
    config = _config(enabled=True, topology=topology, **kwargs)
    environ = {}

    with pytest.raises(ValueError, match=match):
        _load_helpers().apply_dte_config_envvars(config, environ=environ)

    assert environ == {}
    assert config.actor.scheduling_spec[0].env_vars == {}
    assert config.rollout.scheduling_spec[0].env_vars == {}


def test_dte_requires_one_ppo_minibatch_per_weight_update():
    pytest.importorskip("httpx")
    from areal.api.cli_args import PPOActorConfig

    assert (
        PPOActorConfig(
            enable_delta_weight_update=True, ppo_n_minibatches=1
        ).ppo_n_minibatches
        == 1
    )

    with pytest.raises(ValueError, match="requires ppo_n_minibatches=1"):
        PPOActorConfig(enable_delta_weight_update=True, ppo_n_minibatches=2)

    assert PPOActorConfig(enable_delta_weight_update=False).ppo_n_minibatches == 4


def test_delta_weight_transfer_fields_belong_to_train_engine_config():
    pytest.importorskip("httpx")
    from dataclasses import fields

    from areal.api.cli_args import TrainEngineConfig

    field_names = {config_field.name for config_field in fields(TrainEngineConfig)}

    assert "dte" not in field_names
    assert {
        "enable_delta_weight_update",
        "weight_update_delta_method",
        "weight_update_anchor_interval",
    } <= field_names
    assert TrainEngineConfig().enable_delta_weight_update is False
