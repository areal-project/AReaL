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


def _config(*, enabled=False, topology="separation", **overrides):
    dte = {
        "enabled": enabled,
        "transfer": "delta",
        "delta_method": "adamw",
        "anchor_interval": 20,
        **overrides,
    }
    actor_spec = SimpleNamespace(env_vars={})
    rollout_spec = SimpleNamespace(env_vars={})
    return SimpleNamespace(
        actor=SimpleNamespace(
            dte=SimpleNamespace(**dte), scheduling_spec=(actor_spec,)
        ),
        rollout=SimpleNamespace(
            scheduling_strategy=SimpleNamespace(type=topology),
            scheduling_spec=(rollout_spec,),
        ),
    )


def test_dte_disabled_does_not_change_environment():
    config = _config()

    exported = _load_helpers().apply_dte_config_envvars(config, environ={})

    assert exported == {}
    assert config.actor.scheduling_spec[0].env_vars == {}
    assert config.rollout.scheduling_spec[0].env_vars == {}


def test_dte_enabled_exports_only_separation_adamw_switches():
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
        ({"transfer": "full"}, "transfer must be 'delta'"),
        ({"delta_method": "snapshot"}, "delta_method must be 'adamw'"),
        ({"anchor_interval": -1}, "anchor_interval must be non-negative"),
    ],
)
def test_dte_rejects_out_of_scope_modes(kwargs, match):
    kwargs = dict(kwargs)
    topology = kwargs.pop("topology", "separation")
    config = _config(enabled=True, topology=topology, **kwargs)

    with pytest.raises(ValueError, match=match):
        _load_helpers().apply_dte_config_envvars(config, environ={})


def test_dte_requires_one_ppo_minibatch_per_weight_update():
    pytest.importorskip("httpx")
    from areal.api.cli_args import DTEConfig, PPOActorConfig

    enabled = DTEConfig(enabled=True)
    assert PPOActorConfig(dte=enabled, ppo_n_minibatches=1).ppo_n_minibatches == 1

    with pytest.raises(ValueError, match="requires ppo_n_minibatches=1"):
        PPOActorConfig(dte=enabled, ppo_n_minibatches=2)

    disabled = DTEConfig(enabled=False)
    assert PPOActorConfig(dte=disabled).ppo_n_minibatches == 4
