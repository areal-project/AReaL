from __future__ import annotations

from collections.abc import Callable

import pytest

from areal.api.cli_args import (
    ClusterSpecConfig,
    PPOActorConfig,
    PPOConfig,
    PPOCriticConfig,
)
from areal.utils.awex_runtime import prepare_awex_runtime


def _create_ppo_config(weight_update_mode: str = "disk") -> PPOConfig:
    config = PPOConfig()
    config.experiment_name = "test_experiment"
    config.trial_name = "test_trial"
    config.allocation_mode = "vllm:d4p1t2+d8p1t1"
    config.cluster = ClusterSpecConfig()
    config.cluster.fileroot = "/tmp/areal_test"
    config.actor = PPOActorConfig()
    config.actor.weight_update_mode = weight_update_mode
    config.critic = PPOCriticConfig()
    return config


def test_prepare_awex_runtime_noop_for_non_awex():
    config = _create_ppo_config(weight_update_mode="disk")

    handle = prepare_awex_runtime(config)

    assert handle.meta_server_addr is None
    assert handle.owns_meta_server is False
    assert config.awex.meta_server_addr == ""


def test_prepare_awex_runtime_reuses_explicit_config_addr(
    monkeypatch: pytest.MonkeyPatch,
):
    config = _create_ppo_config(weight_update_mode="awex")
    config.awex.meta_server_addr = "127.0.0.1:23456"

    def _unexpected_import() -> tuple[Callable[..., object], Callable[..., object]]:
        raise AssertionError("meta server should not be started for explicit addr")

    monkeypatch.setattr(
        "areal.utils.awex_runtime._import_meta_server_fns", _unexpected_import
    )

    handle = prepare_awex_runtime(config)

    assert handle.meta_server_addr == "127.0.0.1:23456"
    assert handle.owns_meta_server is False
    assert config.awex.meta_server_addr == "127.0.0.1:23456"


def test_prepare_awex_runtime_starts_local_server_for_auto(
    monkeypatch: pytest.MonkeyPatch,
):
    config = _create_ppo_config(weight_update_mode="awex")
    config.awex.meta_server_addr = "auto"
    state = {"stopped": False}

    def _fake_start():
        return "127.0.0.1", 34567

    def _fake_stop():
        state["stopped"] = True
        return True

    monkeypatch.setattr(
        "areal.utils.awex_runtime._import_meta_server_fns",
        lambda: (_fake_start, _fake_stop),
    )
    monkeypatch.setattr(
        "areal.utils.awex_runtime.is_single_controller",
        lambda: True,
    )

    handle = prepare_awex_runtime(config)

    assert handle.meta_server_addr == "127.0.0.1:34567"
    assert handle.owns_meta_server is True
    assert config.awex.meta_server_addr == "127.0.0.1:34567"

    handle.close()

    assert state["stopped"] is True


def test_prepare_awex_runtime_prefers_env_override(monkeypatch: pytest.MonkeyPatch):
    config = _create_ppo_config(weight_update_mode="awex")
    config.awex.meta_server_addr = "auto"
    monkeypatch.setenv("AREAL_AWEX_META_SERVER_ADDR", "127.0.0.1:45678")

    def _unexpected_import() -> tuple[Callable[..., object], Callable[..., object]]:
        raise AssertionError(
            "meta server should not be started when env override exists"
        )

    monkeypatch.setattr(
        "areal.utils.awex_runtime._import_meta_server_fns", _unexpected_import
    )

    handle = prepare_awex_runtime(config)

    assert handle.meta_server_addr == "127.0.0.1:45678"
    assert handle.owns_meta_server is False
    assert config.awex.meta_server_addr == "127.0.0.1:45678"


def test_prepare_awex_runtime_requires_explicit_addr_in_spmd(
    monkeypatch: pytest.MonkeyPatch,
):
    config = _create_ppo_config(weight_update_mode="awex")
    config.awex.meta_server_addr = "auto"
    monkeypatch.setattr(
        "areal.utils.awex_runtime.is_single_controller",
        lambda: False,
    )

    with pytest.raises(ValueError, match="single-controller mode"):
        prepare_awex_runtime(config)
