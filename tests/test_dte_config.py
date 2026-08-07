from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from areal.api.cli_args import DTEConfig
from areal.trainer.rl_trainer import PPOTrainer


def _make_trainer(dte_config: DTEConfig):
    actor_spec = SimpleNamespace(env_vars={})
    rollout_spec = SimpleNamespace(env_vars={})
    trainer = object.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        actor=SimpleNamespace(dte=dte_config, scheduling_spec=[actor_spec]),
        rollout=SimpleNamespace(scheduling_spec=[rollout_spec]),
    )
    return trainer, actor_spec, rollout_spec


def test_dte_config_unset_preserves_legacy_environment(monkeypatch):
    monkeypatch.setenv("AWEX_DELTA_TRANSFER", "1")
    monkeypatch.delenv("DTE_DELTA_TRANSFER", raising=False)
    trainer, actor_spec, rollout_spec = _make_trainer(DTEConfig())

    trainer._apply_dte_config_envvars()

    assert "DTE_DELTA_TRANSFER" not in actor_spec.env_vars
    assert "DTE_DELTA_TRANSFER" not in rollout_spec.env_vars


def test_dte_delta_config_enables_colocate_and_inversion():
    trainer, actor_spec, rollout_spec = _make_trainer(
        DTEConfig(transfer="delta", delta_method="adamw")
    )

    trainer._apply_dte_config_envvars()

    expected = {
        "DTE_COLOCATE_WEIGHT_UPDATE": "1",
        "DTE_DELTA_TRANSFER": "1",
        "DTE_DELTA_DETECTOR": "inversion",
    }
    for name, value in expected.items():
        assert actor_spec.env_vars[name] == value
        assert rollout_spec.env_vars[name] == value


def test_explicit_full_transfer_enables_colocate():
    trainer, actor_spec, _ = _make_trainer(DTEConfig(transfer="full"))

    trainer._apply_dte_config_envvars()

    assert actor_spec.env_vars["DTE_COLOCATE_WEIGHT_UPDATE"] == "1"
    assert actor_spec.env_vars["DTE_DELTA_TRANSFER"] == "0"


def test_explicit_dte_enable_overrides_legacy_delta_environment(monkeypatch):
    monkeypatch.setenv("AWEX_DELTA_TRANSFER", "1")
    trainer, actor_spec, _ = _make_trainer(DTEConfig(enabled=True))

    trainer._apply_dte_config_envvars()

    assert actor_spec.env_vars["DTE_COLOCATE_WEIGHT_UPDATE"] == "1"
    assert actor_spec.env_vars["DTE_DELTA_TRANSFER"] == "0"


def test_dte_config_rejects_invalid_transfer():
    trainer, _, _ = _make_trainer(DTEConfig(transfer="invalid"))

    with pytest.raises(ValueError, match="actor.dte.transfer"):
        trainer._apply_dte_config_envvars()


def test_delta_rollout_offload_preserves_weight_base():
    trainer = object.__new__(PPOTrainer)
    trainer.rollout = MagicMock()
    trainer.eval_rollout = None
    trainer._keep_rollout_weights_resident = True

    trainer._offload_rollout()

    trainer.rollout.offload.assert_called_once_with(tags=["kv_cache"])
