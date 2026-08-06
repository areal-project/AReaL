"""Tests for topology-specific DTE runtime gates."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_HELPER_PATH = Path(__file__).resolve().parent.parent / "areal" / "utils" / "dte.py"

_DTE_ENV_KEYS = {
    "DTE_COLOCATE_WEIGHT_UPDATE",
    "DTE_SEPARATION_WEIGHT_UPDATE",
    "DTE_DELTA_TRANSFER",
    "DTE_DELTA_DETECTOR",
    "DTE_DELTA_ANCHOR_INTERVAL",
    "DTE_DELTA_BYTES_RATIO",
    "DTE_DELTA_VERIFY_SNAPSHOT",
    "DTE_RELEASE_TRAIN_WEIGHTS_AFTER_UPDATE",
    "DTE_SYNC_MODEL_PARAMS_BEFORE_PAYLOAD",
    "DTE_DELTA_INVERSION_DEBUG",
    "DTE_DELTA_INVERSION_BF16_MARGIN_REL",
}


def _load_dte_helpers():
    spec = importlib.util.spec_from_file_location("areal_utils_dte", _HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def clear_dte_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure each test observes only env vars exported by the helper."""
    for key in _DTE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize(
    ("topology", "expected_colocate", "expected_separation"),
    [
        ("colocation", True, False),
        ("separation", False, True),
    ],
)
def test_dte_enabled_returns_only_matching_topology_gate(
    topology: str,
    expected_colocate: bool,
    expected_separation: bool,
) -> None:
    """DTE must activate exactly one transport for the configured topology."""
    helpers = _load_dte_helpers()

    actual = helpers.dte_weight_update_topology_gates(True, topology)

    assert actual == (expected_colocate, expected_separation)


@pytest.mark.parametrize("topology", ["colocation", "separation"])
def test_dte_disabled_returns_no_topology_gate(topology: str) -> None:
    """Disabling DTE must leave both weight-update transports inactive."""
    helpers = _load_dte_helpers()

    actual = helpers.dte_weight_update_topology_gates(False, topology)

    assert actual == (False, False)


def test_dte_unknown_topology_is_rejected() -> None:
    """Unknown topology values must not silently select a DTE transport."""
    helpers = _load_dte_helpers()

    with pytest.raises(ValueError, match="Unsupported DTE scheduling topology"):
        helpers.dte_weight_update_topology_gates(True, "shared")


def _make_dte_config(**kwargs):
    defaults = {
        "enabled": None,
        "transfer": "full",
        "delta_method": None,
        "anchor_interval": 0,
        "bytes_ratio": None,
        "verify_snapshot": False,
        "release_train_weights_after_update": None,
        "release_initial_rollout_weights": None,
        "sync_model_params_before_payload": None,
        "inversion_debug": None,
        "inversion_bf16_margin_rel": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_config(dte, topology: str):
    actor_spec = SimpleNamespace(env_vars={})
    rollout_spec = SimpleNamespace(env_vars={})
    return SimpleNamespace(
        actor=SimpleNamespace(dte=dte, scheduling_spec=(actor_spec,)),
        rollout=SimpleNamespace(
            scheduling_strategy=SimpleNamespace(type=topology),
            scheduling_spec=(rollout_spec,),
        ),
    )


@pytest.mark.parametrize(
    ("topology", "expected_colocate", "expected_separation"),
    [
        ("colocation", "1", "0"),
        ("separation", "0", "1"),
    ],
)
def test_dte_delta_config_exports_topology_and_transfer_env(
    monkeypatch: pytest.MonkeyPatch,
    topology: str,
    expected_colocate: str,
    expected_separation: str,
) -> None:
    """actor.dte enables DTE only through matching topology-specific gates."""
    helpers = _load_dte_helpers()
    config = _make_config(
        _make_dte_config(enabled=True, transfer="delta", delta_method="adamw"),
        topology,
    )

    env = helpers.apply_dte_config_envvars(config, environ={})

    expected = {
        "DTE_COLOCATE_WEIGHT_UPDATE": expected_colocate,
        "DTE_SEPARATION_WEIGHT_UPDATE": expected_separation,
        "DTE_DELTA_TRANSFER": "1",
        "DTE_DELTA_DETECTOR": "inversion",
        "DTE_DELTA_ANCHOR_INTERVAL": "0",
        "DTE_DELTA_VERIFY_SNAPSHOT": "0",
    }
    for key, value in expected.items():
        assert env[key] == value
        assert config.actor.scheduling_spec[0].env_vars[key] == value
        assert config.rollout.scheduling_spec[0].env_vars[key] == value


def test_dte_default_config_does_not_override_runtime_env() -> None:
    """Leaving actor.dte.enabled unset must preserve env-var fallback behavior."""
    helpers = _load_dte_helpers()
    config = _make_config(_make_dte_config(), "separation")

    env = helpers.apply_dte_config_envvars(config, environ={})

    assert env == {}
    assert config.actor.scheduling_spec[0].env_vars == {}
    assert config.rollout.scheduling_spec[0].env_vars == {}


def test_dte_delta_defaults_to_inversion_and_implied_enabled() -> None:
    """transfer=delta is a shorthand for enabling DTE with AdamW inversion."""
    helpers = _load_dte_helpers()
    config = _make_config(_make_dte_config(transfer="delta"), "separation")

    helpers.apply_dte_config_envvars(config, environ={})

    env_vars = config.actor.scheduling_spec[0].env_vars
    assert env_vars["DTE_SEPARATION_WEIGHT_UPDATE"] == "1"
    assert env_vars["DTE_COLOCATE_WEIGHT_UPDATE"] == "0"
    assert env_vars["DTE_DELTA_TRANSFER"] == "1"
    assert env_vars["DTE_DELTA_DETECTOR"] == "inversion"


@pytest.mark.parametrize(
    ("dte", "match"),
    [
        (
            _make_dte_config(enabled=False, transfer="delta"),
            "actor.dte.enabled=false conflicts",
        ),
        (
            _make_dte_config(enabled=True, transfer="sparse"),
            "actor.dte.transfer",
        ),
        (
            _make_dte_config(
                enabled=True,
                transfer="delta",
                delta_method="unknown",
            ),
            "actor.dte.delta_method",
        ),
    ],
)
def test_dte_invalid_config_is_rejected(dte, match: str) -> None:
    """Invalid actor.dte combinations fail before worker launch."""
    helpers = _load_dte_helpers()
    config = _make_config(dte, "separation")

    with pytest.raises(ValueError, match=match):
        helpers.apply_dte_config_envvars(config, environ={})
