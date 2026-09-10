"""Tests for Megatron deterministic-mode activation semantics."""

import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from areal.engine.megatron_utils.deterministic import set_deterministic_algorithms


@pytest.fixture(autouse=True)
def _restore_global_state(monkeypatch):
    monkeypatch.delenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", raising=False)
    monkeypatch.delenv("NCCL_ALGO", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("NCCL_NVLS_ENABLE", raising=False)
    prev_deterministic = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    yield

    torch.use_deterministic_algorithms(prev_deterministic, warn_only=prev_warn_only)


def _make_config(**extra):
    return SimpleNamespace(
        deterministic_mode=False,
        cross_entropy_loss_fusion=True,
        bias_dropout_fusion=True,
        **extra,
    )


def test_set_deterministic_algorithms_prebuild_selects_flash_backend():
    enums = pytest.importorskip("megatron.core.transformer.enums")
    config = _make_config(attention_backend=enums.AttnBackend.auto)

    set_deterministic_algorithms(config, prebuild=True)

    assert config.deterministic_mode is True
    assert config.cross_entropy_loss_fusion is False
    assert config.bias_dropout_fusion is False
    assert config.attention_backend == enums.AttnBackend.flash


def test_set_deterministic_algorithms_postbuild_keeps_attention_backend():
    sentinel = object()
    config = _make_config(attention_backend=sentinel)

    set_deterministic_algorithms(config)

    assert config.deterministic_mode is True
    assert config.attention_backend is sentinel


def test_set_deterministic_algorithms_te_imported_without_env_warns(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "transformer_engine", types.ModuleType("transformer_engine")
    )
    config = _make_config()

    with mock.patch("areal.engine.megatron_utils.deterministic.logger") as mock_logger:
        set_deterministic_algorithms(config)

    assert mock_logger.warning.called


def test_set_deterministic_algorithms_env_preset_does_not_warn(monkeypatch):
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    monkeypatch.setitem(
        sys.modules, "transformer_engine", types.ModuleType("transformer_engine")
    )
    config = _make_config()

    with mock.patch("areal.engine.megatron_utils.deterministic.logger") as mock_logger:
        set_deterministic_algorithms(config)

    assert not mock_logger.warning.called
