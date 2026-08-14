# SPDX-License-Identifier: Apache-2.0
"""Tests for shared environment-variable parsing helpers."""

import importlib.util
import logging as stdlib_logging
import sys
import types
from pathlib import Path

_ENVIRON_PATH = Path(__file__).parents[1] / "areal/utils/environ.py"


def _load_environ(monkeypatch):
    logging_mod = types.ModuleType("areal.utils.logging")
    logging_mod.getLogger = stdlib_logging.getLogger
    utils_mod = types.ModuleType("areal.utils")
    utils_mod.logging = logging_mod
    monkeypatch.setitem(sys.modules, "areal", types.ModuleType("areal"))
    monkeypatch.setitem(sys.modules, "areal.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "areal.utils.logging", logging_mod)

    spec = importlib.util.spec_from_file_location("areal_test_environ", _ENVIRON_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_get_env_var_uses_primary_then_fallback(monkeypatch):
    environ = _load_environ(monkeypatch)
    monkeypatch.delenv("AREAL_TEST_PRIMARY", raising=False)
    monkeypatch.setenv("AREAL_TEST_LEGACY", "legacy")

    assert (
        environ.get_env_var(
            "AREAL_TEST_PRIMARY",
            fallback_names=("AREAL_TEST_LEGACY",),
        )
        == "legacy"
    )

    monkeypatch.setenv("AREAL_TEST_PRIMARY", "primary")
    assert (
        environ.get_env_var(
            "AREAL_TEST_PRIMARY",
            fallback_names=("AREAL_TEST_LEGACY",),
        )
        == "primary"
    )


def test_get_env_var_empty_value_fallback_is_configurable(monkeypatch):
    environ = _load_environ(monkeypatch)
    monkeypatch.setenv("AREAL_TEST_PRIMARY", "")
    monkeypatch.setenv("AREAL_TEST_LEGACY", "legacy")

    assert (
        environ.get_env_var(
            "AREAL_TEST_PRIMARY",
            fallback_names=("AREAL_TEST_LEGACY",),
        )
        == "legacy"
    )
    assert (
        environ.get_env_var(
            "AREAL_TEST_PRIMARY",
            fallback_names=("AREAL_TEST_LEGACY",),
            allow_empty=True,
        )
        == ""
    )


def test_get_bool_env_var_preserves_defaults_and_supports_opt_in_values(monkeypatch):
    environ = _load_environ(monkeypatch)
    monkeypatch.setenv("AREAL_TEST_BOOL", "yes")

    assert environ.get_bool_env_var("AREAL_TEST_BOOL") is False
    assert (
        environ.get_bool_env_var(
            "AREAL_TEST_BOOL",
            truthy_values=("true", "1", "yes", "on"),
            falsy_values=("false", "0", "no", "off"),
        )
        is True
    )


def test_get_bool_env_var_can_strip_legacy_dte_values(monkeypatch):
    environ = _load_environ(monkeypatch)
    monkeypatch.setenv("AREAL_TEST_BOOL", " on ")

    assert (
        environ.get_bool_env_var(
            "AREAL_TEST_BOOL",
            truthy_values=("true", "1", "yes", "on"),
            falsy_values=("false", "0", "no", "off"),
            strip_value=True,
        )
        is True
    )
