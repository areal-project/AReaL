# SPDX-License-Identifier: Apache-2.0

"""Task entrypoint wiring without starting training workers."""

import json
import sys
import types
from importlib.util import find_spec, module_from_spec

import pytest

import areal
from areal.elastic.infra.adapter import ReconfiguratorAdapter


def load_entrypoint(name):
    """Execute a fresh entrypoint module without caching its patched dependencies."""
    spec = find_spec(name)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config_api(monkeypatch):
    """Replace the config-reader boundary before importing the training entrypoint."""
    module = types.ModuleType("areal.api.cli_args")
    module.PPOConfig = object
    module.load_expr_config = lambda *args: (object(), None)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.mark.parametrize("enabled", [False, True])
def test_hermes_entry_preserves_config_arguments_and_optionally_attaches_native_reconfigurator(
    monkeypatch, tmp_path, config_api, enabled
):
    config = object()
    seen = {}

    def load(args, config_type):
        seen["args"] = args
        return config, None

    class Trainer:
        def __init__(self, actual_config, **kwargs):
            assert actual_config is config
            seen.update(kwargs)

        def __enter__(self):
            return self

        def train(self):
            seen["trained"] = True

        def __exit__(self, *args):
            if "elastic_reconfigurator" in seen:
                seen["elastic_reconfigurator"].close()

    monkeypatch.setattr(areal, "PPOTrainer", Trainer, raising=False)
    config_api.load_expr_config = load
    train = load_entrypoint("examples.hermes.train")

    monkeypatch.setattr(train, "load_expr_config", load)
    monkeypatch.setattr(train, "PPOTrainer", Trainer)
    config_args = [
        "--config",
        "examples/hermes/config.yaml",
        "elastic.enabled=true",
    ]
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"reconfigurator": {}}))
    elastic_args = ["--elastic-profile", str(profile)] if enabled else []
    train.main([*elastic_args, *config_args])
    assert seen["args"] == config_args and seen["trained"]
    assert ("elastic_reconfigurator" in seen) == enabled
    if enabled:
        assert type(seen["elastic_reconfigurator"]) is ReconfiguratorAdapter
        assert seen["elastic_reconfigurator"].closed


@pytest.mark.parametrize(
    "entrypoint", ["examples.hermes.train", "examples.task_workflows.train"]
)
def test_measured_profile_selects_areal_adapter_without_autopilot(
    monkeypatch, tmp_path, config_api, entrypoint
):
    """Profile-only runs use the ordinary adapter and default Monitor."""
    import builtins

    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"reconfigurator": {}}))
    seen = {}

    class Trainer:
        def __init__(self, config, **kwargs):
            seen.update(kwargs)

        def __enter__(self):
            return self

        def train(self):
            seen["trained"] = True

        def __exit__(self, *args):
            seen["elastic_reconfigurator"].close()

    monkeypatch.setattr(areal, "PPOTrainer", Trainer, raising=False)
    train = load_entrypoint(entrypoint)
    original_import = builtins.__import__

    def without_autopilot(name, *args, **kwargs):
        if name == "areal.autopilot" or name.startswith("areal.autopilot."):
            raise AssertionError("Ordinary Online entry must not import AutoPilot")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(train, "load_expr_config", lambda *args: (object(), None))
    monkeypatch.setattr(train, "PPOTrainer", Trainer)
    monkeypatch.setattr(builtins, "__import__", without_autopilot)
    train.main(["--elastic-profile", str(profile), "elastic.enabled=true"])
    assert type(seen["elastic_reconfigurator"]) is ReconfiguratorAdapter
    assert seen["trained"]
