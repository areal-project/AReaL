# SPDX-License-Identifier: Apache-2.0

"""Task entrypoint wiring without starting training workers."""

import json

import pytest

from areal.autopilot.domains.elastic.native import RevisionAdapter


@pytest.mark.parametrize("enabled", [False, True])
def test_hermes_entry_preserves_config_arguments_and_optionally_attaches_native_reconfigurator(
    monkeypatch, enabled
):
    from examples.hermes import train

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

    monkeypatch.setattr(train, "load_expr_config", load)
    monkeypatch.setattr(train, "PPOTrainer", Trainer)
    config_args = [
        "--config",
        "examples/hermes/config.yaml",
        "elastic.enabled=true",
    ]
    train.main([*(["--elastic-revision"] if enabled else []), *config_args])
    assert seen["args"] == config_args and seen["trained"]
    assert ("elastic_reconfigurator" in seen) == enabled
    if enabled:
        assert isinstance(seen["elastic_reconfigurator"], RevisionAdapter)
        assert seen["elastic_reconfigurator"].closed


def test_hermes_measured_profile_selects_areal_adapter_without_autopilot(
    monkeypatch, tmp_path
):
    """Profile-only runs use the ordinary adapter and default Monitor."""
    import builtins

    from examples.hermes import train

    from areal.elastic.infra.adapter import ReconfiguratorAdapter

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
    assert seen["elastic_reconfigurator"].hooks is None and seen["trained"]
