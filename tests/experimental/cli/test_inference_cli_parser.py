# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from click.testing import CliRunner

from areal.experimental.cli.main import cli


def test_inference_commands_expose_service_flag():
    runner = CliRunner()

    for command in (
        "run",
        "stop",
        "status",
        "register",
        "deregister",
        "models",
        "collect",
        "reward",
        "logs",
    ):
        result = runner.invoke(cli, ["inf", command, "--help"])
        assert result.exit_code == 0
        assert "--service" in result.output


def test_inference_ps_lists_services_not_models():
    runner = CliRunner()

    result = runner.invoke(cli, ["inf", "ps", "--help"])

    assert result.exit_code == 0
    assert "--all" in result.output
    assert "--service" not in result.output


def test_inference_model_name_options_are_exposed():
    runner = CliRunner()

    register = runner.invoke(cli, ["inf", "register", "--help"])
    deregister = runner.invoke(cli, ["inf", "deregister", "--help"])
    collect = runner.invoke(cli, ["inf", "collect", "--help"])

    assert register.exit_code == 0
    assert "--model-name" in register.output
    assert deregister.exit_code == 0
    assert "--model-name" in deregister.output
    assert collect.exit_code == 0
    assert "--model-name" in collect.output


def test_inference_reward_session_api_key_option_is_exposed():
    runner = CliRunner()

    result = runner.invoke(cli, ["inf", "reward", "--help"])

    assert result.exit_code == 0
    assert "--session-api-key" in result.output


def test_inference_register_parses_model_name_option(monkeypatch):
    from areal.experimental.cli.inference.commands import register

    captured = {}

    def fake_register(model_name, opts, *, service=None):
        captured["model_name"] = model_name
        captured["opts"] = opts
        captured["service"] = service
        return 0

    monkeypatch.setattr(register, "do_register", fake_register)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "inf",
            "register",
            "--service",
            "svc",
            "--model-name",
            "m",
            "--backend",
            "sglang:tp=1,dp=1",
            "--model-path",
            "/models/m",
        ],
    )

    assert result.exit_code == 0
    assert captured["model_name"] == "m"
    assert captured["service"] == "svc"
    assert captured["opts"]["backend"] == "sglang:tp=1,dp=1"
    assert captured["opts"]["model_path"] == "/models/m"


def test_inference_deregister_parses_model_name_option(monkeypatch):
    from areal.experimental.cli.inference.commands import deregister

    captured = {}

    def fake_deregister(model_name, grace, force, *, service=None):
        captured["model_name"] = model_name
        captured["grace"] = grace
        captured["force"] = force
        captured["service"] = service
        return 0

    monkeypatch.setattr(deregister, "do_deregister", fake_deregister)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "inf",
            "deregister",
            "--service",
            "svc",
            "--model-name",
            "m",
            "--force",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "model_name": "m",
        "grace": 10.0,
        "force": True,
        "service": "svc",
    }


def test_inference_reward_parses_session_api_key_option(monkeypatch):
    from areal.experimental.cli.inference.commands import reward

    captured = {}

    def fake_reward(session_api_key, reward_value, model, *, service=None):
        captured["session_api_key"] = session_api_key
        captured["reward_value"] = reward_value
        captured["model"] = model
        captured["service"] = service
        return 0

    monkeypatch.setattr(reward, "do_reward", fake_reward)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "inf",
            "reward",
            "--service",
            "svc",
            "--session-api-key",
            "sk",
            "--model",
            "m",
            "1.25",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "session_api_key": "sk",
        "reward_value": 1.25,
        "model": "m",
        "service": "svc",
    }


def test_inference_collect_parses_model_name_option(monkeypatch):
    from areal.experimental.cli.inference.commands import collect

    captured = {}

    def fake_collect(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(collect, "do_collect", fake_collect)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "inf",
            "collect",
            "--service",
            "svc",
            "--model-name",
            "m",
            "--batch-size",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert captured["model"] == "m"
    assert captured["service"] == "svc"
    assert captured["batch_size"] == 2
