# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from click.testing import CliRunner

from areal.v2.cli.main import cli


def test_inference_commands_expose_service_flag():
    runner = CliRunner()

    for command in (
        "run",
        "stop",
        "status",
        "register",
        "deregister",
        "models",
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

    assert register.exit_code == 0
    assert "--model-name" in register.output
    assert deregister.exit_code == 0
    assert "--model-name" in deregister.output


def test_inference_register_parses_model_name_option(monkeypatch):
    from areal.v2.cli.inference.commands import register

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
            "sglang:d1",
            "--model-path",
            "/models/m",
        ],
    )

    assert result.exit_code == 0
    assert captured["model_name"] == "m"
    assert captured["service"] == "svc"
    assert captured["opts"]["backend"] == "sglang:d1"
    assert captured["opts"]["model_path"] == "/models/m"


def test_inference_deregister_parses_model_name_option(monkeypatch):
    from areal.v2.cli.inference.commands import deregister

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
