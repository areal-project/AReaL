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
