# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from click.testing import CliRunner

from areal.v2.cli.cli import cli


def test_agent_help_contains_expected_commands():
    result = CliRunner().invoke(cli, ["agent", "--help"])

    assert result.exit_code == 0
    assert "run" in result.output
    assert "stop" in result.output
    assert "status" in result.output
    assert "ps" in result.output
    assert "logs" in result.output


def test_agent_help_does_not_register_session_commands():
    runner = CliRunner()

    for verb in ("new_session", "switch_session", "chat", "reward"):
        result = runner.invoke(cli, ["agent", verb])
        assert result.exit_code != 0
        assert "No such command" in result.output


def test_agent_run_does_not_accept_initial_session_key():
    result = CliRunner().invoke(cli, ["agent", "run", "--help"])

    assert result.exit_code == 0
    assert "--session-key" not in result.output
