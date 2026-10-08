"""Focused tests for the optional τ³-Bench workflow adapter."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("tau2")

from examples.tau3 import agent as tau3_agent  # noqa: E402
from examples.tau3.train import get_tau3_dataset  # noqa: E402


def test_tau3_dataset_repeats_tasks_to_fill_batch(monkeypatch):
    monkeypatch.setattr(
        tau3_agent.registry,
        "get_task_splits_loader",
        lambda domain: lambda: {"train": ["task-1", "task-2", "task-3"]},
    )

    dataset = get_tau3_dataset("airline", "train", min_size=5)

    assert dataset["task_id"] == ["task-1", "task-2", "task-3", "task-1", "task-2"]


@pytest.fixture
def simulator_stubs(monkeypatch):
    task = SimpleNamespace(id="task-1", user_scenario="Book a flight")
    environment = MagicMock()
    environment.get_tools.return_value = []
    environment.get_user_tools.return_value = []
    monkeypatch.setattr(
        tau3_agent.registry, "get_tasks_loader", lambda domain: lambda split: [task]
    )
    monkeypatch.setattr(
        tau3_agent.registry,
        "get_env_constructor",
        lambda domain: lambda **kwargs: environment,
    )
    agent_constructor = MagicMock()
    user_constructor = MagicMock()
    orchestrator = MagicMock()
    orchestrator.return_value.run.return_value = SimpleNamespace()
    monkeypatch.setattr(tau3_agent, "LLMAgent", agent_constructor)
    monkeypatch.setattr(tau3_agent, "UserSimulator", user_constructor)
    monkeypatch.setattr(tau3_agent, "Orchestrator", orchestrator)
    monkeypatch.setattr(
        tau3_agent,
        "evaluate_simulation",
        lambda **kwargs: SimpleNamespace(reward=0.75),
    )
    return user_constructor, orchestrator


def test_tau3_agent_with_key_file_uses_simulated_user(
    tmp_path, monkeypatch, simulator_stubs
):
    """A real user simulator receives the key from a file, not process env."""
    user_constructor, orchestrator = simulator_stubs
    key_file = tmp_path / "user.key"
    key_file.write_text("test-key\n")
    monkeypatch.setenv("TAU3_USER_API_KEY_FILE", str(key_file))
    workflow = tau3_agent.Tau3AgentWorkflow(
        econfig={
            "domain": "airline",
            "user_llm_base_url": "https://example.invalid/v1",
            "user_llm": "gpt-6-luna",
        }
    )

    reward = workflow._run_sync(
        {"task_id": "task-1", "split": "train"}, "http://proxy", "proxy-key"
    )

    assert reward == 0.75
    assert user_constructor.call_args.kwargs["llm_args"]["api_key"] == "test-key"
    assert orchestrator.call_args.kwargs["solo_mode"] is False


def test_tau3_agent_without_key_file_fails_before_simulation(
    monkeypatch, simulator_stubs
):
    """Normal simulator mode explains which credential file is missing."""
    _, orchestrator = simulator_stubs
    monkeypatch.delenv("TAU3_USER_API_KEY_FILE", raising=False)
    workflow = tau3_agent.Tau3AgentWorkflow(
        econfig={
            "domain": "airline",
            "user_llm_base_url": "https://example.invalid/v1",
            "user_llm": "gpt-6-luna",
        }
    )

    with pytest.raises(ValueError, match="TAU3_USER_API_KEY_FILE"):
        workflow._run_sync({"task_id": "task-1"}, "http://proxy", "proxy-key")

    orchestrator.assert_not_called()
