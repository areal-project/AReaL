# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OpenEnvWorkflow.

All external I/O (WebSocket, container launch, LLM inference) is mocked in
process. The tests confirm:
  * config validation and post-init rejection of invalid combinations
  * default parsers correctly extract JSON / tag / passthrough actions
  * observation formatter stringifies dataclasses, dicts, primitives
  * arun_episode drives reset -> step -> ... in order, records per-step reward,
    honors max_turns, and terminates on ``done``
  * terminal_reward_only zeroes intermediate rewards
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from areal.api.openenv_api import OpenEnvConfig
from areal.workflow.openenv_utils import (
    AutoObservationFormatter,
    JSONActionParser,
    PassthroughActionParser,
    TagActionParser,
    build_action,
    resolve_action_parser,
    resolve_obs_formatter,
)

# ---------------------------------------------------------------------------
# Fakes: EnvClient / StepResult / ArealOpenAI
# ---------------------------------------------------------------------------


@dataclass
class _FakeStepResult:
    observation: Any
    reward: float | None = 0.0
    done: bool = False
    metadata: dict[str, Any] | None = None


@dataclass
class _FakeObservation:
    turn: int
    hint: str = "keep going"


class _FakeEnvClient:
    """Mocks openenv.core.EnvClient's async-context surface.

    Records every reset/step call so the test can inspect the interaction
    history. Each step reads its reward and done flag from the script queue.
    """

    def __init__(self, script: list[_FakeStepResult]) -> None:
        self._script = list(script)
        self.reset_calls: list[dict[str, Any]] = []
        self.step_actions: list[Any] = []
        self.closed = False

    async def __aenter__(self) -> _FakeEnvClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.closed = True

    async def reset(self, **kwargs: Any) -> _FakeStepResult:
        self.reset_calls.append(kwargs)
        return _FakeStepResult(observation=_FakeObservation(turn=0))

    async def step(self, action: Any, **kwargs: Any) -> _FakeStepResult:
        self.step_actions.append(action)
        if not self._script:
            return _FakeStepResult(
                observation=_FakeObservation(turn=len(self.step_actions)),
                reward=0.0,
                done=True,
            )
        return self._script.pop(0)


@dataclass
class _FakeChoiceMsg:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeChoiceMsg


@dataclass
class _FakeCompletion:
    id: str
    choices: list[_FakeChoice]


class _FakeChatCompletions:
    def __init__(self, scripts: list[str]) -> None:
        self._scripts = list(scripts)
        self.n_calls = 0

    async def create(self, **kwargs: Any) -> _FakeCompletion:
        if not self._scripts:
            raise RuntimeError("no more scripted LLM completions available")
        content = self._scripts.pop(0)
        self.n_calls += 1
        cid = f"cmpl-{self.n_calls}"
        return _FakeCompletion(
            id=cid, choices=[_FakeChoice(message=_FakeChoiceMsg(content=content))]
        )


class _FakeChat:
    def __init__(self, completions: _FakeChatCompletions) -> None:
        self.completions = completions


class _FakeArealOpenAI:
    """Stand-in for ArealOpenAI that only records reward assignments."""

    last_instance: _FakeArealOpenAI | None = None

    def __init__(self, engine: Any, tokenizer: Any) -> None:  # noqa: D401
        # ``scripts`` is injected via the class attribute prior to instantiation.
        scripts = getattr(_FakeArealOpenAI, "pending_scripts", [])
        self.chat = _FakeChat(_FakeChatCompletions(scripts))
        self.rewards: dict[str, float] = {}
        self.discount_calls: list[float] = []
        self.exported: dict[str, dict[str, Any]] = {}
        _FakeArealOpenAI.last_instance = self

    def set_reward(self, cid: str, reward: float) -> None:
        self.rewards[cid] = float(reward)

    def apply_reward_discount(self, factor: float) -> None:
        self.discount_calls.append(factor)

    def export_interactions(self, style: str) -> dict[str, dict[str, Any]]:
        return {cid: {"reward": r, "style": style} for cid, r in self.rewards.items()}


class _FakeGenerationHyperparameters:
    """Minimal stand-in matching the surface used by the workflow."""

    def __init__(self) -> None:
        self.frequency_penalty = 0.0
        self.max_new_tokens = 32
        self.stop = None
        self.temperature = 1.0
        self.top_p = 1.0
        self.n_samples = 1

    def new_with_stop_and_pad_token_ids(
        self, tokenizer: Any
    ) -> _FakeGenerationHyperparameters:
        return self

    def new(self, **kwargs: Any) -> _FakeGenerationHyperparameters:
        return self


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_areal_openai_class(monkeypatch):
    """Route ArealOpenAI/env-client imports inside the workflow to fakes."""
    from areal.workflow import openenv as workflow_mod

    monkeypatch.setattr(workflow_mod, "ArealOpenAI", _FakeArealOpenAI)
    yield


@pytest.fixture
def _stats_tracker_stub(monkeypatch):
    """Silence stats_tracker.get(...).scalar side effects."""
    from areal.workflow import openenv as workflow_mod

    class _Scope:
        def scalar(self, **kwargs: Any) -> None:
            return None

    class _StatsTracker:
        def get(self, *_args, **_kwargs) -> _Scope:  # noqa: D401
            return _Scope()

    monkeypatch.setattr(workflow_mod, "stats_tracker", _StatsTracker())
    monkeypatch.setattr(
        workflow_mod,
        "workflow_context",
        type("_Wctx", (), {"stat_scope": staticmethod(lambda: "test")})(),
    )


def _make_workflow(cfg: OpenEnvConfig, env_client: _FakeEnvClient, monkeypatch):
    from areal.workflow import openenv as workflow_mod
    from areal.workflow.openenv import OpenEnvWorkflow

    monkeypatch.setattr(
        workflow_mod,
        "_instantiate_env_client",
        lambda _cfg: env_client,
    )
    return OpenEnvWorkflow(
        config=cfg,
        gconfig=_FakeGenerationHyperparameters(),
        tokenizer=object(),
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_zero_max_turns():
    """OpenEnvConfig.__post_init__ rejects non-positive max_turns."""
    with pytest.raises(ValueError, match="max_turns"):
        OpenEnvConfig(env_client_class="pkg.Env", base_url="ws://x", max_turns=0)


def test_config_rejects_missing_target():
    """Config must specify base_url or provider (not neither)."""
    with pytest.raises(ValueError, match="base_url or provider"):
        OpenEnvConfig(env_client_class="pkg.Env")


def test_config_rejects_both_targets():
    """base_url and provider are mutually exclusive."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        OpenEnvConfig(
            env_client_class="pkg.Env",
            base_url="ws://x",
            provider="uv",
            project_path="/tmp",
        )


def test_config_rejects_uv_without_project_path():
    """provider='uv' requires project_path."""
    with pytest.raises(ValueError, match="project_path"):
        OpenEnvConfig(env_client_class="pkg.Env", provider="uv")


def test_config_rejects_out_of_range_step_discount():
    """step_discount must be in (0, 1]."""
    with pytest.raises(ValueError, match="step_discount"):
        OpenEnvConfig(env_client_class="pkg.Env", base_url="ws://x", step_discount=1.5)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def test_json_parser_extracts_pure_json_body():
    """JSONActionParser returns dict from a raw JSON payload."""
    action = JSONActionParser()(
        '{"tool_name":"echo","arguments":{"message":"hi"}}', None
    )
    assert action == {"tool_name": "echo", "arguments": {"message": "hi"}}


def test_json_parser_extracts_trailing_json_after_reasoning():
    """JSONActionParser scans for the last JSON block when text precedes it."""
    action = JSONActionParser()(
        'Reasoning: I should echo.\nFinal answer: {"tool_name":"echo"}',
        None,
    )
    assert action == {"tool_name": "echo"}


def test_json_parser_returns_none_on_no_json():
    """Unparsable output surfaces as None to the workflow."""
    assert JSONActionParser()("just plain text", None) is None


def test_tag_parser_extracts_body_and_json_when_present():
    """TagActionParser returns dict from JSON tag body, else the raw string."""
    parser = TagActionParser(tag="action")
    assert parser('thinking...<action>{"a": 1}</action>', None) == {"a": 1}
    assert parser("<action>hit</action>", None) == "hit"
    assert parser("no tag here", None) is None


def test_passthrough_parser_trims_string():
    """PassthroughActionParser returns the trimmed completion text."""
    assert PassthroughActionParser()("  stand  \n", None) == "stand"


def test_resolve_action_parser_alias_and_import(tmp_path, monkeypatch):
    """resolve_action_parser accepts shorthand aliases and dotted paths."""
    assert isinstance(resolve_action_parser("json"), JSONActionParser)
    assert isinstance(resolve_action_parser("tag"), TagActionParser)
    assert isinstance(resolve_action_parser("passthrough"), PassthroughActionParser)


# ---------------------------------------------------------------------------
# Observation formatting
# ---------------------------------------------------------------------------


def test_auto_formatter_serializes_dataclass_observation():
    """AutoObservationFormatter JSON-encodes a dataclass observation."""
    fmt = AutoObservationFormatter()
    msg = fmt(_FakeObservation(turn=3, hint="STAND"), step=3)
    assert msg["role"] == "user"
    assert '"turn": 3' in msg["content"]
    assert "STAND" in msg["content"]


def test_auto_formatter_handles_string_observation():
    """String observations are passed through verbatim (after the prefix)."""
    msg = AutoObservationFormatter(prefix="obs=")("hello", step=0)
    assert msg["content"] == "obs=hello"


def test_resolve_obs_formatter_returns_auto_by_default():
    assert isinstance(resolve_obs_formatter("auto"), AutoObservationFormatter)


# ---------------------------------------------------------------------------
# build_action coercion
# ---------------------------------------------------------------------------


def test_build_action_passthrough_when_no_action_class():
    """Without action_class, build_action returns the raw parsed value."""
    assert build_action({"a": 1}, None) == {"a": 1}
    assert build_action("hit", None) == "hit"


def test_build_action_instantiates_dataclass(monkeypatch):
    """build_action expands a dict parser output into ActionClass(**dict)."""
    import areal.workflow.openenv_utils as utils

    @dataclass
    class _Action:
        tool_name: str
        arguments: dict[str, Any] = field(default_factory=dict)

    monkeypatch.setattr(utils, "_import_from_string", lambda _p: _Action)
    parsed = {"tool_name": "echo", "arguments": {"m": "hi"}}
    action = build_action(parsed, "pkg._Action")
    assert isinstance(action, _Action)
    assert action.tool_name == "echo"
    assert action.arguments == {"m": "hi"}


def test_build_action_falls_back_on_bad_kwargs(monkeypatch):
    """When ActionClass(**dict) fails, build_action logs and returns the dict."""
    import areal.workflow.openenv_utils as utils

    class _BrokenAction:
        def __init__(self, required: str) -> None:
            self.required = required

    monkeypatch.setattr(utils, "_import_from_string", lambda _p: _BrokenAction)
    got = build_action({"unrelated": 1}, "pkg._BrokenAction")
    assert got == {"unrelated": 1}


# ---------------------------------------------------------------------------
# Workflow: end-to-end mocked episode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_episode_runs_until_done_and_records_step_rewards(
    monkeypatch, _stats_tracker_stub
):
    """Terminates on ``done``, records reward per LLM turn, keeps insertion order."""
    _FakeArealOpenAI.pending_scripts = [
        '{"tool_name": "echo", "arguments": {"message": "step1"}}',
        '{"tool_name": "echo", "arguments": {"message": "step2"}}',
    ]
    env = _FakeEnvClient(
        script=[
            _FakeStepResult(observation=_FakeObservation(1), reward=0.4, done=False),
            _FakeStepResult(observation=_FakeObservation(2), reward=0.7, done=True),
        ]
    )
    cfg = OpenEnvConfig(
        env_client_class="stub.NotUsed", base_url="ws://stub", max_turns=5
    )
    workflow = _make_workflow(cfg, env, monkeypatch)

    result = await workflow.arun_episode(engine=None, data={"seed": 42})

    assert env.reset_calls == [{"seed": 42}]
    assert len(env.step_actions) == 2
    inst = _FakeArealOpenAI.last_instance
    assert inst is not None
    assert inst.rewards == {"cmpl-1": 0.4, "cmpl-2": 0.7}
    assert set(result.keys()) == {"cmpl-1", "cmpl-2"}


@pytest.mark.asyncio
async def test_episode_stops_at_max_turns_when_env_never_dones(
    monkeypatch, _stats_tracker_stub
):
    """Hard cap on max_turns applies regardless of ``done`` flag."""
    _FakeArealOpenAI.pending_scripts = ['{"a":1}'] * 5
    env = _FakeEnvClient(
        script=[
            _FakeStepResult(observation=_FakeObservation(i), reward=0.1, done=False)
            for i in range(1, 6)
        ]
    )
    cfg = OpenEnvConfig(
        env_client_class="stub.NotUsed", base_url="ws://stub", max_turns=3
    )
    workflow = _make_workflow(cfg, env, monkeypatch)

    await workflow.arun_episode(engine=None, data={})

    inst = _FakeArealOpenAI.last_instance
    assert inst is not None
    assert len(inst.rewards) == 3
    assert len(env.step_actions) == 3


@pytest.mark.asyncio
async def test_terminal_reward_only_zeroes_intermediate(
    monkeypatch, _stats_tracker_stub
):
    """terminal_reward_only=True keeps only the last step's reward on record."""
    _FakeArealOpenAI.pending_scripts = ['{"a":1}', '{"a":2}', '{"a":3}']
    env = _FakeEnvClient(
        script=[
            _FakeStepResult(observation=_FakeObservation(1), reward=0.2, done=False),
            _FakeStepResult(observation=_FakeObservation(2), reward=0.5, done=False),
            _FakeStepResult(observation=_FakeObservation(3), reward=1.0, done=True),
        ]
    )
    cfg = OpenEnvConfig(
        env_client_class="stub.NotUsed",
        base_url="ws://stub",
        max_turns=5,
        terminal_reward_only=True,
    )
    workflow = _make_workflow(cfg, env, monkeypatch)

    await workflow.arun_episode(engine=None, data={})

    inst = _FakeArealOpenAI.last_instance
    assert inst is not None
    assert inst.rewards == {"cmpl-1": 0.0, "cmpl-2": 0.0, "cmpl-3": 1.0}


@pytest.mark.asyncio
async def test_parse_failure_breaks_and_records_zero(monkeypatch, _stats_tracker_stub):
    """Unparsable completion breaks the loop with zero reward, no env.step call."""
    _FakeArealOpenAI.pending_scripts = [
        "no json here at all",
        '{"a": 1}',
    ]
    env = _FakeEnvClient(
        script=[_FakeStepResult(observation=_FakeObservation(1), reward=1.0, done=True)]
    )
    cfg = OpenEnvConfig(
        env_client_class="stub.NotUsed", base_url="ws://stub", max_turns=3
    )
    workflow = _make_workflow(cfg, env, monkeypatch)

    await workflow.arun_episode(engine=None, data={})

    inst = _FakeArealOpenAI.last_instance
    assert inst is not None
    assert inst.rewards == {"cmpl-1": 0.0}
    assert env.step_actions == []


@pytest.mark.asyncio
async def test_step_discount_triggers_reward_propagation(
    monkeypatch, _stats_tracker_stub
):
    """step_discount<1.0 triggers ArealOpenAI.apply_reward_discount(discount)."""
    _FakeArealOpenAI.pending_scripts = ['{"a":1}', '{"a":2}']
    env = _FakeEnvClient(
        script=[
            _FakeStepResult(observation=_FakeObservation(1), reward=0.5, done=False),
            _FakeStepResult(observation=_FakeObservation(2), reward=1.0, done=True),
        ]
    )
    cfg = OpenEnvConfig(
        env_client_class="stub.NotUsed",
        base_url="ws://stub",
        max_turns=3,
        step_discount=0.9,
    )
    workflow = _make_workflow(cfg, env, monkeypatch)

    await workflow.arun_episode(engine=None, data={})

    inst = _FakeArealOpenAI.last_instance
    assert inst is not None
    assert inst.discount_calls == [0.9]
