import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import torch
import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "examples/hermes/config.yaml"


def test_hermes_math_tool_captures_real_python_output():
    from examples.task_workflows.environments.python import run_limited

    result = run_limited(
        [sys.executable, "-c", "print(6 * 7)"], timeout=10, output_bytes=1024
    )
    assert result["status"] == "completed"
    assert result["exit_code"] == 0
    assert result["output"].strip() == "42"


def test_hermes_math_sampling_keeps_episode_on_direct_and_recreated_clients(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from examples.task_workflows import service
    from examples.task_workflows.environments import python as python_environment
    from examples.task_workflows.harnesses import hermes, hermes_runtime
    from examples.task_workflows.harnesses.hermes import _Upstream

    requests = []
    clock = SimpleNamespace(value=0.0)

    def chunks():
        for text in ("hello", "world"):
            clock.value += 2.0
            yield text

    def client():
        def create(**kwargs):
            requests.append(kwargs)
            clock.value += 1.0
            return chunks() if kwargs.get("stream") else kwargs

        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

    class Agent:
        def __init__(self, **kwargs):
            assert kwargs["enabled_toolsets"] == [
                python_environment.PythonSandbox.toolset
            ]
            self.request_overrides = kwargs.get("request_overrides", {})
            self.client = client()

        def _create_openai_client(self, *args, **kwargs):
            return client()

    module = ModuleType("run_agent")
    module.AIAgent = Agent
    monkeypatch.setitem(sys.modules, "run_agent", module)
    monkeypatch.setenv("AREAL_TASK_PYTHON_IMAGE", "local-python-image")
    monkeypatch.setenv("AREAL_TASK_TOOL_ROOT", str(tmp_path))
    monkeypatch.setenv("AREAL_TASK_MAX_TOKENS", "123")
    monkeypatch.setattr(
        hermes,
        "time",
        SimpleNamespace(time=lambda: clock.value, perf_counter=lambda: clock.value),
    )
    monkeypatch.setattr(python_environment, "time", hermes.time)
    monkeypatch.setattr(hermes_runtime, "time", hermes.time)
    monkeypatch.setattr(service.importlib.metadata, "version", lambda name: "0.19.0")
    monkeypatch.setattr(service, "register_environment_tools", lambda provider: None)
    monkeypatch.setattr(python_environment.shutil, "which", lambda name: "/bin/docker")
    monkeypatch.setattr(
        python_environment.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="sha256:test-image\n"),
    )
    identity = {"incarnation_id": "math-source", "trajectory_id": 2}
    adapter = service.TaskAgent()
    agent = adapter._build_agent(
        _Upstream("http://inference/v1", "session-key", "test", identity)
    )
    agent.client.chat.completions.create(model="test", messages=[])
    stream = agent._create_openai_client().chat.completions.create(
        model="test", messages=[], stream=True
    )
    assert agent._areal_inference_calls[1]["finished"] is None
    assert list(stream) == ["hello", "world"]
    stream.close()
    assert [call["seconds"] for call in agent._areal_inference_calls] == [1.0, 5.0]
    tool = SimpleNamespace(call_timings=[])

    def execute(code):
        clock.value += 4.0
        return "42"

    tool.execute = execute
    tool.invoke = lambda name, arguments: python_environment.PythonSandbox.invoke(
        tool, name, arguments
    )
    monkeypatch.setitem(hermes._runtimes, "timed-task", tool)
    assert (
        hermes._dispatch_tool(
            "math_python", {"code": "print(6 * 7)"}, task_id="timed-task"
        )
        == "42"
    )
    assert tool.call_timings[0]["seconds"] == 4.0
    assert len(requests) == 2
    for request in requests:
        assert request["extra_body"]["areal_episode"] == identity
        assert request["max_tokens"] == 123
        assert request["temperature"] == request["top_p"] == 1.0


@pytest.mark.asyncio
async def test_hermes_math_client_scores_and_submits_native_reward(tmp_path):
    import httpx

    from examples.task_workflows.benchmarks.dapo_math import MathTask, question_id
    from examples.task_workflows.client import ClientSettings, EpisodeClient

    output = tmp_path / "client"
    output.mkdir()
    identity = {"incarnation_id": "math-incarnation", "trajectory_id": 0}
    session_id = "math-session"
    observed = []

    def respond(request):
        body = json.loads(request.content)
        observed.append((request.url.path, body))
        if request.url.path == "/rl/start_session":
            return httpx.Response(
                201,
                json={
                    "sessions": [
                        {
                            "session_id": session_id,
                            "session_api_key": "session-secret",
                            "areal_episode": identity,
                        }
                    ]
                },
            )
        if request.url.path == "/v1/responses":
            assert "gold_answer" not in request.content.decode()
            assert "42" not in body["input"][0]["content"]
            assert body["inf_episode"] == identity
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": r"\boxed{42}"}],
                        }
                    ],
                    "metadata": {
                        "completed": True,
                        "api_calls": 1,
                        "environment": {"calls": []},
                    },
                },
            )
        if request.url.path == "/rl/set_reward":
            assert body == {"reward": 1.0, "areal_episode": identity}
            return httpx.Response(
                200,
                json={
                    "trajectory_id": 0,
                    "trajectory_ready": True,
                    "interaction_count": 1,
                },
            )
        if request.url.path == "/sessions/close":
            return httpx.Response(200, json={"status": "closed"})
        raise AssertionError(request.url.path)

    settings = ClientSettings(
        agent_url="http://agent",
        inference_url="http://inference",
        agent_key="agent-secret",
        inference_key="inference-secret",
        model="test-model",
        mode="resident",
        output=output,
        concurrency=1,
    )
    question = "Compute six times seven."
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        summary = await EpisodeClient(settings, http).run(
            [MathTask(question_id(question), question, "42")]
        )
    assert summary["completed"] == summary["correct"] == 1
    row = json.loads((output / "episodes.jsonl").read_text())
    assert row["reward_receipt"]["trajectory_ready"]
    assert "session-secret" not in json.dumps(row)
    assert [path for path, _ in observed] == [
        "/rl/start_session",
        "/v1/responses",
        "/rl/set_reward",
        "/sessions/close",
    ]


def test_hermes_math_config_shares_training_settings_with_elastic_enabled(
    monkeypatch, tmp_path
):
    from omegaconf import OmegaConf

    path = _CONFIG_PATH.parent.parent / "task_workflows/configs/dapo.yaml"
    for key, value in {
        "AREAL_TASK_RUN_ROOT": str(tmp_path / "runs"),
        "AREAL_TASK_STATE_ROOT": str(tmp_path / "state"),
        "AREAL_MODEL_PATH": str(tmp_path / "model"),
        "AREAL_INFERENCE_ADMIN_KEY": "inference-test",
        "AREAL_ACTOR_ADMIN_KEY": "actor-test",
        "AREAL_CHECKPOINT_ESTIMATED_BYTES": str(80 << 30),
    }.items():
        monkeypatch.setenv(key, value)
    configs = []
    for enabled in (False, True):
        monkeypatch.setenv("AREAL_ELASTIC_ENABLED", str(enabled).lower())
        config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        assert config["elastic"].pop("enabled") is enabled
        assert config["train_dataset"]["batch_size"] == 4
        assert config["rollout"]["consumer_batch_size"] == 4
        assert config["rollout"]["agent"]["set_reward_finish_timeout"] == 0
        configs.append(config)
    assert configs[0] == configs[1]


@pytest.mark.asyncio
async def test_hermes_math_eval_does_not_create_or_reward_training_sessions(tmp_path):
    import httpx

    from examples.task_workflows.benchmarks.dapo_math import MathTask, question_id
    from examples.task_workflows.client import ClientSettings, EpisodeClient

    paths = []

    def respond(request):
        paths.append(request.url.path)
        body = json.loads(request.content)
        if request.url.path == "/v1/responses":
            assert body["session_api_key"] == "inference-admin"
            assert "inf_episode" not in body
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": r"\boxed{42}"}],
                        }
                    ],
                    "metadata": {
                        "completed": True,
                        "api_calls": 1,
                        "inference_calls": [
                            {"started": 10.0, "finished": 13.0, "seconds": 3.0}
                        ],
                        "environment": {
                            "timing_calls": [
                                {"started": 12.0, "finished": 14.0, "seconds": 2.0}
                            ],
                            "calls": [
                                {
                                    "status": "completed",
                                    "started": 13.0,
                                    "finished": 14.0,
                                    "seconds": 0.5,
                                }
                            ],
                        },
                    },
                },
            )
        assert request.url.path == "/sessions/close"
        return httpx.Response(200, json={"status": "closed"})

    question = "Compute six times seven."
    settings = ClientSettings(
        agent_url="http://agent",
        inference_url="http://inference",
        agent_key="agent-admin",
        inference_key="inference-admin",
        model="test-model",
        mode="eval",
        output=tmp_path,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        result = await EpisodeClient(settings, http).run_task(
            MathTask(question_id(question), question, "42")
        )
    assert result["score"]["reward"] == 1
    assert result["timing"]["total_seconds"] >= 0
    assert result["timing"]["inference_seconds"] == 3.0
    assert result["timing"]["tool_seconds"] == 2.0
    assert paths == ["/v1/responses", "/sessions/close"]


def _load_config() -> dict:
    return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))


def test_hermes_singleton_online_config_preserves_each_outcome_signal():
    from areal.api.cli_args import PPOActorConfig

    config = _load_config()
    actor = config["actor"]

    assert config["gconfig"]["n_samples"] == 1
    assert config["train_dataset"]["batch_size"] == 1
    assert actor.get("reward_norm") is None
    assert actor.get("adv_norm") is None

    counterfactual_outcomes = torch.tensor([0.0, 1.0])
    actor_defaults = PPOActorConfig()
    reward_bias = actor.get("reward_bias", actor_defaults.reward_bias)
    reward_scaling = actor.get("reward_scaling", actor_defaults.reward_scaling)
    signed_rewards = (counterfactual_outcomes + reward_bias) * reward_scaling

    for signed_reward in signed_rewards:
        single_trajectory_token_advantages = signed_reward.expand(3)
        assert torch.count_nonzero(single_trajectory_token_advantages).item() == 3

    assert signed_rewards[0] < 0 < signed_rewards[1]
    assert not torch.equal(signed_rewards[0], signed_rewards[1])


@pytest.mark.parametrize("on_demand", [False, True])
def test_hermes_carries_episode_identity_into_every_llm_request(monkeypatch, on_demand):
    from examples.hermes.hermes import HermesAgent, _Upstream

    module = ModuleType("run_agent")
    module.AIAgent = MagicMock()
    monkeypatch.setitem(sys.modules, "run_agent", module)
    metadata = {
        "base_url": "http://inference",
        "api_key": "session-key",
        "model": "Qwen",
    }
    if on_demand:
        metadata["episode"] = {"incarnation_id": "source", "trajectory_id": 3}
    upstream = _Upstream.from_inference(metadata)
    HermesAgent()._build_agent(upstream)
    kwargs = module.AIAgent.call_args.kwargs
    if on_demand:
        assert kwargs["request_overrides"] == {
            "extra_body": {"areal_episode": metadata["episode"]}
        }
        assert (
            _Upstream.from_inference(
                {
                    **metadata,
                    "episode": {"incarnation_id": "source", "trajectory_id": 4},
                }
            )
            != upstream
        )
    else:
        assert "request_overrides" not in kwargs


@pytest.mark.parametrize("recreated", [False, True])
def test_hermes_direct_summary_and_recreated_clients_keep_bound_episode(recreated):
    from types import SimpleNamespace

    import httpx
    from openai import OpenAI

    from examples.hermes.hermes import _bind_episode_requests

    requests = []
    clients = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "summary",
                "object": "chat.completion",
                "created": 0,
                "model": "default",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    def create_client(*args, **kwargs):
        client = OpenAI(
            api_key="test-session",
            base_url="http://inference/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        )
        clients.append(client)
        return client

    identity = {"incarnation_id": "source", "trajectory_id": 3}
    agent = SimpleNamespace(client=create_client(), _create_openai_client=create_client)
    _bind_episode_requests(agent, identity)
    identity["trajectory_id"] = 99
    client = (
        agent._create_openai_client({}, reason="summary", shared=True)
        if recreated
        else agent.client
    )
    try:
        result = client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": "summarize"}],
            extra_body={"other_metadata": "preserved"},
        )
        assert result.choices[0].message.content == "done"
        assert requests == [
            {
                "model": "default",
                "messages": [{"role": "user", "content": "summarize"}],
                "other_metadata": "preserved",
                "areal_episode": {"incarnation_id": "source", "trajectory_id": 3},
            }
        ]
        with pytest.raises(ValueError, match="conflicts"):
            client.chat.completions.create(
                model="default",
                messages=[],
                extra_body={"areal_episode": identity},
            )
        assert len(requests) == 1
    finally:
        for client in clients:
            client.close()


@pytest.mark.asyncio
async def test_hermes_replaced_episode_cannot_run_a_closed_instance():
    from types import SimpleNamespace

    from examples.hermes.hermes import HermesAgent

    agent = HermesAgent()
    old = SimpleNamespace(session_key="session", agent=MagicMock())
    agent._sessions["session"] = SimpleNamespace(
        session_key="session", agent=MagicMock()
    )
    with pytest.raises(RuntimeError, match="replaced"):
        await agent._run_conversation(old, "late", [])
    old.agent.run_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_hermes_direct_turn_cannot_inherit_cached_training_identity():
    from types import SimpleNamespace

    from examples.hermes.hermes import HermesAgent, _Upstream

    from areal.v2.agent_service.types import AgentRequest

    agent = HermesAgent()
    agent._sessions["session"] = SimpleNamespace(
        upstream=_Upstream(
            "http://inference",
            "key",
            "Qwen",
            {"incarnation_id": "source", "trajectory_id": 1},
        )
    )
    with pytest.raises(RuntimeError, match="requires an episode identity"):
        await agent._resolve_state(AgentRequest("delayed", "session", "old"))


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["group", "history", "missing_budget"])
async def test_gsm8k_agent_rejects_changed_episode_semantics_before_inference(
    monkeypatch, invalid
):
    from examples.hermes.hermes import GSM8KAgent

    from areal.v2.agent_service.types import AgentRequest
    from areal.v2.agent_service.worker.app import _CollectingEmitter

    client = MagicMock()
    monkeypatch.setattr("openai.AsyncOpenAI", client)
    metadata = {
        "areal_inference": {"base_url": "http://inference", "api_key": "key"},
        "generation": {"max_tokens": 128},
    }
    history = []
    if invalid == "group":
        metadata["generation"]["n"] = 2
    elif invalid == "history":
        history = [{"role": "user", "content": "previous problem"}]
    else:
        metadata["generation"] = {}
    with pytest.raises(ValueError):
        await GSM8KAgent().run(
            AgentRequest(
                "question", "session", "run", history=history, metadata=metadata
            ),
            emitter=_CollectingEmitter(),
        )
    client.assert_not_called()
