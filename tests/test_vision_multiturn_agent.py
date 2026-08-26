# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the multi-turn VLM agent's turn loop and reward accounting.

The agent talks to the proxy over the OpenAI SDK, so the client is stubbed and
only the agent's own decisions -- when to stop, and what scalar reward to hand
back to the proxy -- are exercised here.
"""

from types import SimpleNamespace

import pytest
from PIL import Image

from areal.workflow.vision_env import EnvResetResult, EnvStepResult, MultiTurnVisionEnv


class _ScriptedEnv(MultiTurnVisionEnv):
    """Env that replays a fixed list of per-turn rewards."""

    rewards: list[float] = []
    dones: list[bool] = []

    def reset(self, data):
        self.turn = 0
        return EnvResetResult(
            messages_chat=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": ""}},
                        {"type": "text", "text": "find x"},
                    ],
                }
            ],
            images=[Image.new("RGB", (8, 8), (12, 34, 56))],
        )

    def step(self, assistant_text: str) -> EnvStepResult:
        reward = self.rewards[self.turn]
        done = self.dones[self.turn]
        self.turn += 1
        return EnvStepResult(reward=reward, done=done, observation="try again")

    def get_metrics(self):
        return {}


def _make_env_class(rewards, dones):
    return type("_Env", (_ScriptedEnv,), {"rewards": rewards, "dones": dones})


class _StubCompletions:
    async def create(self, **kwargs):
        message = SimpleNamespace(
            content="x = 12", model_dump=lambda **kw: {"role": "assistant"}
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(total_tokens=64),
        )


class _StubAsyncOpenAI:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=_StubCompletions())


@pytest.fixture
def run_agent(monkeypatch):
    """Run the agent against a scripted env and a stubbed proxy client."""
    import asyncio

    from areal.workflow import vision_multiturn

    monkeypatch.setattr(vision_multiturn, "AsyncOpenAI", _StubAsyncOpenAI)

    def _run(rewards, dones, max_turns=2):
        env_cls = _make_env_class(rewards, dones)
        monkeypatch.setattr(vision_multiturn, "import_from_string", lambda _: env_cls)
        agent = vision_multiturn.VisionMultiTurnAgent(
            env_factory="tests.stub.Env",
            max_turns=max_turns,
            max_completion_tokens=16,
        )
        return asyncio.run(agent.run({}))

    return _run


@pytest.mark.parametrize(
    "rewards,dones,expected",
    [
        ([-1.0], [True], -1.0),  # penalty-only env must not be clipped to zero
        ([-1.0, -0.5], [False, True], -0.5),  # least-bad turn wins
        ([0.0, 1.0], [False, True], 1.0),  # success on the final turn
        ([1.0], [True], 1.0),  # immediate success
    ],
)
def test_agent_reward_preserves_negative_outcomes(run_agent, rewards, dones, expected):
    """Test that the episode reward is the best turn reward, negatives included."""
    assert run_agent(rewards, dones) == pytest.approx(expected)


def test_agent_stops_when_the_env_signals_done(run_agent):
    """Test that a done on turn 0 does not consume the second turn's reward."""
    # Turn 1 would score 1.0, but the env terminates first, so it must not run.
    assert run_agent([0.25, 1.0], [True, True]) == pytest.approx(0.25)
