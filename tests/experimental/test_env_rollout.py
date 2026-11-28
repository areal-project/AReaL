# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest

from areal.experimental.env.openenv import OpenEnvBackend
from areal.experimental.env.protocol import StepResult
from areal.experimental.env.rollout import EnvRolloutRejected, run_environment


class FakeSession:
    def __init__(self, steps):
        self.steps = iter(steps)
        self.closed = False
        self.actions = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        await self.aclose()

    async def aclose(self):
        self.closed = True

    async def reset(self, seed=None):
        self.seed = seed
        return 0

    async def step(self, action):
        self.actions.append(action)
        return next(self.steps)


class FakeBackend:
    def __init__(self, steps):
        self.session = FakeSession(steps)

    async def acquire(self, task_data):
        self.task_data = task_data
        return self.session


async def decide(observation):
    return f"turn-{observation}", observation + 10


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, missing_policy, expected",
    [
        ("episode", "reject", 0.25),
        ("dense", "reject", None),
        ("dense", "ignore", {"turn-1": 0.0, "turn-2": 0.25}),
        ("dense", "zero", {"turn-0": 0.0, "turn-1": 0.0, "turn-2": 0.25}),
        ("dense", "terminal_fallback", {"turn-0": 0.25, "turn-1": 0.0, "turn-2": 0.25}),
    ],
)
async def test_run_environment_assigns_rewards_by_completion(
    mode, missing_policy, expected
):
    backend = FakeBackend(
        [StepResult(1), StepResult(2, reward=0.0), StepResult(3, 0.25, terminated=True)]
    )
    data = {"task": "sample"}

    result = await run_environment(
        backend,
        data,
        decide,
        max_turns=5,
        reward_mode=mode,
        missing_reward_policy=missing_policy,
        seed=42,
    )

    assert result.reward == expected
    assert result.terminated and not result.truncated
    assert result.turns == 3
    assert backend.session.actions == [10, 11, 12]
    assert backend.session.seed == 42
    assert backend.task_data is data
    assert backend.session.closed


@pytest.mark.asyncio
async def test_run_environment_ignores_all_missing_dense_rewards():
    """An episode with no assignments still has a valid proxy reward map."""
    backend = FakeBackend([StepResult(1), StepResult(2, terminated=True)])

    result = await run_environment(
        backend,
        {},
        decide,
        max_turns=2,
        reward_mode="dense",
        missing_reward_policy="ignore",
    )

    assert result.terminated
    assert result.proxy_reward() == {}
    assert backend.session.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("native_stop", [True, False])
@pytest.mark.parametrize(
    "policy, last_reward, expected",
    [
        ("drop", 0.5, None),
        ("retain_terminal", 0.5, {"turn-0": 0.25, "turn-1": 0.5}),
        ("retain_terminal", None, None),
        ("retain_zero", None, {"turn-0": 0.25, "turn-1": 0.0}),
        ("retain_zero", 0.5, {"turn-0": 0.25, "turn-1": 0.0}),
    ],
)
async def test_run_environment_applies_truncation_policy(
    native_stop, policy, last_reward, expected
):
    backend = FakeBackend(
        [
            StepResult(1, 0.25),
            StepResult(
                2,
                last_reward,
                truncated=native_stop,
                stop_reason="time_limit" if native_stop else None,
            ),
        ]
    )

    result = await run_environment(
        backend,
        {},
        decide,
        max_turns=4 if native_stop else 2,
        reward_mode="dense",
        missing_reward_policy="zero",
        truncation_policy=policy,
    )

    assert result.truncated and not result.terminated
    assert result.stop_reason == ("time_limit" if native_stop else "max_turns")
    if expected is None:
        with pytest.raises(EnvRolloutRejected, match=result.stop_reason):
            result.proxy_reward()
    else:
        assert result.proxy_reward() == expected
    assert backend.session.closed


@pytest.mark.asyncio
async def test_run_environment_closes_session_on_cancellation():
    backend = FakeBackend([])
    waiting = asyncio.Event()

    async def cancelled_decision(_observation):
        waiting.set()
        await asyncio.Future()

    task = asyncio.create_task(
        run_environment(backend, {}, cancelled_decision, max_turns=1)
    )
    await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert backend.session.closed


@pytest.mark.asyncio
async def test_run_environment_propagates_backend_error_and_closes_session():
    backend = FakeBackend([StepResult(1, terminated=True, error="lost session")])

    with pytest.raises(RuntimeError, match="lost session"):
        await run_environment(backend, {}, decide, max_turns=1)

    assert backend.session.closed


@pytest.mark.asyncio
async def test_openenv_backend_owns_distinct_clients_and_maps_done():
    closed = []

    class Client:
        def __init__(self, task):
            self.task = task

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, *_exc):
            closed.append((self.task, exc_type))

        async def reset(self, **kwargs):
            return SimpleNamespace(observation=kwargs, done=False)

        async def step(self, action):
            return SimpleNamespace(
                observation=action,
                reward=2.0,
                done=True,
                metadata={"backend": "openenv"},
            )

    backend = OpenEnvBackend(lambda task: Client(task["name"]))
    session_a = await backend.acquire({"name": "a"})
    session_b = await backend.acquire({"name": "b"})

    async with session_a, session_b:
        assert await session_a.reset(seed=7) == {"seed": 7}
        assert await session_b.reset() == {}
        assert await session_b.step("move") == StepResult(
            "move", reward=2.0, terminated=True, info={"backend": "openenv"}
        )
        await session_a.aclose()

    assert closed == [("a", None), ("b", None)]


@pytest.mark.asyncio
async def test_openenv_backend_closes_client_when_connect_is_cancelled():
    connecting = asyncio.Event()
    closed = asyncio.Event()

    class Client:
        async def __aenter__(self):
            connecting.set()
            await asyncio.Future()

        async def __aexit__(self, *_exc):
            pytest.fail("The client context did not finish entering")

        async def close(self):
            closed.set()

    backend = OpenEnvBackend(lambda _task: Client())
    task = asyncio.create_task(run_environment(backend, {}, decide, max_turns=1))
    await connecting.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
