"""Unit tests for scaffolding controllers (TraceTrajectoryMaker, PipelineTrajectoryMaker).

Tests use fake workers/controllers that simulate LLM inference responses
without requiring an SGLang backend or GPU.

Design Notes
------------
- ``FakeGenerationController`` fills ``GenerationTask`` fields in-memory
  (single-turn generation).
- ``FakeChatRolloutController`` appends assistant messages to ``ChatTask``
  across multiple yields (multi-turn chat). It manually calls
  ``ChatTracer.before_yield / after_yield`` because the lightweight
  ``with_task_collection`` decorator in ``_compat.py`` only attaches the
  ``ChatTracer`` instance to the class; it does NOT wrap ``process`` to
  invoke hooks automatically (that is the tensorrt_llm implementation's
  responsibility).
- ``FakeChatRewardController`` assigns predetermined rewards to
  ``ChatRewardTask`` objects.
- The lightweight ``with_task_collection`` creates the ``ChatTracer`` as a
  **class-level** attribute, so all ``TraceTrajectoryMaker`` instances share
  one tracer.  Each test that uses it must call ``tracer.clear()`` (or use a
  fresh instance) to avoid inter-test pollution.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from examples.scaffolding._compat import (
    AssistantMessage,
    ChatTask,
    Controller,
    GenerationTask,
    Task,
)
from examples.scaffolding.controllers import (
    ChatTracer,
    PipelineTrajectoryMaker,
    RLVRRewardController,
    TraceTrajectoryMaker,
)
from examples.scaffolding.task import (
    ChatRewardTask,
    RLVRRewardTask,
    TraceGenerationTask,
)

from areal.api.io_struct import ModelResponse
from areal.experimental.openai.types import InteractionWithTokenLogpReward

# ---------------------------------------------------------------------------
# Fake / stub helpers
# ---------------------------------------------------------------------------

FAKE_INPUT_TOKENS = [101, 102, 103]
FAKE_OUTPUT_TOKENS = [201, 202, 203, 204]
FAKE_OUTPUT_STR = "42"
FAKE_PROMPT_STR = "What is the answer to life?"


def _simple_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    **kwargs,
) -> float:
    """Deterministic reward: 1.0 if completion contains the ground-truth answer, else 0.0."""
    answer = kwargs.get("answer", "")
    return 1.0 if answer and answer in completions else 0.0


class FakeGenerationController(Controller):
    """Fills ``GenerationTask`` fields without calling any LLM backend."""

    def __init__(
        self,
        output_str: str = FAKE_OUTPUT_STR,
        output_tokens: list[int] | None = None,
        input_tokens: list[int] | None = None,
        output_logprobs: list[float] | None = None,
        output_versions: list[int] | None = None,
    ):
        super().__init__()
        self.output_str = output_str
        self.output_tokens = output_tokens or FAKE_OUTPUT_TOKENS
        self.input_tokens = input_tokens or FAKE_INPUT_TOKENS
        self.output_logprobs = output_logprobs
        self.output_versions = output_versions

    def process(self, tasks: list[Task], **kwargs) -> Any:
        for task in tasks:
            if isinstance(task, GenerationTask):
                task.output_str = self.output_str
                task.output_tokens = self.output_tokens
                if task.input_tokens is None:
                    task.input_tokens = self.input_tokens
                if self.output_logprobs is not None:
                    task.customized_result_fields["output_logprobs"] = (
                        self.output_logprobs
                    )
                if self.output_versions is not None:
                    task.customized_result_fields["output_versions"] = (
                        self.output_versions
                    )
        yield tasks


def _make_fake_completion(completion_id: str = "cmpl-001") -> MagicMock:
    """Create a minimal fake ``ChatCompletion`` object."""
    completion = MagicMock()
    completion.id = completion_id
    completion.created = 1000
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = FAKE_OUTPUT_STR
    completion.choices[0].finish_reason = "stop"
    return completion


class FakeChatRolloutController(Controller):
    """Simulates a multi-turn chat rollout by appending assistant messages.

    Each turn:
    1. Populates the ``ChatTask`` with a fake completion and tokens.
    2. Calls ``ChatTracer.before_yield`` / ``after_yield`` manually.
    3. Yields the tasks.
    4. Appends a follow-up user message before the next turn.
    """

    def __init__(
        self,
        n_turns: int = 2,
        responses: list[str] | None = None,
        output_tokens_per_turn: list[list[int]] | None = None,
        input_tokens_per_turn: list[list[int]] | None = None,
    ):
        super().__init__()
        self.n_turns = n_turns
        self.responses = responses or [f"response_{i}" for i in range(n_turns)]
        self.output_tokens_per_turn = output_tokens_per_turn or [
            [300 + i * 10 + j for j in range(4)] for i in range(n_turns)
        ]
        self.input_tokens_per_turn = input_tokens_per_turn or [
            FAKE_INPUT_TOKENS for _ in range(n_turns)
        ]
        # Set by the test to allow manual ChatTracer hook invocation.
        self._tracer: ChatTracer | None = None

    def process(self, tasks: list[Task], **kwargs) -> Any:
        for turn_idx in range(self.n_turns):
            for task in tasks:
                if isinstance(task, ChatTask):
                    task.messages.append(
                        AssistantMessage(content=self.responses[turn_idx])
                    )
                    task.output_tokens = self.output_tokens_per_turn[turn_idx]
                    task.input_tokens = self.input_tokens_per_turn[turn_idx]
                    task.completion = _make_fake_completion(
                        completion_id=f"cmpl-turn-{turn_idx}"
                    )

            if self._tracer is not None:
                self._tracer.before_yield(tasks)

            yield tasks

            if self._tracer is not None:
                self._tracer.after_yield(tasks)

            if turn_idx < self.n_turns - 1:
                for task in tasks:
                    if isinstance(task, ChatTask):
                        task.messages.append(
                            {"role": "user", "content": f"follow-up-{turn_idx}"}
                        )


class FakeChatRewardController(Controller):
    """Assigns predetermined rewards to ``ChatRewardTask`` objects."""

    def __init__(self, rewards: list[float] | None = None, default_reward: float = 1.0):
        super().__init__()
        self.rewards = rewards
        self.default_reward = default_reward

    def process(self, tasks: list[Task], **kwargs) -> Any:
        for i, task in enumerate(tasks):
            if isinstance(task, ChatRewardTask):
                reward = (
                    self.rewards[i]
                    if self.rewards is not None and i < len(self.rewards)
                    else self.default_reward
                )
                task.reward = reward
                if task.interaction is not None:
                    task.interaction.reward = reward
        yield tasks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_shared_tracer():
    """Reset the class-level ChatTracer before each test.

    The lightweight ``with_task_collection`` stores a single ``ChatTracer``
    instance on ``TraceTrajectoryMaker`` (class-level).  We must clear it
    between tests so that cached interactions don't leak.
    """
    yield
    tracer = TraceTrajectoryMaker.task_collections.get("chat_tracer")
    if tracer is not None:
        tracer.clear()


# ===========================================================================
# PipelineTrajectoryMaker tests
# ===========================================================================


class TestPipelineTrajectoryMaker:
    """Tests for PipelineTrajectoryMaker (single-turn generation + reward)."""

    def test_basic_pipeline_correct_answer(self):
        """Pipeline should produce interaction with reward=1.0 for a correct answer."""
        gen_ctrl = FakeGenerationController(output_str="The answer is 42.")
        reward_ctrl = RLVRRewardController(_simple_reward_fn)

        maker = PipelineTrajectoryMaker(
            generation_controller=gen_ctrl,
            reward_controller=reward_ctrl,
            task_data={"answer": "42"},
            prompt_str=FAKE_PROMPT_STR,
        )

        task = GenerationTask(input_str=FAKE_PROMPT_STR, input_tokens=FAKE_INPUT_TOKENS)
        results = list(maker.process([task]))

        # Only generation yields (reward computed locally, no dict yield)
        assert len(results) == 1

        # Interactions stored on task
        interactions = task.customized_result_fields["interactions"]
        assert isinstance(interactions, dict)
        assert len(interactions) == 1

        interaction = list(interactions.values())[0]
        assert isinstance(interaction, InteractionWithTokenLogpReward)
        assert interaction.reward == 1.0

    def test_basic_pipeline_wrong_answer(self):
        """Pipeline should produce interaction with reward=0.0 for a wrong answer."""
        gen_ctrl = FakeGenerationController(output_str="I don't know")
        reward_ctrl = RLVRRewardController(_simple_reward_fn)

        maker = PipelineTrajectoryMaker(
            generation_controller=gen_ctrl,
            reward_controller=reward_ctrl,
            task_data={"answer": "42"},
            prompt_str=FAKE_PROMPT_STR,
        )

        task = GenerationTask(input_str=FAKE_PROMPT_STR, input_tokens=FAKE_INPUT_TOKENS)
        list(maker.process([task]))

        interactions = task.customized_result_fields["interactions"]
        interaction = list(interactions.values())[0]
        assert interaction.reward == 0.0

    def test_pipeline_multiple_tasks(self):
        """Pipeline should handle multiple GenerationTasks in a single batch."""
        gen_ctrl = FakeGenerationController(output_str="42")
        reward_ctrl = RLVRRewardController(_simple_reward_fn)

        maker = PipelineTrajectoryMaker(
            generation_controller=gen_ctrl,
            reward_controller=reward_ctrl,
            task_data={"answer": "42"},
            prompt_str=FAKE_PROMPT_STR,
        )

        tasks = [
            GenerationTask(input_str=FAKE_PROMPT_STR, input_tokens=FAKE_INPUT_TOKENS),
            GenerationTask(input_str="Another prompt", input_tokens=[111, 112]),
        ]
        list(maker.process(tasks))

        # Both tasks should have the same interactions dict
        interactions = tasks[0].customized_result_fields["interactions"]
        assert len(interactions) == 2
        for interaction in interactions.values():
            assert interaction.reward == 1.0

    def test_pipeline_interaction_has_model_response(self):
        """Each interaction should contain a valid ModelResponse with tokens."""
        gen_ctrl = FakeGenerationController(
            output_str="42",
            output_tokens=[201, 202],
            output_logprobs=[0.1, 0.2],
            output_versions=[1, 1],
        )
        reward_ctrl = RLVRRewardController(_simple_reward_fn)

        maker = PipelineTrajectoryMaker(
            generation_controller=gen_ctrl,
            reward_controller=reward_ctrl,
            task_data={"answer": "42"},
            prompt_str=FAKE_PROMPT_STR,
        )

        task = GenerationTask(input_str=FAKE_PROMPT_STR, input_tokens=FAKE_INPUT_TOKENS)
        list(maker.process([task]))

        interaction = list(task.customized_result_fields["interactions"].values())[0]
        mr = interaction.model_response
        assert isinstance(mr, ModelResponse)
        assert mr.input_tokens == FAKE_INPUT_TOKENS
        assert mr.output_tokens == [201, 202]
        assert mr.output_logprobs == [0.1, 0.2]
        assert mr.output_versions == [1, 1]

    def test_pipeline_uses_constructor_prompt_str(self):
        """Reward controller should receive the prompt_str provided at construction."""
        received_prompts = []

        def _capture(prompt, completions, prompt_ids, completion_ids, **kw):
            received_prompts.append(prompt)
            return 1.0

        gen_ctrl = FakeGenerationController(output_str="42")
        reward_ctrl = RLVRRewardController(_capture)

        maker = PipelineTrajectoryMaker(
            generation_controller=gen_ctrl,
            reward_controller=reward_ctrl,
            task_data={"answer": "42"},
            prompt_str="custom prompt",
        )

        task = GenerationTask(input_str="input str", input_tokens=FAKE_INPUT_TOKENS)
        list(maker.process([task]))

        assert received_prompts == ["custom prompt"]

    def test_pipeline_falls_back_to_input_str(self):
        """When prompt_str is empty, should fall back to task.input_str."""
        received_prompts = []

        def _capture(prompt, completions, prompt_ids, completion_ids, **kw):
            received_prompts.append(prompt)
            return 1.0

        gen_ctrl = FakeGenerationController(output_str="42")
        reward_ctrl = RLVRRewardController(_capture)

        maker = PipelineTrajectoryMaker(
            generation_controller=gen_ctrl,
            reward_controller=reward_ctrl,
            task_data={"answer": "42"},
            prompt_str="",
        )

        task = GenerationTask(
            input_str="fallback prompt", input_tokens=FAKE_INPUT_TOKENS
        )
        list(maker.process([task]))

        assert received_prompts[0] == "fallback prompt"

    def test_pipeline_reward_scores_tracked(self):
        """RLVRRewardController should track scores in self.scores."""
        gen_ctrl = FakeGenerationController(output_str="42")
        reward_ctrl = RLVRRewardController(_simple_reward_fn)

        maker = PipelineTrajectoryMaker(
            generation_controller=gen_ctrl,
            reward_controller=reward_ctrl,
            task_data={"answer": "42"},
            prompt_str=FAKE_PROMPT_STR,
        )

        tasks = [
            GenerationTask(input_str=FAKE_PROMPT_STR, input_tokens=FAKE_INPUT_TOKENS),
            GenerationTask(input_str=FAKE_PROMPT_STR, input_tokens=FAKE_INPUT_TOKENS),
        ]
        list(maker.process(tasks))

        assert reward_ctrl.scores == [1.0, 1.0]

    def test_pipeline_default_logprobs_and_versions(self):
        """When no logprobs/versions provided, interaction should use placeholders."""
        gen_ctrl = FakeGenerationController(
            output_str="42",
            output_tokens=[201, 202],
            # No logprobs/versions supplied
        )
        reward_ctrl = RLVRRewardController(_simple_reward_fn)

        maker = PipelineTrajectoryMaker(
            generation_controller=gen_ctrl,
            reward_controller=reward_ctrl,
            task_data={"answer": "42"},
            prompt_str=FAKE_PROMPT_STR,
        )

        task = GenerationTask(input_str=FAKE_PROMPT_STR, input_tokens=FAKE_INPUT_TOKENS)
        list(maker.process([task]))

        interaction = list(task.customized_result_fields["interactions"].values())[0]
        mr = interaction.model_response
        # Placeholders: [0.0] * output_len
        assert mr.output_logprobs == [0.0, 0.0]
        assert mr.output_versions == [-1, -1]


# ===========================================================================
# ChatTracer tests
# ===========================================================================


class TestChatTracer:
    """Tests for ChatTracer (TaskCollection for tracing multi-turn chats)."""

    def test_after_yield_creates_interaction(self):
        """after_yield should create an interaction for each ChatTask."""
        tracer = ChatTracer(export_style="individual")
        task = ChatTask(messages=[{"role": "user", "content": "hello"}])
        task.completion = _make_fake_completion("cmpl-001")
        task.input_tokens = [1, 2, 3]
        task.output_tokens = [4, 5]

        tracer.after_yield([task])

        results = tracer.get_trace_results()
        assert len(results) == 1
        assert "cmpl-001" in results

    def test_multiple_turns_traced(self):
        """Calling after_yield with different completions should trace all."""
        tracer = ChatTracer(export_style="individual")
        task = ChatTask(messages=[{"role": "user", "content": "hello"}])

        # Turn 1
        task.completion = _make_fake_completion("cmpl-turn-0")
        task.input_tokens = [1, 2]
        task.output_tokens = [3, 4]
        tracer.after_yield([task])

        # Turn 2 (same ChatTask, new completion)
        task.messages.append(AssistantMessage(content="first response"))
        task.completion = _make_fake_completion("cmpl-turn-1")
        task.input_tokens = [1, 2, 3, 4]
        task.output_tokens = [5, 6]
        tracer.after_yield([task])

        results = tracer.get_trace_results()
        assert len(results) == 2
        assert "cmpl-turn-0" in results
        assert "cmpl-turn-1" in results

    def test_tracer_interaction_has_model_response(self):
        """Traced interactions should have ModelResponse with correct tokens."""
        tracer = ChatTracer(export_style="individual")
        task = ChatTask(messages=[{"role": "user", "content": "hello"}])
        task.completion = _make_fake_completion("cmpl-001")
        task.input_tokens = [10, 20]
        task.output_tokens = [30, 40, 50]

        tracer.after_yield([task])

        interaction = tracer.get_trace_results()["cmpl-001"]
        assert interaction.model_response is not None
        assert interaction.model_response.input_tokens == [10, 20]
        assert interaction.model_response.output_tokens == [30, 40, 50]

    def test_tracer_clear(self):
        """clear() should remove all traced data."""
        tracer = ChatTracer(export_style="individual")
        task = ChatTask(messages=[{"role": "user", "content": "hello"}])
        task.completion = _make_fake_completion("cmpl-001")
        task.input_tokens = [1]
        task.output_tokens = [2]

        tracer.after_yield([task])
        assert len(tracer.get_trace_results()) == 1

        tracer.clear()
        assert tracer.get_trace_results() == {}

    def test_tracer_ignores_non_chat_tasks(self):
        """after_yield should skip non-ChatTask objects."""
        tracer = ChatTracer(export_style="individual")
        tracer.after_yield([GenerationTask(input_str="hello")])
        assert tracer.get_trace_results() == {}

    def test_tracer_empty_returns_empty(self):
        """get_trace_results on a fresh tracer should return empty dict."""
        tracer = ChatTracer(export_style="individual")
        assert tracer.get_trace_results() == {}


# ===========================================================================
# TraceTrajectoryMaker tests
# ===========================================================================


class TestTraceTrajectoryMaker:
    """Tests for TraceTrajectoryMaker (multi-turn tracing + reward pipeline)."""

    @staticmethod
    def _make_trace_maker(
        n_turns: int = 2,
        responses: list[str] | None = None,
        rewards: list[float] | None = None,
        default_reward: float = 1.0,
    ) -> tuple[TraceTrajectoryMaker, FakeChatRolloutController]:
        """Build a ``TraceTrajectoryMaker`` with fake sub-controllers.

        Also wires the class-level ``ChatTracer`` into the fake rollout
        controller so it can call ``before_yield`` / ``after_yield`` hooks.
        """
        rollout_ctrl = FakeChatRolloutController(n_turns=n_turns, responses=responses)
        reward_ctrl = FakeChatRewardController(
            rewards=rewards, default_reward=default_reward
        )
        maker = TraceTrajectoryMaker(
            rollout_controller=rollout_ctrl,
            reward_controller=reward_ctrl,
        )
        chat_tracer = maker.task_collections["chat_tracer"]
        rollout_ctrl._tracer = chat_tracer
        return maker, rollout_ctrl

    def test_basic_trace_single_turn(self):
        """Single-turn trace should produce one traced interaction with reward."""
        maker, _ = self._make_trace_maker(
            n_turns=1,
            responses=["The answer is 42"],
            default_reward=1.0,
        )

        task = TraceGenerationTask.create_from_prompt("What is 6*7?")
        list(maker.process([task]))

        assert task.trace_results is not None
        assert len(task.trace_results) == 1
        interaction = list(task.trace_results.values())[0]
        assert interaction.reward == 1.0

    def test_multi_turn_trace(self):
        """Multi-turn trace should produce one interaction per turn."""
        maker, _ = self._make_trace_maker(
            n_turns=3,
            responses=["step 1", "step 2", "final answer: 42"],
            rewards=[0.0, 0.0, 1.0],
        )

        task = TraceGenerationTask.create_from_prompt("Solve step by step")
        list(maker.process([task]))

        assert task.trace_results is not None
        assert len(task.trace_results) == 3

    def test_trace_rewards_assigned_correctly(self):
        """Each traced interaction should get its designated reward."""
        maker, _ = self._make_trace_maker(
            n_turns=2,
            responses=["thinking...", "42"],
            rewards=[0.5, 1.0],
        )

        task = TraceGenerationTask.create_from_prompt("What is the answer?")
        list(maker.process([task]))

        assert task.trace_results is not None
        rewards = [i.reward for i in task.trace_results.values()]
        assert rewards == [0.5, 1.0]

    def test_trace_results_stored_in_task(self):
        """trace_results should be set on the TraceGenerationTask after processing."""
        maker, _ = self._make_trace_maker(n_turns=1, default_reward=0.0)

        task = TraceGenerationTask.create_from_prompt("hello")
        assert task.trace_results is None

        list(maker.process([task]))

        assert task.trace_results is not None
        assert isinstance(task.trace_results, dict)

    def test_trace_with_plain_task_fallback(self):
        """process should not crash when given a plain ChatTask."""
        maker, _ = self._make_trace_maker(n_turns=1, default_reward=1.0)

        chat_task = ChatTask.create_from_prompt("direct chat task")
        list(maker.process([chat_task]))
        # No assertion on trace_results — plain ChatTask doesn't store them.

    def test_trace_generation_task_create_from_chat_task(self):
        """TraceGenerationTask.create_from_chat_task should wrap correctly."""
        chat_task = ChatTask.create_from_prompt("hello")
        trace_task = TraceGenerationTask.create_from_chat_task(chat_task)

        assert trace_task.generation_task is chat_task
        assert trace_task.trace_results is None

    def test_trace_generation_task_scaffolding_output(self):
        """create_scaffolding_output should reflect generation_task fields and trace_results."""
        gen_task = GenerationTask(output_str="result text", output_tokens=[10, 20, 30])
        trace_task = TraceGenerationTask(generation_task=gen_task)
        trace_task.trace_results = {"id-1": "fake_interaction"}

        output = trace_task.create_scaffolding_output()
        assert output.text == "result text"
        assert output.token_ids == [10, 20, 30]
        assert output.data == {"id-1": "fake_interaction"}

    def test_trace_generation_task_scaffolding_output_empty(self):
        """create_scaffolding_output with no generation_task should return empty."""
        trace_task = TraceGenerationTask()
        output = trace_task.create_scaffolding_output()
        assert output.text == ""
        assert output.token_ids == []
        assert output.data is None

    def test_no_reward_tasks_when_no_traces(self):
        """If rollout produces no traceable outputs, reward step should be skipped."""

        class EmptyRolloutController(Controller):
            def process(self, tasks, **kwargs):
                yield tasks

        reward_ctrl = FakeChatRewardController(default_reward=1.0)
        maker = TraceTrajectoryMaker(
            rollout_controller=EmptyRolloutController(),
            reward_controller=reward_ctrl,
        )
        assert maker.task_collections.get("chat_tracer") is not None

        task = TraceGenerationTask.create_from_prompt("hello")
        list(maker.process([task]))

        assert task.trace_results is not None
        assert len(task.trace_results) == 0

    def test_trace_interaction_model_response_tokens(self):
        """Traced interactions should carry correct per-turn tokens."""
        maker, _ = self._make_trace_maker(
            n_turns=2,
            responses=["r0", "r1"],
            default_reward=1.0,
        )
        # Use distinct per-turn tokens
        rollout_ctrl = maker.rollout_controller
        rollout_ctrl.output_tokens_per_turn = [[10, 11], [20, 21, 22]]
        rollout_ctrl.input_tokens_per_turn = [[1, 2], [1, 2, 3]]

        task = TraceGenerationTask.create_from_prompt("prompt")
        list(maker.process([task]))

        interactions = list(task.trace_results.values())
        assert interactions[0].model_response.output_tokens == [10, 11]
        assert interactions[0].model_response.input_tokens == [1, 2]
        assert interactions[1].model_response.output_tokens == [20, 21, 22]
        assert interactions[1].model_response.input_tokens == [1, 2, 3]


# ===========================================================================
# RLVRRewardController tests
# ===========================================================================


class TestRLVRRewardController:
    """Tests for RLVRRewardController (reward computation)."""

    def test_compute_reward_correct(self):
        """Should return 1.0 when completion contains the answer."""
        ctrl = RLVRRewardController(_simple_reward_fn)
        task = RLVRRewardTask(
            prompt_str="What is 2+2?",
            completion_str="The answer is 4",
            input_tokens=[1, 2, 3],
            output_tokens=[4, 5],
            task_data={"answer": "4"},
        )

        list(ctrl.process([task]))
        assert task.reward == 1.0
        assert ctrl.scores == [1.0]

    def test_compute_reward_wrong(self):
        """Should return 0.0 when completion does not contain the answer."""
        ctrl = RLVRRewardController(_simple_reward_fn)
        task = RLVRRewardTask(
            prompt_str="What is 2+2?",
            completion_str="I think it's 5",
            input_tokens=[1, 2, 3],
            output_tokens=[4, 5],
            task_data={"answer": "4"},
        )

        list(ctrl.process([task]))
        assert task.reward == 0.0

    def test_compute_reward_updates_interaction(self):
        """Reward should be propagated to the attached interaction object."""
        ctrl = RLVRRewardController(_simple_reward_fn)
        interaction = InteractionWithTokenLogpReward(
            model_response=ModelResponse(
                input_tokens=[1],
                output_tokens=[2],
                output_logprobs=[0.0],
                output_versions=[-1],
            ),
            messages=[],
        )
        task = RLVRRewardTask(
            prompt_str="Q",
            completion_str="42",
            input_tokens=[1],
            output_tokens=[2],
            task_data={"answer": "42"},
            interaction=interaction,
        )

        list(ctrl.process([task]))
        assert interaction.reward == 1.0

    def test_compute_reward_batch(self):
        """Should process multiple tasks and track all scores."""
        ctrl = RLVRRewardController(_simple_reward_fn)
        tasks = [
            RLVRRewardTask(
                prompt_str="Q1",
                completion_str="42",
                task_data={"answer": "42"},
            ),
            RLVRRewardTask(
                prompt_str="Q2",
                completion_str="wrong",
                task_data={"answer": "42"},
            ),
            RLVRRewardTask(
                prompt_str="Q3",
                completion_str="also 42",
                task_data={"answer": "42"},
            ),
        ]

        list(ctrl.process(tasks))
        assert ctrl.scores == [1.0, 0.0, 1.0]

    def test_reward_from_generation_task(self):
        """Should handle GenerationTask via customized_result_fields path."""
        ctrl = RLVRRewardController(_simple_reward_fn)
        gen_task = GenerationTask(
            input_str="What is 2+2?",
            output_str="4",
            input_tokens=[1, 2],
            output_tokens=[3],
        )

        list(
            ctrl.process(
                [gen_task],
                task_data={"answer": "4"},
                prompt_str="What is 2+2?",
            )
        )
        assert gen_task.customized_result_fields["reward"] == 1.0


# ===========================================================================
# RLVRRewardTask tests
# ===========================================================================


class TestRLVRRewardTask:
    """Tests for RLVRRewardTask creation and data flow."""

    def test_create_from_generation_task(self):
        """create_from_generation_task should correctly populate all fields."""
        gen_task = GenerationTask(
            input_str="prompt",
            output_str="completion text",
            input_tokens=[1, 2, 3],
            output_tokens=[4, 5],
        )
        gen_task.customized_result_fields["output_logprobs"] = [0.1, 0.2]
        gen_task.customized_result_fields["output_versions"] = [1, 1]

        reward_task = RLVRRewardTask.create_from_generation_task(
            gen_task=gen_task,
            prompt_str="original prompt",
            task_data={"answer": "42"},
        )

        assert reward_task.prompt_str == "original prompt"
        assert reward_task.completion_str == "completion text"
        assert reward_task.input_tokens == [1, 2, 3]
        assert reward_task.output_tokens == [4, 5]
        assert reward_task.output_logprobs == [0.1, 0.2]
        assert reward_task.output_versions == [1, 1]
        assert reward_task.task_data == {"answer": "42"}
        assert reward_task.reward is None

    def test_create_from_generation_task_no_output(self):
        """Should handle GenerationTask with None output gracefully."""
        gen_task = GenerationTask()

        reward_task = RLVRRewardTask.create_from_generation_task(
            gen_task=gen_task,
            prompt_str="prompt",
            task_data={},
        )

        assert reward_task.completion_str == ""
        assert reward_task.input_tokens == []
        assert reward_task.output_tokens == []


# ===========================================================================
# ChatRewardTask tests
# ===========================================================================


class TestChatRewardTask:
    """Tests for ChatRewardTask creation."""

    def test_create_from_trace_result(self):
        """create_from_trace_result should wrap an interaction correctly."""
        interaction = InteractionWithTokenLogpReward(
            model_response=ModelResponse(
                input_tokens=[1, 2],
                output_tokens=[3, 4],
                output_logprobs=[0.0, 0.0],
                output_versions=[-1, -1],
            ),
            messages=[{"role": "user", "content": "hello"}],
        )

        task = ChatRewardTask.create_from_trace_result("id-001", interaction)

        assert task.interaction is interaction
        assert task.interaction_id == "id-001"
        assert task.reward is None


# ===========================================================================
# End-to-end integration tests
# ===========================================================================


class TestEndToEnd:
    """Integration tests that exercise the full scaffolding rollout pipeline."""

    def test_pipeline_e2e_tensor_dict_compatible(self):
        """PipelineTrajectoryMaker interactions should be convertible to tensor dicts."""
        gen_ctrl = FakeGenerationController(
            output_str="42",
            output_tokens=[201, 202],
        )
        reward_ctrl = RLVRRewardController(_simple_reward_fn)

        maker = PipelineTrajectoryMaker(
            generation_controller=gen_ctrl,
            reward_controller=reward_ctrl,
            task_data={"answer": "42"},
            prompt_str=FAKE_PROMPT_STR,
        )

        task = GenerationTask(input_str=FAKE_PROMPT_STR, input_tokens=FAKE_INPUT_TOKENS)
        list(maker.process([task]))
        interaction = list(task.customized_result_fields["interactions"].values())[0]

        td = interaction.to_tensor_dict()
        assert "input_ids" in td
        assert "loss_mask" in td
        assert "logprobs" in td
        assert "rewards" in td
        assert td["rewards"].item() == 1.0

    def test_trace_e2e_multi_turn_with_rewards(self):
        """Full multi-turn TraceTrajectoryMaker E2E with per-turn rewards."""
        rollout_ctrl = FakeChatRolloutController(
            n_turns=3,
            responses=["Let me think...", "Calculating...", "The answer is 42"],
        )
        reward_ctrl = FakeChatRewardController(rewards=[0.0, 0.5, 1.0])

        maker = TraceTrajectoryMaker(
            rollout_controller=rollout_ctrl,
            reward_controller=reward_ctrl,
        )
        chat_tracer = maker.task_collections["chat_tracer"]
        rollout_ctrl._tracer = chat_tracer

        task = TraceGenerationTask.create_from_prompt("Solve step by step: 6*7")
        list(maker.process([task]))

        assert task.trace_results is not None
        assert len(task.trace_results) == 3
        rewards = [i.reward for i in task.trace_results.values()]
        assert rewards == [0.0, 0.5, 1.0]

    def test_trace_e2e_single_turn_generates_output(self):
        """Single-turn trace should produce a valid interaction with reward."""
        rollout_ctrl = FakeChatRolloutController(n_turns=1, responses=["42"])
        reward_ctrl = FakeChatRewardController(default_reward=1.0)

        maker = TraceTrajectoryMaker(
            rollout_controller=rollout_ctrl,
            reward_controller=reward_ctrl,
        )
        chat_tracer = maker.task_collections["chat_tracer"]
        rollout_ctrl._tracer = chat_tracer

        task = TraceGenerationTask.create_from_prompt("What is 6*7?")
        list(maker.process([task]))

        assert task.trace_results is not None
        assert len(task.trace_results) == 1
        interaction = list(task.trace_results.values())[0]
        assert interaction.reward == 1.0
        assert interaction.model_response is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
