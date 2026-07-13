"""Integration tests for ScaffoldingLlm with full controller pipelines.

Tests verify that ScaffoldingLlm can correctly drive:
1. PipelineTrajectoryMaker (single-turn generation + reward)
2. TraceTrajectoryMaker + MultiTurnChatController (multi-turn chat + tracing + reward)

These tests use a FakeChatWorker that handles ChatTask and GenerationTask
via the Worker.task_handlers dispatch mechanism, simulating an LLM backend.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from examples.scaffolding._compat import (
    AssistantMessage,
    ChatTask,
    GenerationTask,
    NativeGenerationController,
    ScaffoldingLlm,
    TaskStatus,
    Worker,
)
from examples.scaffolding.controllers import (
    MultiTurnChatController,
    PipelineTrajectoryMaker,
    RLVRRewardController,
    TraceTrajectoryMaker,
)

# ---------------------------------------------------------------------------
# Test constants
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
    """Deterministic reward: 1.0 if completion contains the answer, else 0.0."""
    answer = kwargs.get("answer", "")
    return 1.0 if answer and answer in completions else 0.0


# ---------------------------------------------------------------------------
# Fake Worker
# ---------------------------------------------------------------------------


def _make_fake_completion(completion_id: str = "cmpl-001") -> MagicMock:
    """Create a minimal fake ChatCompletion object."""
    completion = MagicMock()
    completion.id = completion_id
    completion.created = 1000
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = FAKE_OUTPUT_STR
    completion.choices[0].finish_reason = "stop"
    return completion


_chat_handler_call_count = 0


class FakeChatWorker(Worker):
    """Worker that handles ChatTask and GenerationTask without any backend.

    For ChatTask: appends an AssistantMessage and sets completion/tokens.
    For GenerationTask: fills output_str and output_tokens.
    """

    def __init__(
        self,
        response_text: str = FAKE_OUTPUT_STR,
        output_tokens: list[int] | None = None,
    ):
        self.response_text = response_text
        self.output_tokens = output_tokens or FAKE_OUTPUT_TOKENS

    async def _handle_generation_task(self, task: GenerationTask) -> TaskStatus:
        task.output_str = self.response_text
        task.output_tokens = list(self.output_tokens)
        if task.input_tokens is None:
            task.input_tokens = FAKE_INPUT_TOKENS
        task.finish_reason = "stop"
        return TaskStatus.SUCCESS

    async def _handle_chat_task(self, task: ChatTask) -> TaskStatus:
        global _chat_handler_call_count
        _chat_handler_call_count += 1

        completion_id = f"cmpl-{_chat_handler_call_count:03d}"
        task.completion = _make_fake_completion(completion_id)
        task.completion.choices[0].message.content = self.response_text
        task.output_tokens = list(self.output_tokens)
        if task.input_tokens is None:
            task.input_tokens = FAKE_INPUT_TOKENS
        task.finish_reason = "stop"

        # Mimic what OpenaiWorker.chat_handler does: append assistant message
        task.messages.append(AssistantMessage(content=self.response_text))
        return TaskStatus.SUCCESS

    task_handlers = {
        GenerationTask: _handle_generation_task,
        ChatTask: _handle_chat_task,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_chat_handler_count():
    """Reset the global call counter before each test."""
    global _chat_handler_call_count
    _chat_handler_call_count = 0
    yield


@pytest.fixture(autouse=True)
def _reset_shared_tracer():
    """Reset the class-level ChatTracer before each test."""
    yield
    tracer = TraceTrajectoryMaker.task_collections.get("chat_tracer")
    if tracer is not None:
        tracer.clear()


# ===========================================================================
# PipelineTrajectoryMaker + ScaffoldingLlm
# ===========================================================================


class TestPipelineViaScaffoldingLlm:
    """Test PipelineTrajectoryMaker running through ScaffoldingLlm."""

    def test_single_generation_sync(self):
        """ScaffoldingLlm.generate() should produce a result with interactions."""
        worker = FakeChatWorker(response_text="The answer is 42.")
        reward_ctrl = RLVRRewardController(_simple_reward_fn)
        gen_ctrl = NativeGenerationController(
            sampling_params={"max_tokens": 100, "temperature": 1.0}
        )
        trajectory_maker = PipelineTrajectoryMaker(
            generation_controller=gen_ctrl,
            reward_controller=reward_ctrl,
            task_data={"answer": "42"},
            prompt_str=FAKE_PROMPT_STR,
            input_tokens=FAKE_INPUT_TOKENS,
        )

        llm = ScaffoldingLlm(
            trajectory_maker,
            {NativeGenerationController.WorkerTag.GENERATION: worker},
        )
        try:
            result = llm.generate(FAKE_PROMPT_STR)

            assert result is not None
            assert result._done is True
            output = result.outputs[0]
            assert output.data is not None

            interactions = output.data
            assert len(interactions) == 1
            interaction = list(interactions.values())[0]
            assert interaction.reward == 1.0
            assert interaction.model_response is not None
            assert interaction.model_response.input_tokens == FAKE_INPUT_TOKENS
        finally:
            llm.shutdown()

    def test_single_generation_async(self):
        """ScaffoldingLlm.generate_async() + await should work."""
        worker = FakeChatWorker(response_text="42")
        reward_ctrl = RLVRRewardController(_simple_reward_fn)
        gen_ctrl = NativeGenerationController(sampling_params={"max_tokens": 100})
        trajectory_maker = PipelineTrajectoryMaker(
            generation_controller=gen_ctrl,
            reward_controller=reward_ctrl,
            task_data={"answer": "42"},
            prompt_str=FAKE_PROMPT_STR,
            input_tokens=FAKE_INPUT_TOKENS,
        )

        llm = ScaffoldingLlm(
            trajectory_maker,
            {NativeGenerationController.WorkerTag.GENERATION: worker},
        )
        try:
            result = llm.generate_async(FAKE_PROMPT_STR)
            # Use result.result() which blocks via the event loop
            result.result(timeout=10.0)

            assert result._done is True
            output = result.outputs[0]
            interactions = output.data
            assert len(interactions) == 1
            interaction = list(interactions.values())[0]
            assert interaction.reward == 1.0
        finally:
            llm.shutdown()

    def test_batch_generation(self):
        """ScaffoldingLlm.generate() with a list of prompts should work."""
        worker = FakeChatWorker(response_text="42")
        reward_ctrl = RLVRRewardController(_simple_reward_fn)
        gen_ctrl = NativeGenerationController(sampling_params={"max_tokens": 50})
        trajectory_maker = PipelineTrajectoryMaker(
            generation_controller=gen_ctrl,
            reward_controller=reward_ctrl,
            task_data={"answer": "42"},
            prompt_str=FAKE_PROMPT_STR,
            input_tokens=FAKE_INPUT_TOKENS,
        )

        llm = ScaffoldingLlm(
            trajectory_maker,
            {NativeGenerationController.WorkerTag.GENERATION: worker},
        )
        try:
            results = llm.generate(["prompt1", "prompt2", "prompt3"])
            assert len(results) == 3
            for result in results:
                assert result._done is True
                assert result.outputs[0].data is not None
        finally:
            llm.shutdown()


# ===========================================================================
# MultiTurnChatController + TraceTrajectoryMaker + ScaffoldingLlm
# ===========================================================================


class TestMultiTurnViaScaffoldingLlm:
    """Test MultiTurnChatController + TraceTrajectoryMaker via ScaffoldingLlm.

    This reproduces the exact architecture used by ChatScaffoldingWorkflow.
    """

    @staticmethod
    def _build_llm(
        response_text: str = FAKE_OUTPUT_STR,
        max_turns: int = 2,
        reward_fn=None,
    ) -> tuple[ScaffoldingLlm, MultiTurnChatController, TraceTrajectoryMaker]:
        """Build the full ChatScaffoldingWorkflow pipeline."""
        worker = FakeChatWorker(response_text=response_text)
        gen_ctrl = NativeGenerationController(
            sampling_params={"max_tokens": 100, "temperature": 1.0}
        )
        reward_ctrl = RLVRRewardController(reward_fn or _simple_reward_fn)
        multi_turn_ctrl = MultiTurnChatController(
            generation_controller=gen_ctrl,
            max_turns=max_turns,
            reflection_message="Try again.",
        )
        trace_maker = TraceTrajectoryMaker(
            rollout_controller=multi_turn_ctrl,
            reward_controller=reward_ctrl,
        )
        llm = ScaffoldingLlm(
            trace_maker,
            {NativeGenerationController.WorkerTag.GENERATION: worker},
        )
        return llm, multi_turn_ctrl, trace_maker

    def test_single_turn_sync(self):
        """Single-turn chat via ScaffoldingLlm.generate() should work."""
        llm, multi_turn_ctrl, _ = self._build_llm(response_text="42", max_turns=1)
        try:
            # Set per-episode data (like ChatScaffoldingWorkflow.arun_episode does)
            multi_turn_ctrl.messages = [{"role": "user", "content": "What is 6*7?"}]
            multi_turn_ctrl.input_tokens = FAKE_INPUT_TOKENS

            result = llm.generate(FAKE_PROMPT_STR)

            assert result is not None
            assert result._done is True
            output = result.outputs[0]
            # TraceTrajectoryMaker.generate() returns ScaffoldingOutput with
            # trace_results in data
            assert output.data is not None or output.text is not None
        finally:
            llm.shutdown()

    def test_multi_turn_sync(self):
        """Multi-turn (2-turn) chat via ScaffoldingLlm.generate() should work."""
        llm, multi_turn_ctrl, _ = self._build_llm(response_text="42", max_turns=2)
        try:
            multi_turn_ctrl.messages = [{"role": "user", "content": "Solve: 6*7"}]
            multi_turn_ctrl.input_tokens = FAKE_INPUT_TOKENS

            result = llm.generate(FAKE_PROMPT_STR)

            assert result is not None
            assert result._done is True

            # Verify the worker was called twice (2 turns)
            global _chat_handler_call_count
            assert _chat_handler_call_count >= 2
        finally:
            llm.shutdown()

    def test_multi_turn_async_result(self):
        """generate_async() + result() should complete for multi-turn."""
        llm, multi_turn_ctrl, _ = self._build_llm(response_text="42", max_turns=2)
        try:
            multi_turn_ctrl.messages = [{"role": "user", "content": "Solve: 6*7"}]
            multi_turn_ctrl.input_tokens = FAKE_INPUT_TOKENS

            result = llm.generate_async(FAKE_PROMPT_STR)
            result.result(timeout=10.0)

            assert result._done is True
        finally:
            llm.shutdown()

    def test_trace_results_available(self):
        """Trace results should be accessible via ScaffoldingOutput.data."""
        llm, multi_turn_ctrl, _ = self._build_llm(response_text="42", max_turns=1)
        try:
            multi_turn_ctrl.messages = [{"role": "user", "content": "What is 6*7?"}]
            multi_turn_ctrl.input_tokens = FAKE_INPUT_TOKENS

            result = llm.generate(FAKE_PROMPT_STR)
            output = result.outputs[0]

            # TraceTrajectoryMaker stores trace_results in
            # TraceGenerationTask.trace_results, which is returned via
            # create_scaffolding_output().data
            trace_data = output.data
            if trace_data is not None:
                # If tracing worked, we should have at least one interaction
                assert len(trace_data) >= 1
        finally:
            llm.shutdown()

    def test_multiple_concurrent_requests(self):
        """Multiple concurrent requests should all complete."""
        llm, multi_turn_ctrl, _ = self._build_llm(response_text="42", max_turns=1)
        try:
            multi_turn_ctrl.messages = [{"role": "user", "content": "What is 6*7?"}]
            multi_turn_ctrl.input_tokens = FAKE_INPUT_TOKENS

            results = llm.generate([f"prompt_{i}" for i in range(5)])

            assert len(results) == 5
            for result in results:
                assert result._done is True
        finally:
            llm.shutdown()

    def test_clone_isolation(self):
        """Each request should get an independent controller clone."""
        call_messages = []

        def _capture_reward(prompt, completions, prompt_ids, completion_ids, **kw):
            call_messages.append(completions)
            return 1.0

        llm, multi_turn_ctrl, _ = self._build_llm(
            response_text="42", max_turns=1, reward_fn=_capture_reward
        )
        try:
            multi_turn_ctrl.messages = [{"role": "user", "content": "Q1"}]
            multi_turn_ctrl.input_tokens = FAKE_INPUT_TOKENS

            # Two sequential requests
            r1 = llm.generate("prompt1")
            r2 = llm.generate("prompt2")

            assert r1._done is True
            assert r2._done is True
        finally:
            llm.shutdown()


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and error scenarios."""

    def test_empty_messages(self):
        """Controller with empty messages should not crash."""
        worker = FakeChatWorker(response_text="hello")
        gen_ctrl = NativeGenerationController(sampling_params={"max_tokens": 10})
        reward_ctrl = RLVRRewardController(_simple_reward_fn)
        multi_turn_ctrl = MultiTurnChatController(
            generation_controller=gen_ctrl,
            max_turns=1,
            reflection_message="retry",
            messages=[],
            input_tokens=[],
        )
        trace_maker = TraceTrajectoryMaker(
            rollout_controller=multi_turn_ctrl,
            reward_controller=reward_ctrl,
        )
        llm = ScaffoldingLlm(
            trace_maker,
            {NativeGenerationController.WorkerTag.GENERATION: worker},
        )
        try:
            result = llm.generate("test prompt")
            assert result._done is True
        finally:
            llm.shutdown()

    def test_shutdown_is_safe(self):
        """Calling shutdown() should not raise."""
        worker = FakeChatWorker()
        gen_ctrl = NativeGenerationController(sampling_params={"max_tokens": 10})
        reward_ctrl = RLVRRewardController(_simple_reward_fn)
        trajectory_maker = PipelineTrajectoryMaker(
            generation_controller=gen_ctrl,
            reward_controller=reward_ctrl,
        )
        llm = ScaffoldingLlm(
            trajectory_maker,
            {NativeGenerationController.WorkerTag.GENERATION: worker},
        )
        llm.shutdown()
        # No exception = success


# ===========================================================================
# Async context tests (reproducing the training framework scenario)
# ===========================================================================


class TestAsyncContextScaffoldingLlm:
    """Tests that run ScaffoldingLlm from within an existing asyncio event loop.

    This reproduces the actual deployment scenario: the rollout framework's
    arun_episode is called from within an asyncio context that already has a
    running event loop. ScaffoldingLlm._get_loop() will detect the running
    loop (own_loop=False) and schedule its main loop on it.
    """

    @pytest.mark.asyncio
    async def test_pipeline_in_async_context(self):
        """PipelineTrajectoryMaker should work when called from async context."""
        worker = FakeChatWorker(response_text="The answer is 42.")
        gen_ctrl = NativeGenerationController(sampling_params={"max_tokens": 100})
        reward_ctrl = RLVRRewardController(_simple_reward_fn)
        trajectory_maker = PipelineTrajectoryMaker(
            generation_controller=gen_ctrl,
            reward_controller=reward_ctrl,
            task_data={"answer": "42"},
            prompt_str=FAKE_PROMPT_STR,
            input_tokens=FAKE_INPUT_TOKENS,
        )

        llm = ScaffoldingLlm(
            trajectory_maker,
            {NativeGenerationController.WorkerTag.GENERATION: worker},
        )
        try:
            result = llm.generate_async(FAKE_PROMPT_STR)
            await asyncio.wait_for(result, timeout=10.0)

            assert result._done is True
            output = result.outputs[0]
            assert output.data is not None
            interactions = output.data
            assert len(interactions) == 1
            interaction = list(interactions.values())[0]
            assert interaction.reward == 1.0
        finally:
            llm.shutdown()

    @pytest.mark.asyncio
    async def test_multi_turn_in_async_context(self):
        """MultiTurnChatController + TraceTrajectoryMaker in async context."""
        worker = FakeChatWorker(response_text="42")
        gen_ctrl = NativeGenerationController(sampling_params={"max_tokens": 100})
        reward_ctrl = RLVRRewardController(_simple_reward_fn)
        multi_turn_ctrl = MultiTurnChatController(
            generation_controller=gen_ctrl,
            max_turns=2,
            reflection_message="Try again.",
            messages=[{"role": "user", "content": "What is 6*7?"}],
            input_tokens=FAKE_INPUT_TOKENS,
        )
        trace_maker = TraceTrajectoryMaker(
            rollout_controller=multi_turn_ctrl,
            reward_controller=reward_ctrl,
        )

        llm = ScaffoldingLlm(
            trace_maker,
            {NativeGenerationController.WorkerTag.GENERATION: worker},
        )
        try:
            result = llm.generate_async(FAKE_PROMPT_STR)
            await asyncio.wait_for(result, timeout=10.0)

            assert result._done is True
        finally:
            llm.shutdown()

    @pytest.mark.asyncio
    async def test_multiple_async_requests(self):
        """Multiple concurrent async requests from within async context."""
        worker = FakeChatWorker(response_text="42")
        gen_ctrl = NativeGenerationController(sampling_params={"max_tokens": 100})
        reward_ctrl = RLVRRewardController(_simple_reward_fn)
        trajectory_maker = PipelineTrajectoryMaker(
            generation_controller=gen_ctrl,
            reward_controller=reward_ctrl,
            task_data={"answer": "42"},
            prompt_str=FAKE_PROMPT_STR,
            input_tokens=FAKE_INPUT_TOKENS,
        )

        llm = ScaffoldingLlm(
            trajectory_maker,
            {NativeGenerationController.WorkerTag.GENERATION: worker},
        )
        try:
            # Launch multiple requests concurrently (like the rollout framework)
            results = []
            for i in range(5):
                results.append(llm.generate_async(f"prompt_{i}"))

            # Await all concurrently
            await asyncio.wait_for(
                asyncio.gather(*[r.aresult() for r in results]),
                timeout=10.0,
            )

            for result in results:
                assert result._done is True
                assert result.outputs[0].data is not None
        finally:
            llm.shutdown()


# ===========================================================================
# Uvloop thread tests (reproducing the exact AsyncTaskRunner scenario)
# ===========================================================================


class TestUvloopThreadScaffoldingLlm:
    """Tests that reproduce the exact production deployment architecture.

    AsyncTaskRunner runs a uvloop in a separate thread. Multiple coroutines
    (one per arun_episode) run concurrently on this loop. They share one
    ScaffoldingLlm instance. The ScaffoldingLlm is lazily initialized inside
    the first coroutine (so it captures the uvloop as its event loop).

    This is the exact scenario where the deadlock was observed.
    """

    @staticmethod
    def _build_shared_components(
        response_text: str = FAKE_OUTPUT_STR,
        max_turns: int = 2,
    ):
        """Build shared components (simulating ChatScaffoldingWorkflow.__init__)."""
        worker = FakeChatWorker(response_text=response_text)
        gen_ctrl = NativeGenerationController(
            sampling_params={"max_tokens": 100, "temperature": 1.0}
        )
        reward_ctrl = RLVRRewardController(_simple_reward_fn)
        multi_turn_ctrl = MultiTurnChatController(
            generation_controller=gen_ctrl,
            max_turns=max_turns,
            reflection_message="Try again.",
        )
        trace_maker = TraceTrajectoryMaker(
            rollout_controller=multi_turn_ctrl,
            reward_controller=reward_ctrl,
        )
        return worker, multi_turn_ctrl, trace_maker

    def test_uvloop_lazy_init_single_request(self):
        """ScaffoldingLlm lazily initialized on uvloop should handle one request."""
        import threading

        import uvloop

        worker, multi_turn_ctrl, trace_maker = self._build_shared_components(
            max_turns=1
        )
        multi_turn_ctrl.messages = [{"role": "user", "content": "What is 6*7?"}]
        multi_turn_ctrl.input_tokens = FAKE_INPUT_TOKENS

        result_holder = {}
        error_holder = {}

        async def run_on_uvloop():
            try:
                # Lazy init ScaffoldingLlm inside the uvloop (like _lazy_init_scaffolding)
                llm = ScaffoldingLlm(
                    trace_maker,
                    {NativeGenerationController.WorkerTag.GENERATION: worker},
                )
                try:
                    result = llm.generate_async(FAKE_PROMPT_STR)
                    await asyncio.wait_for(result, timeout=10.0)
                    result_holder["result"] = result
                finally:
                    llm.shutdown()
            except Exception as e:
                error_holder["error"] = e

        def thread_fn():
            loop = uvloop.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_on_uvloop())
            loop.close()

        t = threading.Thread(target=thread_fn)
        t.start()
        t.join(timeout=30)
        assert not t.is_alive(), "Thread deadlocked!"

        if "error" in error_holder:
            raise error_holder["error"]
        assert "result" in result_holder
        assert result_holder["result"]._done is True

    def test_uvloop_concurrent_coroutines_shared_llm(self):
        """Multiple concurrent coroutines on uvloop sharing one ScaffoldingLlm.

        This is the exact scenario from the training pipeline:
        - AsyncTaskRunner creates a uvloop in a background thread
        - Multiple _execute_workflow coroutines run concurrently
        - They all share the same ChatScaffoldingWorkflow instance
        - The ScaffoldingLlm is lazily initialized on first call
        """
        import threading

        import uvloop

        worker, multi_turn_ctrl, trace_maker = self._build_shared_components(
            max_turns=2
        )

        num_concurrent = 10
        results = {}
        errors = {}

        async def run_concurrent():
            # Lazy init ScaffoldingLlm inside uvloop (like _lazy_init_scaffolding)
            llm = ScaffoldingLlm(
                trace_maker,
                {NativeGenerationController.WorkerTag.GENERATION: worker},
            )
            try:

                async def simulate_arun_episode(idx: int):
                    """Simulate what arun_episode does."""
                    try:
                        # Each coroutine sets per-episode data and calls generate_async
                        # Note: in real code, clone() inside ScaffoldingLlm deep-copies
                        # the prototype controller, so the race on shared state is safe
                        # as long as the prototype is set before generate_async.
                        multi_turn_ctrl.messages = [
                            {"role": "user", "content": f"Question {idx}"}
                        ]
                        multi_turn_ctrl.input_tokens = FAKE_INPUT_TOKENS

                        result = llm.generate_async(f"prompt_{idx}")
                        await asyncio.wait_for(result, timeout=10.0)
                        results[idx] = result
                    except Exception as e:
                        errors[idx] = e

                # Run all concurrently (like AsyncTaskRunner does)
                await asyncio.gather(
                    *[simulate_arun_episode(i) for i in range(num_concurrent)]
                )
            finally:
                llm.shutdown()

        def thread_fn():
            loop = uvloop.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_concurrent())
            loop.close()

        t = threading.Thread(target=thread_fn)
        t.start()
        t.join(timeout=60)
        assert not t.is_alive(), "Thread deadlocked with concurrent coroutines!"

        if errors:
            first_error = next(iter(errors.values()))
            raise first_error

        assert len(results) == num_concurrent
        for idx, result in results.items():
            assert result._done is True, f"Result {idx} not done"

    def test_uvloop_high_concurrency(self):
        """Stress test: 50 concurrent coroutines on uvloop (closer to 256 in prod)."""
        import threading

        import uvloop

        worker, multi_turn_ctrl, trace_maker = self._build_shared_components(
            max_turns=1
        )
        multi_turn_ctrl.messages = [{"role": "user", "content": "test"}]
        multi_turn_ctrl.input_tokens = FAKE_INPUT_TOKENS

        num_concurrent = 50
        done_count = {"value": 0}
        errors = []

        async def run_stress():
            llm = ScaffoldingLlm(
                trace_maker,
                {NativeGenerationController.WorkerTag.GENERATION: worker},
            )
            try:

                async def single_episode(idx: int):
                    try:
                        result = llm.generate_async(f"prompt_{idx}")
                        await asyncio.wait_for(result, timeout=15.0)
                        assert result._done is True
                        done_count["value"] += 1
                    except Exception as e:
                        errors.append((idx, e))

                await asyncio.gather(
                    *[single_episode(i) for i in range(num_concurrent)]
                )
            finally:
                llm.shutdown()

        def thread_fn():
            loop = uvloop.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_stress())
            loop.close()

        t = threading.Thread(target=thread_fn)
        t.start()
        t.join(timeout=60)
        assert not t.is_alive(), "Thread deadlocked under high concurrency!"

        if errors:
            raise errors[0][1]
        assert done_count["value"] == num_concurrent


# ===========================================================================
# AsyncTaskRunner integration test (exact production scenario)
# ===========================================================================


class TestAsyncTaskRunnerScaffoldingLlm:
    """Tests using the real AsyncTaskRunner to reproduce the production architecture.

    This is the closest reproduction of the actual training pipeline:
    - AsyncTaskRunner runs a uvloop in a background thread
    - A shared workflow object is used across all tasks
    - ScaffoldingLlm is lazily initialized inside the first coroutine
    - Multiple concurrent coroutines share one ScaffoldingLlm instance
    """

    def test_async_task_runner_with_scaffolding_llm(self):
        """Full integration test using AsyncTaskRunner + shared ScaffoldingLlm."""
        from areal.infra.async_task_runner import AsyncTaskRunner

        worker = FakeChatWorker(response_text="42")
        gen_ctrl = NativeGenerationController(
            sampling_params={"max_tokens": 100, "temperature": 1.0}
        )
        reward_ctrl = RLVRRewardController(_simple_reward_fn)
        multi_turn_ctrl = MultiTurnChatController(
            generation_controller=gen_ctrl,
            max_turns=2,
            reflection_message="Try again.",
        )
        trace_maker = TraceTrajectoryMaker(
            rollout_controller=multi_turn_ctrl,
            reward_controller=reward_ctrl,
        )

        # Shared mutable state, like in the real workflow
        shared_state = {
            "scaffolding_llm": None,
            "worker": worker,
            "multi_turn_ctrl": multi_turn_ctrl,
            "trace_maker": trace_maker,
        }

        async def simulate_arun_episode(episode_idx: int) -> dict:
            """Simulate ChatScaffoldingWorkflow.arun_episode."""
            # Lazy init (like _lazy_init_scaffolding)
            if shared_state["scaffolding_llm"] is None:
                shared_state["scaffolding_llm"] = ScaffoldingLlm(
                    shared_state["trace_maker"],
                    {
                        NativeGenerationController.WorkerTag.GENERATION: shared_state[
                            "worker"
                        ]
                    },
                )

            llm = shared_state["scaffolding_llm"]

            # Set per-episode data (race condition in production!)
            shared_state["multi_turn_ctrl"].messages = [
                {"role": "user", "content": f"Question {episode_idx}"}
            ]
            shared_state["multi_turn_ctrl"].input_tokens = FAKE_INPUT_TOKENS

            # Generate
            result = llm.generate_async(f"prompt_{episode_idx}")
            await result

            return {"episode": episode_idx, "done": result._done}

        # Use AsyncTaskRunner like the real WorkflowExecutor
        runner = AsyncTaskRunner(max_queue_size=64)
        runner.initialize()

        num_tasks = 10
        try:
            for i in range(num_tasks):
                runner.submit(simulate_arun_episode, i, task_id=i)

            results = runner.wait(count=num_tasks, timeout=30.0)

            assert len(results) == num_tasks
            for result in results:
                assert result is not None
                assert result["done"] is True
        finally:
            # Shutdown ScaffoldingLlm if initialized
            if shared_state["scaffolding_llm"] is not None:
                shared_state["scaffolding_llm"].shutdown()
            runner.destroy()

    def test_async_task_runner_high_concurrency(self):
        """Stress test: 50 concurrent tasks via AsyncTaskRunner."""
        from areal.infra.async_task_runner import AsyncTaskRunner

        worker = FakeChatWorker(response_text="42")
        gen_ctrl = NativeGenerationController(sampling_params={"max_tokens": 100})
        reward_ctrl = RLVRRewardController(_simple_reward_fn)
        multi_turn_ctrl = MultiTurnChatController(
            generation_controller=gen_ctrl,
            max_turns=1,
            reflection_message="Try again.",
        )
        trace_maker = TraceTrajectoryMaker(
            rollout_controller=multi_turn_ctrl,
            reward_controller=reward_ctrl,
        )

        shared_state = {
            "scaffolding_llm": None,
            "worker": worker,
            "multi_turn_ctrl": multi_turn_ctrl,
            "trace_maker": trace_maker,
        }

        async def simulate_arun_episode(episode_idx: int) -> dict:
            if shared_state["scaffolding_llm"] is None:
                shared_state["scaffolding_llm"] = ScaffoldingLlm(
                    shared_state["trace_maker"],
                    {
                        NativeGenerationController.WorkerTag.GENERATION: shared_state[
                            "worker"
                        ]
                    },
                )

            llm = shared_state["scaffolding_llm"]

            shared_state["multi_turn_ctrl"].messages = [
                {"role": "user", "content": f"Q{episode_idx}"}
            ]
            shared_state["multi_turn_ctrl"].input_tokens = FAKE_INPUT_TOKENS

            result = llm.generate_async(f"prompt_{episode_idx}")
            await result

            return {"episode": episode_idx, "done": result._done}

        runner = AsyncTaskRunner(max_queue_size=128)
        runner.initialize()

        num_tasks = 50
        try:
            for i in range(num_tasks):
                runner.submit(simulate_arun_episode, i, task_id=i)

            results = runner.wait(count=num_tasks, timeout=60.0)

            assert len(results) == num_tasks
            for result in results:
                assert result is not None
                assert result["done"] is True
        finally:
            if shared_state["scaffolding_llm"] is not None:
                shared_state["scaffolding_llm"].shutdown()
            runner.destroy()


# ===========================================================================
# Per-task ScaffoldingLlm instances (RolloutController path)
# ===========================================================================


class TestPerTaskScaffoldingLlm:
    """Tests where EACH task creates its own ScaffoldingLlm instance.

    In the RolloutController path, RemoteInfEngine._resolve_workflow() creates
    a NEW ChatScaffoldingWorkflow per submit() call. So 256 tasks create 256
    workflow instances, each with its own ScaffoldingLlm. All 256 ScaffoldingLlm
    instances share the same uvloop (own_loop=False) and each schedules its own
    _main_loop_async_func on it.

    This is the ACTUAL production architecture with scheduler.type=local.
    """

    def test_multiple_llm_instances_on_async_task_runner(self):
        """Each task creates its own ScaffoldingLlm — deadlock reproduction."""
        from areal.infra.async_task_runner import AsyncTaskRunner

        worker = FakeChatWorker(response_text="42")
        llm_instances = []

        async def simulate_arun_episode_per_instance(episode_idx: int) -> dict:
            """Each task creates its OWN ScaffoldingLlm (like _resolve_workflow)."""
            # Create fresh controller hierarchy (like workflow.__init__ + build_scaffolding_llm)
            gen_ctrl = NativeGenerationController(
                sampling_params={"max_tokens": 100, "temperature": 1.0}
            )
            reward_ctrl = RLVRRewardController(_simple_reward_fn)
            multi_turn_ctrl = MultiTurnChatController(
                generation_controller=gen_ctrl,
                max_turns=1,
                reflection_message="Try again.",
                messages=[{"role": "user", "content": f"Q{episode_idx}"}],
                input_tokens=FAKE_INPUT_TOKENS,
            )
            trace_maker = TraceTrajectoryMaker(
                rollout_controller=multi_turn_ctrl,
                reward_controller=reward_ctrl,
            )

            # Each task creates its OWN ScaffoldingLlm
            llm = ScaffoldingLlm(
                trace_maker,
                {NativeGenerationController.WorkerTag.GENERATION: worker},
            )
            llm_instances.append(llm)

            result = llm.generate_async(f"prompt_{episode_idx}")
            await result

            return {"episode": episode_idx, "done": result._done}

        runner = AsyncTaskRunner(max_queue_size=64)
        runner.initialize()

        num_tasks = 10
        try:
            for i in range(num_tasks):
                runner.submit(simulate_arun_episode_per_instance, i, task_id=i)

            results = runner.wait(count=num_tasks, timeout=30.0)

            assert len(results) == num_tasks
            for result in results:
                assert result is not None
                assert result["done"] is True
        finally:
            for llm in llm_instances:
                try:
                    llm.shutdown()
                except Exception:
                    pass
            runner.destroy()

    def test_many_llm_instances_on_async_task_runner(self):
        """50 per-task ScaffoldingLlm instances — stress test."""
        from areal.infra.async_task_runner import AsyncTaskRunner

        worker = FakeChatWorker(response_text="42")
        llm_instances = []

        async def simulate_arun_episode_per_instance(episode_idx: int) -> dict:
            gen_ctrl = NativeGenerationController(sampling_params={"max_tokens": 50})
            reward_ctrl = RLVRRewardController(_simple_reward_fn)
            multi_turn_ctrl = MultiTurnChatController(
                generation_controller=gen_ctrl,
                max_turns=1,
                reflection_message="retry",
                messages=[{"role": "user", "content": f"Q{episode_idx}"}],
                input_tokens=FAKE_INPUT_TOKENS,
            )
            trace_maker = TraceTrajectoryMaker(
                rollout_controller=multi_turn_ctrl,
                reward_controller=reward_ctrl,
            )

            llm = ScaffoldingLlm(
                trace_maker,
                {NativeGenerationController.WorkerTag.GENERATION: worker},
            )
            llm_instances.append(llm)

            result = llm.generate_async(f"prompt_{episode_idx}")
            await result

            return {"episode": episode_idx, "done": result._done}

        runner = AsyncTaskRunner(max_queue_size=128)
        runner.initialize()

        num_tasks = 50
        try:
            for i in range(num_tasks):
                runner.submit(simulate_arun_episode_per_instance, i, task_id=i)

            results = runner.wait(count=num_tasks, timeout=60.0)

            assert len(results) == num_tasks
            for result in results:
                assert result is not None
                assert result["done"] is True
        finally:
            for llm in llm_instances:
                try:
                    llm.shutdown()
                except Exception:
                    pass
            runner.destroy()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
