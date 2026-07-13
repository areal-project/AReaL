"""Tests proving the vendored scaffolding modules are self-contained.

These tests verify that:
1. All scaffolding core modules import without tensorrt_llm installed.
2. Every public symbol in core/__init__.py and _compat.py is importable.
3. Core primitives (Task, Controller, Worker, ScaffoldingLlm, TaskCollection,
   math_utils) function correctly in isolation.
4. AReaL wrapper modules (controllers, task, worker, workflow) import cleanly.
5. No source file under examples/scaffolding/ contains a live
   ``import tensorrt_llm`` or ``from tensorrt_llm`` statement.
"""

from __future__ import annotations

import ast
import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ============================================================================
# 1. Import isolation — tensorrt_llm must NOT be importable
# ============================================================================


class TestNoTensorRTLLMDependency:
    """Verify that tensorrt_llm is not required at runtime."""

    def test_tensorrt_llm_not_installed(self):
        """tensorrt_llm should not be importable in the test environment."""
        assert "tensorrt_llm" not in sys.modules or sys.modules["tensorrt_llm"] is None

    def test_no_live_tensorrt_llm_imports_in_source(self):
        """No .py file under scaffolding/ should have a live import of tensorrt_llm.

        Comments and docstrings are allowed; only top-level or function-level
        ``import tensorrt_llm`` / ``from tensorrt_llm import ...`` are flagged.
        """
        scaffolding_root = Path(__file__).resolve().parents[1]
        violations = []
        for py_file in scaffolding_root.rglob("*.py"):
            try:
                tree = ast.parse(py_file.read_text(), filename=str(py_file))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "tensorrt_llm" or alias.name.startswith(
                            "tensorrt_llm."
                        ):
                            violations.append(
                                f"{py_file.relative_to(scaffolding_root)}:{node.lineno}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module and (
                        node.module == "tensorrt_llm"
                        or node.module.startswith("tensorrt_llm.")
                    ):
                        violations.append(
                            f"{py_file.relative_to(scaffolding_root)}:{node.lineno}"
                        )

        assert violations == [], "Found live tensorrt_llm imports in:\n" + "\n".join(
            violations
        )


# ============================================================================
# 2. Core module imports — every public symbol is importable
# ============================================================================


class TestCoreImports:
    """Verify every public symbol in core/ is importable."""

    def test_core_init_imports(self):
        """All symbols in core/__init__.py should import successfully."""
        from examples.scaffolding.core import (  # noqa: F401
            AssistantMessage,
            BestOfNController,
            ChatTask,
            Controller,
            GenerationTask,
            MajorityVoteController,
            NativeChatController,
            NativeGenerationController,
            NativeRewardController,
            OpenAIToolDescription,
            OpenaiWorker,
            ParallelProcess,
            RoleMessage,
            ScaffoldingLlm,
            ScaffoldingOutput,
            StreamGenerationTask,
            SystemMessage,
            Task,
            TaskCollection,
            TaskStatus,
            UserMessage,
            Worker,
            extract_answer_from_boxed,
            extract_answer_with_regex,
            get_digit_majority_vote_result,
            with_task_collection,
        )

        # Spot-check a few are real classes/functions
        assert callable(Controller)
        assert callable(extract_answer_from_boxed)
        assert callable(with_task_collection)

    def test_compat_reexports_match_core(self):
        """_compat.py should re-export the same symbols as core/__init__.py
        (minus math_utils which is not in _compat)."""
        from examples.scaffolding import _compat, core

        compat_all = set(_compat.__all__)
        # _compat intentionally omits math_utils functions
        math_utils_names = {
            "extract_answer_from_boxed",
            "extract_answer_with_regex",
            "get_digit_majority_vote_result",
        }
        core_all_minus_math = set(core.__all__) - math_utils_names
        assert compat_all == core_all_minus_math

    def test_core_submodule_imports(self):
        """Each core submodule should import independently."""
        import examples.scaffolding.core.controller  # noqa: F401
        import examples.scaffolding.core.math_utils  # noqa: F401
        import examples.scaffolding.core.result  # noqa: F401
        import examples.scaffolding.core.scaffolding_llm  # noqa: F401
        import examples.scaffolding.core.task  # noqa: F401
        import examples.scaffolding.core.task_collection  # noqa: F401
        import examples.scaffolding.core.worker  # noqa: F401


# ============================================================================
# 3. Core primitives — functional tests
# ============================================================================


class TestTask:
    """Tests for Task, GenerationTask, ChatTask, and related dataclasses."""

    def test_task_creation(self):
        from examples.scaffolding.core.task import Task

        t = Task()
        assert t.worker_tag is None
        assert t.streaming_output_flag is False
        assert t.streaming_output_list == []

    def test_generation_task_create_from_prompt(self):
        from examples.scaffolding.core.task import GenerationTask

        t = GenerationTask.create_from_prompt("Hello world")
        assert t.input_str == "Hello world"
        assert t.skip_tokenizer is False
        assert t.skip_detokenizer is False

    def test_generation_task_scaffolding_output(self):
        from examples.scaffolding.core.task import GenerationTask

        t = GenerationTask(output_str="result", output_tokens=[10, 20])
        output = t.create_scaffolding_output()
        assert output.text == "result"
        assert output.token_ids == [10, 20]

    def test_stream_generation_task_from_generation(self):
        from examples.scaffolding.core.task import (
            GenerationTask,
            StreamGenerationTask,
        )

        gen = GenerationTask(input_str="prompt", max_tokens=100)
        stream = StreamGenerationTask.create_from_generation_task(gen, streaming_step=5)
        assert stream.input_str == "prompt"
        assert stream.max_tokens == 100
        assert stream.streaming_step == 5

    def test_chat_task_create_from_prompt(self):
        from examples.scaffolding.core.task import (
            ChatTask,
            SystemMessage,
        )

        t = ChatTask.create_from_prompt(
            "What is 2+2?",
            system_prompts=[SystemMessage("You are a math tutor")],
        )
        assert len(t.messages) == 2
        assert t.messages[0].role == "system"
        assert t.messages[1].role == "user"

    def test_chat_task_create_from_messages(self):
        from examples.scaffolding.core.task import (
            ChatTask,
            UserMessage,
        )

        msgs = [UserMessage("hi"), UserMessage("hello")]
        t = ChatTask.create_from_messages(msgs)
        assert len(t.messages) == 2

    def test_chat_task_add_message(self):
        from examples.scaffolding.core.task import (
            AssistantMessage,
            ChatTask,
            UserMessage,
        )

        t = ChatTask()
        t.add_message(UserMessage("Q"))
        t.add_message(AssistantMessage("A"))
        assert len(t.messages) == 2
        assert t.messages[0].role == "user"
        assert t.messages[1].role == "assistant"

    def test_chat_task_messages_to_dict(self):
        from examples.scaffolding.core.task import (
            ChatTask,
            UserMessage,
        )

        t = ChatTask()
        t.add_message(UserMessage("hello"))
        dicts = t.messages_to_dict_content()
        assert dicts == [{"role": "user", "content": "hello"}]

    def test_role_message_str_repr(self):
        from examples.scaffolding.core.task import UserMessage

        m = UserMessage("hi")
        assert '"role": "user"' in str(m)
        assert "user" in repr(m)

    def test_role_message_from_dict(self):
        from examples.scaffolding.core.task import RoleMessage

        m = RoleMessage.from_dict({"role": "user", "content": "test"})
        assert m.role == "user"
        assert m.content == "test"

    def test_assistant_message_with_reasoning(self):
        from examples.scaffolding.core.task import AssistantMessage

        m = AssistantMessage("answer", reasoning="chain of thought")
        assert m.role == "assistant"
        assert m.reasoning == "chain of thought"
        assert '"reasoning"' in str(m)

    def test_openai_tool_description(self):
        from examples.scaffolding.core.task import OpenAIToolDescription

        tool = OpenAIToolDescription(
            "my_func", "Does something", {"x": {"type": "int"}}
        )
        d = tool.to_dict()
        assert d["type"] == "function"
        assert d["function"]["name"] == "my_func"
        assert d["function"]["description"] == "Does something"

    def test_task_status_values(self):
        from examples.scaffolding.core.task import TaskStatus

        assert TaskStatus.SUCCESS.value == "success"
        assert TaskStatus.WORKER_NOT_SUPPORTED.value == "worker_not_supported"
        assert TaskStatus.WORKER_EXECEPTION.value == "worker_exception"


class TestController:
    """Tests for Controller and built-in controller subclasses."""

    def test_controller_is_abstract(self):
        from examples.scaffolding.core.controller import Controller

        ctrl = Controller()
        assert hasattr(ctrl, "task_collections")
        with pytest.raises(NotImplementedError):
            list(ctrl.process([]))

    def test_controller_clone(self):
        from examples.scaffolding.core.controller import Controller

        class MyCtrl(Controller):
            def __init__(self, val):
                super().__init__()
                self.val = val

            def process(self, tasks, **kw):
                yield tasks

        original = MyCtrl(42)
        cloned = original.clone()
        assert cloned.val == 42
        assert cloned is not original

    def test_native_generation_controller(self):
        from examples.scaffolding.core.controller import (
            NativeGenerationController,
        )
        from examples.scaffolding.core.task import GenerationTask

        ctrl = NativeGenerationController(
            sampling_params={"temperature": 0.7, "max_tokens": 100}
        )
        task = GenerationTask(input_str="test")
        results = list(ctrl.process([task]))

        assert len(results) == 1
        assert task.worker_tag == NativeGenerationController.WorkerTag.GENERATION
        assert task.temperature == 0.7
        assert task.max_tokens == 100

    def test_native_generation_controller_ignores_invalid_params(self):
        from examples.scaffolding.core.controller import (
            NativeGenerationController,
        )

        ctrl = NativeGenerationController(
            sampling_params={"invalid_param_xyz": 999, "temperature": 0.5}
        )
        assert "temperature" in ctrl.sampling_params
        assert "invalid_param_xyz" not in ctrl.sampling_params

    def test_native_chat_controller(self):
        from examples.scaffolding.core.controller import (
            NativeChatController,
        )
        from examples.scaffolding.core.task import (
            ChatTask,
            GenerationTask,
        )

        ctrl = NativeChatController()
        task = GenerationTask(input_str="What is 2+2?")
        results = list(ctrl.process([task]))

        assert len(results) == 1
        # NativeChatController wraps in ChatTask
        yielded_tasks = results[0]
        assert isinstance(yielded_tasks[0], ChatTask)

    def test_native_reward_controller(self):
        from examples.scaffolding.core.controller import (
            NativeRewardController,
        )
        from examples.scaffolding.core.task import GenerationTask

        ctrl = NativeRewardController()
        task = GenerationTask(input_str="test")
        results = list(ctrl.process([task]))

        assert len(results) == 1
        assert task.worker_tag == NativeRewardController.WorkerTag.REWARD

    def test_parallel_process_creation(self):
        from examples.scaffolding.core.controller import (
            NativeGenerationController,
            ParallelProcess,
        )
        from examples.scaffolding.core.task import GenerationTask

        ctrl1 = NativeGenerationController()
        ctrl2 = NativeGenerationController()
        tasks1 = [GenerationTask(input_str="a")]
        tasks2 = [GenerationTask(input_str="b")]

        pp = ParallelProcess(
            controllers=[ctrl1, ctrl2],
            tasks_list=[tasks1, tasks2],
            kwargs_list=[{}, {}],
        )
        assert len(pp.sub_gens) == 2


class TestWorker:
    """Tests for Worker base class."""

    def test_worker_is_abstract(self):
        from examples.scaffolding.core.worker import Worker

        w = Worker()
        assert hasattr(w, "task_handlers")

    @pytest.mark.asyncio
    async def test_worker_unsupported_task(self):
        from examples.scaffolding.core.task import Task, TaskStatus
        from examples.scaffolding.core.worker import Worker

        class EmptyWorker(Worker):
            task_handlers = {}

        w = EmptyWorker()
        status = await w.run_task(Task())
        assert status == TaskStatus.WORKER_NOT_SUPPORTED

    @pytest.mark.asyncio
    async def test_worker_register_handler(self):
        from examples.scaffolding.core.task import (
            GenerationTask,
            TaskStatus,
        )
        from examples.scaffolding.core.worker import Worker

        class MyWorker(Worker):
            task_handlers = {}

        async def my_handler(self, task):
            task.output_str = "handled"
            return TaskStatus.SUCCESS

        w = MyWorker()
        w.register_task_handler(GenerationTask, my_handler)
        task = GenerationTask(input_str="test")
        status = await w.run_task(task)
        assert status == TaskStatus.SUCCESS
        assert task.output_str == "handled"

    def test_worker_context_manager(self):
        from examples.scaffolding.core.worker import Worker

        w = Worker()
        result = w.__enter__()
        assert result is w

    def test_is_deterministic_mode(self):
        from examples.scaffolding.core.worker import is_deterministic_mode

        assert is_deterministic_mode() is False

        with patch.dict("os.environ", {"SCAFFOLDING_DETERMINISTIC": "1"}):
            assert is_deterministic_mode() is True


class TestResult:
    """Tests for ScaffoldingOutput and ScaffoldingResult."""

    def test_scaffolding_output(self):
        from examples.scaffolding.core.result import ScaffoldingOutput

        o = ScaffoldingOutput(text="hello", token_ids=[1, 2, 3])
        assert o.text == "hello"
        assert o.token_ids == [1, 2, 3]
        assert o.data is None

    def test_scaffolding_output_with_data(self):
        from examples.scaffolding.core.result import ScaffoldingOutput

        payload = {"key": "value", "nested": [1, 2, 3]}
        o = ScaffoldingOutput(text="hello", token_ids=[1, 2, 3], data=payload)
        assert o.text == "hello"
        assert o.token_ids == [1, 2, 3]
        assert o.data is payload

    def test_scaffolding_result_set_output(self):
        from examples.scaffolding.core.result import (
            ScaffoldingOutput,
            ScaffoldingResult,
        )

        result = ScaffoldingResult()
        assert not result._done

        output = ScaffoldingOutput("text", [1, 2])
        result.set_output(output)

        # After set_output, we should be able to get the result
        # by draining the queue
        loop = asyncio.new_event_loop()
        try:
            done = loop.run_until_complete(result.aresult())
            assert done._done
            assert done.outputs[0].text == "text"
            assert done.outputs[0].token_ids == [1, 2]
        finally:
            loop.close()

    def test_scaffolding_result_set_output_none(self):
        from examples.scaffolding.core.result import ScaffoldingResult

        result = ScaffoldingResult()
        result.set_output(None)

        loop = asyncio.new_event_loop()
        try:
            done = loop.run_until_complete(result.aresult())
            assert done._done
        finally:
            loop.close()

    def test_scaffolding_result_task_collections(self):
        from examples.scaffolding.core.result import ScaffoldingResult

        result = ScaffoldingResult()
        assert result.task_collections is None
        result.set_task_collections({"key": "value"})
        assert result.task_collections == {"key": "value"}


class TestTaskCollection:
    """Tests for TaskCollection and the with_task_collection decorator."""

    def test_task_collection_base(self):
        from examples.scaffolding.core.task_collection import TaskCollection

        tc = TaskCollection()
        tc.before_yield([])  # Should not raise
        tc.after_yield([])  # Should not raise
        assert TaskCollection.get_global_info() is None

    def test_with_task_collection_decorator(self):
        from examples.scaffolding.core.controller import Controller
        from examples.scaffolding.core.task import Task
        from examples.scaffolding.core.task_collection import (
            TaskCollection,
            with_task_collection,
        )

        class MyCollection(TaskCollection):
            def __init__(self):
                super().__init__()
                self.before_count = 0
                self.after_count = 0

            def before_yield(self, tasks):
                self.before_count += 1

            def after_yield(self, tasks):
                self.after_count += 1

        @with_task_collection("my_tc", MyCollection)
        class MyController(Controller):
            def process(self, tasks, **kwargs):
                yield tasks

        ctrl = MyController()
        assert "my_tc" in ctrl.task_collections
        tc = ctrl.task_collections["my_tc"]
        assert isinstance(tc, MyCollection)

        list(ctrl.process([Task()]))
        assert tc.before_count == 1
        assert tc.after_count == 1

    def test_generation_token_counter(self):
        from examples.scaffolding.core.task import GenerationTask
        from examples.scaffolding.core.task_collection import (
            GenerationTokenCounter,
        )

        counter = GenerationTokenCounter()
        task = GenerationTask(output_tokens=[1, 2, 3])

        counter.before_yield([task])
        # Simulate worker adding tokens
        task.output_tokens = [1, 2, 3, 4, 5]
        counter.after_yield([task])

        assert counter.generation_token_count == 2  # 5 - 3 = 2 new tokens

    def test_task_metrics_collector_reset(self):
        from examples.scaffolding.core.task_collection import (
            TaskMetricsCollector,
        )

        TaskMetricsCollector.statistics["test_ctrl"] = [{"data": 1}]
        TaskMetricsCollector.reset("test_ctrl")
        assert TaskMetricsCollector.statistics["test_ctrl"] == []
        TaskMetricsCollector.reset()
        assert TaskMetricsCollector.statistics == {}


class TestMathUtils:
    """Tests for math_utils functions."""

    def test_extract_answer_from_boxed_simple(self):
        from examples.scaffolding.core.math_utils import (
            extract_answer_from_boxed,
        )

        assert extract_answer_from_boxed("The answer is \\boxed{42}") == "42"

    def test_extract_answer_from_boxed_nested(self):
        from examples.scaffolding.core.math_utils import (
            extract_answer_from_boxed,
        )

        assert extract_answer_from_boxed("\\boxed{x^{2}}") == "x^{2}"

    def test_extract_answer_from_boxed_none(self):
        from examples.scaffolding.core.math_utils import (
            extract_answer_from_boxed,
        )

        assert extract_answer_from_boxed("No boxed answer here") is None

    def test_extract_answer_with_regex(self):
        from examples.scaffolding.core.math_utils import (
            extract_answer_with_regex,
        )

        result = extract_answer_with_regex("The final answer is 42")
        assert result == "42"

    def test_extract_answer_with_regex_no_match(self):
        from examples.scaffolding.core.math_utils import (
            extract_answer_with_regex,
        )

        assert extract_answer_with_regex("Nothing relevant") is None

    def test_get_digit_majority_vote_result(self):
        from examples.scaffolding.core.math_utils import (
            get_digit_majority_vote_result,
        )

        results = [
            "The answer is \\boxed{42}",
            "Therefore \\boxed{42}",
            "I get \\boxed{99}",
        ]
        index, answer = get_digit_majority_vote_result(results)
        assert answer == "42"

    def test_get_digit_majority_vote_no_valid(self):
        from examples.scaffolding.core.math_utils import (
            get_digit_majority_vote_result,
        )

        results = ["no boxed", "nothing here"]
        index, answer = get_digit_majority_vote_result(results)
        assert answer is None


# ============================================================================
# 4. AReaL wrapper modules — import and basic function
# ============================================================================


class TestWrapperImports:
    """Verify AReaL wrapper modules import without tensorrt_llm."""

    def test_compat_module_imports(self):
        from examples.scaffolding._compat import (  # noqa: F401
            AssistantMessage,
            BestOfNController,
            ChatTask,
            Controller,
            GenerationTask,
            MajorityVoteController,
            NativeChatController,
            NativeGenerationController,
            NativeRewardController,
            OpenAIToolDescription,
            OpenaiWorker,
            ParallelProcess,
            RoleMessage,
            ScaffoldingLlm,
            ScaffoldingOutput,
            StreamGenerationTask,
            SystemMessage,
            Task,
            TaskCollection,
            TaskStatus,
            UserMessage,
            Worker,
            with_task_collection,
        )

    def test_controllers_module_imports(self):
        from examples.scaffolding.controllers import (  # noqa: F401
            ChatTracer,
            PipelineTrajectoryMaker,
            RLVRRewardController,
            TraceTrajectoryMaker,
        )

    def test_task_module_imports(self):
        from examples.scaffolding.task import (  # noqa: F401
            ChatRewardTask,
            RLVRRewardTask,
            TraceGenerationTask,
        )

    def test_worker_module_imports(self):
        from examples.scaffolding.worker import (  # noqa: F401
            CreateWorkerFromEngine,
            SGLangWorker,
        )

    def test_workflow_module_imports(self):
        from examples.scaffolding.workflow import (
            ScaffoldingWorkflow,  # noqa: F401
        )

    def test_top_level_package_imports(self):
        from examples.scaffolding import (  # noqa: F401
            ChatRewardTask,
            ChatTracer,
            CreateWorkerFromEngine,
            PipelineTrajectoryMaker,
            RLVRRewardController,
            RLVRRewardTask,
            ScaffoldingWorkflow,
            SGLangWorker,
            TraceGenerationTask,
            TraceTrajectoryMaker,
        )


# ============================================================================
# 5. Cross-module integration — core + wrappers work together
# ============================================================================


class TestCrossModuleIntegration:
    """Verify that core primitives and AReaL wrappers interoperate."""

    def test_rlvr_reward_task_inherits_from_core_task(self):
        from examples.scaffolding.core.task import Task
        from examples.scaffolding.task import RLVRRewardTask

        t = RLVRRewardTask(prompt_str="Q", completion_str="A")
        assert isinstance(t, Task)

    def test_sglang_worker_inherits_from_core_openai_worker(self):
        from examples.scaffolding.core.worker import OpenaiWorker
        from examples.scaffolding.worker import SGLangWorker

        assert issubclass(SGLangWorker, OpenaiWorker)

    def test_rlvr_reward_controller_inherits_from_core_controller(self):
        from examples.scaffolding.controllers import RLVRRewardController
        from examples.scaffolding.core.controller import Controller

        assert issubclass(RLVRRewardController, Controller)

    def test_pipeline_trajectory_maker_inherits_from_core_controller(self):
        from examples.scaffolding.controllers import PipelineTrajectoryMaker
        from examples.scaffolding.core.controller import Controller

        assert issubclass(PipelineTrajectoryMaker, Controller)

    def test_native_gen_controller_process_with_compat_import(self):
        """Using _compat imports should produce the same result as core imports."""
        from examples.scaffolding._compat import (
            GenerationTask,
            NativeGenerationController,
        )

        ctrl = NativeGenerationController(sampling_params={"temperature": 0.5})
        task = GenerationTask.create_from_prompt("test")
        results = list(ctrl.process([task]))
        assert len(results) == 1
        assert task.temperature == 0.5

    def test_scaffolding_workflow_is_rollout_workflow(self):
        from examples.scaffolding.workflow import ScaffoldingWorkflow

        from areal.api.workflow_api import RolloutWorkflow

        assert issubclass(ScaffoldingWorkflow, RolloutWorkflow)

    def test_compat_classes_are_same_as_core_classes(self):
        """Verify _compat re-exports are the exact same class objects as core."""
        from examples.scaffolding._compat import (
            Controller as CompatController,
        )
        from examples.scaffolding._compat import (
            GenerationTask as CompatGenTask,
        )
        from examples.scaffolding._compat import (
            ScaffoldingLlm as CompatLlm,
        )
        from examples.scaffolding._compat import Task as CompatTask
        from examples.scaffolding._compat import Worker as CompatWorker
        from examples.scaffolding.core.controller import Controller
        from examples.scaffolding.core.scaffolding_llm import ScaffoldingLlm
        from examples.scaffolding.core.task import GenerationTask, Task
        from examples.scaffolding.core.worker import Worker

        assert CompatTask is Task
        assert CompatGenTask is GenerationTask
        assert CompatController is Controller
        assert CompatWorker is Worker
        assert CompatLlm is ScaffoldingLlm


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
