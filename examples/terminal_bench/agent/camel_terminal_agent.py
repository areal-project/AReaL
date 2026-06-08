from __future__ import annotations

import asyncio
import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from agent_rl_config import TaskTimeouts
from camel.messages import BaseMessage
from camel.toolkits import FunctionTool, TerminalToolkit
from transformers import PreTrainedTokenizerFast

from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.parsers.base_parser import UnitTestStatus
from terminal_bench.parsers.parser_factory import ParserFactory
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from terminal_bench.terminal.terminal import Terminal

from areal.experimental.camel.openai_model import AReaLOpenAICompatibleModel
from areal.utils.perf_tracer import (
    Category,
    atrace_scope,
    atrace_session_phase,
    session_context,
    trace_perf,
    trace_scope,
)

from .chat_agent_trace import ChatAgentTrace
from .prompts import get_developer_agent_prompt

DATASET_ROOT = Path(__file__).resolve().parents[3] / "dataset"


class CamelTerminalAgent:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast | None = None,
        max_tokens_per_turn: int = 1024,
        max_total_tokens: int = 40000,
        output_path: str = "CamelTerminalAgent_Output",
        max_iteration: int = 50,
        executor: ThreadPoolExecutor | None = None,
        task_timeouts: TaskTimeouts | None = None,
        non_think_mode: bool = True,
        encourage_completion_reward: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_total_tokens = max_total_tokens
        self.output_path = output_path
        self.max_iteration = max_iteration
        self.task_timeouts = task_timeouts or TaskTimeouts()
        self.executor = executor
        self.non_think_mode = non_think_mode
        self.encourage_completion_reward = encourage_completion_reward
        assert self.executor is not None, (
            "Executor must be provided to CamelTerminalAgent"
        )

    @session_context()
    @trace_perf("CamelTerminalAgent.run_agent", category=Category.COMPUTE)
    async def run_agent(
        self,
        data,
        client,
        uid: str | None = None,
        traj_i: int = 0,
    ) -> float | None:
        """Execute a complete agent workflow: setup environment, run agent, cleanup."""
        task_name = data.get("task_name")
        self.task_name = task_name
        self.uid = uid
        self.traj_i = traj_i
        self.meta_info = {}
        reward = None

        print(f"Running task {task_name}")

        try:
            async with atrace_scope(
                f"reset_env:{task_name}, traj:{traj_i}",
                args={"uid": uid, "timeout": self.task_timeouts._reset_env},
            ):
                prompt = await self.run_in_executor(
                    self._reset_env,
                    data,
                    uid,
                    timeout=self.task_timeouts._reset_env,
                )
            print(f"env started: {task_name}")

            async with atrace_scope(
                f"reset_agent:{task_name}, traj:{traj_i}",
                args={"uid": uid, "timeout": self.task_timeouts._reset_agent},
            ):
                await self.run_in_executor(
                    self._reset_agent,
                    client,
                    timeout=self.task_timeouts._reset_agent,
                )

            try:
                async with atrace_scope(
                    f"agent_astep:{task_name}, traj:{traj_i}",
                    args={"uid": uid, "timeout": self.task_timeouts.agent_astep},
                ):
                    self.response = await self.agent.astep(prompt)
            except TimeoutError as exc:
                print(f"Agent step timeout for task {task_name}: {exc}")
            print(f"Task {task_name}: agent responded")

            async with atrace_session_phase(
                "reward",
                start_payload={
                    "task_name": task_name,
                    "traj_i": traj_i,
                    "uid": uid,
                    "timeout": self.task_timeouts._evaluate_completion_sync,
                },
            ):
                async with atrace_scope(
                    f"evaluate_completion_sync:{task_name}, traj:{traj_i}",
                    args={
                        "uid": uid,
                        "timeout": self.task_timeouts._evaluate_completion_sync,
                    },
                ):
                    print("try to set rewards")
                    reward = await self.run_in_executor(
                        self._evaluate_completion_sync,
                        timeout=self.task_timeouts._evaluate_completion_sync,
                    )
                    print(f"reward from run in executor is set as {reward}")
            client.set_last_reward(reward)

        except TimeoutError as exc:
            print(f"Timeout for task {task_name}: {exc}")
        except Exception as exc:
            print(f"Error in task {task_name}: {exc}")
            import traceback

            traceback.print_exc()
        finally:
            try:
                if hasattr(self, "terminal") and self.terminal is not None:
                    async with atrace_scope(
                        f"cleanup_env:{task_name}, traj:{traj_i}",
                        args={"uid": uid, "timeout": self.task_timeouts._cleanup},
                    ):
                        await self.run_in_executor(
                            self._close_env,
                            timeout=self.task_timeouts._cleanup,
                        )
                    print(f"Task {task_name}: cleaned up")
            except Exception as exc:
                print(f"Cleanup error for task {task_name}: {exc}")
            finally:
                return reward

    def _close_env(self):
        if self.terminal:
            self.terminal.stop(timeout=self.task_timeouts._cleanup)

    async def run_in_executor(self, fn, *args, timeout: float | None = None, **kwargs):
        loop = asyncio.get_running_loop()
        executor_task = loop.run_in_executor(
            self.executor,
            partial(fn, *args, **kwargs),
        )
        if timeout is not None:
            return await asyncio.wait_for(executor_task, timeout=timeout)
        return await executor_task

    def _reset_env(self, task: dict, uid: str | None):
        output_path = Path(self.output_path).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        task_path = DATASET_ROOT / task.get("task_path")
        print(f"Task path: {task_path}")
        instruction = task.get("instruction")
        task_id = task.get("task_name")

        self.trial_handler = TrialHandler(
            trial_name=f"{task_id}.{uid}.areal-run",
            input_path=task_path,
            output_path=output_path,
        )

        task_config = self.trial_handler.task
        self.parser = ParserFactory.get_parser(task_config.parser_name)

        self.client_container_name = f"{self.trial_handler.client_container_name}"
        self.terminal = Terminal(
            client_container_name=self.trial_handler.client_container_name,
            client_image_name=self.trial_handler.client_image_name,
            docker_compose_path=self.trial_handler.task_paths.docker_compose_path,
            docker_image_name_prefix=self.trial_handler.docker_image_name_prefix,
            sessions_logs_path=self.trial_handler.trial_paths.sessions_path,
            agent_logs_path=self.trial_handler.trial_paths.agent_logging_dir,
            no_rebuild=True,
            cleanup=False,
        )
        with trace_scope(
            f"reset_env.start_terminal:{task_id}, traj:{self.traj_i}",
            args={"uid": uid},
        ):
            self.terminal.start(timeout=self.task_timeouts._reset_env)

        return f"Task name:{self.task_name}\nTask instruction: {instruction}"

    def _reset_agent(self, client):
        session_logs_dir = (
            self.trial_handler.trial_paths.sessions_path
            / "terminal_toolkit_session_logs"
        )
        terminal_toolkit = TerminalToolkit(
            timeout=20.0,
            working_directory=None,
            use_docker_backend=True,
            docker_container_name=self.trial_handler.client_container_name,
            session_logs_dir=session_logs_dir,
            safe_mode=False,
        )
        tools = [
            FunctionTool(terminal_toolkit.shell_exec),
            FunctionTool(terminal_toolkit.shell_view),
            FunctionTool(terminal_toolkit.shell_write_to_process),
            FunctionTool(terminal_toolkit.shell_write_content_to_file),
        ]

        system_message = get_developer_agent_prompt(
            current_date=str(datetime.date.today()),
            system="Linux (in Docker)",
            machine="aarch64",
            is_workforce=False,
            non_think_mode=self.non_think_mode,
        )
        print("starting chat agent")
        os.environ["CAMEL_MODEL_LOG_ENABLED"] = "True"
        os.environ["CAMEL_LOG_DIR"] = str(
            self.trial_handler.trial_paths.sessions_path.parent / "CAMEL_LOG_DIR"
        )
        model = AReaLOpenAICompatibleModel(
            openai_client=client,
            tokenizer=self.tokenizer,
            model_type="areal",
            model_config_dict={
                "max_completion_tokens": self.max_tokens_per_turn,
            },
        )
        self.agent = ChatAgentTrace(
            system_message=BaseMessage.make_assistant_message(
                role_name="Developer Agent",
                content=system_message,
            ),
            model=model,
            tools=tools,
            token_limit=self.max_total_tokens,
            step_timeout=self.task_timeouts.agent_astep,
        )
        self.agent.reset()
        self.agent.max_iteration = self.max_iteration
        print(f"{self.task_name}: agent started")

    def _evaluate_completion_sync(self) -> float:
        assert self.trial_handler is not None and self.terminal is not None

        paths = [self.trial_handler.task_paths.run_tests_path]
        if self.trial_handler.task_paths.test_dir.exists():
            paths.append(self.trial_handler.task_paths.test_dir)
        with trace_scope(
            f"evaluate_completion_sync.copy_tests:{self.task_name}, traj:{self.traj_i}"
        ):
            self.terminal.copy_to_container(
                paths=paths,
                container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR),
            )

        print("running tests in a new shell")
        with trace_scope(
            f"evaluate_completion_sync.create_test_session:{self.task_name}, traj:{self.traj_i}"
        ):
            test_session = self.terminal.create_session(
                "tests",
                is_active_stream=False,
                as_configured_user=False,
            )

        test_script_path = str(DockerComposeManager.CONTAINER_TEST_DIR / "run-tests.sh")
        try:
            with trace_scope(
                f"evaluate_completion_sync.run_tests:{self.task_name}, traj:{self.traj_i}"
            ):
                test_session.send_keys(
                    [f"bash {test_script_path}", "Enter"],
                    block=True,
                    max_timeout_sec=min(
                        self.task_timeouts._evaluate_completion_sync,
                        4 * self.trial_handler.task.max_test_timeout_sec,
                    ),
                )
            test_output = test_session.capture_pane(capture_entire=True)
            parser_results = self.parser.parse(test_output)

            all_passed = parser_results and all(
                status == UnitTestStatus.PASSED for status in parser_results.values()
            )
            pass_ratio = (
                sum(
                    1
                    for status in parser_results.values()
                    if status == UnitTestStatus.PASSED
                )
                / len(parser_results)
                if parser_results
                else 0.0
            )
            results_path = str(
                self.trial_handler.trial_paths.sessions_path.parent
                / "test_results.json"
            )
            result_dict = {
                "test_results": {
                    k: (v == UnitTestStatus.PASSED) for k, v in parser_results.items()
                },
                "all_passed": all_passed,
                "pass_ratio": pass_ratio,
            }
            try:
                result_dict["iteration"] = len(self.response.info["tool_calls"])
                result_dict.update(self.response.info["usage"])
            except Exception:
                pass
            with open(results_path, "w") as f:
                json.dump(result_dict, f, indent=4)

        except Exception as exc:
            print(exc)
            all_passed = False
            pass_ratio = 0.0

        if self.encourage_completion_reward and pass_ratio == 1.0:
            pass_ratio += 1.0

        return pass_ratio
