# SPDX-License-Identifier: Apache-2.0

"""Normal-path offline checks; Docker transport and model responses are doubles."""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import httpx
import pytest

from examples.task_workflows.benchmarks.dapo_math import (
    DapoBenchmark,
    MathTask,
    question_id,
    score_answer,
    write_splits,
)
from examples.task_workflows.benchmarks.swe_rebench import (
    SWERebenchBenchmark,
    SWERebenchTask,
    load_tasks,
    write_tasks,
)
from examples.task_workflows.client import ClientSettings, EpisodeClient
from examples.task_workflows.environments.repository import (
    RepositoryEnvironment,
    RepositorySettings,
)
from examples.task_workflows.types import GradeResult, TaskResult


def _git(directory, *args):
    return subprocess.run(
        ["git", "-C", str(directory), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repository_case(tmp_path):
    """A real git repository, candidate patch and hidden pytest test patch."""
    repo = tmp_path / "image" / "demo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "offline@example.invalid")
    _git(repo, "config", "user.name", "Offline test")
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    commit = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    reference = _git(repo, "diff", "HEAD")
    _git(repo, "reset", "--hard", "HEAD")
    (repo / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(6, 7) == 13\n"
    )
    _git(repo, "add", "test_calc.py")
    tests = _git(repo, "diff", "--cached", "HEAD")
    _git(repo, "reset", "--hard", "HEAD")
    return {
        "instance_id": "example__demo-1",
        "repo": "example/demo",
        "base_commit": commit,
        "image_name": "offline-image",
        "problem_statement": "Fix add so it adds its arguments.",
        "language": "python",
        "patch": reference,
        "test_patch": tests,
        "FAIL_TO_PASS": ["test_calc.py::test_add"],
        "PASS_TO_PASS": [],
        "install_config": {
            "test_cmd": [f"{sys.executable} -m pytest -rA test_calc.py"],
            "log_parser": "parse_log_pytest",
        },
        "_test_image_directory": str(repo.parent),
    }


@pytest.fixture
def docker_transport(tmp_path, monkeypatch, repository_case):
    """Fake only Docker transport; commands, git diffs and pytest really execute."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "docker"
    executable.write_text(
        "#!"
        + sys.executable
        + "\n"
        + r"""
import json, os, pathlib, shutil, subprocess, sys
args = sys.argv[1:]
root = pathlib.Path(os.environ["OFFLINE_DOCKER_ROOT"])
root.mkdir(exist_ok=True)
if args[:2] == ["image", "inspect"]:
    print("sha256:offline")
elif args[0] == "create":
    name = args[args.index("--name") + 1]
    target = root / name
    shutil.copytree(os.environ["OFFLINE_DOCKER_IMAGE"], target)
    workdir = args[args.index("--workdir") + 1] if "--workdir" in args else "/work"
    cwd = target / workdir.lstrip("/")
    cwd.mkdir(parents=True, exist_ok=True)
    (target / "_metadata.json").write_text(json.dumps({"cwd": str(cwd)}))
    print(name)
elif args[0] == "start":
    print(args[1])
elif args[0] == "exec":
    target = root / args[1]
    cwd = json.loads((target / "_metadata.json").read_text())["cwd"]
    command = args[2:]
    command = [item.replace("/patches", str(target / "patches")) for item in command]
    if command == ["bash", "-lc", "command -v timeout >/dev/null"]:
        sys.exit(0)  # The simulated Linux image supplies GNU timeout.
    timeout = None
    if command[0] == "timeout":
        timeout, command = float(command[2]) + 5, command[3:]
    command = ["-c" if item == "-lc" else item for item in command]
    sys.exit(subprocess.run(command, cwd=cwd, timeout=timeout).returncode)
elif args[0] == "cp":
    source, destination = args[1:]
    name, destination = destination.split(":", 1)
    shutil.copytree(source, root / name / destination.lstrip("/"), dirs_exist_ok=True)
elif args[0] == "rm":
    shutil.rmtree(root / args[-1], ignore_errors=True)
else:
    raise RuntimeError(args)
"""
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("OFFLINE_DOCKER_ROOT", str(tmp_path / "containers"))
    monkeypatch.setenv("OFFLINE_DOCKER_IMAGE", repository_case["_test_image_directory"])
    return tmp_path / "containers"


def test_dapo_benchmark_preserves_frozen_task_and_grading_contract(tmp_path):
    """Loading keeps the frozen file's exact order, prompts and scoring labels."""
    tasks = [
        MathTask(question_id(f"Question {i}"), f"Question {i}", str(i))
        for i in range(6)
    ]
    folder = tmp_path / "tasks"
    write_splits(
        tasks,
        folder,
        source={"revision": "frozen"},
        seed=1,
        pilot=2,
        train=2,
        evaluation=2,
    )
    frozen = [
        json.loads(line) for line in (folder / "train.jsonl").read_text().splitlines()
    ]
    benchmark = DapoBenchmark()
    new, manifest = benchmark.load_tasks(folder / "train.jsonl")
    assert [asdict(task) for task in new] == frozen
    assert manifest["source"]["revision"] == "frozen"
    for task in new:
        public = asdict(benchmark.prepare(task))
        assert "gold_answer" not in public
        assert public["message"] == task.prompt()
        result = TaskResult(r"\boxed{" + task.gold_answer + "}", True, {})
        assert benchmark.grade(task, result, tmp_path) == score_answer(
            result.answer, task.gold_answer
        )


def test_repository_environment_exports_real_edits_and_new_files(
    tmp_path, docker_transport, repository_case
):
    """Exercise real git changes through the repository tool transport."""
    environment = RepositoryEnvironment(
        RepositorySettings("offline-image", tmp_path / "tools"),
        "episode-one",
        image_id="sha256:offline",
        repo=repository_case["repo"],
        base_commit=repository_case["base_commit"],
    )
    try:
        outcome = json.loads(
            environment.invoke(
                "workspace_exec",
                {
                    "command": "printf 'new file\\n' > added.txt; printf 'repository note\\n' >&2"
                },
            )
        )
        assert outcome["exit_code"] == 0
        assert outcome["output"] == "repository note\n"
        assert "stderr" not in outcome
        artifacts = environment.export_artifacts()
        assert "new file" in artifacts["patch"] and "added.txt" in artifacts["patch"]
        assert artifacts["base_commit"] == repository_case["base_commit"]
        assert environment.call_timings[0]["seconds"] >= 0
        assert environment.records[0]["command"].startswith("printf")
    finally:
        environment.close()
    assert not list(docker_transport.iterdir())


def test_math_tool_separates_stdout_and_stderr(tmp_path, docker_transport):
    """Keep diagnostics out of calculation output and persist both streams."""
    from examples.task_workflows.environments.python import PythonSandbox, ToolSettings

    runtime = PythonSandbox(
        ToolSettings("offline-image", tmp_path / "tools"),
        "separate-streams",
        image_id="sha256:offline",
    )
    try:
        result = json.loads(
            runtime.execute(
                "import sys; print(6 * 7); print('tool diagnostic', file=sys.stderr)"
            )
        )
        assert result["exit_code"] == 0
        assert result["output"] == result["stdout"] == "42\n"
        assert result["stderr"] == "tool diagnostic\n"
        record = json.loads((runtime.directory / "tools.jsonl").read_text())
        assert record["stdout"] == "42\n"
        assert record["stderr"] == result["stderr"]
    finally:
        runtime.close()
    assert not list(docker_transport.iterdir())


def test_swe_task_manifest_and_public_input_keep_grading_material_private(
    tmp_path, repository_case
):
    """Only issue text and public environment initialization reach the harness."""
    folder = tmp_path / "swe"
    manifest = write_tasks([repository_case], folder, {"revision": "frozen"})
    tasks, actual = load_tasks(folder / "tasks.jsonl")
    assert manifest == actual
    public = json.dumps(asdict(SWERebenchBenchmark().prepare(tasks[0])))
    assert "Fix add" in public
    assert repository_case["test_patch"] not in public
    assert (
        "FAIL_TO_PASS" not in public
        and "install_config" not in public
        and '"patch"' not in public
    )


@pytest.mark.parametrize("benchmark_name", ["dapo_math", "swe_rebench"])
@pytest.mark.asyncio
async def test_shared_client_submits_and_rewards_both_benchmarks(
    tmp_path, monkeypatch, repository_case, benchmark_name
):
    """Both benchmarks use the same session, result, timing and reward flow."""
    benchmark = (
        DapoBenchmark() if benchmark_name == "dapo_math" else SWERebenchBenchmark()
    )
    task = (
        MathTask(
            question_id("Compute six times seven."), "Compute six times seven.", "42"
        )
        if benchmark_name == "dapo_math"
        else SWERebenchTask(repository_case)
    )
    if benchmark_name == "swe_rebench":
        monkeypatch.setattr(
            benchmark,
            "grade",
            lambda task, result, output: GradeResult(1.0, "resolved"),
        )
    observed = []

    def respond(request):
        body = json.loads(request.content)
        observed.append((request.url.path, body))
        if request.url.path == "/rl/start_session":
            return httpx.Response(
                201,
                json={
                    "sessions": [
                        {"session_id": "session", "session_api_key": "private-key"}
                    ]
                },
            )
        if request.url.path == "/v1/responses":
            assert (
                "gold_answer" not in request.content.decode()
                and "test_patch" not in request.content.decode()
            )
            assert (
                body["metadata"]["task_environment"]["kind"]
                == benchmark.prepare(task).environment.kind
            )
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
                        "api_calls": 2,
                        "inference_calls": [{"seconds": 3}],
                        "environment": {
                            "calls": [{"status": "completed"}],
                            "timing_calls": [{"started": 1, "finished": 2}],
                        },
                    },
                },
            )
        if request.url.path == "/rl/set_reward":
            assert body == {"reward": 1.0}
            return httpx.Response(
                200, json={"trajectory_id": 0, "trajectory_ready": True}
            )
        assert request.url.path == "/sessions/close"
        return httpx.Response(200, json={"status": "closed"})

    settings = ClientSettings(
        "http://agent",
        "http://inference",
        "admin",
        "inference-admin",
        "Qwen",
        "resident",
        tmp_path,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        summary = await EpisodeClient(settings, http, benchmark).run([task])
    assert summary["completed"] == summary["correct"] == 1
    assert summary["model_calls"] == 2 and summary["tool_executions"] == 1
    row = json.loads((tmp_path / "episodes.jsonl").read_text())
    assert row["task_index"] == 0 and row["available_at"] <= row["started"]
    assert row["arrival_mode"] == "all_ready"
    assert row["client_admission_started"] <= row["client_admitted"] <= row["submitted"]
    assert (
        row["timing"]["inference_seconds"] == 3 and row["timing"]["tool_seconds"] == 1
    )
    assert "private-key" not in json.dumps(row)
    assert [path for path, _ in observed] == [
        "/rl/start_session",
        "/v1/responses",
        "/rl/set_reward",
        "/sessions/close",
    ]


@pytest.mark.parametrize("kind", ["python", "repository"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_task_service_runs_multi_call_tools_with_both_environments(
    tmp_path, monkeypatch, docker_transport, repository_case, kind, streaming
):
    """A scripted harness exercises actual SDK routing and real environment commands."""
    from openai import OpenAI

    from examples.task_workflows import service
    from examples.task_workflows.harnesses import hermes

    from areal.v2.agent_service.types import AgentRequest

    seen = []

    def respond(request):
        seen.append(json.loads(request.content))
        if seen[-1].get("stream"):
            chunks = [
                {"delta": {"content": "done"}, "finish_reason": None},
                {"delta": {}, "finish_reason": "stop"},
            ]
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text="".join(
                    "data: "
                    + json.dumps(
                        {
                            "id": "offline",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": "Qwen",
                            "choices": [{"index": 0, **choice}],
                            "usage": {
                                "prompt_tokens": 100,
                                "completion_tokens": 20,
                                "total_tokens": 120,
                            }
                            if i
                            else None,
                        }
                    )
                    + "\n\n"
                    for i, choice in enumerate(chunks)
                )
                + "data: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": "Qwen",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    class ScriptedAgent:
        def __init__(self, **kwargs):
            self.request_overrides = kwargs.get("request_overrides", {})
            self.client = self._create_openai_client()

        def _create_openai_client(self):
            return OpenAI(
                base_url="http://inference/v1",
                api_key="session",
                http_client=httpx.Client(transport=httpx.MockTransport(respond)),
            )

        def run_conversation(self, message, task_id):
            def infer(text):
                result = self.client.chat.completions.create(
                    model="Qwen",
                    messages=[{"role": "user", "content": text}],
                    stream=streaming,
                )
                if streaming:
                    list(result)

            infer(message)
            name, arguments = (
                ("math_python", {"code": "print(6 * 7)"})
                if kind == "python"
                else ("workspace_exec", {"command": "printf 'done\\n' > new.txt"})
            )
            hermes._dispatch_tool(name, arguments, task_id=task_id)
            infer("continue")
            return {"final_response": r"\boxed{42}", "completed": True, "api_calls": 2}

        def close(self):
            self.client.close()

    module = ModuleType("run_agent")
    module.AIAgent = ScriptedAgent
    monkeypatch.setitem(sys.modules, "run_agent", module)
    monkeypatch.setattr(service.importlib.metadata, "version", lambda name: "0.19.0")
    monkeypatch.setattr(service, "register_environment_tools", lambda provider: None)
    monkeypatch.setenv("AREAL_TASK_TOOL_ROOT", str(tmp_path / "tools"))
    monkeypatch.setenv("AREAL_TASK_PYTHON_IMAGE", "offline-image")
    task = (
        MathTask("math", "Compute six times seven.", "42")
        if kind == "python"
        else SWERebenchTask(repository_case)
    )
    benchmark = DapoBenchmark() if kind == "python" else SWERebenchBenchmark()
    task_input = benchmark.prepare(task)
    identity = {"incarnation_id": "source", "trajectory_id": 3}
    request = AgentRequest(
        task_input.message,
        "session",
        "run",
        metadata={
            "task_environment": asdict(task_input.environment),
            "areal_inference": {
                "base_url": "http://inference/v1",
                "api_key": "session",
                "model": "Qwen",
                "episode": identity,
            },
        },
    )
    text = []

    class Emitter:
        async def emit_delta(self, value):
            text.append(value)

    agent = service.TaskAgent()
    try:
        response = await agent.run(request, emitter=Emitter())
    finally:
        await agent.close_all_sessions()
    assert text == [r"\boxed{42}"]
    assert len(seen) == len(response.metadata["inference_calls"]) == 2
    assert all(row["areal_episode"] == identity for row in seen)
    environment = response.metadata["environment"]
    assert environment["kind"] == kind and len(environment["calls"]) == 1
    assert len(environment["timing_calls"]) == 1
    trace = response.metadata["simulation_trace"]
    assert trace["lengths_complete"] and trace["dependencies_complete"]
    first, tool, second = trace["nodes"]
    assert first["input_tokens"] == second["input_tokens"] == 100
    assert first["output_tokens"] == second["output_tokens"] == 20
    assert first["cached_prefix_tokens"] is None
    assert first["predecessors"] == []
    assert tool["predecessors"] == [first["name"]]
    assert second["predecessors"] == [tool["name"]]
    assert tool["queued"] <= tool["started"] <= tool["finished"]
    assert tool["seconds"] >= 0 and trace["external_pools"][tool["pool"]] == 1
    if kind == "repository":
        assert "new.txt" in environment["artifacts"]["patch"]
    assert not list(docker_transport.iterdir())


def test_task_trace_parallel_tools_join_before_next_request(tmp_path, docker_transport):
    """Both tools depend on the model, with a join and separately measured queue."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from examples.task_workflows.environments.python import PythonSandbox, ToolSettings
    from examples.task_workflows.trace import CallTrace

    runtime = PythonSandbox(
        ToolSettings("offline-image", tmp_path / "tools"),
        "fanout",
        image_id="sha256:offline",
    )
    trace = CallTrace()
    trace.bind_tools(runtime)
    first = {"started": 0.0, "finished": 1.0}
    trace.request(first)
    barrier = threading.Barrier(2)

    def invoke(_):
        barrier.wait(timeout=10)
        return json.loads(runtime.invoke("math_python", {"code": "print(6 * 7)"}))

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(invoke, range(2)))
        assert all(row["output"].strip() == "42" for row in results)
        second = {"started": 2.0, "finished": 3.0}
        trace.request(second)
        tools = [node for node in trace.nodes if node["kind"] == "external"]
        assert len(tools) == 2
        assert all(node["predecessors"] == [first["name"]] for node in tools)
        assert set(second["predecessors"]) == {node["name"] for node in tools}
        assert all(
            node["queued"] <= node["started"] <= node["finished"] for node in tools
        )
        assert all(node["seconds"] >= 0 for node in tools)
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_task_scheduled_arrivals_continue_during_client_concurrency_wait(
    tmp_path,
):
    """External arrivals keep their schedule while the single client slot is busy."""
    import asyncio
    import time

    class ScheduledClient(EpisodeClient):
        async def run_task(self, task, **kwargs):
            started = time.time()
            await asyncio.sleep(0.1)
            return {
                "task_id": task.task_id,
                "started": started,
                "status": "completed",
                "score": {"reward": 0},
                "metadata": {"api_calls": 0, "environment": {"calls": []}},
            }

    settings = ClientSettings(
        "http://agent",
        "http://inference",
        "admin",
        "admin",
        "Qwen",
        "eval",
        tmp_path,
        concurrency=1,
        arrival_offsets=(0.0, 0.01, 0.02),
    )
    tasks = [MathTask(str(i), "Question", "42") for i in range(3)]
    async with httpx.AsyncClient() as http:
        summary = await ScheduledClient(settings, http).run(tasks)
    rows = [
        json.loads(line)
        for line in (tmp_path / "episodes.jsonl").read_text().splitlines()
    ]
    assert summary["completed"] == 3
    assert all(row["arrival_mode"] == "scheduled" for row in rows)
    assert rows[2]["available_at"] - rows[0]["available_at"] == pytest.approx(0.02)
    assert rows[2]["arrival_observed_at"] < rows[1]["started"]


def test_task_trace_records_tool_payload_and_model_visible_tool_message():
    """Tool payload token counts remain distinct from bytes and message wrappers."""
    from types import SimpleNamespace

    from examples.task_workflows.trace import CallTrace

    class Tokenizer:
        def encode(self, text, **kwargs):
            return text.split()

    runtime = SimpleNamespace(
        kind="python", task_id="tokens", invoke=lambda *args: "答案 is 42"
    )
    trace = CallTrace(tokenizer=Tokenizer())
    trace.bind_tools(runtime)
    trace.request({"started": 0, "finished": 1})
    runtime.invoke("python", {})
    trace.request(
        {"started": 2, "finished": 3},
        messages=[
            {"role": "tool", "tool_call_id": "call-1", "content": "Result: 答案 is 42"}
        ],
    )
    first, tool, second = trace.nodes
    assert tool["returned_text"] == "答案 is 42"
    assert tool["output_tokens"] == 3
    assert tool["output_utf8_bytes"] > len(tool["returned_text"])
    assert second["tool_messages"][0]["tokens"] == 4
    assert second["predecessors"] == [tool["name"]]
    assert tool["predecessors"] == [first["name"]]
    assert trace.snapshot()["tool_lengths_complete"]


def test_swe_grading_uses_real_upstream_parser_on_candidate_patch(
    tmp_path, monkeypatch, docker_transport, repository_case
):
    """Real upstream evaluator + git apply + pytest; only Docker transport is fake."""
    evaluator = os.environ.get("AREAL_SWE_EVALUATOR_ROOT")
    if not evaluator:
        pytest.skip(
            "Set AREAL_SWE_EVALUATOR_ROOT to run the pinned upstream offline check"
        )
    task = SWERebenchTask(repository_case)
    benchmark = SWERebenchBenchmark()
    result = TaskResult(
        "done",
        True,
        {
            "image_id": "sha256:offline",
            "artifacts": {
                "patch": repository_case["patch"],
                "base_commit": repository_case["base_commit"],
            },
        },
    )
    grade = benchmark.grade(task, result, tmp_path / "grade")
    assert grade.reward == 1
    assert grade.details["passed_actual"] == repository_case["FAIL_TO_PASS"]
    assert not list(docker_transport.iterdir())


def test_task_training_configs_compose_same_resident_online_recipe(
    tmp_path, monkeypatch
):
    """Hydra resolves both example recipes without loading the training stack."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    values = {
        "AREAL_TASK_RUN_ROOT": str(tmp_path / "run"),
        "AREAL_TASK_STATE_ROOT": str(tmp_path / "state"),
        "AREAL_MODEL_PATH": str(tmp_path / "model"),
        "AREAL_ACTOR_LIFECYCLE": "resident",
        "AREAL_CHECKPOINT_ESTIMATED_BYTES": "10000000000",
        "AREAL_ACTOR_ADMIN_KEY": "actor",
        "AREAL_INFERENCE_ADMIN_KEY": "inference",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    root = Path(__file__).resolve().parents[1] / "examples/task_workflows/configs"
    configs = []
    for name in ("dapo", "swe"):
        with initialize_config_dir(config_dir=str(root), version_base=None):
            config = OmegaConf.to_container(compose(config_name=name), resolve=True)
        assert config["actor_lifecycle"]["mode"] == "resident"
        assert config["rollout"]["agent"]["mode"] == "online"
        assert config["rollout"]["agent"]["set_reward_finish_timeout"] == 0
        assert (
            config["train_dataset"]["batch_size"]
            == config["rollout"]["consumer_batch_size"]
        )
        configs.append(config)
    assert configs[0]["actor"].pop("trial_name") == "trial0"
    assert configs[1]["actor"].pop("trial_name") == "swe-smoke"
    assert configs[0]["actor"] == configs[1]["actor"]


@pytest.mark.asyncio
async def test_shared_cli_reads_frozen_tasks_and_runs_eval(tmp_path, monkeypatch):
    """Exercise the canonical CLI setup, frozen tasks and evaluation output."""
    from argparse import Namespace

    from examples.task_workflows import client

    tasks = [
        MathTask(question_id(f"Question {i}"), f"Question {i}", "42") for i in range(3)
    ]
    folder = tmp_path / "tasks"
    write_splits(tasks, folder, source={}, seed=1, pilot=1, train=1, evaluation=1)
    monkeypatch.setenv("AREAL_AGENT_ADMIN_KEY", "agent")
    monkeypatch.setenv("AREAL_INFERENCE_ADMIN_KEY", "inference")

    def respond(request):
        if request.url.path == "/sessions/close":
            return httpx.Response(200, json={"status": "closed"})
        assert request.url.path == "/v1/responses"
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "42"}],
                    }
                ],
                "metadata": {
                    "completed": True,
                    "api_calls": 1,
                    "environment": {"calls": []},
                },
            },
        )

    async_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    output = tmp_path / "result"
    args = Namespace(
        tasks=folder / "eval.jsonl",
        output=output,
        agent_url="http://agent",
        inference_url="http://inference",
        model="Qwen",
        mode="eval",
        state_root=None,
        concurrency=1,
        request_timeout=60,
        handoff_timeout=60,
        limit=None,
    )
    assert await client._main(args) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed"] == summary["correct"] == 1
    assert summary["infrastructure_errors"] == 0


@pytest.mark.parametrize("case", ["actor", "rollout", "joint"])
def test_reconfiguration_prepare_freezes_twelve_tasks_without_services(
    tmp_path, monkeypatch, case
):
    from argparse import Namespace

    from examples.task_workflows.reconfiguration import prepare

    for name in (
        "AREAL_TASK_RUN_ROOT",
        "AREAL_TASK_STATE_ROOT",
        "AREAL_MODEL_PATH",
        "AREAL_CHECKPOINT_ESTIMATED_BYTES",
        "AREAL_INFERENCE_ADMIN_KEY",
        "AREAL_ACTOR_ADMIN_KEY",
    ):
        monkeypatch.setenv(name, "acceptance-private-key")
    model = tmp_path / "model"
    model.mkdir()
    tasks = [
        MathTask(question_id(f"Compute {i}+1."), f"Compute {i}+1.", str(i + 1))
        for i in range(14)
    ]
    write_splits(
        tasks, tmp_path / "tasks", source={}, seed=1, pilot=1, train=12, evaluation=1
    )
    args = Namespace(
        case=case,
        devices=list(range(8)),
        model=model,
        tasks=tmp_path / "tasks/train.jsonl",
        output=tmp_path / case,
        estimate_gib=8,
        context_length=8192,
        max_new_tokens=512,
    )
    config, selected, targets, output = prepare(args)
    plan = json.loads((output / "plan.json").read_text())
    assert len(selected) == 12 and len(targets) == 2
    assert plan["updates"] == config.total_train_steps == 3
    assert plan["ordered_task_ids"] == [task.task_id for task in selected]
    assert config.actor_lifecycle.mode == "on_demand"
    assert (
        config.train_dataset.batch_size == 4 and config.rollout.backend == "sglang:d4"
    )
    assert "acceptance-private-key" not in (output / "config.json").read_text()
    assert sorted(path.name for path in output.iterdir()) == [
        "config.json",
        "plan.json",
    ]
    again = prepare(args)
    assert [task.task_id for task in again[1]] == plan["ordered_task_ids"]


@pytest.mark.asyncio
async def test_shared_client_reuses_rl_credentials_without_starting_new_session(
    tmp_path,
):
    from unittest.mock import AsyncMock

    task = MathTask(
        question_id("Compute six times seven."), "Compute six times seven.", "42"
    )
    credentials = {
        "session_id": "retained",
        "session_api_key": "retained-key",
        "areal_episode": {"incarnation_id": "incarnation", "trajectory_id": 1},
    }
    paths = []

    def respond(request):
        paths.append(request.url.path)
        body = json.loads(request.content)
        if request.url.path == "/v1/responses":
            assert body["session_api_key"] == credentials["session_api_key"]
            assert body["inf_episode"] == credentials["areal_episode"]
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
                    "metadata": {"completed": True, "environment": {"calls": []}},
                },
            )
        if request.url.path == "/rl/set_reward":
            assert body["areal_episode"]["trajectory_id"] == 1
            return httpx.Response(
                200, json={"trajectory_id": 1, "trajectory_ready": True}
            )
        assert request.url.path == "/sessions/close"
        return httpx.Response(200, json={"status": "closed"})

    settings = ClientSettings(
        "http://agent",
        "http://inference",
        "admin",
        "inference",
        "math",
        "on_demand",
        tmp_path,
        state_root=tmp_path.absolute(),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = EpisodeClient(settings, http)
        client._admit = AsyncMock()
        client._handoff = AsyncMock(return_value={"verified": True})
        row = await client.run_task(task, credentials=credentials)
    assert row["status"] == "completed" and row["session_id"] == "retained"
    assert row["handoff"]["verified"]
    assert "/rl/start_session" not in paths
    assert "retained-key" not in json.dumps(row)


def test_reconfiguration_telemetry_requires_correlated_phases_and_backend_calls(
    tmp_path,
):
    import time

    from examples.task_workflows.reconfiguration import validate_telemetry

    from areal.elastic.control.transition_timing import TransitionTiming

    phases = (
        "checkpoint",
        "actor_release",
        "actor_start",
        "actor_restore",
        "awex_connect",
        "publish",
        "application_and_commit",
        "admission_open",
        "generation_resume",
        "total",
    )
    for index in (1, 2):
        timing = TransitionTiming(
            tmp_path,
            f"accept-actor-{index}",
            {"weights_version": index},
            {"world_size": 4 if index == 1 else 2},
        )
        timing.record("begin", source=timing.source, target=timing.target)
        for phase in phases:
            with timing.phase(phase):
                pass
        timing.record("end", status="applied", generation=index)
        now = time.monotonic()
        timing.record(
            "first_inference_snapshot",
            proxies=[
                {
                    "proxy": "fixture",
                    "first_call": {
                        "transition_id": timing.transition_id,
                        "status": "completed",
                        "admitted_to_response_seconds": 0.1,
                        "response_ready_monotonic": now + 0.1,
                        "backend_attempts": [
                            {
                                "status": "completed",
                                "version": index,
                                "input_tokens": 12,
                                "output_tokens": 4,
                                "seconds": 0.1,
                                "response_monotonic": now + 0.1,
                            }
                        ],
                    },
                }
            ],
        )
    result = validate_telemetry(tmp_path, "actor")
    assert result["status"] == "complete"
    assert [row["observed_proxies"] for row in result["transitions"]] == [1, 1]
