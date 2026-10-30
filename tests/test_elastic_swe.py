# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import subprocess
import sys
import textwrap

import httpx
import pytest

from examples.elastic.swe.grading import validate_grading
from examples.elastic.swe.sandbox import SandboxProcess
from examples.elastic.swe.workflow import SWEAgent


def test_multitask_dataset_preserves_provenance_and_removes_gold(tmp_path):
    from examples.elastic.swe.prepare_data import write_dataset
    from examples.elastic.swe.workflow import load_tasks

    paths = []
    for task in ("repo__project-1", "repo__project-2", "repo__project-3"):
        path = tmp_path / (task + ".json")
        path.write_text(
            json.dumps(
                {
                    "instance_id": task,
                    "problem_statement": "Fix this issue",
                    "base_commit": "a" * 40,
                    "eval_script": "run-reference-tests",
                    "patch": "secret-gold-patch",
                    "hints_text": "secret-hint",
                    "source_image": "upstream/" + task,
                    "provenance": {"dataset": "SWE-Gym/SWE-Gym", "split": "train"},
                }
            )
        )
        paths.append(path)
    source = {"dataset": "SWE-Gym/SWE-Gym", "revision": "pinned", "split": "train"}
    output = tmp_path / "tasks.jsonl"
    images = [
        {
            "task": path.stem,
            "image": "upstream/" + path.stem,
            "digest": "sha256:" + "a" * 64,
        }
        for path in paths
    ]
    manifest = write_dataset(
        paths, output, repeats=2, seed=1, source=source, images=images
    )
    dataset = load_tasks(output)
    ids = [row["instance_id"] for row in dataset]
    assert len(ids) == 6 and len(set(ids[:3])) == len(set(ids[3:])) == 3
    assert (
        "secret-gold" not in output.read_text()
        and "secret-hint" not in output.read_text()
    )
    assert manifest["source"] == source and manifest["unique_tasks"] == 3
    assert manifest["images"]["sha256"]
    assert json.loads(output.with_suffix(".images.json").read_text()) == images
    assert all(
        task["fixture_sha256"] and task["source_image"] for task in manifest["tasks"]
    )
    second = tmp_path / "repeat.jsonl"
    write_dataset(paths, second, repeats=2, seed=1, source=source)
    assert output.read_bytes() == second.read_bytes()
    with pytest.raises(ValueError, match="Duplicate task"):
        write_dataset(
            [paths[0], paths[0]],
            tmp_path / "bad.jsonl",
            repeats=1,
            seed=1,
            source=source,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cleanup_fails", "grading_case"),
    [
        (False, "zero"),
        (False, "one"),
        (False, "telemetry_error"),
        (False, "incomplete_import"),
        (False, "incomplete_segfault"),
        (False, "sdk_error"),
        (False, "deadline"),
        (False, "cancelled"),
        (False, "patch_error"),
        (False, "tool_error"),
        (True, "zero"),
        (True, "incomplete_import"),
    ],
)
async def test_native_swe_tools_grade_live_worktree_then_confirm_cleanup(
    tmp_path, monkeypatch, cleanup_fails, grading_case
):
    calls = []

    class Sandbox:
        instance_id = "test-owned-sandbox"
        tool_call_failed_count = 0

        def __init__(self, *args):
            from types import SimpleNamespace

            self.config = SimpleNamespace(repo_dir="/testbed")
            self.calls = []

        async def start(self, data):
            calls.append("start")

        async def execute_bash(self, command):
            assert command == "pytest tests/test_bug.py"
            calls.append("tool")
            if grading_case == "tool_error":
                raise RuntimeError("Tool transport failed before grading")
            return {"returncode": 0, "output": "1 passed"}

        async def submit_final_solution(self):
            calls.append("patch")
            if grading_case == "patch_error":
                return {"returncode": 1, "output": "git diff failed"}
            return {"returncode": 0, "output": ""}

        async def request(self, method, **kwargs):
            assert method == "get_reward" and kwargs == {
                "patch": "",
                "return_output": True,
            }
            calls.append("grade")
            if grading_case.startswith("incomplete_"):
                error = (
                    "AttributeError: 'list' object has no attribute 'items'"
                    if grading_case == "incomplete_import"
                    else "Fatal Python error: Segmentation fault"
                )
                return 0.0, error + "\nEvaluation omitted reference tests: test_bug"
            if grading_case == "sdk_error":
                raise RuntimeError("Sandbox RPC response stream closed")
            if grading_case == "deadline":
                await asyncio.Event().wait()
            if grading_case == "cancelled":
                raise asyncio.CancelledError()
            if grading_case in ("one", "telemetry_error"):
                return 1.0, "AUTOPILOT_SWE_RESOLVED=true"
            return 0.0, "AUTOPILOT_SWE_RESOLVED=false"

        async def close(self):
            calls.append("cleanup")
            if cleanup_fails:
                raise RuntimeError("owned pod remains")

    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        message = {"role": "assistant", "content": "Finished."}
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "execute_bash",
                            "arguments": json.dumps(
                                {"command": "pytest tests/test_bug.py"}
                            ),
                        },
                    }
                ],
            }
        else:
            assert body["messages"][:2] == requests[0]["messages"]
            assert body["messages"][-1]["tool_call_id"] == "call-1"
        return httpx.Response(
            200,
            json={
                "id": "completion",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                },
            },
        )

    monkeypatch.setattr("examples.elastic.swe.workflow.SandboxProcess", Sandbox)
    monkeypatch.setattr("examples.elastic.swe.workflow.load_runtime", lambda _: {})
    if grading_case == "telemetry_error":

        async def progress_disk_error(*args):
            raise OSError("progress snapshot unavailable")

        monkeypatch.setattr(
            "examples.elastic.swe.progress.aiofiles.os.replace", progress_disk_error
        )
    agent = SWEAgent(log_path=str(tmp_path / "episodes.jsonl"), runtime_config="test")
    if grading_case == "deadline":
        agent.episode_timeout = 0.5
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        run = agent.run(
            {"instance_id": "test-task", "problem_statement": "Fix the bug."},
            base_url="https://native.invalid/v1",
            api_key="session-test",
            http_client=http,
        )
        if cleanup_fails:
            with pytest.raises(RuntimeError, match="owned pod remains"):
                await run
            assert not (tmp_path / "episodes.jsonl").exists()
        elif grading_case == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await run
            assert not (tmp_path / "episodes.jsonl").exists()
        elif grading_case == "tool_error":
            with pytest.raises(RuntimeError, match="before grading"):
                await run
            assert not (tmp_path / "episodes.jsonl").exists()
        else:
            assert await run == (
                1.0 if grading_case in ("one", "telemetry_error") else 0.0
            )
            record = json.loads((tmp_path / "episodes.jsonl").read_text())
            report = json.loads(
                next(tmp_path.glob("swe-episodes/*/report.json")).read_text()
            )
            assert record["cleanup_verified"] and record["turns"] == 2
            assert report["status"] == "graded" and record["status"] == "completed"
            if grading_case not in ("zero", "one", "telemetry_error"):
                assert report["grading_kind"] == "exception_zero"
                assert record["grading_kind"] == "exception_zero"
                assert report["grading_error_type"] == (
                    "TimeoutError" if grading_case == "deadline" else "RuntimeError"
                )
                assert report["grading_error_type"] in report["grading_error_traceback"]
                assert report["grading_stage"] in (
                    "extract_patch",
                    "evaluate",
                    "validate",
                )
                if grading_case.startswith("incomplete_"):
                    assert report["raw_reward"] == 0.0
                    assert "omitted reference tests" in report["grading_output"]
                    assert "patch" in report and "patch_sha256" in report
                if grading_case == "patch_error":
                    assert report["patch_extraction_error_output"] == "git diff failed"
            else:
                assert record["grading_kind"] == "completed_reference_report"
                assert "grading_error" not in record
                if grading_case == "telemetry_error":
                    assert "snapshot unavailable" in report["progress_error"]
    expected = ["start", "tool"]
    if grading_case != "tool_error":
        expected.append("patch")
        if grading_case != "patch_error":
            expected += ["patch", "grade"]
    assert calls == [*expected, "cleanup"]
    if cleanup_fails or grading_case in ("cancelled", "tool_error"):
        report = json.loads(
            next(tmp_path.glob("swe-episodes/*/report.json")).read_text()
        )
        assert report["status"] == (
            "cancelled" if grading_case == "cancelled" else "failed"
        )


def test_swe_grading_rejects_infrastructure_errors_and_incomplete_reports():
    for reward, output, failures in [
        (0, "get_reward error", 0),
        (1, "AUTOPILOT_SWE_RESOLVED=true", 1),
        (0, "AUTOPILOT_SWE_RESOLVED=true", 0),
    ]:
        with pytest.raises(RuntimeError):
            validate_grading(reward, output, failures)


@pytest.mark.asyncio
@pytest.mark.parametrize("gym", [False, True])
async def test_syntax_reward_requires_unchanged_candidate_and_working_baseline(
    monkeypatch, gym
):
    from copy import deepcopy
    from types import SimpleNamespace

    from examples.elastic.swe.grading import verify_candidate_syntax_failure

    snapshot = {
        "python": "/opt/miniconda3/envs/testbed/bin/python",
        "files": {
            "pkg/core.py": {
                "source_sha256": "candidate",
                "baseline_sha256": "base",
                "baseline_compiles": True,
                "candidate_error": {
                    "type": "SyntaxError",
                    "line": 7,
                    "message": "invalid syntax",
                },
            }
        },
    }
    after = deepcopy(snapshot)

    async def compile_again(*args):
        return after

    monkeypatch.setattr("examples.elastic.swe.grading.compile_snapshot", compile_again)
    marker = (
        "Evaluation omitted reference tests: ['test_case']"
        if gym
        else "Evaluation did not report every reference test\n>>>>> Start Test Output\n>>>>> End Test Output"
    )
    output = (
        marker
        + '\nTesting result:\nFile "pkg/core.py", line 7\nSyntaxError: invalid syntax'
    )
    env = SimpleNamespace(tool_call_failed_count=0)
    assert await verify_candidate_syntax_failure(env, "patch", 0, output, snapshot)
    after["files"]["pkg/core.py"]["source_sha256"] = "changed-candidate"
    assert (
        await verify_candidate_syntax_failure(env, "patch", 0, output, snapshot) is None
    )
    after = deepcopy(snapshot)
    snapshot["files"]["pkg/core.py"]["baseline_compiles"] = False
    after["files"]["pkg/core.py"]["baseline_compiles"] = False
    assert (
        await verify_candidate_syntax_failure(env, "patch", 0, output, snapshot) is None
    )


@pytest.mark.asyncio
async def test_swe_close_waits_for_owned_cleanup_before_propagating_cancellation(
    tmp_path, monkeypatch
):
    env = SandboxProcess({}, tmp_path, None)
    started, finish = asyncio.Event(), asyncio.Event()
    completed = []

    async def close():
        started.set()
        await finish.wait()
        completed.append(True)

    monkeypatch.setattr(env, "_close", close)
    task = asyncio.create_task(env.close())
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert completed == [True]


def test_actor_tree_release_preserves_live_uvloop_subprocess_reaping():
    """Run libuv and Actor-style cleanup together; a handler reset hangs wait()."""
    code = textwrap.dedent("""
        import asyncio, concurrent.futures, subprocess, sys, threading
        import uvloop
        from areal.infra.utils.proc import kill_process_tree
        started, release = threading.Event(), threading.Event()
        result = concurrent.futures.Future()
        async def workflow():
            child = await asyncio.create_subprocess_exec(
                sys.executable, '-c', 'import sys; sys.stdin.read(1); print("tool-done")',
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
            started.set()
            await asyncio.to_thread(release.wait)
            child.stdin.write(b'x')
            await child.stdin.drain()
            output, _ = await asyncio.wait_for(child.communicate(), timeout=3)
            assert child.returncode == 0 and output.strip() == b'tool-done'
        def worker():
            loop = uvloop.new_event_loop()
            try:
                loop.run_until_complete(workflow())
                result.set_result(True)
            except BaseException as error:
                result.set_exception(error)
            finally:
                loop.close()
        target = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        assert started.wait(10)
        kill_process_tree(target.pid, timeout=1)
        release.set()
        assert result.result(timeout=10)
        thread.join(timeout=3)
        assert not thread.is_alive()
    """)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=25
    )
    assert result.returncode == 0, result.stdout + result.stderr
