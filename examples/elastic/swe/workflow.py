# SPDX-License-Identifier: Apache-2.0

"""Real repository edits and tests inside AReaL's native inline agent session."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path

import aiofiles
from openai import AsyncOpenAI

from .grading import (
    compile_snapshot,
    grade_live_candidate,
    validate_grading,
    verify_candidate_syntax_failure,
)
from .sandbox import SandboxProcess, load_runtime


def tool(name: str, description: str, properties: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    tool(
        "execute_bash",
        "Run a shell command in the persistent repository shell. Inspect files and run targeted tests.",
        {"command": {"type": "string"}},
    ),
    tool(
        "search_replace",
        "Replace an exact source string in a repository file.",
        {key: {"type": "string"} for key in ("filepath", "old_str", "new_str")},
    ),
    tool(
        "task_done",
        "Submit the current repository changes when the issue is resolved.",
        {},
    ),
]


def load_tasks(path: Path):
    from datasets import Dataset

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("SWE dataset is empty")
    for row in rows:
        for key in ("instance_id", "problem_statement", "eval_script", "base_commit"):
            if not row.get(key):
                raise ValueError(f"SWE fixture is missing {key}")
        # Gold is only used by the separate environment validation command.
        row.pop("patch", None)
        row.pop("hints_text", None)
        row["task_id"] = row["instance_id"]
        row["messages"] = [{"role": "user", "content": row["problem_statement"]}]
    return Dataset.from_list(rows)


class SWEAgent:
    def __init__(
        self,
        *,
        log_path: str,
        runtime_config: str,
        max_turns: int = 24,
        max_tokens: int = 4096,
        context_tokens: int = 16384,
        episode_timeout: float = 1800,
    ):
        self.log_path = Path(log_path)
        self.runtime = load_runtime(runtime_config)
        self.max_turns, self.max_tokens = max_turns, max_tokens
        self.context_tokens, self.episode_timeout = context_tokens, episode_timeout

    async def run(self, data: dict, **kwargs) -> float:
        from areal.infra import workflow_context

        started = time.monotonic()
        directory = self.log_path.parent / "swe-episodes" / uuid.uuid4().hex
        await asyncio.to_thread(directory.mkdir, parents=True)
        diagnostics = await asyncio.to_thread((directory / "sandbox.log").open, "w")
        env = SandboxProcess(self.runtime, directory, diagnostics)
        report = {
            "task_id": data["instance_id"],
            "native_task_id": workflow_context.get().task_id,
            "status": "running",
            "turns": 0,
            "tool_calls": 0,
            "llm_seconds": 0.0,
            "tool_seconds": 0.0,
        }
        messages = []
        try:
            async with asyncio.timeout(self.episode_timeout):
                before = time.monotonic()
                await env.start(data)
                report["init_seconds"] = time.monotonic() - before
                client = AsyncOpenAI(
                    base_url=kwargs["base_url"],
                    api_key=kwargs["api_key"],
                    http_client=kwargs.get("http_client"),
                    max_retries=0,
                )
                messages = [
                    {
                        "role": "system",
                        "content": f"You are a software engineer fixing an issue in {env.config.repo_dir}. Inspect the repository, reproduce the bug, edit source files, and run targeted tests. Use execute_bash and search_replace to make actual changes. Do not modify the reference tests to hide failures. Call task_done when finished. Your final repository diff is evaluated by real tests.",
                    },
                    {"role": "user", "content": data["problem_statement"]},
                ]
                total_tokens = 0
                previous_length = 0
                report["exit_reason"] = "turn_budget"
                for turn in range(1, self.max_turns + 1):
                    pending = messages[previous_length:]
                    pending_bytes = len(
                        json.dumps(pending, ensure_ascii=False).encode()
                    )
                    # Byte-BPE conservative admission. Never crop existing concat history.
                    framing = 512 + (
                        len(json.dumps(TOOLS).encode()) if not total_tokens else 0
                    )
                    if (
                        total_tokens + pending_bytes + self.max_tokens + framing
                        > self.context_tokens
                    ):
                        report["exit_reason"] = "context_budget"
                        break
                    before = time.monotonic()
                    response = await client.chat.completions.create(
                        model="default",
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                        temperature=1.0,
                        max_tokens=self.max_tokens,
                    )
                    report["llm_seconds"] += time.monotonic() - before
                    report["turns"] = turn
                    if response.usage is None:
                        raise RuntimeError("Native completion is missing token usage")
                    total_tokens = response.usage.total_tokens
                    message = response.choices[0].message
                    messages.append(message.model_dump(exclude_none=True))
                    previous_length = len(messages)
                    if not message.tool_calls:
                        report["exit_reason"] = "final_answer"
                        break
                    done = False
                    for call in message.tool_calls:
                        before = time.monotonic()
                        try:
                            arguments = json.loads(call.function.arguments)
                            if not isinstance(arguments, dict):
                                raise ValueError("Tool arguments must be an object")
                            schema = next(
                                (
                                    t["function"]["parameters"]
                                    for t in TOOLS
                                    if t["function"]["name"] == call.function.name
                                ),
                                None,
                            )
                            if (
                                schema is None
                                or set(arguments) != set(schema["required"])
                                or any(
                                    not isinstance(v, str) for v in arguments.values()
                                )
                            ):
                                raise ValueError("Unknown tool or invalid arguments")
                        except (ValueError, TypeError) as error:
                            result = {"error": str(error)}
                        else:
                            if call.function.name == "task_done":
                                done, result = True, {"submitted": True}
                            else:
                                result = await getattr(env, call.function.name)(
                                    **arguments
                                )
                        report["tool_seconds"] += time.monotonic() - before
                        report["tool_calls"] += 1
                        content = json.dumps(result, ensure_ascii=False)
                        if len(content) > 6000:
                            content = (
                                content[:6000]
                                + "\n[tool output truncated at 6000 characters]"
                            )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": content,
                            }
                        )
                    if done:
                        report["exit_reason"] = "task_done"
                        break
                if not report["turns"]:
                    raise RuntimeError("SWE prompt does not fit the configured context")
                before = time.monotonic()
                final = await env.submit_final_solution()
                if final["returncode"]:
                    raise RuntimeError("Could not extract candidate patch")
                patch = final["output"]
                report.update(
                    patch=patch,
                    patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
                )
                before_compile = await compile_snapshot(env, patch)
                reward, output = await grade_live_candidate(env, patch)
                # Preserve evidence even if validation rejects an incomplete
                # grading result. Only validated rewards enter training.
                report.update(raw_reward=reward, grading_output=output)
                candidate_failure = await verify_candidate_syntax_failure(
                    env, patch, reward, output, before_compile
                )
                kind = validate_grading(
                    reward, output, env.tool_call_failed_count, candidate_failure
                )
                report.update(
                    reward=float(reward),
                    grading_kind=kind,
                    reward_seconds=time.monotonic() - before,
                    candidate_syntax_failure=candidate_failure,
                    status="graded",
                )
        except BaseException as error:
            report.update(
                status="cancelled"
                if isinstance(error, asyncio.CancelledError)
                else "failed",
                error=f"{type(error).__name__}: {error}",
            )
            raise
        finally:
            before = time.monotonic()
            try:
                await env.close()
                report["cleanup_verified"] = True
            except BaseException as error:
                if isinstance(error, asyncio.CancelledError) and env.cleanup_verified:
                    report.update(status="cancelled", cleanup_verified=True)
                else:
                    report.update(
                        status="failed",
                        cleanup_error=f"{type(error).__name__}: {error}",
                    )
                raise
            finally:
                report.update(
                    cleanup_seconds=time.monotonic() - before,
                    seconds=time.monotonic() - started,
                    wall_time=time.time(),
                    monotonic=time.monotonic(),
                    instance_id=env.instance_id,
                    calls=env.calls,
                    messages=messages,
                )
                await asyncio.to_thread(diagnostics.close)
                async with aiofiles.open(directory / "report.json", "w") as stream:
                    await stream.write(json.dumps(report, indent=2))
        report["status"] = "completed"
        record = {
            k: v
            for k, v in report.items()
            if k not in ("messages", "calls", "patch", "grading_output")
        }
        record["report"] = str(directory / "report.json")
        async with aiofiles.open(self.log_path, "a") as stream:
            await stream.write(json.dumps(record) + "\n")
        return report["reward"]
