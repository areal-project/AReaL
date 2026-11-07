# SPDX-License-Identifier: Apache-2.0
"""Async legacy Arena transport, preflight and recoverable task execution."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote_to_bytes

from examples.task_workflows.arena.config import ArenaConfig, canonical, digest
from examples.task_workflows.types import (
    GradeResult,
    TaskExecution,
    TaskInput,
    TaskResult,
)


def write_private(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


class ArenaCLI:
    """The official CLI owns authentication. JSON bodies travel only on stdin."""

    def __init__(self, config: ArenaConfig, timeout: float = 120):
        self.config, self.timeout = config, timeout

    async def call(
        self, *args: str, body: dict | None = None, missing: bool = False
    ) -> Any:
        arena = self.config.value["arena"]
        process = await asyncio.create_subprocess_exec(
            arena["binary"],
            "--profile",
            arena["profile"],
            "--baseurl",
            arena["base_url"],
            "--json",
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            umask=0o077,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(canonical(body) if body is not None else None),
                self.timeout,
            )
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.communicate()
            raise
        envelope = json.loads(stdout)
        if not envelope.get("ok"):
            error = json.dumps(envelope.get("error", {})).lower()
            if missing and any(
                word in error for word in ("404", "not_found", "not found", "不存在")
            ):
                return None
            # API errors can echo request bodies, including model credentials.
            raise RuntimeError(
                f"Arena {' '.join(args[:2])} failed; inspect the platform task"
            )
        return envelope.get("data")


async def check_combination(
    config: ArenaConfig, api: ArenaCLI, tasks: list[TaskInput]
) -> dict:
    value = config.value
    stream = await api.call("stream", "get", value["arena"]["stream_id"])
    stream = stream.get("stream", stream)
    if stream["status"] != "ACTIVE":
        raise ValueError("Stream is not ACTIVE")
    for kind in ("harness", "reward"):
        ref = value[kind + "_ref"]
        version = await api.call(kind, "version-get", ref["key"], ref["version"])
        if version["status"] != "PUBLISHED" or digest(
            version["spec"]
        ) != config.spec_hash(ref):
            raise ValueError(f"Published {kind} differs from the pinned configuration")
        if stream.get("default_" + kind + "_ref") != ref:
            raise ValueError(
                f"Stream default {kind} differs from the pinned combination"
            )
        if (
            kind == "reward"
            and version["spec"].get("accepted_oracle_schema") != value["oracle_schema"]
        ):
            raise ValueError(
                "Reward oracle schema differs from the configured task protocol"
            )
        if kind == "harness":
            expected = "smoke" if value["model_mode"] == "mock" else "runner"
            if (
                version["spec"]
                .get("agent", {})
                .get("env", {})
                .get("AUTOPILOT_ENTRYPOINT")
                != expected
            ):
                raise ValueError("Harness entrypoint differs from declared model mode")
    ids, offset = [], 0
    while True:
        page = await api.call(
            "api",
            "get",
            f"/streams/{stream['stream_id']}/dataset",
            "--query",
            "limit=1000",
            "--query",
            f"offset={offset}",
        )
        ids.extend(page["data_ids"])
        offset += len(page["data_ids"])
        if offset >= page["total"]:
            break
        if not page["data_ids"]:
            raise RuntimeError("Stream pagination stopped before its reported total")
    expected_ids = {row["data_id"] for row in config.tasks.values()}
    if len(ids) != len(expected_ids) or set(ids) != expected_ids:
        raise ValueError("Stream dataset differs from the pinned task index")
    for task in tasks:
        row = config.match(task.task_id, task.message)
        key, version = row["data_id"][4:].rsplit("@", 1)
        env = await api.call("env", "version-get", key, version)
        spec = env["spec"]
        if (
            env["status"] != "PUBLISHED"
            or spec.get("oracle_schema") != value["oracle_schema"]
        ):
            raise ValueError(
                "Task version is unpublished or has a different oracle protocol"
            )
        query = spec["query"]
        if query["uri"].startswith("data:"):
            header, encoded = query["uri"].split(",", 1)
            payload = (
                base64.b64decode(encoded)
                if header.endswith(";base64")
                else unquote_to_bytes(encoded)
            )
            actual = hashlib.sha256(payload).hexdigest()
        else:
            actual = query.get("digest", "").removeprefix("sha256:")
        if actual != row["prompt_sha256"]:
            raise ValueError("Published task query differs from its pinned prompt")
    return {
        "status": "OK",
        "configuration_sha256": config.fingerprint,
        "dataset_count": len(ids),
        "selected_prompts_checked": len(tasks),
        "scope": "Registry, dataset membership and input contract only; no model or training verification",
    }


class ArenaExecutor:
    def __init__(
        self,
        config: ArenaConfig,
        api: ArenaCLI,
        output: Path,
        *,
        identity: dict[str, Any],
        timeout: float = 900,
        poll_seconds: float = 2,
    ):
        self.config, self.api, self.output = config, api, output
        self.identity, self.timeout, self.poll_seconds = identity, timeout, poll_seconds
        self._locks: dict[str, asyncio.Lock] = {}

    def directory(self, request_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", request_id):
            raise ValueError("Invalid episode request ID")
        return self.output / request_id

    async def load(self, request_id: str) -> dict:
        path = self.directory(request_id) / "state.json"
        value = await asyncio.to_thread(
            lambda: json.loads(path.read_text()) if path.exists() else {}
        )
        if value and (
            value["configuration_sha256"] != self.config.fingerprint
            or value["identity"] != self.identity
        ):
            raise ValueError(
                "Resume configuration, inference target or model differs from the saved run"
            )
        return value

    async def save(self, request_id: str, **values: Any) -> None:
        async with self._locks.setdefault(request_id, asyncio.Lock()):
            previous = await self.load(request_id)
            previous.update(
                values,
                configuration_sha256=self.config.fingerprint,
                identity=self.identity,
            )
            await asyncio.to_thread(
                write_private, self.directory(request_id) / "state.json", previous
            )

    async def execute(
        self,
        task: TaskInput,
        *,
        request_id: str,
        base_url: str,
        api_key: str,
        model: str,
        episode: dict | None,
        session_id: str | None = None,
    ) -> TaskExecution:
        row = self.config.match(task.task_id, task.message)
        if task.environment.kind != "python":
            raise ValueError(
                "This published Arena adapter requires the Python environment"
            )
        state = await self.load(request_id)
        value = self.config.value
        # Legacy explicit IDs require short-name-YYMMDDHHMM. Persist the chosen
        # timestamp with the intent so a later resume retains the same identity.
        job_id = state.get("arena_job_id") or (
            "a"
            + hashlib.sha256(request_id.encode()).hexdigest()[:12]
            + "-"
            + datetime.now(UTC).strftime("%y%m%d%H%M")
        )
        env_key = row["data_id"][4:].rsplit("@", 1)[0]
        expected_task_id = f"{job_id}:{env_key}:0"
        if state.get("launch_intent"):
            arena_task = await self.api.call(
                "task", "get", expected_task_id, missing=True
            )
            if arena_task is None:
                raise RuntimeError(
                    "Arena submission outcome is unknown; resume after inspecting the saved job ID"
                )
        else:
            await self.save(
                request_id,
                launch_intent=True,
                arena_job_id=job_id,
                arena_task_id=expected_task_id,
            )
            envs = {"AUTOPILOT_SAMPLING_JSON": json.dumps(value["sampling"])}
            if episode:
                envs["AUTOPILOT_AREAL_EPISODE_JSON"] = json.dumps(episode)
            if session_id:
                envs["AUTOPILOT_AREAL_SESSION_ID"] = session_id
            launched = await self.api.call(
                "stream",
                "launch",
                value["arena"]["stream_id"],
                "--file",
                "-",
                body={
                    "job_id": job_id,
                    "data_id": row["data_id"],
                    "base_url": base_url,
                    "api_key": api_key,
                    "model_name": model,
                    "session_id": request_id,
                    "harness_ref": value["harness_ref"],
                    "reward_ref": value["reward_ref"],
                    "envs": envs,
                },
            )
            if launched["task_id"] != expected_task_id or launched["job_id"] != job_id:
                raise RuntimeError("Arena returned a different job/task identity")
            await self.save(request_id, accepted=True)
        deadline = time.monotonic() + self.timeout
        while True:
            arena_task = await self.api.call("task", "get", expected_task_id)
            if arena_task["status"] == "DONE":
                break
            if arena_task["status"] in ("FAILED", "CANCELED", "CANCELLED", "STOPPED"):
                raise RuntimeError("Arena task did not complete successfully")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Arena task is still active; resume using the saved run directory"
                )
            await asyncio.sleep(self.poll_seconds)
        for field in ("env_ref", "harness_ref", "reward_ref"):
            expected = (
                {"key": env_key, "version": row["data_id"].rsplit("@", 1)[1]}
                if field == "env_ref"
                else value[field]
            )
            if arena_task.get(field) != expected:
                raise RuntimeError("Arena executed a different component version")
        result = await self.api.call("task", "result", expected_task_id)
        if result["status"] != "OK":
            raise RuntimeError("Arena Reward failed; this is not a zero-reward answer")
        artifacts = await self.api.call("task", "artifacts", expected_task_id)
        required = {"answer.json", "environment.json", "call_trace.json"}
        if not required <= {a["name"] for a in artifacts}:
            raise RuntimeError(
                "Arena task is missing required answer/environment/trace artifacts"
            )
        directory = self.directory(request_id)
        for name in sorted(
            required | {a["name"] for a in artifacts if a["name"] == "tools.jsonl"}
        ):
            target = directory / "artifacts" / name
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            await self.api.call(
                "task", "download", expected_task_id, name, "-o", str(target)
            )

        def read(name):
            return json.loads((directory / "artifacts" / name).read_text())

        answer, environment, trace = (
            read("answer.json"),
            read("environment.json"),
            read("call_trace.json"),
        )
        if (
            answer["model_mode"] != value["model_mode"]
            or answer.get("areal_episode") != episode
            or answer.get("areal_session_id") != session_id
            or answer["sampling"] != value["sampling"]
        ):
            raise RuntimeError("Arena answer provenance differs from this episode")
        if (session_id or episode) and (
            value["model_mode"] != "external" or not answer.get("training_eligible")
        ):
            raise RuntimeError("Scripted or unidentified samples cannot enter training")
        if environment.get("infrastructure_error") or any(
            c.get("status") == "infrastructure_error"
            for c in environment.get("calls", [])
        ):
            raise RuntimeError("Arena tool infrastructure failed")
        reward = result["score"]
        if not 0 <= reward <= 1:
            raise RuntimeError("Arena reward is outside the normalized range")
        await asyncio.to_thread(write_private, directory / "result.json", result)
        await asyncio.to_thread(
            write_private,
            directory / "execution.json",
            {
                "task_id": expected_task_id,
                "job_id": job_id,
                "env_ref": arena_task["env_ref"],
                "harness_ref": value["harness_ref"],
                "reward_ref": value["reward_ref"],
            },
        )
        if any(
            result.get("raw", {}).get(key) != answer[key]
            for key in ("model_mode", "training_eligible")
        ):
            raise RuntimeError("Reward provenance differs from the Agent answer")
        requests = [
            node for node in trace.get("nodes", []) if node["kind"] == "request"
        ]
        task_result = TaskResult(
            answer["answer"],
            answer["completed"],
            environment,
            requests,
            answer["model_calls"],
        )
        metadata = {
            "completed": task_result.completed,
            "api_calls": task_result.model_calls,
            "environment": environment,
            "simulation_trace": trace,
            "inference_calls": task_result.inference_calls,
            "arena": {"task_id": expected_task_id, "job_id": job_id},
        }
        raw = result.get("raw", {})
        return TaskExecution(
            task_result,
            metadata,
            GradeResult(
                float(reward),
                raw.get("reason", "arena_reward"),
                raw.get("extracted_answer"),
                raw,
            ),
        )
