# SPDX-License-Identifier: Apache-2.0

"""Isolated SDK RPC with per-episode process and Kubernetes ownership."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from areal.elastic.control.processes import OWNER_ENV, stop_owned_processes


def load_runtime(path: str) -> dict:
    runtime = json.loads(Path(path).read_text())
    for key in (
        "python",
        "pythonpath",
        "system_url",
        "environment",
        "kubectl",
        "kubeconfig",
        "sandbox_namespace",
    ):
        if not runtime.get(key):
            raise ValueError(f"SWE runtime is missing {key}")
    for key in ("python", "kubectl", "kubeconfig"):
        if not Path(runtime[key]).is_file():
            raise ValueError(f"SWE runtime {key} does not exist")
    runtime.setdefault("transport_timeout", 1900)
    if runtime["transport_timeout"] < 1805:
        raise ValueError(
            "SWE HTTP read timeout must cover the SDK's 1805s tool deadline"
        )
    return runtime


class SandboxProcess:
    def __init__(self, runtime: dict, directory: Path, stderr):
        self.runtime, self.directory, self.stderr = runtime, directory, stderr
        self.ownership_file = directory / "ownership.json"
        self.owner = f"swe-{uuid.uuid4().hex}"
        self.process = None
        self.instance_id = None
        self.config = SimpleNamespace(repo_dir="")
        self.tool_call_failed_count = 0
        self.calls = []
        self._lock = asyncio.Lock()
        self._sequence = 0
        self._broken = False
        self.cleanup_verified = False

    async def start(self, data: dict) -> None:
        env = {
            **os.environ,
            "PYTHONPATH": self.runtime["pythonpath"],
            "AENV_SYSTEM_URL": self.runtime["system_url"],
            "LOG_DIR": str(self.directory / "sdk-logs"),
            OWNER_ENV: self.owner,
        }
        self.process = await asyncio.create_subprocess_exec(
            self.runtime["python"],
            "-u",
            str(Path(__file__).with_name("sandbox_worker.py")),
            str(self.ownership_file),
            env=env,
            start_new_session=True,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=self.stderr,
            limit=16 << 20,
        )
        result = await self.request(
            "initialize",
            data={k: v for k, v in data.items() if k != "patch"},
            environment=self.runtime["environment"],
            transport_timeout=self.runtime.get("transport_timeout", 1900),
        )
        self.config.repo_dir = result["repo_dir"]

    async def request(self, method: str, **params):
        if self.process is None or self._broken:
            raise RuntimeError("Sandbox RPC is unavailable")
        async with self._lock:
            self._sequence += 1
            before = time.monotonic()
            try:
                self.process.stdin.write(
                    (
                        json.dumps(
                            {"id": self._sequence, "method": method, "params": params}
                        )
                        + "\n"
                    ).encode()
                )
                await self.process.stdin.drain()
                timeout = (
                    30
                    if method == "release"
                    else 900
                    if method == "get_reward"
                    else 300
                )
                line = await asyncio.wait_for(self.process.stdout.readline(), timeout)
                if not line:
                    raise RuntimeError("Sandbox RPC response stream closed")
                response = json.loads(line)
                if response.get("id") != self._sequence:
                    raise RuntimeError("Sandbox RPC response ID mismatch")
                self.instance_id = response.get("instance_id", self.instance_id)
                self.tool_call_failed_count = response.get(
                    "tool_call_failed_count", self.tool_call_failed_count
                )
                if "error" in response:
                    raise RuntimeError(response["error"])
                if self.tool_call_failed_count:
                    raise RuntimeError(
                        "Sandbox SDK reported a tool infrastructure failure"
                    )
                self.calls.append(
                    {"method": method, "seconds": time.monotonic() - before}
                )
                return response["result"]
            except BaseException:
                self._broken = True
                raise

    async def execute_bash(self, command: str) -> dict:
        return await self.request("execute_bash", command=command)

    async def search_replace(self, filepath: str, old_str: str, new_str: str) -> dict:
        return await self.request(
            "search_replace", filepath=filepath, old_str=old_str, new_str=new_str
        )

    async def submit_final_solution(self) -> dict:
        return await self.request("submit_final_solution")

    async def _kubectl(self, *arguments: str) -> str:
        process = await asyncio.create_subprocess_exec(
            self.runtime["kubectl"],
            "--kubeconfig",
            self.runtime["kubeconfig"],
            "--request-timeout=10s",
            "-n",
            self.runtime["sandbox_namespace"],
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 75)
        except BaseException:
            if process.returncode is None:
                process.kill()
            await asyncio.wait_for(process.communicate(), 10)
            raise
        if process.returncode:
            raise RuntimeError(f"Sandbox cleanup API failed: {stderr.decode()}")
        return stdout.decode()

    async def close(self) -> None:
        # Cancellation must not detach SDK/pod cleanup from the native task.
        cleanup = asyncio.create_task(self._close())
        cancelled = None
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as error:
                cancelled = error
        cleanup.result()
        self.cleanup_verified = True
        if cancelled is not None:
            raise cancelled

    async def _close(self) -> None:
        """No reward is admitted until the exact pod and SDK descendants are gone."""
        errors = []
        try:
            if self.process is not None:
                try:
                    if not self._broken and self.process.returncode is None:
                        await self.request("release")
                except Exception:
                    # Independent owned-pod deletion below is authoritative.
                    pass
                finally:
                    if self.process.stdin is not None:
                        self.process.stdin.close()
                    await asyncio.to_thread(stop_owned_processes, self.owner)
                    await asyncio.wait_for(self.process.wait(), 10)
        except Exception as error:
            errors.append(error)
        try:
            ids = {self.instance_id} if self.instance_id else set()
            if await asyncio.to_thread(self.ownership_file.exists):
                journal = json.loads(
                    await asyncio.to_thread(self.ownership_file.read_text)
                )
                ids.add(journal["instance_id"])
            for instance_id in ids:
                if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,252}", instance_id):
                    raise RuntimeError("Invalid sandbox identity")
                await self._kubectl(
                    "delete",
                    "pod",
                    instance_id,
                    "--ignore-not-found=true",
                    "--wait=true",
                    "--timeout=60s",
                )
                remaining = json.loads(
                    await self._kubectl(
                        "get",
                        "pods",
                        "--field-selector",
                        f"metadata.name={instance_id}",
                        "-o",
                        "json",
                    )
                )
                if remaining["items"]:
                    raise RuntimeError(f"Sandbox {instance_id} still exists")
        except Exception as error:
            errors.append(error)
        if errors:
            raise ExceptionGroup("SWE cleanup failed", errors)
