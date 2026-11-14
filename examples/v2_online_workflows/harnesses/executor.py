# SPDX-License-Identifier: Apache-2.0
"""Persist Hermes request identities and read the three required result artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from examples.v2_online_workflows.artifacts import write_json
from examples.v2_online_workflows.outcome import require_successful_execution
from examples.v2_online_workflows.types import (
    TaskExecution,
    TaskResult,
    TerminalExecutionError,
)


class LocalExecutor:
    def __init__(
        self,
        root,
        http,
        service_url,
        service_key,
        sampling,
        service_root,
        resume_service_identity=None,
    ):
        self.root, self.http = Path(root), http
        self.url, self.key, self.sampling = service_url, service_key, sampling
        self.service_root = Path(service_root).resolve()
        self.resume_service_identity = resume_service_identity

    def path(self, request_id: str) -> Path:
        if not request_id or any(
            char
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for char in request_id
        ):
            raise ValueError("Invalid request id")
        return self.root / "state" / f"{request_id}.json"

    async def load(self, request_id: str) -> dict[str, Any]:
        path = self.path(request_id)
        return json.loads(path.read_text()) if path.exists() else {}

    async def save(self, request_id: str, **values: Any) -> None:
        old = await self.load(request_id)
        write_json(self.path(request_id), {**old, **values})

    async def execute(
        self,
        task,
        *,
        request_id,
        base_url,
        api_key,
        model,
        episode,
        session_id=None,
    ) -> TaskExecution:
        saved = await self.load(request_id)
        if saved.get("local_receipt"):
            receipt = saved["local_receipt"]
        else:
            if saved.get("local_intent"):
                # Reattach to the exact live service whose ledger still owns this ID.
                identity = json.loads(
                    (self.service_root / "service-process.json").read_text()
                )
                expected = self.resume_service_identity
                if not expected or any(
                    identity.get(key) != expected.get(key)
                    for key in ("pid", "proc_start", "host", "port")
                ):
                    raise RuntimeError(
                        "Local execution outcome unknown; retain original request"
                    )
            await self.save(request_id, local_intent=True)
            response = await self.http.post(
                self.url + "/run",
                headers={"Authorization": "Bearer " + self.key},
                json={
                    "request_id": request_id,
                    "task_id": task.task_id,
                    "prompt": task.message,
                    "base_url": base_url,
                    "api_key": api_key,
                    "model": model,
                    "episode": episode,
                    "session_id": session_id,
                    "sampling": self.sampling,
                },
            )
            response.raise_for_status()
            receipt = response.json()
            await self.save(request_id, local_receipt=receipt)
        output = Path(receipt["output"]).resolve()
        if output != self.service_root / "episodes" / request_id:
            raise RuntimeError("Local receipt targets a different task output")
        if receipt.get("execution_failed"):
            raise TerminalExecutionError(
                "Local Hermes execution failed: " + receipt["error_type"]
            )
        answer = json.loads((output / "answer.json").read_text())
        environment = json.loads((output / "environment.json").read_text())
        trace = json.loads((output / "call_trace.json").read_text())
        calls = [node for node in trace["nodes"] if node.get("kind") == "request"]
        require_successful_execution(answer, environment, calls)
        if not answer.get("completed") or answer.get("completed_normally") is False:
            raise TerminalExecutionError("Hermes episode did not complete normally")
        result = TaskResult(
            answer["answer"],
            answer["completed"],
            environment,
            inference_calls=calls,
            model_calls=answer["model_calls"],
        )
        return TaskExecution(
            result,
            {
                "environment": environment,
                "inference_calls": calls,
                "execution_backend": "local-hermes",
                "local_receipt": receipt,
            },
        )
