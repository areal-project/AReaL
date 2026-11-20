# SPDX-License-Identifier: Apache-2.0

"""Call the existing examples.hermes.hermes.HermesAgent via Agent Service."""

import httpx

from examples.v2_online_workflows.types import TaskInput, TaskResult


class HermesHarness:
    def __init__(self, *, http: httpx.AsyncClient, agent_url: str, agent_key: str):
        self.http = http
        self.agent_url = agent_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {agent_key}"}

    async def run(
        self,
        task: TaskInput,
        *,
        request_id: str,
        inference_url: str,
        model: str,
        session_api_key: str,
    ) -> TaskResult:
        response = await self.http.post(
            f"{self.agent_url}/v1/responses",
            headers={**self.headers, "X-AReaL-Session-Key": request_id},
            json={
                "input": [{"type": "message", "content": task.message}],
                "model": "hermes-agent",
                "user": request_id,
                "metadata": {"task_id": task.task_id},
                "inf_base_url": inference_url.rstrip("/") + "/v1",
                "inf_model": model,
                "session_api_key": session_api_key,
            },
        )
        response.raise_for_status()
        body = response.json()
        if body.get("status") != "completed":
            raise RuntimeError("Agent Service request did not complete")
        metadata = body.get("metadata", {})
        if type(metadata.get("completed")) is not bool:
            raise RuntimeError("Hermes response is missing its completion status")
        answer = "".join(
            block.get("text", "")
            for item in body.get("output", [])
            if item.get("type") == "message"
            for block in item.get("content", [])
            if block.get("type") == "output_text"
        )
        return TaskResult(answer, metadata["completed"], metadata)

    async def close(self, request_id: str) -> None:
        response = await self.http.post(
            f"{self.agent_url}/sessions/close",
            headers=self.headers,
            json={"session_key": request_id},
        )
        response.raise_for_status()
