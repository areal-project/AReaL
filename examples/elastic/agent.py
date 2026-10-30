# SPDX-License-Identifier: Apache-2.0

"""Event-triggered tool calls to a configured OpenAI-compatible control LLM."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

SYSTEM = """You control a fixed-size training Actor group. Inference stays resident.
Choose one action from the tools using the supplied current observation. Minimize
idle training GPU occupancy while making progress on eligible trajectories.
You are called when collection starts, immediately after a completed update with
no next batch, after release, on coalesced progress changes or your wake timer.
Start when a complete batch is ready and Actor is stopped. You may also start
BEFORE a batch is complete to overlap restoration with remaining SWE work.
Use workload.batch_remaining_seconds and lifecycle_costs when available. The
program estimates start_within_seconds = ETA low - startup - restore margin.
When Actor is stopped and this is at most 5 seconds, start now. Otherwise wait
no longer than that estimate (maximum 300 seconds) before checking fresh progress.
Once prestarted, Actor stays resident until an update.
When Actor is online with no batch, consider saving and releasing immediately
only if the expected remaining collection time can cover checkpoint, release,
startup and margin with a useful free window. There is no fixed idle threshold.
ETA is an empirical hint, not a guarantee. Missing/overdue history is unknown,
not zero; with unknown ETA prefer starting a stopped Actor or retaining an online
one to avoid blocking training. Lifecycle configured priors are estimates, not
current-run measurements. Graded samples still need cleanup and native export.
Use wait_for_event with a bounded 5-300 second delay; the coordinator handles
timers and significant progress changes. No future arrival schedule is supplied.
The runtime enforces completed updates, publication and full checkpointing before
release. Return a short operational reason, not a chain of thought. Use exactly one
tool call at a time; querying status is optional. Never invent successful execution.
Use only allowed_actions from the current observation. On action_rejected, the
proposal was NOT executed; correct it using the fresh status and rejection_reason.
"""


def tools(allowed_actions: list[str] | None = None) -> list[dict[str, Any]]:
    result = []
    for name, description in (
        ("get_status", "Read the current coordinator observation."),
        (
            "start_actor",
            "Start or restore the stopped training group, including before a batch is ready.",
        ),
        ("stop_actor", "Save complete state and release idle training workers."),
        ("wait_for_event", "Wait for a changed event or a bounded timeout."),
    ):
        if (
            allowed_actions is not None
            and name != "get_status"
            and name not in allowed_actions
        ):
            continue
        properties = {} if name == "get_status" else {"reason": {"type": "string"}}
        required = [] if name == "get_status" else ["reason"]
        if name == "wait_for_event":
            properties["wake_after_seconds"] = {
                "type": "number",
                "minimum": 5,
                "maximum": 300,
            }
            required.append("wake_after_seconds")
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }
        )
    return result


class ControlLLM:
    def __init__(self, *, client: httpx.Client | None = None) -> None:
        path = Path(
            os.environ.get(
                "AREAL_CONTROL_LLM_CONFIG",
                Path.home() / ".config/areal-elastic/openrouter.json",
            )
        )
        config = json.loads(path.read_text()) if path.exists() else {}
        self.base_url = (
            os.getenv("AREAL_CONTROL_LLM_BASE_URL") or config.get("base_url", "")
        ).rstrip("/")
        self.model = (
            os.getenv("AREAL_CONTROL_LLM_MODEL")
            or os.getenv("OPENROUTER_MODEL")
            or config.get("model")
        )
        self.api_key = (
            os.getenv("AREAL_CONTROL_LLM_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or config.get("api_key")
        )
        if not self.base_url or not self.model or not self.api_key:
            raise ValueError(
                "Configure OpenRouter with python -m examples.elastic.configure_openrouter or set control LLM environment variables"
            )
        self.client = client or httpx.Client(timeout=60.0)
        self._owns_client = client is None
        self.observe: Callable[[], dict[str, Any]] | None = None

    def decide(self, observation: dict[str, Any]) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(observation, sort_keys=True)},
        ]
        for _ in range(3):
            response = self.client.post(
                self.base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + self.api_key},
                json={
                    "model": self.model,
                    "messages": messages,
                    "tools": tools(observation.get("allowed_actions")),
                    "tool_choice": "required",
                    "parallel_tool_calls": False,
                },
            )
            # Do not include response bodies (or credentials) in the run log.
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            calls = message.get("tool_calls", [])
            if len(calls) != 1:
                raise ValueError("Control LLM must select exactly one tool")
            call = calls[0]
            action = call["function"]["name"]
            arguments = json.loads(call["function"]["arguments"])
            if action != "get_status":
                return {**arguments, "action": action, "model": self.model}
            if self.observe is not None:
                observation = {**observation, **self.observe()}
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": message.get("content"),
                        "tool_calls": calls,
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(observation),
                    },
                ]
            )
        raise RuntimeError("Control LLM did not decide within three tool rounds")

    def close(self) -> None:
        if self._owns_client:
            self.client.close()
