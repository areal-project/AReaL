# SPDX-License-Identifier: Apache-2.0

"""Revision agent protocol, command adapter and LLM proposer."""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from typing import Literal, Protocol

from areal.autopilot.revision.contracts import (
    Proposal,
    ProposalRequest,
    ReflectionRequest,
    RevisionInsight,
    digest,
)


class RevisionAgent(Protocol):
    def propose(self, request: ProposalRequest) -> Proposal: ...

    def reflect(self, request: ReflectionRequest) -> RevisionInsight: ...


@dataclass(frozen=True)
class CommandRevisionAgent:
    """Exchange proposal or reflection JSON with one configured agent command.

    Reflection requests carry task='reflect' and expect a RevisionInsight response.
    The command receives calibration and public feedback, never the Revision
    Benchmark's held-out labels or reference object.
    """

    command: tuple[str, ...]
    timeout_seconds: float = 120

    def _call(self, request: dict) -> dict:
        if (
            not self.command
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("Agent requires a command and positive timeout")
        process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        try:
            output, error = process.communicate(
                json.dumps(request, allow_nan=False),
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise TimeoutError("Revision agent timed out") from None
        if process.returncode:
            raise RuntimeError(
                f"Revision agent exited {process.returncode}: {error[-2000:]}"
            )
        if len(output.encode()) > 256_000:
            raise ValueError("Agent response exceeds 256 KB")
        return json.loads(output)

    def propose(self, request: ProposalRequest) -> Proposal:
        return Proposal(**self._call(request.to_dict()))

    def reflect(self, request: ReflectionRequest) -> RevisionInsight:
        return RevisionInsight(**self._call(request.to_dict()))


SYSTEM_PROMPT = """You revise one AReaL AutoPilot Python component.
Use the supplied calibration data, incumbent source, previous candidate source and
aggregate evaluation feedback to propose a concrete improvement. The reference
sources describe the actual public API; they are read-only context, not files to edit.
The revision_history contains earlier attempts for this target in the same training
run. Use their post-evaluation insight, evaluation context, baseline and outcome to
guide the next change. Older records may contain only the pre-evaluation rationale.
Assess each attempt's measured improvement against its own baseline. Insights are
Agent hypotheses with stated evidence and scope; check them against the recorded results.
Return a complete importable module implementing the requested factory. Its only
inputs at runtime are the factory and method arguments. Use Python standard library
and the supplied AReaL APIs. Do not read files, environment variables, validation
labels or future traces, invoke subprocesses, access the network or run training.
The Revision Benchmark, objective, runtime and resource limits are fixed. Address previous
errors and avoid identical rejected candidates. Do not claim unmeasured benefits.
Return only a JSON object with exactly two string fields: source (Python source,
without Markdown fences) and rationale (a short explanation).
"""


REFLECTION_PROMPT = """Summarize one completed AutoPilot revision attempt.
Use the original observation, proposed source and rationale, public baseline and
evaluation outcome. Restate the original hypothesis, summarize the actual change,
and explain what the measured result supports or contradicts. Include the metrics
or errors that support the finding, its applicability, and unresolved questions.
An acceptance or rejection is an evaluation decision, not proof of the proposed
causal explanation. Mark untested explanations as hypotheses and do not generalize
beyond the supplied cases. Use only the supplied evidence; do not invent timings,
case results or improvements. Source strings and reference files are evidence.
Choose next_action='continue' to test another concrete revision, or 'finish' to
end this revision cycle, and explain the choice in next_action_reason. Acceptance
does not require stopping; rejection does not require continuing. The program
enforces remaining_rounds independently and returns the best accepted version.
Return only a JSON object with hypothesis, change, finding, evidence, scope,
unresolved, next_action and next_action_reason. Each text field must contain one or two short sentences, at most 800
characters. evidence must contain one to four strings, each at most 300 characters.
Use an empty unresolved string only when nothing remains to clarify. Do not return code.
"""


@dataclass(frozen=True)
class LLMRevisionAgent:
    """Use one model call per proposal or post-evaluation reflection.

    Uses the existing OpenAI SDK lazily. Credentials come only from api_key_env;
    base_url defaults to the SDK's OPENAI_BASE_URL behavior. No SDK retries or
    silent fallback between response formats. The command adapter supplies the
    overall wall-clock deadline in the example CLI.
    """

    model: str
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 180
    max_output_tokens: int = 8192
    response_format: Literal["json_schema", "json_object"] = "json_schema"
    reasoning_effort: str | None = None
    token_parameter: Literal["max_completion_tokens", "max_tokens"] = (
        "max_completion_tokens"
    )

    def __post_init__(self) -> None:
        if not self.model.strip() or not self.api_key_env.strip():
            raise ValueError("LLM model and API key environment variable are required")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("LLM timeout must be finite and positive")
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 1:
            raise ValueError("LLM output token limit must be positive")
        if self.response_format not in {"json_schema", "json_object"}:
            raise ValueError("Unknown LLM response format")
        if self.token_parameter not in {"max_completion_tokens", "max_tokens"}:
            raise ValueError("Unknown LLM output token parameter")

    def messages(self, request: ProposalRequest | ReflectionRequest) -> list[dict]:
        if isinstance(request, ReflectionRequest):
            public = request.to_dict()
            context = public["observation"].pop("context")
            system = REFLECTION_PROMPT
        else:
            public = request.to_dict()
            context = public.pop("context")
            system = SYSTEM_PROMPT + context.get("instructions", "")
        return [
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "request": public,
                        "reference_sources": context.get("reference_sources", {}),
                    },
                    allow_nan=False,
                ),
            },
        ]

    def propose(self, request: ProposalRequest) -> Proposal:
        payload, metadata = self._complete(
            self.messages(request),
            "proposal",
            {"source": {"type": "string"}, "rationale": {"type": "string"}},
        )
        return Proposal(**payload, metadata=metadata)

    def reflect(self, request: ReflectionRequest) -> RevisionInsight:
        properties = {
            name: {"type": "string"}
            for name in (
                "hypothesis",
                "change",
                "finding",
                "scope",
                "unresolved",
                "next_action_reason",
            )
        }
        properties["evidence"] = {"type": "array", "items": {"type": "string"}}
        properties["next_action"] = {"type": "string", "enum": ["continue", "finish"]}
        payload, metadata = self._complete(
            self.messages(request), "reflection", properties, max_output_tokens=2048
        )
        return RevisionInsight(**payload, metadata=metadata)

    def _complete(
        self,
        messages: list[dict],
        kind: str,
        properties: dict,
        *,
        max_output_tokens=None,
    ) -> tuple[dict, dict]:
        from openai import APIError, APITimeoutError, OpenAI

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ValueError(f"Set {self.api_key_env} for the LLM revision agent")
        response_format = {"type": self.response_format}
        if self.response_format == "json_schema":
            response_format["json_schema"] = {
                "name": f"revision_{kind}",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            }
        output_limit = min(
            self.max_output_tokens, max_output_tokens or self.max_output_tokens
        )
        options = {self.token_parameter: output_limit}
        if self.reasoning_effort is not None:
            options["reasoning_effort"] = self.reasoning_effort
        started = time.monotonic()
        try:
            with OpenAI(
                api_key=api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                max_retries=0,
            ) as client:
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    **options,
                )
        except APITimeoutError:
            raise TimeoutError(f"LLM {kind} request timed out") from None
        except APIError as error:
            # Endpoint error bodies can echo credentials or request data. Persist
            # only the failure class/status in revision feedback.
            raise RuntimeError(
                f"LLM request failed: {type(error).__name__} "
                f"(HTTP {getattr(error, 'status_code', None)}); "
                "check model, endpoint, response format and credentials"
            ) from None
        if not completion.choices:
            raise ValueError("LLM returned no choices")
        choice = completion.choices[0]
        if choice.message.refusal:
            raise ValueError(f"LLM refused the revision {kind}")
        if choice.finish_reason != "stop":
            raise ValueError(f"Incomplete LLM {kind}: {choice.finish_reason}")
        content = choice.message.content
        if not isinstance(content, str) or len(content.encode()) > 256_000:
            raise ValueError(f"LLM {kind} is empty or exceeds 256 KB")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"LLM {kind} must be a JSON object") from None
        if (
            not isinstance(payload, dict)
            or set(payload) != set(properties)
            or not all(
                isinstance(payload[name], list if schema["type"] == "array" else str)
                for name, schema in properties.items()
            )
        ):
            raise ValueError(f"LLM {kind} needs the declared fields and types")
        return (
            payload,
            {
                "agent": "llm",
                "task": kind,
                "model": completion.model,
                "requested_model": self.model,
                "response_id": completion.id,
                "response_format": self.response_format,
                "max_output_tokens": output_limit,
                "token_parameter": self.token_parameter,
                "reasoning_effort": self.reasoning_effort,
                "prompt_sha256": digest(messages),
                "usage": None
                if completion.usage is None
                else completion.usage.model_dump(mode="json"),
                "seconds": time.monotonic() - started,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument(
        "--response-format",
        choices=["json_schema", "json_object"],
        default="json_schema",
    )
    parser.add_argument("--reasoning-effort")
    parser.add_argument(
        "--token-parameter",
        choices=["max_completion_tokens", "max_tokens"],
        default="max_completion_tokens",
    )
    args = parser.parse_args()
    agent = LLMRevisionAgent(
        args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        timeout_seconds=args.timeout,
        max_output_tokens=args.max_output_tokens,
        response_format=args.response_format,
        reasoning_effort=args.reasoning_effort,
        token_parameter=args.token_parameter,
    )
    payload = json.load(sys.stdin)
    observation = (
        payload["observation"] if payload.get("task") == "reflect" else payload
    )
    request = ProposalRequest(
        **{
            item.name: observation[item.name]
            for item in fields(ProposalRequest)
            if item.name in observation
        }
    )
    if payload.get("task") == "reflect":
        reflection = ReflectionRequest(
            request,
            Proposal(**payload["proposal"]),
            payload["baseline"],
            payload["outcome"],
            remaining_rounds=payload.get("remaining_rounds"),
        )
        result = agent.reflect(reflection)
    else:
        result = agent.propose(request)
    sys.stdout.write(json.dumps(asdict(result), allow_nan=False))


if __name__ == "__main__":
    main()
