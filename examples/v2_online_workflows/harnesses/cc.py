# SPDX-License-Identifier: Apache-2.0

"""Claude Code's SDK loop with task-local MCP tools and AReaL inference routing."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from aiohttp import ClientSession, ClientTimeout, web

from examples.v2_online_workflows.artifacts import write_json
from examples.v2_online_workflows.trace import (
    CallTrace,
    observe_anthropic_data,
    observe_openai,
)

if TYPE_CHECKING:
    from areal.v2.agent_service.types import AgentResponse


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {
            "type": type(value).__name__,
            **{
                item.name: _serialize(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_serialize(item) for item in value]
    return value


class InferenceRelay:
    """Bind every CLI HTTP generation (also compaction) to one inference episode.

    A loopback relay is needed because the closed CLI cannot attach AReaL's JSON
    identity itself. It never changes process-wide environment variables.
    """

    def __init__(self, metadata: dict[str, Any]) -> None:
        upstream = metadata.get("areal_inference") or {}
        self.base_url = (
            str(upstream.get("base_url", "")).rstrip("/").removesuffix("/v1")
        )
        self.api_key = str(upstream.get("api_key", ""))
        self.model = str(upstream.get("model", ""))
        self.episode = upstream.get("episode")
        self.protocol = upstream.get("protocol", "messages")
        if self.protocol not in {"messages", "openai"}:
            raise ValueError("Unsupported CC upstream protocol")
        self.model_input_path = metadata.get("model_input_path")
        self.model_headers = dict(upstream.get("model_headers") or {})
        if self.model_headers and (
            set(self.model_headers) != {"X-Session-Id", "X-Session-Token"}
            or any(
                not isinstance(v, str) or not v or "\n" in v or "\r" in v
                for v in self.model_headers.values()
            )
        ):
            raise ValueError("Invalid session routing headers")
        if (
            urlsplit(self.base_url).scheme not in ("http", "https")
            or not self.api_key
            or not self.model
        ):
            raise ValueError(
                "Claude Code tasks require explicit AReaL inference URL, key and model"
            )
        self.key = secrets.token_urlsafe(32)
        self.trace = CallTrace()
        self.calls = self.trace.requests
        self._runner = None
        self._http = None

    async def __aenter__(self):
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/v1/messages", self._messages)
        self._http = ClientSession(timeout=ClientTimeout(total=None, sock_connect=30))
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.url = "http://127.0.0.1:" + str(self._runner.addresses[0][1])
        return self

    async def __aexit__(self, *args):
        try:
            await self._runner.cleanup()
        finally:
            await self._http.close()

    async def _messages(self, request: web.Request) -> web.StreamResponse:
        token = request.headers.get("x-api-key") or request.headers.get(
            "authorization", ""
        ).removeprefix("Bearer ")
        if not secrets.compare_digest(token, self.key):
            raise web.HTTPUnauthorized()
        body = await request.json()
        if body.get("areal_episode") not in (None, self.episode):
            raise web.HTTPConflict(text="Inference episode identity mismatch")
        body["model"] = self.model
        if self.episode is not None:
            body["areal_episode"] = self.episode
        body.update(
            max_tokens=int(os.environ.get("AREAL_TASK_MAX_TOKENS", "4096")),
            temperature=float(os.environ.get("AREAL_TASK_TEMPERATURE", "1.0")),
            top_p=float(os.environ.get("AREAL_TASK_TOP_P", "1.0")),
        )
        if self.model_input_path and not Path(self.model_input_path).exists():
            write_json(
                Path(self.model_input_path),
                {
                    "scope": "first CC HTTP request body before upstream serialization",
                    **{
                        key: body[key]
                        for key in (
                            "model",
                            "system",
                            "messages",
                            "tools",
                            "tool_choice",
                            "max_tokens",
                            "temperature",
                            "top_p",
                            "thinking",
                            "context_management",
                            "output_config",
                        )
                        if key in body
                    },
                },
            )
        if self.protocol == "openai":
            return await self._chat_bridge(request, body)
        clock = time.perf_counter()
        record = {
            "started": time.time(),
            "finished": None,
            "seconds": None,
            "model": self.model,
            "status": None,
            "timing_scope": "relay_receive_to_response_end",
        }
        self.trace.request(record)
        buffer = bytearray()
        try:
            async with self._http.post(
                self.base_url + "/v1/messages",
                json=body,
                headers={
                    "Authorization": "Bearer " + self.api_key,
                    "anthropic-version": "2023-06-01",
                    **self.model_headers,
                },
            ) as upstream:
                record["status"] = upstream.status
                response = web.StreamResponse(
                    status=upstream.status,
                    headers={
                        "Content-Type": upstream.headers.get(
                            "Content-Type", "application/json"
                        )
                    },
                )
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                    buffer.extend(chunk)
                    if upstream.content_type == "text/event-stream":
                        while b"\n" in buffer:
                            line, _, remaining = buffer.partition(b"\n")
                            buffer[:] = remaining
                            if line.startswith(b"data:"):
                                payload = line[5:].strip()
                                if payload and payload != b"[DONE]":
                                    observe_anthropic_data(record, payload)
                if upstream.content_type != "text/event-stream" and buffer:
                    observe_anthropic_data(record, bytes(buffer))
                if upstream.status >= 400 and buffer:
                    text = buffer[:8192].decode("utf-8", errors="replace")
                    record["error_response"] = text.replace(self.api_key, "[REDACTED]")
                await response.write_eof()
                return response
        finally:
            record.update(finished=time.time(), seconds=time.perf_counter() - clock)
            if self.model_input_path:
                write_json(
                    Path(self.model_input_path).with_name(
                        "cc-request-observation.json"
                    ),
                    self.trace.snapshot(),
                )

    async def _chat_bridge(
        self, request: web.Request, body: dict
    ) -> web.StreamResponse:
        """Use the same tested protocol conversion as the native AReaL proxy."""
        import httpx
        from openai import APIStatusError, AsyncOpenAI

        from examples.v2_online_workflows.harnesses.protocol_adapter import (
            anthropic_adapter,
        )

        adapter = anthropic_adapter()
        converted = adapter.translate_anthropic_request(body)
        # Claude server-side context edits have no OpenAI/Qwen equivalent.
        # The local CC loop retains its own history and compaction behavior.
        converted.pop("context_management", None)
        output_config = converted.pop("output_config", {}) or {}
        if set(output_config) - {"effort"}:
            raise ValueError("Unsupported Anthropic output_config fields")
        if output_config.get("effort"):
            converted["reasoning_effort"] = output_config["effort"]
        converted.update(
            model=self.model,
            max_tokens=body["max_tokens"],
            temperature=body["temperature"],
            top_p=body["top_p"],
        )
        streaming = bool(body.get("stream"))
        converted["stream"] = streaming
        if streaming:
            converted["stream_options"] = {"include_usage": True}
        clock = time.perf_counter()
        record = {
            "started": time.time(),
            "finished": None,
            "seconds": None,
            "model": self.model,
            "status": None,
            "protocol": "anthropic_to_openai",
        }
        self.trace.request(record)
        try:
            async with AsyncOpenAI(
                base_url=self.base_url + "/v1",
                api_key=self.api_key,
                default_headers=self.model_headers,
                max_retries=0,
                timeout=1800,
                # Arena may provide its CA bundle through SSL_CERT_FILE/DIR.
                # Match the native clients' verified TLS/environment behavior.
                http_client=httpx.AsyncClient(),
            ) as client:
                result = await client.chat.completions.create(**converted)
                record["status"] = 200
                if streaming:
                    response = web.StreamResponse(
                        headers={"Content-Type": "text/event-stream"}
                    )
                    await response.prepare(request)

                    async def observed():
                        async for chunk in result:
                            observe_openai(record, chunk)
                            yield chunk

                    async for event in adapter.translate_anthropic_stream(
                        observed(), self.model
                    ):
                        await response.write(
                            event.encode() if isinstance(event, str) else event
                        )
                    await response.write_eof()
                    return response
                observe_openai(record, result)
                translated = adapter.translate_anthropic_response(result)
                return web.json_response(translated.model_dump())
        except APIStatusError as exc:
            record["status"] = exc.status_code
            record["error_response"] = exc.response.text[:8192].replace(
                self.api_key, "[REDACTED]"
            )
            return web.json_response(
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": record["error_response"]},
                },
                status=exc.status_code,
            )
        except Exception as exc:
            record["status"] = 500
            record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc).replace(self.api_key, "[REDACTED]"),
            }
            causes = []
            cause = exc.__cause__
            while cause is not None and len(causes) < 4:
                causes.append(
                    {
                        "type": type(cause).__name__,
                        "message": str(cause).replace(self.api_key, "[REDACTED]"),
                    }
                )
                cause = cause.__cause__
            record["error"]["causes"] = causes
            record["tls_environment"] = {
                name: bool(os.environ.get(name))
                for name in ("SSL_CERT_FILE", "SSL_CERT_DIR")
            }
            raise
        finally:
            record.update(finished=time.time(), seconds=time.perf_counter() - clock)
            if self.model_input_path:
                write_json(
                    Path(self.model_input_path).with_name(
                        "cc-request-observation.json"
                    ),
                    self.trace.snapshot(),
                )


class ClaudeCodeHarness:
    """Native SDK orchestration; benchmark preparation and grading stay outside."""

    async def run(self, request: Any, *, runtime: Any, emitter: Any) -> AgentResponse:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKClient,
            ResultMessage,
            create_sdk_mcp_server,
            tool,
        )

        from areal.v2.agent_service.types import AgentResponse

        schemas = runtime.tool_schemas
        tools = []
        for schema in schemas:

            def make_handler(name):
                async def invoke(arguments):
                    await emitter.emit_tool_call(name=name, args=json.dumps(arguments))
                    result = await asyncio.to_thread(runtime.invoke, name, arguments)
                    await emitter.emit_tool_result(name=name, result=result)
                    return {"content": [{"type": "text", "text": result}]}

                return invoke

            tools.append(
                tool(schema["name"], schema["description"], schema["parameters"])(
                    make_handler(schema["name"])
                )
            )
        server = create_sdk_mcp_server("task", tools=tools)
        directory = runtime.directory / "cc"
        await asyncio.to_thread(directory.mkdir, parents=True)
        transcript = [
            {
                "sent": time.time(),
                "message": {"type": "UserMessage", "content": request.message},
            }
        ]
        final = None
        async with InferenceRelay(request.metadata) as relay:
            relay.trace.bind_tools(runtime)
            env = {
                "ANTHROPIC_BASE_URL": relay.url,
                "ANTHROPIC_API_KEY": relay.key,
                "ANTHROPIC_AUTH_TOKEN": "",
                "ANTHROPIC_CUSTOM_HEADERS": "",
                "CLAUDE_CODE_OAUTH_TOKEN": "",
                "ANTHROPIC_MODEL": relay.model,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": relay.model,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": relay.model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": relay.model,
                "CLAUDE_CODE_SUBAGENT_MODEL": relay.model,
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_USE_BEDROCK": "0",
                "CLAUDE_CODE_USE_VERTEX": "0",
                "CLAUDE_CODE_USE_FOUNDRY": "0",
                "CLAUDE_CONFIG_DIR": str(directory / "config"),
                "NO_PROXY": "127.0.0.1,localhost,::1",
                "MCP_TOOL_TIMEOUT": str(int((runtime.settings.timeout + 180) * 1000)),
            }
            options = ClaudeAgentOptions(
                model=relay.model,
                env=env,
                cwd=str(directory),
                tools=[],
                allowed_tools=["mcp__task__" + schema["name"] for schema in schemas],
                mcp_servers={"task": server},
                setting_sources=[],
                system_prompt={
                    "type": "preset",
                    "preset": "claude_code",
                    "append": "All task files and execution are in the provided MCP task environment. "
                    "Use those tools for all inspection, edits and tests; local paths are not task paths.",
                },
                extra_args={"strict-mcp-config": None, "no-session-persistence": None},
            )
            if os.environ.get("AREAL_CC_CLI_PATH"):
                options.cli_path = os.environ["AREAL_CC_CLI_PATH"]
            if os.environ.get("AREAL_CC_MAX_TURNS"):
                options.max_turns = int(os.environ["AREAL_CC_MAX_TURNS"])

            async def prompt():
                yield {
                    "type": "user",
                    "message": {"role": "user", "content": request.message},
                }

            try:
                async with ClaudeSDKClient(options=options) as client:
                    await client.query(prompt())
                    async for message in client.receive_response():
                        transcript.append(
                            {"received": time.time(), "message": _serialize(message)}
                        )
                        if isinstance(message, AssistantMessage) and getattr(
                            message, "error", None
                        ):
                            raise RuntimeError(
                                "Claude Code model request failed: "
                                + str(message.error)
                            )
                        if isinstance(message, ResultMessage):
                            final = message
                if final is None:
                    raise RuntimeError("Claude Code ended without a result")
                if final.is_error and final.subtype not in (
                    "error_max_turns",
                    "error_max_budget_usd",
                ):
                    raise RuntimeError("Claude Code execution failed: " + final.subtype)
                answer = final.result or ""
                if not answer:
                    for item in reversed(transcript):
                        message = item["message"]
                        if message["type"] == "AssistantMessage":
                            answer = "".join(
                                block["text"]
                                for block in message["content"]
                                if block["type"] == "TextBlock"
                            )
                            break
                if answer:
                    await emitter.emit_delta(answer)
                return AgentResponse(
                    summary=answer,
                    metadata={
                        "harness": "cc",
                        "completed": not final.is_error
                        and final.stop_reason != "max_tokens",
                        "api_calls": len(relay.calls),
                        "inference_calls": relay.calls,
                        "simulation_trace": relay.trace.snapshot(),
                        "transcript": transcript,
                        "cc_result": _serialize(final),
                    },
                )
            finally:
                await asyncio.to_thread(
                    (directory / "transcript.json").write_text,
                    json.dumps(
                        {
                            "messages": transcript,
                            "inference_calls": relay.calls,
                            "simulation_trace": relay.trace.snapshot(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
