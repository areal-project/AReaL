import json
import uuid
from typing import Any

from loguru import logger
from pydantic import BaseModel

try:
    from tau2.agent.base import LocalAgent, ValidAgentInputMessage
    from tau2.agent.llm_agent import (
        AGENT_INSTRUCTION,
        is_valid_agent_history_message,
    )
    from tau2.agent.llm_agent import (
        SYSTEM_PROMPT as AGENT_SYSTEM_PROMPT,
    )
    from tau2.data_model.message import (
        APICompatibleMessage,
        AssistantMessage,
        Message,
        SystemMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )
    from tau2.environment.tool import Tool
except ImportError as exc:
    logger.error(
        "Failed to import tau2: {}\nPlease install tau2-bench: pip install -e ../tau2-bench",
        exc,
    )
    raise

from .service import OpenClawConfig, OpenClawService
from .workspace_manager import OpenClawWorkspaceManager


class MessageTranslator:
    @staticmethod
    def to_openclaw(messages: list[Message]) -> list[dict[str, Any]]:
        translated: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, UserMessage):
                translated.append({"role": "user", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                translated.append(
                    {
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": msg.tool_call_id,
                    }
                )
            elif isinstance(msg, SystemMessage):
                translated.append({"role": "system", "content": msg.content})
            elif isinstance(msg, AssistantMessage):
                payload = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    payload["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": json.dumps(call.function.arguments),
                            },
                        }
                        for call in msg.tool_calls
                    ]
                translated.append(payload)
            else:
                logger.warning("Unknown message type: {}", type(msg))
        return translated

    @staticmethod
    def from_openclaw(openclaw_msg: dict[str, Any]) -> AssistantMessage:
        tool_calls = openclaw_msg.get("tool_calls") or None
        return AssistantMessage(
            role="assistant",
            content=openclaw_msg.get("content") or None,
            tool_calls=[
                ToolCall(
                    id=call["id"],
                    type="function",
                    function={
                        "name": call["function"]["name"],
                        "arguments": json.loads(call["function"]["arguments"])
                        if isinstance(call["function"]["arguments"], str)
                        else call["function"]["arguments"],
                    },
                )
                for call in tool_calls
            ]
            if tool_calls
            else None,
        )


class OpenClawAgentState(BaseModel):
    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    openclaw_session_id: str | None = None


class OpenClawAgent(LocalAgent[OpenClawAgentState]):
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        socket_server_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.socket_server_config = socket_server_config
        self.config = OpenClawConfig.from_env()
        self.message_translator = MessageTranslator()
        self.service = OpenClawService(
            cli_command=self.config.cli_command,
            timeout=self.config.timeout,
            api_base=self.config.api_base,
            api_key=self.config.api_key,
            model=self.config.model,
        )
        self.workspace_manager: OpenClawWorkspaceManager | None = (
            OpenClawWorkspaceManager(cli_command=self.config.cli_command)
        )
        self.agent_id: str | None = f"tau2-{uuid.uuid4().hex[:8]}"
        try:
            self.workspace_manager.create_agent_workspace(
                agent_id=self.agent_id, tools=tools, agent_name="TAU² Evaluation Agent"
            )
            if self.socket_server_config:
                self._inject_socket_tools(self.socket_server_config)
        except Exception as exc:
            logger.warning(
                "Failed to create workspace, continuing without isolation: {}", exc
            )
            self.workspace_manager = None
            self.agent_id = None
        logger.info(
            "OpenClawAgent initialized: agent_id={} tools={} cli={} timeout={} socket_server={}",
            self.agent_id or "default",
            len(tools),
            self.config.cli_command,
            self.config.timeout,
            bool(self.socket_server_config),
        )

    def _inject_socket_tools(self, server_config: dict) -> None:
        if not self.workspace_manager or not self.agent_id:
            return
        from ..tau2_env import create_openclaw_tool_script

        tools_dir = (
            self.workspace_manager.get_workspace_path(self.agent_id) / "socket_tools"
        )
        tools_dir.mkdir(exist_ok=True)
        for tool in self.tools:
            script_path = tools_dir / f"{tool.name}.py"
            script_path.write_text(
                create_openclaw_tool_script(
                    tool_name=tool.name, server_config=server_config
                ),
                encoding="utf-8",
            )
            script_path.chmod(0o755)
        (tools_dir / "server_config.json").write_text(
            json.dumps(server_config, indent=2), encoding="utf-8"
        )
        logger.info("Injected {} socket tools into {}", len(self.tools), tools_dir)

    @property
    def system_prompt(self) -> str:
        return AGENT_SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        ) + (
            "\n\n## Available Tools\n\n"
            "Before using tools, read `skills/tau2-tools/SKILL.md`. "
            "Tools are exposed as Python scripts in `socket_tools/` and should be invoked like "
            '`python socket_tools/<tool_name>.py \'{"param": "value"}\'`.'
        )

    def get_init_state(
        self, message_history: list[Message] | None = None
    ) -> OpenClawAgentState:
        message_history = message_history or []
        assert all(
            is_valid_agent_history_message(message) for message in message_history
        ), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return OpenClawAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
            openclaw_session_id=str(uuid.uuid4()),
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: OpenClawAgentState
    ) -> tuple[AssistantMessage, OpenClawAgentState]:
        state.messages.append(message)
        try:
            response = self.service.chat(
                messages=self.message_translator.to_openclaw(
                    state.system_messages + state.messages
                ),
                session_id=state.openclaw_session_id,
                agent_id=self.agent_id,
            )
            state.openclaw_session_id = response.get(
                "session_id", state.openclaw_session_id
            )
            assistant_msg = self.message_translator.from_openclaw(response["message"])
        except Exception:
            logger.exception("Error in OpenClaw agent")
            assistant_msg = AssistantMessage(
                role="assistant",
                content="I apologize, but I encountered an error processing your request. Please try again or rephrase your question.",
            )
        state.messages.append(assistant_msg)
        return assistant_msg, state

    def stop(
        self,
        message: ValidAgentInputMessage | None = None,
        state: OpenClawAgentState | None = None,
    ) -> None:
        _ = message
        cleanups = []
        if state and state.openclaw_session_id:
            cleanups.append(
                (
                    "session",
                    lambda: self.service.cleanup_session(state.openclaw_session_id),
                )
            )
        if self.workspace_manager and self.agent_id:
            cleanups.append(
                (
                    "workspace",
                    lambda: self.workspace_manager.delete_agent_workspace(
                        self.agent_id
                    ),
                )
            )
        for label, cleanup in cleanups:
            try:
                cleanup()
            except Exception as exc:
                logger.warning("Failed to cleanup {}: {}", label, exc)

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        return False

    def set_seed(self, seed: int) -> None:
        logger.warning(
            "set_seed({}) called but OpenClaw may not support deterministic seeding",
            seed,
        )
