import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger
from tau2.environment.tool import Tool


class WorkspaceManagerError(Exception):
    pass


class OpenClawWorkspaceManager:
    def __init__(
        self,
        cli_command: str = "openclaw",
        base_workspace_dir: Path | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
    ):
        self.cli_command = cli_command
        self.llm_api_key = llm_api_key or os.environ.get("OPENCLAW_API_KEY")
        self.llm_base_url = llm_base_url or os.environ.get("OPENCLAW_API_BASE")
        self.llm_model = llm_model or os.environ.get("OPENCLAW_MODEL", "gpt-4o")
        self.base_workspace_dir = Path(
            base_workspace_dir
            or Path(tempfile.gettempdir()) / "openclaw-tau2-workspaces"
        )
        self.base_workspace_dir.mkdir(parents=True, exist_ok=True)
        self.created_agents: set[str] = set()
        logger.info("Workspace manager initialized: {}", self.base_workspace_dir)

    @property
    def openclaw_dir(self) -> Path:
        return Path.home() / ".openclaw"

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _provider_config(self) -> dict[str, Any]:
        model_name = self.llm_model or "model"
        return {
            "baseUrl": self.llm_base_url or "http://127.0.0.1:30000",
            "apiKey": self.llm_api_key or "dummy",
            "api": "openai-completions",
            "models": [{"id": model_name, "name": model_name}],
        }

    def _tool_markdown(self, tool: Tool) -> str:
        schema = tool.openai_schema.get("function", {})
        params = schema.get("parameters", {})
        properties = params.get("properties", {})
        required = set(params.get("required", []))
        parameter_lines = "\n".join(
            f"- `{name}` ({info.get('type', 'string')}, {'required' if name in required else 'optional'}): {info.get('description', '')}"
            for name, info in properties.items()
        )
        example = json.dumps(
            {name: f"<{name}>" for name in properties if name in required} or {}
        )
        return "\n".join(
            filter(
                None,
                [
                    f"### {tool.name}",
                    "",
                    schema.get("description", tool.short_desc),
                    "",
                    f"**Script**: `socket_tools/{tool.name}.py`",
                    "",
                    "**Parameters**:" if properties else "",
                    parameter_lines,
                    "" if properties else "",
                    "**Example**:",
                    "```bash",
                    f"python socket_tools/{tool.name}.py '{example}'",
                    "```",
                    "",
                ],
            )
        )

    def _build_skill_markdown(self, tools: list[Tool]) -> str:
        header = '---\nname: tau2-tools\ndescription: TAU²-Bench Environment Tools - Execute tools via Socket Server\nmetadata: {"openclaw":{"emoji":"🔧","requires":{"bins":["python"]}}}\n---\n\n# TAU²-Bench Environment Tools\n\nUse `python socket_tools/<tool_name>.py \'<json-args>\'` to call the shared environment.\nGenerated scripts live in `socket_tools/` and require only the Python standard library.\n\n## Available Tools\n'
        footer = "## Notes\n\n- Socket tools are generated automatically when the socket server is enabled.\n- Check `socket_tools/server_config.json` for connection details.\n- Changes are applied to the shared environment immediately.\n"
        body = "\n".join(self._tool_markdown(tool) for tool in tools)
        return f"{header}\n{body}\n{footer}"

    def _setup_workspace_tools(self, workspace_path: Path, tools: list[Tool]) -> None:
        skill_dir = workspace_path / "skills" / "tau2-tools"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            self._build_skill_markdown(tools), encoding="utf-8"
        )
        tools_dir = workspace_path / "tools"
        tools_dir.mkdir(exist_ok=True)
        self._write_json(
            tools_dir / "tau2_tools.json",
            {
                "version": "1.0",
                "tools": [tool.openai_schema for tool in tools],
                "metadata": {
                    "source": "tau2-bench-socket-server",
                    "count": len(tools),
                    "socket_tools_dir": "socket_tools/",
                },
            },
        )

    def create_agent_workspace(
        self, agent_id: str, tools: list[Tool], agent_name: str | None = None
    ) -> str:
        workspace_path = self.get_workspace_path(agent_id)
        agent_dir = self.openclaw_dir / "agents" / agent_id / "agent"
        workspace_path.mkdir(parents=True, exist_ok=True)
        try:
            agent_dir.mkdir(parents=True, exist_ok=True)
            (self.openclaw_dir / "workspace").mkdir(exist_ok=True)
            self._write_json(
                self.openclaw_dir / "openclaw.json",
                {
                    "models": {"providers": {"sglang": self._provider_config()}},
                    "agents": {
                        "defaults": {"maxConcurrent": 4},
                        "list": [
                            {"id": "main"},
                            {
                                "id": agent_id,
                                "name": agent_name or agent_id,
                                "workspace": str(workspace_path),
                                "agentDir": str(agent_dir),
                            },
                        ],
                    },
                },
            )
            self.created_agents.add(agent_id)
            self._setup_workspace_tools(workspace_path, tools)
        except Exception as exc:
            logger.exception("Error creating agent workspace")
            raise WorkspaceManagerError(f"Failed to create agent: {exc}") from exc
        logger.info("Agent '{}' created with {} tools", agent_id, len(tools))
        return agent_id

    def delete_agent_workspace(self, agent_id: str) -> None:
        try:
            result = subprocess.run(
                [self.cli_command, "agents", "delete", agent_id, "--force"],
                capture_output=True,
                text=True,
                timeout=30,
                input="y\n",
            )
            if result.returncode:
                logger.warning(
                    "Failed to delete agent '{}': {}",
                    agent_id,
                    result.stderr or result.stdout,
                )
        except Exception as exc:
            logger.debug("Agent deletion via CLI failed for '{}': {}", agent_id, exc)
        shutil.rmtree(self.get_workspace_path(agent_id), ignore_errors=True)
        shutil.rmtree(self.openclaw_dir / "agents" / agent_id, ignore_errors=True)
        self.created_agents.discard(agent_id)
        logger.info("Agent '{}' deleted", agent_id)

    def cleanup_all(self) -> None:
        for agent_id in tuple(self.created_agents):
            self.delete_agent_workspace(agent_id)
        try:
            self.base_workspace_dir.rmdir()
        except OSError:
            pass
        except Exception as exc:
            logger.error("Error removing base workspace: {}", exc)

    def list_agents(self) -> list[dict[str, Any]]:
        try:
            result = subprocess.run(
                [self.cli_command, "agents", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode:
                logger.warning("Failed to list agents: {}", result.stderr)
                return []
            return json.loads(result.stdout)
        except Exception as exc:
            logger.error("Error listing agents: {}", exc)
            return []

    def get_workspace_path(self, agent_id: str) -> Path:
        return self.base_workspace_dir / agent_id

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all()
