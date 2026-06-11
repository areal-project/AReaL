import json
import os
import subprocess
import time
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

load_dotenv()


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


class OpenClawConfig(BaseModel):
    cli_command: str = "openclaw"
    timeout: int = 600
    max_retries: int = 3
    api_base: str | None = None
    api_key: str | None = None
    model: str | None = None

    @classmethod
    def from_env(cls) -> "OpenClawConfig":
        env = os.getenv
        return cls(
            cli_command=env("OPENCLAW_CLI_COMMAND", "openclaw"),
            timeout=int(env("OPENCLAW_TIMEOUT", "120")),
            max_retries=int(env("OPENCLAW_MAX_RETRIES", "3")),
            api_base=env("OPENCLAW_API_BASE"),
            api_key=env("OPENCLAW_API_KEY"),
            model=env("OPENCLAW_MODEL"),
        )

    class Config:
        extra = "forbid"


class OpenClawServiceError(Exception):
    pass


class OpenClawService:
    def __init__(
        self,
        cli_command: str = "openclaw",
        timeout: int = 60,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.cli_command = cli_command
        self.timeout = timeout
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self._verify_cli()

    def _verify_cli(self) -> None:
        try:
            result = subprocess.run(
                [self.cli_command, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except FileNotFoundError:
            logger.error("OpenClaw CLI not found: {}", self.cli_command)
            return
        except Exception as exc:
            logger.error("Error checking OpenClaw CLI: {}", exc)
            return
        if result.returncode:
            logger.warning("OpenClaw CLI check returned code {}", result.returncode)
            return
        version = result.stdout.strip().split()
        logger.info(
            "OpenClaw CLI available: {}", version[1] if len(version) > 1 else "unknown"
        )

    def _build_message_text(self, messages: list[dict[str, Any]]) -> str:
        system_parts, user_messages = [], []
        for message in messages:
            text = _message_text(message.get("content", ""))
            if not text.strip():
                continue
            if message.get("role") == "system":
                system_parts.append(text)
            elif message.get("role") == "user":
                user_messages.append(text)
        return (
            "\n\n".join((*system_parts, "---", user_messages[0]))
            if len(user_messages) == 1 and system_parts
            else (user_messages[-1] if user_messages else "")
        )

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.pop("OPENAI_BASE_URL", None)
        env.pop("OPENAI_API_BASE", None)
        if self.api_key:
            env["OPENAI_API_KEY"] = self.api_key
        return env

    def _parse_result(
        self, result: subprocess.CompletedProcess[str], session_id: str
    ) -> dict[str, Any]:
        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse JSON: {}", result.stdout[:500])
            raise OpenClawServiceError(f"Invalid JSON response: {exc}") from exc
        payloads = output.get("payloads")
        if payloads is None and output.get("status") == "ok":
            payloads = output.get("result", {}).get("payloads")
        if payloads is None:
            raise OpenClawServiceError(
                f"OpenClaw returned error: {output.get('error', str(output)[:500])}"
            )
        if not payloads:
            logger.error(
                "[No payloads] output_keys={} output_preview={} stderr_preview={}",
                list(output),
                json.dumps(output, default=str)[:1000],
                (result.stderr or "")[:500],
            )
            raise OpenClawServiceError("No payloads in response")
        payload = payloads[0]
        if not payload.get("text", ""):
            logger.warning(
                "[Empty text] payload={} output_preview={}",
                json.dumps(payload, default=str)[:500],
                json.dumps(output, default=str)[:500],
            )
        return {
            "message": {
                "role": "assistant",
                "content": payload.get("text", ""),
                "tool_calls": None,
            },
            "session_id": session_id,
            "raw_output": output,
        }

    def _retry_enoent(self, error: Any, attempt: int, attempts: int) -> bool:
        if "ENOENT" not in str(error) or attempt >= attempts - 1:
            return False
        logger.warning(
            "Session file not found, retrying in 1.0s ({}/{})", attempt + 1, attempts
        )
        time.sleep(1.0)
        return True

    def chat(
        self,
        messages: list[dict[str, Any]],
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        message_text = self._build_message_text(messages)
        if not message_text:
            raise OpenClawServiceError("No user message found in messages")
        session_id = session_id or "tau2-bench-session"
        cmd = [
            self.cli_command,
            "agent",
            "--message",
            message_text,
            "--session-id",
            session_id,
            "--local",
            "--json",
        ]
        if agent_id:
            cmd.extend(["--agent", agent_id])
        logger.info(
            "[OpenClaw CLI] Running: {} | api_base={} model={}",
            " ".join(cmd),
            self.api_base,
            self.model,
        )
        attempts = 3
        last_error: Any = None
        for attempt in range(attempts):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=self._build_env(),
                )
                if not result.returncode:
                    return self._parse_result(result, session_id)
                last_error = result.stderr or result.stdout
                if self._retry_enoent(last_error, attempt, attempts):
                    continue
                raise OpenClawServiceError(f"OpenClaw CLI failed: {last_error}")
            except subprocess.TimeoutExpired as exc:
                self._cleanup_lock_files(agent_id)
                raise OpenClawServiceError(
                    f"OpenClaw CLI timeout after {self.timeout}s"
                ) from exc
            except OpenClawServiceError:
                raise
            except Exception as exc:
                last_error = exc
                if self._retry_enoent(exc, attempt, attempts):
                    continue
                logger.exception("Error calling OpenClaw CLI")
                raise OpenClawServiceError(f"OpenClaw CLI call failed: {exc}") from exc
        raise OpenClawServiceError(
            f"OpenClaw CLI failed after {attempts} attempts: {last_error}"
        )

    def _cleanup_lock_files(self, agent_id: str | None) -> None:
        if not agent_id:
            return
        try:
            from .workspace_manager import OpenClawWorkspaceManager

            for lock_path in (
                OpenClawWorkspaceManager(
                    cli_command=self.cli_command
                ).get_workspace_path(agent_id)
                / ".openclaw"
            ).rglob("*.lock"):
                lock_path.unlink(missing_ok=True)
                logger.warning("Cleaned up stale lock: {}", lock_path)
        except Exception as exc:
            logger.debug("Lock cleanup failed (non-fatal): {}", exc)

    def cleanup_session(self, session_id: str) -> None:
        logger.debug("Session cleanup not needed for CLI mode: {}", session_id)
