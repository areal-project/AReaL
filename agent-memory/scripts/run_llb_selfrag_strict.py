#!/usr/bin/env python3
"""Launch run_llb with strict, observable Self-RAG critique semantics.

This patch is process-local: it does not modify the shared/root-owned LLB runner.
"""
from __future__ import annotations

import json
import logging
import os
import re
import runpy
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from memrl.run.llb_rl_runner import LLBRunner
from memrl.configs.config import MempConfig

logger = logging.getLogger("memrl.run.llb_rl_runner")


def _matrix_credential(model: str) -> str:
    """Read the current Matrix credential without printing or persisting it."""
    env_value = os.environ.get("MATRIX_API_KEY")
    if env_value:
        return env_value

    import yaml

    credential_config = Path(
        os.environ.get(
            "MATRIX_CREDENTIAL_CONFIG",
            "/storage/openpsi/users/yl/cfuse/rq3_v11_clean_20260719/"
            "config_multisurface_isolated.yaml",
        )
    )
    data = yaml.safe_load(credential_config.read_text(encoding="utf-8")) or {}
    aliases = [model, "text-embedding-3-large"]
    for alias in aliases:
        for item in data.get("model_list", []):
            if item.get("model_name") != alias:
                continue
            value = (item.get("litellm_params") or {}).get("api_key")
            if isinstance(value, str) and value.startswith("os.environ/"):
                value = os.environ.get(value.split("/", 1)[1])
            if value:
                return value
    raise RuntimeError(f"No Matrix credential mapping for model {model!r}")


def _private_runtime_config() -> Path:
    """Copy the requested YAML to node-local /tmp and inject the current key."""
    import yaml

    try:
        index = sys.argv.index("--config")
        source = Path(sys.argv[index + 1])
    except (ValueError, IndexError):
        raise RuntimeError("strict Self-RAG launcher requires --config PATH")
    if not source.is_absolute():
        source = (Path(__file__).resolve().parents[1] / source).resolve()

    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    model = str((config.get("llm") or {}).get("model") or "gpt-4.1-mini-2025-04-14")
    credential = _matrix_credential(model)
    config.setdefault("llm", {})["api_key"] = credential
    config.setdefault("embedding", {})["api_key"] = credential

    fd, name = tempfile.mkstemp(prefix="llb_os_selfrag_", suffix=".yaml", dir="/tmp")
    os.close(fd)
    runtime = Path(name)
    runtime.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    runtime.chmod(0o600)
    sys.argv[index + 1] = str(runtime)
    logger.warning(
        "[Self-RAG] MATRIX CREDENTIAL LOADED from verified mapping; runtime config=%s (mode=0600)",
        runtime,
    )
    return runtime


def _install_config_log_redaction() -> None:
    """Prevent run_llb's config dump from exposing API credentials."""
    original = MempConfig.model_dump_json

    def redacted(self, *args, **kwargs):
        import json

        payload = self.model_dump(mode="json")
        for section in ("llm", "embedding"):
            if isinstance(payload.get(section), dict) and payload[section].get("api_key"):
                payload[section]["api_key"] = "<REDACTED>"
        indent = kwargs.get("indent")
        return json.dumps(payload, ensure_ascii=False, indent=indent, default=str)

    redacted.__name__ = original.__name__
    MempConfig.model_dump_json = redacted


def strict_self_rag_critique(
    self: LLBRunner, task_description: str, selected_mems: List[dict], inject_k: int
) -> List[dict]:
    if not selected_mems:
        return []

    numbered = []
    for i, memory in enumerate(selected_mems):
        content = memory.get("content") or ""
        numbered.append(f"[Memory {i + 1}]\n{content[:2000]}")
    prompt = (
        "You are a relevance judge. Given a task description and a list of retrieved memories "
        "from past problem-solving attempts, decide which memories are RELEVANT and could help "
        "solve the current task.\n\n"
        f"Task: {task_description[:2000]}\n\n"
        "Retrieved memories:\n" + "\n\n".join(numbered) + "\n\n"
        "Return ONLY a JSON list of the relevant memory numbers (1-indexed). "
        "If none are relevant, return an empty list: []\n"
        "Example: [1, 3]"
    )

    attempts = max(1, int(os.environ.get("MEMRL_SELFRAG_MAX_ATTEMPTS", "8")))
    base_delay = max(0.0, float(os.environ.get("MEMRL_SELFRAG_RETRY_DELAY", "10")))
    last_error: Optional[BaseException] = None

    for attempt in range(1, attempts + 1):
        try:
            response = self.llm_provider.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            match = re.search(r"\[[\d\s,]*\]", response or "")
            if not match:
                raise ValueError(f"malformed response: {(response or '')[:160]!r}")
            indices = json.loads(match.group())
            if not isinstance(indices, list) or any(
                isinstance(index, bool) or not isinstance(index, int) for index in indices
            ):
                raise ValueError(f"indices are not an integer list: {indices!r}")

            valid_indices = []
            seen = set()
            for index in indices:
                if 1 <= index <= len(selected_mems) and index not in seen:
                    valid_indices.append(index)
                    seen.add(index)
            filtered = [selected_mems[index - 1] for index in valid_indices]
            logger.info(
                "[Self-RAG] Critique OK attempt=%d/%d kept=%d/%d indices=%s",
                attempt,
                attempts,
                len(filtered),
                len(selected_mems),
                valid_indices,
            )
            return filtered[:inject_k]
        except BaseException as error:
            last_error = error
            logger.warning(
                "[Self-RAG] Critique attempt %d/%d failed: %s",
                attempt,
                attempts,
                error,
            )
            if attempt < attempts and base_delay:
                time.sleep(min(base_delay * (2 ** (attempt - 1)), 120.0))

    raise RuntimeError(
        f"Self-RAG critique failed after {attempts} attempts; refusing silent RAG fallback"
    ) from last_error


LLBRunner._self_rag_critique = strict_self_rag_critique
logger.warning(
    "[Self-RAG] STRICT PATCH ACTIVE: retries=%s delay=%ss; ordinary-RAG fallback disabled",
    os.environ.get("MEMRL_SELFRAG_MAX_ATTEMPTS", "8"),
    os.environ.get("MEMRL_SELFRAG_RETRY_DELAY", "10"),
)

project_root = Path(__file__).resolve().parents[1]
os.chdir(project_root)
_runtime_config = _private_runtime_config()
_install_config_log_redaction()
try:
    runpy.run_path(str(project_root / "run" / "run_llb.py"), run_name="__main__")
finally:
    try:
        _runtime_config.unlink(missing_ok=True)
    except Exception:
        pass
