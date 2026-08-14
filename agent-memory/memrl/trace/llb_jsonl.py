from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _now_ts() -> float:
    return time.time()


def summarize_text(text: str, *, preview_chars: int = 120) -> Dict[str, Any]:
    """Symbolic text summary to avoid bloating traces with repeated long strings."""
    s = text or ""
    return {
        "sha1": _sha1(s),
        "length": len(s),
        "preview": s[:preview_chars],
    }


def _parse_sample_filter(sample_filter: Optional[str]) -> Dict[str, Any]:
    """Parse TRACE_SAMPLE_FILTER.

    Supported:
    - None/empty: trace all
    - digits (e.g. "3"): trace first N tasks (in encounter order)
    - comma/space separated tokens (e.g. "1, 2, abc"): trace only those sample_index values
    """
    if not sample_filter:
        return {"mode": "all"}
    raw = sample_filter.strip()
    if raw.isdigit():
        n = int(raw)
        return {"mode": "limit", "remaining": max(n, 0)}
    tokens: List[str] = []
    for part in raw.replace(",", " ").split():
        p = part.strip()
        if p:
            tokens.append(p)
    return {"mode": "set", "allowed": set(tokens)}


def apply_trace_env_from_experiment_config(experiment_cfg: Any) -> None:
    """Apply TRACE_* env vars from ExperimentConfig.

    Precedence rule (requested behavior):
    - If the YAML (ExperimentConfig) explicitly sets a trace field, it wins over any
      pre-existing environment variables (including allowing YAML to unset them via null/empty).
    - If the YAML does not mention a trace field at all, we leave the env var untouched
      (so callers can still opt-in via env vars without editing YAML).
    """
    try:
        yaml_path = getattr(experiment_cfg, "trace_jsonl_path", None)
        yaml_filter = getattr(experiment_cfg, "trace_sample_filter", None)
    except Exception:
        return

    # Pydantic v2 tracks which fields were explicitly provided via `model_fields_set`.
    # Use this to distinguish:
    # - "key absent in YAML" (do nothing) vs
    # - "key present but null/empty" (explicitly clear env).
    fields_set = None
    try:
        fields_set = getattr(experiment_cfg, "model_fields_set", None)
        if fields_set is None:
            # Pydantic v1 compatibility.
            fields_set = getattr(experiment_cfg, "__fields_set__", None)
    except Exception:
        fields_set = None

    def _explicit(name: str, value: Any) -> bool:
        if fields_set is None:
            # Best-effort fallback: only treat non-None values as explicit.
            return value is not None
        return name in fields_set

    if _explicit("trace_jsonl_path", yaml_path):
        if yaml_path is None or str(yaml_path).strip() == "":
            os.environ.pop("TRACE_JSONL_PATH", None)
        else:
            os.environ["TRACE_JSONL_PATH"] = str(yaml_path)

    if _explicit("trace_sample_filter", yaml_filter):
        if yaml_filter is None or str(yaml_filter).strip() == "":
            os.environ.pop("TRACE_SAMPLE_FILTER", None)
        else:
            os.environ["TRACE_SAMPLE_FILTER"] = str(yaml_filter)


_TLS = threading.local()


@dataclass
class TaskTraceContext:
    sample_index: str
    run_meta: Dict[str, Any]
    task_description: str
    ts_start: float = field(default_factory=_now_ts)
    ts_end: Optional[float] = None

    retrieval: Dict[str, Any] = field(default_factory=dict)
    prompt: Dict[str, Any] = field(default_factory=dict)
    llm_calls: List[Dict[str, Any]] = field(default_factory=list)
    interaction: Dict[str, Any] = field(default_factory=dict)
    outcome: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None

    _call_seq: int = 0

    def set_full_system_prompt(self, full_system_prompt: str) -> None:
        self.prompt["full_system_prompt"] = full_system_prompt
        self.prompt["system_prompt_id"] = _sha1(full_system_prompt)

    @property
    def system_prompt_id(self) -> Optional[str]:
        spid = self.prompt.get("system_prompt_id")
        return str(spid) if spid else None

    def add_llm_call(
        self,
        *,
        system_prompt_id: Optional[str],
        messages_wo_system: List[Dict[str, str]],
        params: Dict[str, Any],
        response_text: Optional[str] = None,
        finish_reason: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._call_seq += 1
        self.llm_calls.append(
            {
                "call_id": self._call_seq,
                "ts": _now_ts(),
                "system_prompt_id": system_prompt_id,
                "messages_wo_system": messages_wo_system,
                "params": params,
                "response_text": response_text,
                "finish_reason": finish_reason,
                "usage": usage,
                "error": error,
            }
        )

    def to_json_obj(self) -> Dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "task_description": self.task_description,
            "run": self.run_meta,
            "ts_start": self.ts_start,
            "ts_end": self.ts_end,
            "retrieval": self.retrieval,
            "prompt": self.prompt,
            "llm_calls": self.llm_calls,
            "interaction": self.interaction,
            "outcome": self.outcome,
            "error": self.error,
        }


class LLBJsonlTracer:
    """Per-task tracer, activated via env vars.

    - TRACE_JSONL_PATH: enable tracing and write to this path
        - If ends with ".jsonl": append one JSON object per line (recommended for large runs)
        - If ends with ".json": write a single JSON array (recommended for small runs / easy viewing)
    - TRACE_SAMPLE_FILTER: optional (see _parse_sample_filter)
    """

    def __init__(self, *, path: Path, sample_filter: Optional[str] = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Backwards compatible: default is JSONL. If the path ends with ".json",
        # we write a single JSON array (nice for small runs and inspection).
        self._format = "json" if self.path.suffix.lower() == ".json" else "jsonl"
        self._lock = threading.Lock()
        self._filter = _parse_sample_filter(sample_filter)

    @classmethod
    def from_env(cls) -> Optional["LLBJsonlTracer"]:
        raw_path = os.environ.get("TRACE_JSONL_PATH", "").strip()
        if not raw_path:
            return None
        p = Path(raw_path)
        if not p.is_absolute():
            # Keep relative paths anchored at cwd so runner scripts behave predictably.
            p = Path(os.getcwd()) / p
        filt = os.environ.get("TRACE_SAMPLE_FILTER", None)
        return cls(path=p, sample_filter=filt)

    def _should_trace(self, sample_index: str) -> bool:
        mode = self._filter.get("mode")
        if mode == "all":
            return True
        if mode == "set":
            allowed = self._filter.get("allowed", set())
            return sample_index in allowed
        if mode == "limit":
            with self._lock:
                rem = int(self._filter.get("remaining", 0))
                if rem <= 0:
                    return False
                self._filter["remaining"] = rem - 1
                return True
        return False

    @contextlib.contextmanager
    def task(
        self, *, sample_index: str, run_meta: Dict[str, Any], task_description: str
    ) -> Iterator[Optional[TaskTraceContext]]:
        if not self._should_trace(sample_index):
            yield None
            return

        ctx = TaskTraceContext(
            sample_index=str(sample_index),
            run_meta=dict(run_meta),
            task_description=str(task_description or ""),
        )
        _TLS.current = ctx
        try:
            yield ctx
        except Exception as e:
            # Best-effort error capture; re-raise after writing.
            ctx.error = {"type": type(e).__name__, "message": str(e)}
            raise
        finally:
            ctx.ts_end = _now_ts()
            _TLS.current = None
            self._write(ctx)

    def current(self) -> Optional[TaskTraceContext]:
        ctx = getattr(_TLS, "current", None)
        return ctx if isinstance(ctx, TaskTraceContext) else None

    def _write(self, ctx: TaskTraceContext) -> None:
        obj = ctx.to_json_obj()
        if self._format == "jsonl":
            line = json.dumps(obj, ensure_ascii=False)
            with self._lock:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            return

        # JSON array mode: read-modify-write with atomic replace.
        with self._lock:
            items: List[Dict[str, Any]] = []
            if self.path.exists():
                try:
                    txt = self.path.read_text(encoding="utf-8").strip()
                    if txt:
                        loaded = json.loads(txt)
                        if isinstance(loaded, list):
                            items = loaded
                        else:
                            # If a user accidentally points to a non-array JSON file,
                            # keep it readable by treating it as a single-entry history.
                            items = [loaded]
                except Exception:
                    # Corrupted/partial file; start fresh rather than crashing training.
                    items = []

            items.append(obj)
            tmp_path = self.path.with_name(self.path.name + ".tmp")
            tmp_path.write_text(
                json.dumps(items, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp_path, self.path)
