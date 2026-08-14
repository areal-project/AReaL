from __future__ import annotations

from typing import Any, Dict, List, Optional

from .llb_jsonl import LLBJsonlTracer, _sha1


def _jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    return str(x)


class TracingLLMProvider:
    """Wrap an OpenAI-like provider to record full model I/O into the current task trace.

    This is intentionally generic: it only assumes the wrapped provider has:
      generate(messages: list[dict[str, str]], **kwargs) -> str
    """

    def __init__(self, provider: Any, *, tracer: Optional[LLBJsonlTracer]) -> None:
        self._provider = provider
        self._tracer = tracer

    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        tracer = self._tracer
        ctx = tracer.current() if tracer is not None else None

        system_prompt: Optional[str] = None
        messages_wo_system: List[Dict[str, str]] = []
        for m in messages or []:
            role = m.get("role")
            if role == "system" and system_prompt is None:
                system_prompt = m.get("content", "")
                continue
            messages_wo_system.append({"role": role or "unknown", "content": m.get("content", "")})

        system_prompt_id = _sha1(system_prompt or "") if system_prompt is not None else None

        try:
            out = self._provider.generate(messages, **kwargs) or ""
            if ctx is not None:
                # Ensure task-level prompt is captured once without duplicating it per call.
                if system_prompt is not None and not ctx.prompt.get("full_system_prompt"):
                    ctx.set_full_system_prompt(system_prompt)
                ctx.add_llm_call(
                    system_prompt_id=system_prompt_id,
                    messages_wo_system=messages_wo_system,
                    params=_jsonable(dict(kwargs)),
                    response_text=str(out),
                )
            return out
        except Exception as e:
            if ctx is not None:
                ctx.add_llm_call(
                    system_prompt_id=system_prompt_id,
                    messages_wo_system=messages_wo_system,
                    params=_jsonable(dict(kwargs)),
                    response_text=None,
                    error={"type": type(e).__name__, "message": str(e)},
                )
            raise
