"""
Official BigCodeBench evaluation helpers.

We vendor BigCodeBench under:
  3rdparty/bigcodebench-main

This module provides:
- `ensure_bigcodebench_on_path` to import the vendored package
- a hard-timeout wrapper around `bigcodebench.eval.untrusted_check`
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BCB_REPO = REPO_ROOT / "3rdparty" / "bigcodebench-main"


def ensure_bigcodebench_on_path(bcb_repo: str | os.PathLike[str] | None = None) -> str:
    """Ensure vendored BigCodeBench is importable and return the resolved path."""
    path = Path(bcb_repo) if bcb_repo else DEFAULT_BCB_REPO
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"bigcodebench repo not found at: {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return str(path)


def _untrusted_check_worker(
    code: str,
    test_code: str,
    entry_point: str,
    max_as_limit: int,
    max_data_limit: int,
    max_stack_limit: int,
    min_time_limit: float,
    gt_time_limit: float,
    q: Any,
    bcb_repo: str | None,
) -> None:
    try:
        # Optional: prepend BCB_EVAL_LIBS_DIR to sys.path so that the eval
        # subprocess can import tensorflow/keras/wordcloud/etc that we keep
        # out of the main vLLM process (numpy 1.x there would clash with vLLM).
        _libs_dir = os.environ.get("BCB_EVAL_LIBS_DIR")
        if _libs_dir and os.path.isdir(_libs_dir):
            if _libs_dir not in sys.path:
                sys.path.insert(0, _libs_dir)
            os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

        ensure_bigcodebench_on_path(bcb_repo)
        from bigcodebench.eval import untrusted_check

        stat, details = untrusted_check(
            code=code,
            test_code=test_code,
            entry_point=entry_point,
            max_as_limit=max_as_limit,
            max_data_limit=max_data_limit,
            max_stack_limit=max_stack_limit,
            min_time_limit=min_time_limit,
            gt_time_limit=gt_time_limit,
        )
        try:
            q.put(("ok", stat, details), block=False)
        except Exception:
            pass
    except Exception as e:
        try:
            q.put(("err", str(e), None), block=False)
        except Exception:
            pass


def run_untrusted_check_with_hard_timeout(
    *,
    code: str,
    test_code: str,
    entry_point: str,
    max_as_limit: int,
    max_data_limit: int,
    max_stack_limit: int,
    min_time_limit: float,
    gt_time_limit: float,
    hard_timeout_s: float = 120.0,
    bcb_repo: str | None = None,
) -> Tuple[Any | None, Any | None, str | None, bool]:
    """Run `bigcodebench.eval.untrusted_check` in a subprocess with a hard timeout.

    Returns:
        (stat, details, error, hard_timed_out)
    """
    import multiprocessing as mp

    try:
        methods = mp.get_all_start_methods()
        ctx = mp.get_context("spawn") if "spawn" in methods else mp.get_context()
    except Exception:
        ctx = mp.get_context()

    q: Any = ctx.Queue(maxsize=1)
    p = ctx.Process(
        target=_untrusted_check_worker,
        args=(
            code,
            test_code,
            entry_point,
            max_as_limit,
            max_data_limit,
            max_stack_limit,
            min_time_limit,
            gt_time_limit,
            q,
            bcb_repo,
        ),
    )

    p.start()
    p.join(timeout=float(hard_timeout_s))
    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
        try:
            p.join(timeout=5)
        except Exception:
            pass
        return None, None, f"Hard timeout after {hard_timeout_s}s", True

    try:
        status, stat, details = q.get_nowait()
    except Exception:
        return None, None, "no_result_from_untrusted_check", False

    if status == "ok":
        return stat, details, None, False
    return None, None, str(stat), False


def sanitize_code(code: str, entry_point: str, *, bcb_repo: str | None = None) -> str:
    """Use BigCodeBench sanitizer (best-effort)."""
    ensure_bigcodebench_on_path(bcb_repo)
    try:
        from bigcodebench.sanitize import sanitize
    except Exception:
        return code
    try:
        return sanitize(code, entry_point)
    except Exception:
        return code

