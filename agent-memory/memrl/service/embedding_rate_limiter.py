"""Cross-process rate limiting and retry helpers for embedding operations."""
from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
T = TypeVar("T")
_LOCAL_LOCK = threading.Lock()
_LOCAL_NEXT_REQUEST_AT: dict[str, float] = {}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using %s", name, os.environ.get(name), default)
        return default


def _error_text(exc: BaseException) -> str:
    parts = []
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current))
        current = current.__cause__ or current.__context__
    return " ".join(parts)


def is_rate_limit_error(exc: BaseException) -> bool:
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "status_code", None) == 429:
            return True
        if getattr(getattr(current, "response", None), "status_code", None) == 429:
            return True
        current = current.__cause__ or current.__context__
    text = _error_text(exc).lower()
    return (bool(re.search(r"(?:^|\D)429(?:\D|$)", text))
            or "rate limit" in text or "too many requests" in text)


def is_non_retryable_embedding_error(exc: BaseException) -> bool:
    text = _error_text(exc).lower()
    return (bool(re.search(r"(?:^|\D)428(?:\D|$)", text))
            or "content_filter" in text or "content filter" in text
            or "not retryable" in text)




def is_connection_embedding_error(exc: BaseException) -> bool:
    """Best-effort classification for transient transport failures, not 429."""
    text = _error_text(exc).lower()
    return any(token in text for token in (
        "connection error", "connecterror", "connect timeout", "connecttimeout",
        "read timeout", "readtimeout", "remoteprotocolerror",
        "server disconnected", "connection reset", "connection aborted",
        "temporary failure in name resolution",
    ))

def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        headers = getattr(response, "headers", None) or getattr(current, "headers", None)
        if headers:
            value = headers.get("retry-after") or headers.get("Retry-After")
            if value is not None:
                try:
                    return max(0.0, float(value))
                except (TypeError, ValueError):
                    pass
        current = current.__cause__ or current.__context__
    return None


def _limiter_key() -> str:
    raw = os.environ.get("MEMRL_EMBED_RATE_LIMIT_KEY", "default-embedding-api")
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")[:48] or "default"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}"


def _rate_limit_dir() -> Path:
    configured = os.environ.get("MEMRL_EMBED_RATE_LIMIT_DIR")
    return Path(configured).expanduser() if configured else Path.home() / ".cache" / "memrl" / "embedding_rate_limits"


def _reserve_file_slot(key: str, min_interval: float, not_before: Optional[float]) -> float:
    directory = _rate_limit_dir()
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / f"{key}.state"
    with state_path.open("a+", encoding="utf-8") as state:
        assert fcntl is not None
        fcntl.flock(state.fileno(), fcntl.LOCK_EX)
        try:
            state.seek(0)
            try:
                next_at = float(state.read().strip() or "0")
            except ValueError:
                next_at = 0.0
            slot = max(time.time(), next_at, not_before or 0.0)
            state.seek(0)
            state.truncate()
            state.write(f"{slot + min_interval:.6f}\n")
            state.flush()
            os.fsync(state.fileno())
            return slot
        finally:
            fcntl.flock(state.fileno(), fcntl.LOCK_UN)


def _reserve_local_slot(key: str, min_interval: float, not_before: Optional[float]) -> float:
    with _LOCAL_LOCK:
        slot = max(time.time(), _LOCAL_NEXT_REQUEST_AT.get(key, 0.0), not_before or 0.0)
        _LOCAL_NEXT_REQUEST_AT[key] = slot + min_interval
        return slot


def _reserve_slot(min_interval: float, not_before: Optional[float] = None) -> float:
    key = _limiter_key()
    if fcntl is not None:
        try:
            return _reserve_file_slot(key, min_interval, not_before)
        except Exception as exc:
            logger.warning("Shared embedding limiter unavailable (%s); using process-local limiter", exc)
    return _reserve_local_slot(key, min_interval, not_before)


def _global_interval() -> float:
    fallback = _env_float("MEMRL_EMBED_THROTTLE", 0.3)
    return max(0.0, _env_float("MEMRL_EMBED_GLOBAL_MIN_INTERVAL", fallback))


def wait_for_embedding_slot() -> None:
    """Atomically reserve a shared request slot, then sleep without holding its lock."""
    delay = _reserve_slot(_global_interval()) - time.time()
    if delay > 0:
        time.sleep(delay)


def defer_global_embedding_requests(delay: float) -> None:
    """Publish a 429 cooldown so sibling containers using this key also slow down."""
    if delay > 0:
        _reserve_slot(_global_interval(), not_before=time.time() + delay)


def call_embedding_with_retry(
    call: Callable[[], T], *, operation_name: str = "Embedding API call",
    max_retries: int = 5, base_delay: float = 2.0,
) -> T:
    """Call with shared throttling and bounded retry time.

    429 keeps the longer exponential policy. Transport failures fail fast with
    a separate, smaller attempt count and total wall-clock budget so one serial
    memory write cannot block the entire batch for many minutes.
    """
    max_retries = max(1, int(os.environ.get("MEMRL_EMBED_MAX_RETRIES", max_retries)))
    connection_retries = max(1, int(os.environ.get("MEMRL_EMBED_CONNECTION_MAX_RETRIES", "2") or "2"))
    connection_budget = max(1.0, _env_float("MEMRL_EMBED_CONNECTION_RETRY_BUDGET_S", 75.0))
    connection_base = max(0.0, _env_float("MEMRL_EMBED_CONNECTION_BASE_DELAY", 1.0))
    rate_base = max(0.0, _env_float("MEMRL_EMBED_429_BASE_DELAY", 5.0))
    rate_cap = max(rate_base, _env_float("MEMRL_EMBED_429_MAX_DELAY", 60.0))
    jitter_max = max(0.0, _env_float("MEMRL_EMBED_RETRY_JITTER", 1.0))
    started = time.monotonic()
    attempt = 0

    while True:
        attempt += 1
        try:
            wait_for_embedding_slot()
            return call()
        except Exception as exc:
            if is_non_retryable_embedding_error(exc):
                logger.warning("%s is non-retryable: %s", operation_name, _error_text(exc)[:300])
                raise
            rate_limited = is_rate_limit_error(exc)
            connection_error = (not rate_limited) and is_connection_embedding_error(exc)
            attempt_limit = max_retries if rate_limited else connection_retries
            elapsed = time.monotonic() - started
            logger.warning(
                "[Retry %d/%d] %s failed%s: %s", attempt, attempt_limit, operation_name,
                " with 429" if rate_limited else " with connection error" if connection_error else "", exc,
            )
            if attempt >= attempt_limit:
                logger.error("%s reached retry limit (%d attempts)", operation_name, attempt_limit)
                raise
            if not rate_limited and elapsed >= connection_budget:
                logger.error(
                    "%s exceeded non-429 retry budget %.1fs after %.1fs",
                    operation_name, connection_budget, elapsed,
                )
                raise
            if rate_limited:
                nominal = min(rate_cap, rate_base * (2 ** (attempt - 1)))
                retry_after = _retry_after_seconds(exc)
                if retry_after is not None:
                    nominal = max(nominal, retry_after)
            else:
                nominal = connection_base * (2 ** (attempt - 1))
            sleep_time = nominal + random.uniform(0.0, jitter_max)
            if not rate_limited and elapsed + sleep_time >= connection_budget:
                logger.error(
                    "%s would exceed non-429 retry budget %.1fs; failing fast",
                    operation_name, connection_budget,
                )
                raise
            if rate_limited:
                defer_global_embedding_requests(sleep_time)
            logger.warning("Retrying %s in %.2fs", operation_name, sleep_time)
            time.sleep(sleep_time)


def add_text_memory_with_retry(text_mem: Any, item: Any, *, max_retries: int = 5,
                               base_delay: float = 2.0) -> None:
    """Rate-limit and retry text_mem.add(), which embeds internally."""
    call_embedding_with_retry(lambda: text_mem.add([item]),
                              operation_name="text_mem.add embedding call",
                              max_retries=max_retries, base_delay=base_delay)
