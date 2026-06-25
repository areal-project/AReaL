# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import queue
import signal
import threading
import time

from math_verify.grader import verify as math_verify_verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

from areal.utils import logging

logger = logging.getLogger("RewardUtils")


class _MathVerifyTimeoutError(TimeoutError):
    pass


def _raise_math_verify_timeout(signum, frame):
    del signum, frame
    raise _MathVerifyTimeoutError()


def _verify_in_subprocess(
    worker: "MathVerifyWorker",
    response: str,
    ground_truth: str,
    result_queue: mp.Queue,
) -> None:
    try:
        result_queue.put(("ok", worker._verify_impl(response, ground_truth)))
    except Exception as exc:
        result_queue.put(("error", repr(exc)))


class MathVerifyWorker:
    """Thin wrapper over math_verify with configurable extraction/precision.

    ``math_verify`` timeouts use ``SIGALRM``, which is invalid in reward
    worker threads. This wrapper disables those nested timeouts and owns one
    wall-clock bound around parsing + comparison.

    Args:
        try_extract_without_anchor: When False, only answers with explicit anchors
            (e.g., "answer = 1", "final answer = 1") are matched. When True,
            any numeric string in the text may be extracted.
        precision: Number of significant digits that must match.
        timeout: Timeout in seconds for parsing + comparison. ``None`` disables
            the timeout.
    """

    def __init__(
        self,
        try_extract_without_anchor=True,
        precision: int = 6,
        timeout: float | None = 5.0,
    ):
        self.gold_extraction_target = (
            ExprExtractionConfig(try_extract_without_anchor=try_extract_without_anchor),
            LatexExtractionConfig(),
        )
        self.pred_extraction_target = (
            ExprExtractionConfig(try_extract_without_anchor=try_extract_without_anchor),
            LatexExtractionConfig(),
        )
        self.precision = precision
        self.timeout = timeout

    def _verify_impl(self, response: str, ground_truth: str) -> float:
        """Core verification logic without timeout wrapper."""
        gold_parsed = parse(
            ground_truth,
            extraction_config=self.gold_extraction_target,
            parsing_timeout=None,
        )
        pred_parsed = parse(
            response,
            extraction_config=self.pred_extraction_target,
            parsing_timeout=None,
        )
        if not gold_parsed or not pred_parsed:
            return 0.0
        result = math_verify_verify(
            gold_parsed,
            pred_parsed,
            float_rounding=self.precision,
            timeout_seconds=None,
        )
        return 1.0 if result else 0.0

    def _verify_with_signal_timeout(self, response: str, ground_truth: str) -> float:
        assert self.timeout is not None
        previous_handler = signal.getsignal(signal.SIGALRM)
        started = time.monotonic()
        signal.signal(signal.SIGALRM, _raise_math_verify_timeout)
        previous_timer = signal.setitimer(signal.ITIMER_REAL, self.timeout)
        try:
            return self._verify_impl(response, ground_truth)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0] > 0 or previous_timer[1] > 0:
                elapsed = time.monotonic() - started
                delay = max(previous_timer[0] - elapsed, 1e-6)
                signal.setitimer(signal.ITIMER_REAL, delay, previous_timer[1])

    def _verify_with_subprocess_timeout(
        self, response: str, ground_truth: str
    ) -> float:
        assert self.timeout is not None
        ctx = mp.get_context("spawn")
        result_queue: mp.Queue = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=_verify_in_subprocess,
            args=(self, response, ground_truth, result_queue),
        )
        proc.start()
        proc.join(self.timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.kill()
                proc.join()
            raise _MathVerifyTimeoutError()
        try:
            status, payload = result_queue.get_nowait()
        except queue.Empty:
            return 0.0
        if status == "ok":
            return float(payload)
        raise RuntimeError(payload)

    def verify(self, response: str, ground_truth: str) -> float:
        try:
            if self.timeout is None:
                return self._verify_impl(response, ground_truth)
            if threading.current_thread() is threading.main_thread() and hasattr(
                signal, "SIGALRM"
            ):
                return self._verify_with_signal_timeout(response, ground_truth)
            return self._verify_with_subprocess_timeout(response, ground_truth)
        except _MathVerifyTimeoutError:
            logger.warning(
                f"Timeout ({self.timeout}s) in MathVerifyWorker.verify for "
                f"response={response!r} and ground_truth={ground_truth!r}",
            )
            return 0.0
        except Exception:
            logger.warning(
                f"Exception in MathVerifyWorker.verify for response={response} and ground_truth={ground_truth}",
                exc_info=True,
            )
            return 0.0


_MATH_VERIFY_WORKER: MathVerifyWorker | None = None


def get_math_verify_worker() -> MathVerifyWorker:
    global _MATH_VERIFY_WORKER
    if _MATH_VERIFY_WORKER is None:
        _MATH_VERIFY_WORKER = MathVerifyWorker()
    return _MATH_VERIFY_WORKER


__all__ = [
    "MathVerifyWorker",
    "get_math_verify_worker",
    "gsm8k_reward_fn",
    "geometry3k_reward_fn",
    "clevr_count_70k_reward_fn",
]


_LAZY_IMPORTS = {
    "gsm8k_reward_fn": "areal.reward.gsm8k",
    "geometry3k_reward_fn": "areal.reward.geometry3k",
    "clevr_count_70k_reward_fn": "areal.reward.clevr_count_70k",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
