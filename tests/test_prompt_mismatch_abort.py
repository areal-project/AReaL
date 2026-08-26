# SPDX-License-Identifier: Apache-2.0

"""A server refusing a prompt as inexact must stop the run, not a trajectory.

The disagreement is not a property of one request: it may be deployment-wide,
or specific to a class of input such as certain image shapes or concat
histories. Either way the affected rollouts are dropped and reported as
ordinary rejections, so continuing spends the run producing less -- or nothing
-- with no signal saying why.
"""

from pathlib import Path
from unittest.mock import Mock

import aiohttp
import pytest

from areal.api.cli_args import InferenceEngineConfig
from areal.infra.workflow_executor import (
    BatchTaskDispatcher,
    WorkflowExecutor,
    _RolloutTaskInput,
)
from areal.utils import logging
from areal.utils.vision_canary import EXACT_TOKEN_REFUSAL, is_exact_token_refusal


def _rejection() -> aiohttp.ClientResponseError:
    """The error aiohttp raises when the server refuses an inexact prompt."""
    return aiohttp.ClientResponseError(
        Mock(real_url="http://server:8000/inference/v1/generate"),
        (),
        status=400,
        message=(
            "[areal-exact-token] Rendered prompt does not match "
            "expected_token_ids. expected_len=81 actual_len=144 "
            "first_divergence=12"
        ),
        headers={},
    )


# ---------------------------------------------------------------------------
# Recognising the failure
# ---------------------------------------------------------------------------


def test_a_rejection_is_recognised():
    assert is_exact_token_refusal(_rejection())


def test_the_uninspectable_prompt_branch_is_also_a_refusal():
    """Test that both of the patch's refusal branches abort, not just one.

    The server also refuses when it cannot inspect the final prompt at all.
    Recognising only the disagreement branch would let that one through as an
    ordinary dropped trajectory.
    """
    exc = RuntimeError(
        "400, message='[areal-exact-token] Cannot verify expected_token_ids: "
        "this request produced no prompt_token_ids'"
    )

    assert is_exact_token_refusal(exc)


def test_every_refusal_the_patch_emits_carries_the_marker():
    """Test that the marker is on every refusal branch in the patch.

    The predicate matches the marker alone, so a branch added without it would
    silently stop aborting.
    """
    patch = Path("patches/vllm.v0.23.0-exact-token-validation.patch").read_text()
    refusals = [
        line
        for line in patch.splitlines()
        if line.startswith("+") and "create_error_response(" in line
    ]
    assert len(refusals) == 2, refusals
    assert patch.count(EXACT_TOKEN_REFUSAL) == 2


def test_a_rejection_survives_being_wrapped():
    """Test that a chained exception does not hide the reason.

    A refusal rarely reaches the executor bare: the bridge re-raises it as an
    HTTPStatusError and callers chain it further, so the response that explains
    why is not reliably the top-level exception.
    """
    inner = _rejection()
    outer = RuntimeError(f"Request to the inference server failed: {inner!r}")
    outer.__cause__ = inner

    assert is_exact_token_refusal(outer)


def test_unrelated_failures_are_not_mismatches():
    """Test that a timeout or a 503 is not treated as a contract violation.

    Aborting on those would turn ordinary cluster noise into a stopped run.
    """
    assert not is_exact_token_refusal(TimeoutError("read timeout"))
    assert not is_exact_token_refusal(
        aiohttp.ClientResponseError(
            Mock(real_url="http://server:8000/x"),
            (),
            status=503,
            message="Service Unavailable",
            headers={},
        )
    )
    assert not is_exact_token_refusal(None)


def test_a_cyclic_cause_chain_terminates():
    """Test that a cycle in __cause__ cannot hang the rollout path."""
    a, b = RuntimeError("a"), RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a

    assert not is_exact_token_refusal(a)


def test_marker_matches_what_the_patch_emits():
    """Test that the predicate keys off the text the vLLM patch actually sends.

    The two live in different repos, so drift would make every refusal read as
    an unrelated failure and silently disable the abort.
    """
    patch = Path("patches/vllm.v0.23.0-exact-token-validation.patch").read_text()
    assert EXACT_TOKEN_REFUSAL in patch


# ---------------------------------------------------------------------------
# Acting on it
# ---------------------------------------------------------------------------


class _RaisingWorkflow:
    def __init__(self, exc):
        self._exc = exc

    async def arun_episode(self, engine, data):
        raise self._exc


def _executor(abort: bool) -> WorkflowExecutor:
    config = InferenceEngineConfig(
        experiment_name="t",
        trial_name="t",
        consumer_batch_size=1,
        abort_on_prompt_mismatch=abort,
    )
    # A staleness manager is supplied directly so the executor does not need
    # initialize(), which would stand up dispatcher threads this test never uses.
    executor = WorkflowExecutor(
        config=config, inference_engine=Mock(), staleness_manager=Mock()
    )
    # initialize() would set these; it is skipped here because it also stands up
    # dispatcher threads this test never uses. A real logger keeps the error
    # path under test rather than short-circuiting it.
    executor.logger = logging.getLogger("PromptMismatchAbortTest")
    dispatcher = BatchTaskDispatcher(
        max_queue_size=1,
        task_factory=executor._create_workflow_task,
        staleness_manager=Mock(),
    )
    dispatcher.logger = executor.logger
    executor._dispatcher = dispatcher
    return executor


async def _run(executor: WorkflowExecutor, exc: BaseException):
    task = _RolloutTaskInput(task_id=1, data={}, workflow=_RaisingWorkflow(exc))
    return await executor._create_workflow_task(task)()


@pytest.mark.asyncio
async def test_a_mismatch_stops_the_run():
    """Test that the failure is escalated to the fail-fast channel."""
    executor = _executor(abort=True)

    assert await _run(executor, _rejection()) is None
    with pytest.raises(RuntimeError, match="Background thread failed"):
        executor.dispatcher._check_thread_exception()


@pytest.mark.asyncio
async def test_an_unrelated_failure_still_only_drops_the_rollout():
    """Test that ordinary workflow errors keep the existing behaviour."""
    executor = _executor(abort=True)

    assert await _run(executor, ValueError("boom")) is None
    executor.dispatcher._check_thread_exception()  # must not raise


@pytest.mark.asyncio
async def test_the_abort_can_be_turned_off():
    """Test that the escape hatch restores drop-and-continue."""
    executor = _executor(abort=False)

    assert await _run(executor, _rejection()) is None
    executor.dispatcher._check_thread_exception()  # must not raise


def test_the_default_is_to_abort():
    """Test that the safe behaviour is what an unconfigured run gets.

    Silent corruption of every multimodal rollout is worse than a stopped run.
    """
    assert (
        InferenceEngineConfig(
            experiment_name="t", trial_name="t"
        ).abort_on_prompt_mismatch
        is True
    )


class TestFatalErrorReachesTheCaller:
    """The abort is only worth anything if the trainer's own wait sees it.

    The dispatcher's waits check for a fatal error inside their retry loop, so
    a batch whose results were already buffered used to skip the check and hand
    back a healthy-looking batch. These drive the public API rather than
    ``_check_thread_exception`` so that hole cannot reopen unnoticed.
    """

    @staticmethod
    def _dispatcher_with_a_buffered_result():
        from areal.infra.workflow_executor import BatchTaskDispatcher, TimedResult

        d = BatchTaskDispatcher(
            max_queue_size=1, task_factory=lambda t: None, staleness_manager=Mock()
        )
        d.logger = Mock()
        d._pending_results[1] = TimedResult(task_id=1, data="accepted", create_time=0.0)
        d._active_task_ids.add(1)
        return d

    def test_wait_results_surfaces_a_fatal_error_despite_buffered_results(self):
        d = self._dispatcher_with_a_buffered_result()
        d.fail_fast(_rejection())

        with pytest.raises(RuntimeError, match="Background thread failed"):
            d.wait_results(1, timeout=1)

    def test_wait_for_task_surfaces_a_fatal_error_despite_buffered_results(self):
        d = self._dispatcher_with_a_buffered_result()
        d.fail_fast(_rejection())

        with pytest.raises(RuntimeError, match="Background thread failed"):
            d.wait_for_task(1, timeout=1)

    def test_a_healthy_dispatcher_still_returns_its_results(self):
        """Test that the added check does not swallow ordinary batches."""
        d = self._dispatcher_with_a_buffered_result()

        assert d.wait_results(1, timeout=1) == ["accepted"]
