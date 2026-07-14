# SPDX-License-Identifier: Apache-2.0

import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import pytest

from areal.api.reward_api import AsyncRewardWrapper


# Module-level: ProcessPoolExecutor requires picklable callables
def _add_reward(a: float, b: float) -> float:
    return a + b


def _slow_reward(seconds: float) -> float:
    time.sleep(seconds)
    return 1.0


def _crash_worker() -> float:
    os._exit(1)


def _raise_value_error() -> float:
    raise ValueError("intentional error from reward fn")


@pytest.fixture(autouse=True)
def _isolate_executor_state():
    """Shut down and clear all shared executors before and after each test."""
    AsyncRewardWrapper._atexit_shutdown_all()
    yield
    AsyncRewardWrapper._atexit_shutdown_all()


class TestAsyncRewardWrapperBasic:
    @pytest.mark.asyncio
    async def test_returns_correct_result(self):
        wrapper = AsyncRewardWrapper(_add_reward, max_workers=1, max_retries=0)
        result = await wrapper(3.0, 4.0)
        assert result == 7.0

    @pytest.mark.asyncio
    async def test_multiple_calls_return_correct_results(self):
        wrapper = AsyncRewardWrapper(_add_reward, max_workers=2, max_retries=0)
        results = [await wrapper(float(i), 1.0) for i in range(5)]
        assert results == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_shared_executor_for_same_max_workers(self):
        w1 = AsyncRewardWrapper(_add_reward, max_workers=2)
        w2 = AsyncRewardWrapper(_add_reward, max_workers=2)
        assert w1._executor_key == w2._executor_key
        assert AsyncRewardWrapper._executors.get(2) is not None
        assert len(AsyncRewardWrapper._executors) == 1

    def test_different_executor_for_different_max_workers(self):
        AsyncRewardWrapper(_add_reward, max_workers=1)
        AsyncRewardWrapper(_add_reward, max_workers=2)
        assert len(AsyncRewardWrapper._executors) == 2


class TestAsyncRewardWrapperTimeout:
    @pytest.mark.asyncio
    async def test_timeout_returns_zero_after_retries(self):
        wrapper = AsyncRewardWrapper(
            _slow_reward, timeout_seconds=0.1, max_workers=1, max_retries=1
        )
        result = await wrapper(10.0)
        assert isinstance(result, float)
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_timeout_no_retries_returns_zero(self):
        wrapper = AsyncRewardWrapper(
            _slow_reward, timeout_seconds=0.1, max_workers=1, max_retries=0
        )
        result = await wrapper(10.0)
        assert isinstance(result, float)
        assert result == 0.0


class TestAsyncRewardWrapperBrokenPool:
    @pytest.mark.asyncio
    async def test_crash_raises_broken_process_pool(self):
        wrapper = AsyncRewardWrapper(
            _crash_worker, max_workers=1, max_retries=0, timeout_seconds=5
        )
        with pytest.raises(BrokenProcessPool):
            await wrapper()

    @pytest.mark.asyncio
    async def test_recreation_replaces_executor(self):
        wrapper = AsyncRewardWrapper(
            _crash_worker, max_workers=1, max_retries=1, timeout_seconds=5
        )
        executor_before = AsyncRewardWrapper._executors.get(1)
        assert executor_before is not None

        with pytest.raises(BrokenProcessPool):
            await wrapper()

        executor_after = AsyncRewardWrapper._executors.get(1)
        assert executor_after is not executor_before


class TestAsyncRewardWrapperExceptionHandling:
    @pytest.mark.asyncio
    async def test_reward_fn_exception_propagates(self):
        wrapper = AsyncRewardWrapper(_raise_value_error, max_workers=1, max_retries=0)
        with pytest.raises(ValueError, match="intentional error"):
            await wrapper()

    @pytest.mark.asyncio
    async def test_reward_fn_exception_retries_then_raises(self):
        wrapper = AsyncRewardWrapper(_raise_value_error, max_workers=1, max_retries=2)
        with pytest.raises(ValueError, match="intentional error"):
            await wrapper()

    @pytest.mark.asyncio
    async def test_shutdown_then_call_raises_runtime_error(self):
        wrapper = AsyncRewardWrapper(_add_reward, max_workers=1, max_retries=0)
        AsyncRewardWrapper._atexit_shutdown_all()
        with pytest.raises(RuntimeError, match="has been shut down"):
            await wrapper(1.0, 2.0)


class TestAsyncRewardWrapperAtexitCleanup:
    def test_atexit_clears_all_executors(self):
        AsyncRewardWrapper(_add_reward, max_workers=1)
        AsyncRewardWrapper(_add_reward, max_workers=2)
        assert len(AsyncRewardWrapper._executors) == 2

        AsyncRewardWrapper._atexit_shutdown_all()
        assert len(AsyncRewardWrapper._executors) == 0

    def test_atexit_is_idempotent(self):
        AsyncRewardWrapper(_add_reward, max_workers=1)
        AsyncRewardWrapper._atexit_shutdown_all()
        AsyncRewardWrapper._atexit_shutdown_all()
        assert len(AsyncRewardWrapper._executors) == 0


class TestRecreateExecutorRaceSafety:
    def test_recreate_skips_when_already_replaced(self):
        AsyncRewardWrapper(_add_reward, max_workers=1)
        original = AsyncRewardWrapper._executors[1]

        # Simulate a concurrent thread having already replaced the executor
        replacement = ProcessPoolExecutor(max_workers=1)
        AsyncRewardWrapper._executors[1] = replacement

        result = AsyncRewardWrapper._recreate_executor(1, 1, original)
        assert result is replacement

        replacement.shutdown(wait=False)

    def test_recreate_replaces_when_identity_matches(self):
        AsyncRewardWrapper(_add_reward, max_workers=1)
        original = AsyncRewardWrapper._executors[1]

        result = AsyncRewardWrapper._recreate_executor(1, 1, original)
        assert result is not None
        assert result is not original
        assert AsyncRewardWrapper._executors[1] is result
