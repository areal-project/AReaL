"""Unit tests for the generic AsyncTaskRunner.

This test suite verifies the functionality of the AsyncTaskRunner without
any AReaL-specific dependencies, ensuring it can be used as a generic
async task executor.
"""

import asyncio
import time

import pytest

from areal.infra.async_task_runner import AsyncTaskRunner


class TestAsyncTaskRunnerBasic:
    """Test basic functionality of AsyncTaskRunner."""

    def test_initialization_and_shutdown(self):
        """Test that runner can be initialized and shut down cleanly."""
        runner = AsyncTaskRunner[int](max_queue_size=10)
        runner.initialize()
        time.sleep(0.1)  # Allow thread to start
        runner.destroy()

    def test_simple_task_submission(self):
        """Test submitting and waiting for a single task."""
        runner = AsyncTaskRunner[int](max_queue_size=10)
        runner.initialize()

        async def add(a: int, b: int) -> int:
            await asyncio.sleep(0.01)
            return a + b

        runner.submit(add, 5, 10, task_id=1)
        results = runner.wait(count=1, timeout=2.0)

        assert len(results) == 1
        assert results[0] == 15
        runner.destroy()

    def test_multiple_task_submission(self):
        """Test submitting and waiting for multiple tasks."""
        runner = AsyncTaskRunner[int](max_queue_size=50)
        runner.initialize()

        async def compute(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        # Submit 10 tasks
        for i in range(10):
            runner.submit(compute, i, task_id=i)

        results = runner.wait(count=10, timeout=5.0)

        assert len(results) == 10
        # Results are shuffled, so check that all expected values are present
        assert set(results) == {0, 2, 4, 6, 8, 10, 12, 14, 16, 18}
        runner.destroy()

    def test_with_kwargs(self):
        """Test task submission with keyword arguments."""
        runner = AsyncTaskRunner[int](max_queue_size=10)
        runner.initialize()

        async def multiply(a: int, b: int) -> int:
            await asyncio.sleep(0.01)
            return a * b

        runner.submit(multiply, a=5, b=3, task_id=1)
        runner.submit(multiply, 4, b=7, task_id=2)

        results = runner.wait(count=2, timeout=2.0)

        assert len(results) == 2
        assert set(results) == {15, 28}
        runner.destroy()


class TestAsyncTaskRunnerPauseResume:
    """Test pause/resume functionality."""

    def test_pause_prevents_new_tasks(self):
        """Test that pause prevents new tasks from starting."""
        runner = AsyncTaskRunner[int](max_queue_size=50)
        runner.initialize()

        async def slow_task(x: int) -> int:
            await asyncio.sleep(0.2)
            return x

        # Submit some tasks
        for i in range(5):
            runner.submit(slow_task, i, task_id=i)

        # Pause immediately
        runner.pause()
        time.sleep(0.1)

        # Submit more tasks while paused
        for i in range(5, 10):
            runner.submit(slow_task, i, task_id=i)

        # Should not be able to get all results quickly because paused tasks don't start
        start = time.time()
        try:
            runner.wait(count=10, timeout=0.5)
            pytest.fail("Should have timed out")
        except TimeoutError:
            elapsed = time.time() - start
            assert elapsed >= 0.5

        runner.destroy()

    def test_resume_allows_tasks(self):
        """Test that resume allows paused tasks to start."""
        runner = AsyncTaskRunner[int](max_queue_size=50)
        runner.initialize()

        async def fast_task(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        # Pause before submitting
        runner.pause()

        # Submit tasks while paused
        for i in range(5):
            runner.submit(fast_task, i, task_id=i)

        # Resume
        runner.resume()

        # Now should be able to get results
        results = runner.wait(count=5, timeout=2.0)
        assert len(results) == 5
        assert set(results) == {0, 2, 4, 6, 8}

        runner.destroy()


class TestAsyncTaskRunnerTimeout:
    """Test timeout behavior."""

    def test_timeout_on_insufficient_results(self):
        """Test that wait() times out if not enough results are ready."""
        runner = AsyncTaskRunner[int](max_queue_size=10)
        runner.initialize()

        async def slow_task(x: int) -> int:
            await asyncio.sleep(5.0)  # Very slow task
            return x

        runner.submit(slow_task, 1, task_id=1)

        with pytest.raises(TimeoutError, match="Timed out waiting for 1 results"):
            runner.wait(count=1, timeout=0.5)

        runner.destroy()

    def test_partial_results_timeout(self):
        """Test timeout when only partial results are available."""
        runner = AsyncTaskRunner[int](max_queue_size=50)
        runner.initialize()
        time.sleep(0.1)  # Give thread time to start

        async def fast_task(x: int) -> int:
            await asyncio.sleep(0.01)
            return x

        async def slow_task(x: int) -> int:
            await asyncio.sleep(10.0)
            return x

        # Submit 3 fast and 2 slow
        for i in range(3):
            runner.submit(fast_task, i, task_id=i)
        for i in range(2):
            runner.submit(slow_task, i + 100, task_id=i + 100)

        # Should timeout waiting for all 5
        with pytest.raises(TimeoutError, match="only received 3"):
            runner.wait(count=5, timeout=1.0)

        runner.destroy()


class TestAsyncTaskRunnerConcurrency:
    """Test concurrent task execution."""

    def test_concurrent_execution(self):
        """Test that tasks execute concurrently, not sequentially."""
        # Use faster polling for this test
        runner = AsyncTaskRunner[float](
            max_queue_size=50,
            poll_wait_time=0.05,
            poll_sleep_time=0.01,  # Very fast polling for this test
        )
        runner.initialize()
        time.sleep(0.1)  # Give thread time to start

        async def timed_task(duration: float) -> float:
            start = time.time()
            await asyncio.sleep(duration)
            return time.time() - start

        # Submit 5 tasks that each take 0.2 seconds
        for i in range(5):
            runner.submit(timed_task, 0.2, task_id=i)

        start = time.time()
        results = runner.wait(count=5, timeout=2.0)
        total_time = time.time() - start

        # If concurrent, should take ~0.2s. If sequential, would take ~1.0s
        assert total_time < 0.8, f"Tasks run sequentially ({total_time}s)"
        assert len(results) == 5

        runner.destroy()

    def test_many_concurrent_tasks(self):
        """Test handling many concurrent tasks."""
        runner = AsyncTaskRunner[int](max_queue_size=200)
        runner.initialize()

        async def compute(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * x

        # Submit 100 tasks
        for i in range(100):
            runner.submit(compute, i, task_id=i)

        results = runner.wait(count=100, timeout=10.0)

        assert len(results) == 100
        assert set(results) == {i * i for i in range(100)}

        runner.destroy()


class TestAsyncTaskRunnerErrorHandling:
    """Test error handling and edge cases."""

    def test_queue_full_error(self):
        """Test that submitting to a full queue raises an error."""
        runner = AsyncTaskRunner[int](max_queue_size=2)
        runner.initialize()

        async def task(x: int) -> int:
            await asyncio.sleep(10.0)  # Long delay to keep tasks pending
            return x

        # Fill the queue
        runner.submit(task, 1, task_id=1)
        runner.submit(task, 2, task_id=2)

        # Next submission should fail
        with pytest.raises(RuntimeError, match="Input queue full"):
            runner.submit(task, 3, task_id=3)

        runner.destroy()

    def test_task_exception_handling(self):
        """Test that task exceptions are properly handled and don't stop the runner."""
        runner = AsyncTaskRunner[int](max_queue_size=10)
        runner.initialize()

        async def failing_task() -> int:
            await asyncio.sleep(0.01)
            raise ValueError("Task failed!")

        async def working_task() -> int:
            await asyncio.sleep(0.02)
            return 42

        # Submit a failing task and a working task
        runner.submit(failing_task, task_id=1)
        runner.submit(working_task, task_id=2)

        # The working task should still complete and its result should be retrievable.
        # The failing task will return None but won't crash the runner.
        results = runner.wait(count=1, timeout=2.0)
        assert results == [None]
        results = runner.wait(count=1, timeout=2.0)
        assert results == [42]

        # Check that the runner thread is still alive
        runner._check_thread_health()

        runner.destroy()

    def test_shutdown_with_pending_tasks(self):
        """Test clean shutdown with pending tasks."""
        runner = AsyncTaskRunner[int](max_queue_size=50)
        runner.initialize()

        async def long_task(x: int) -> int:
            await asyncio.sleep(10.0)
            return x

        # Submit tasks that won't complete
        for i in range(10):
            runner.submit(long_task, i, task_id=i)

        # Should shut down cleanly, cancelling pending tasks
        runner.destroy()


class TestAsyncTaskRunnerQueueSizes:
    """Test queue size tracking."""

    def test_get_queue_sizes(self):
        """Test that queue sizes can be retrieved."""
        runner = AsyncTaskRunner[int](max_queue_size=50)
        runner.initialize()

        async def task(x: int) -> int:
            await asyncio.sleep(0.1)
            return x

        # Initially, queues should be empty
        input_size, output_size = runner.get_queue_sizes()
        assert input_size == 0
        assert output_size == 0

        # Submit some tasks
        for i in range(5):
            runner.submit(task, i, task_id=i)

        time.sleep(0.05)
        input_size, output_size = runner.get_queue_sizes()
        # Some tasks should have been picked up from input queue
        assert input_size <= 5

        runner.destroy()


class TestAsyncTaskRunnerResultOrdering:
    """Test result ordering and caching."""

    def test_result_cache_accumulation(self):
        """Test that results accumulate in cache across multiple wait() calls."""
        runner = AsyncTaskRunner[int](max_queue_size=50)
        runner.initialize()

        async def fast_task(x: int) -> int:
            await asyncio.sleep(0.01)
            return x

        # Submit 10 tasks
        for i in range(10):
            runner.submit(fast_task, i, task_id=i)

        # Wait for first 3
        results1 = runner.wait(count=3, timeout=2.0)
        assert len(results1) == 3

        # Wait for next 4
        results2 = runner.wait(count=4, timeout=2.0)
        assert len(results2) == 4

        # Wait for remaining 3
        results3 = runner.wait(count=3, timeout=2.0)
        assert len(results3) == 3

        # All results combined should be the full set
        all_results = results1 + results2 + results3
        assert set(all_results) == set(range(10))

        runner.destroy()

    def test_results_are_shuffled(self):
        """Test that results are returned in random order."""
        runner = AsyncTaskRunner[int](max_queue_size=50)
        runner.initialize()

        async def task(x: int) -> int:
            await asyncio.sleep(0.01)
            return x

        # Submit tasks in order
        for i in range(20):
            runner.submit(task, i, task_id=i)

        results = runner.wait(count=20, timeout=5.0)

        # Results should not be in strict sequential order (extremely unlikely
        # with shuffle)
        # Check that at least some results are out of order
        assert results != list(range(20)), "Results should be shuffled"

        runner.destroy()


class TestAsyncTaskRunnerWithDifferentTypes:
    """Test AsyncTaskRunner with various result types."""

    def test_with_string_results(self):
        """Test with string return types."""
        runner = AsyncTaskRunner[str](max_queue_size=10)
        runner.initialize()

        async def concat(a: str, b: str) -> str:
            await asyncio.sleep(0.01)
            return a + b

        runner.submit(concat, "hello", " world", task_id=1)
        results = runner.wait(count=1, timeout=2.0)

        assert results[0] == "hello world"
        runner.destroy()

    def test_with_dict_results(self):
        """Test with dictionary return types."""
        runner = AsyncTaskRunner[dict](max_queue_size=10)
        runner.initialize()

        async def create_dict(key: str, value: int) -> dict:
            await asyncio.sleep(0.01)
            return {key: value}

        runner.submit(create_dict, "a", 1, task_id=1)
        runner.submit(create_dict, "b", 2, task_id=2)

        results = runner.wait(count=2, timeout=2.0)

        assert len(results) == 2
        assert {"a": 1} in results
        assert {"b": 2} in results

        runner.destroy()

    def test_with_none_results(self):
        """Test with None return type (e.g., for filtering)."""
        runner = AsyncTaskRunner[int | None](max_queue_size=20)
        runner.initialize()

        async def filter_even(x: int) -> int | None:
            await asyncio.sleep(0.01)
            return x if x % 2 == 0 else None

        # Submit 10 numbers
        for i in range(10):
            runner.submit(filter_even, i, task_id=i)

        # Get all results (including None values)
        results = runner.wait(count=10, timeout=2.0)

        assert len(results) == 10
        # Filter None values
        even_numbers = [x for x in results if x is not None]
        assert set(even_numbers) == {0, 2, 4, 6, 8}

        runner.destroy()
