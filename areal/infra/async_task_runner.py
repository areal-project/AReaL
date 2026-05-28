# SPDX-License-Identifier: Apache-2.0

"""Generic asynchronous task runner for executing concurrent async Python functions.

This module provides a reusable, thread-based async task executor that can run
any async Python functions concurrently with queue management, pause/resume control,
and health monitoring. It has no dependencies on AReaL-specific logic.

The AsyncTaskRunner manages a background thread running an asyncio event loop (uvloop)
that processes tasks from an input queue and places results in an output queue.
"""

import asyncio
import queue
import threading
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

import uvloop

# Type variable for generic result types
T = TypeVar("T")

# Polling configuration
DEFAULT_POLL_WAIT_TIME = 0.05  # 50ms
DEFAULT_POLL_SLEEP_TIME = 0.5  # 500ms


class TaskQueueFullError(RuntimeError):
    """Raised when an AsyncTaskRunner queue is full."""


@dataclass
class TimedResult(Generic[T]):
    """Wrapper for task results with creation timestamp.

    Attributes
    ----------
    create_time : int
        Task creation time in nanoseconds from time.monotonic_ns().
    data : T
        The actual result data from the completed task.
    task_id : int
        The task ID associated with this result.
    """

    create_time: int
    data: T
    task_id: int


@dataclass
class _TaskInput(Generic[T]):
    """Internal wrapper for task input with async function and arguments."""

    async_fn: Callable[..., Awaitable[T]]
    args: tuple
    kwargs: dict
    task_id: int


@dataclass
class _Task(Generic[T]):
    """Internal wrapper for running task with metadata."""

    create_time: int  # nanoseconds from time.monotonic_ns()
    task: asyncio.Task
    task_input: _TaskInput[T]


class AsyncTaskRunner(Generic[T]):
    """Generic asynchronous task runner with queue management and pause/resume control.

    This class provides a reusable async task executor that runs a background thread
    with an asyncio event loop (using uvloop for performance). It can execute any
    async Python function concurrently with configurable queue sizes and optional
    pause/resume control.

    The runner maintains thread-safe input and output queues, manages task lifecycle,
    and provides health monitoring to detect thread failures.

    Parameters
    ----------
    max_queue_size : int
        Maximum size for input and output queues. Tasks submitted when
        the input queue is full will raise TaskQueueFullError.
    poll_wait_time : float, optional
        Time in seconds to wait for task completion during each poll
        cycle. Default is 0.05 (50ms).
    poll_sleep_time : float, optional
        Time in seconds to sleep between poll cycles.
        Default is 0.5 seconds.
    enable_tracing : bool, optional
        Enable detailed logging of task submission and completion.
        Default is False.

    Attributes
    ----------
    input_queue : queue.Queue
        Thread-safe queue for incoming task submissions.
    output_queue : queue.Queue
        Thread-safe queue for completed task results.
    exiting : threading.Event
        Signal to request thread shutdown.
    paused : threading.Event
        Signal to pause new task creation (existing tasks continue).

    Examples
    --------
    Basic usage with simple async functions:

    >>> import asyncio
    >>> runner = AsyncTaskRunner[int](max_queue_size=100)
    >>> runner.initialize()
    >>>
    >>> async def compute(x: int) -> int:
    ...     await asyncio.sleep(0.1)
    ...     return x * 2
    >>>
    >>> # Submit tasks
    >>> for i in range(5):
    ...     runner.submit(compute, i)
    >>>
    >>> # Wait for results
    >>> results = runner.wait(count=5)
    >>> print(results)  # [0, 2, 4, 6, 8] (order may vary)
    >>>
    >>> runner.destroy()

    Using pause/resume for control:

    >>> runner = AsyncTaskRunner[str](max_queue_size=50)
    >>> runner.initialize()
    >>>
    >>> async def fetch_data(url: str) -> str:
    ...     # Simulate network request
    ...     await asyncio.sleep(0.5)
    ...     return f"Data from {url}"
    >>>
    >>> # Submit some tasks
    >>> for i in range(10):
    ...     runner.submit(fetch_data, f"http://example.com/{i}")
    >>>
    >>> # Pause to prevent new tasks from starting
    >>> runner.pause()
    >>>
    >>> # Wait for currently running tasks
    >>> results = runner.wait(count=5, timeout=2.0)
    >>>
    >>> # Resume and submit more
    >>> runner.resume()
    >>> runner.destroy()

    See Also
    --------
    WorkflowExecutor : AReaL-specific wrapper that adds staleness management
    """

    def __init__(
        self,
        max_queue_size: int,
        poll_wait_time: float = DEFAULT_POLL_WAIT_TIME,
        poll_sleep_time: float = DEFAULT_POLL_SLEEP_TIME,
        enable_tracing: bool = False,
    ):
        """Initialize the AsyncTaskRunner.

        Parameters
        ----------
        max_queue_size : int
            Maximum size for input and output queues.
        poll_wait_time : float, optional
            Time in seconds to wait for task completion during polling.
            Default is 0.05.
        poll_sleep_time : float, optional
            Time in seconds to sleep between poll cycles.
            Default is 0.5.
        enable_tracing : bool, optional
            Enable detailed logging. Default is False.
        """
        self.max_queue_size = max_queue_size
        self.poll_wait_time = poll_wait_time
        self.poll_sleep_time = poll_sleep_time
        self.enable_tracing = enable_tracing

        # Thread control
        self.exiting = threading.Event()
        self.paused = threading.Event()

        # Queues for task management
        self.input_queue: queue.Queue[_TaskInput[T]] = queue.Queue(
            maxsize=max_queue_size
        )
        self.output_queue: queue.Queue[TimedResult[T]] = queue.Queue(
            maxsize=max_queue_size
        )

        # Async loop coordination
        self._loop_ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._input_event: asyncio.Event | None = None

        # Thread exception handling
        self._thread_exception_lock = threading.Lock()
        self._thread_exception: Exception | None = None

        # Task ID tracking for duplicate detection
        self._active_task_ids: set[int] = set()
        self._active_task_ids_lock = threading.Lock()

        # Will be set in initialize()
        self.logger = None
        self.thread: threading.Thread | None = None

    def initialize(self, logger=None):
        """Initialize and start the background thread.

        This method starts the background thread that runs the asyncio
        event loop. Must be called before submitting any tasks.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance for debugging and tracing.
            If None, logging is minimal.
        """
        self.logger = logger

        # Start the background thread (daemon=True for automatic cleanup)
        self.exiting.clear()
        # Always start in resumed state; previous pause() should not leak
        self.paused.clear()

        # Reset the readiness event before spinning up the new worker thread
        self._loop_ready.clear()
        self.thread = threading.Thread(target=self._run_thread, daemon=True)
        self.thread.start()
        self._loop_ready.wait()

    def destroy(self, timeout: float = 30.0):
        """Shutdown the task runner and wait for thread cleanup.

        This method signals the background thread to exit and waits for
        it to complete. All pending tasks will be cancelled.

        Parameters
        ----------
        timeout : float, optional
            Maximum time in seconds to wait for thread to exit.
            Default is 30.0 seconds.
        """
        self.exiting.set()
        self.paused.clear()

        self._signal_new_input()
        if self.thread is not None:
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                if self.logger:
                    self.logger.warning(
                        f"Background thread did not exit within {timeout}s timeout."
                    )

    def _check_thread_health(self):
        """Check if the background thread has encountered a fatal error.

        Raises
        ------
        RuntimeError
            If the background thread has died due to an exception.
        """
        with self._thread_exception_lock:
            if self._thread_exception is not None:
                raise RuntimeError(
                    "AsyncTaskRunner thread has died due to an exception. "
                    "No further tasks can be processed."
                ) from self._thread_exception

    def _run_thread(self):
        """Entry point for the background thread.

        Runs the async event loop and handles exceptions.
        """
        try:
            loop = uvloop.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._loop_ready.set()
            loop.run_until_complete(self._run_async_loop())
        except Exception as e:
            # Store exception for thread-safe access
            with self._thread_exception_lock:
                self._thread_exception = e
            if self.logger:
                self.logger.error(
                    f"AsyncTaskRunner thread failed with exception: {e}",
                    exc_info=True,
                )
        finally:
            # Signal shutdown regardless of success/failure so other threads do not hang.
            self.exiting.set()
            self._loop_ready.set()
            if self._loop is not None:
                loop = self._loop
                self._loop = None
                loop.close()

    async def _run_async_loop(self):
        """Main async event loop that processes tasks.

        This loop:
        1. Pulls tasks from input_queue when not paused
        2. Creates asyncio.Task instances for each
        3. Waits for task completion
        4. Places results in output_queue
        5. Continues until exiting signal is set
        """
        self._input_event = asyncio.Event()
        self._input_event.set()

        running_tasks: dict[str, _Task[T]] = {}

        try:
            while not self.exiting.is_set():
                if self.paused.is_set():
                    await asyncio.sleep(self.poll_sleep_time)
                    continue

                # running_tasks is mutated in-place as we enqueue freshly created asyncio tasks
                self._drain_pending_inputs(running_tasks)

                if not running_tasks:
                    await self._wait_for_new_tasks()
                    continue

                tasks = [t.task for t in running_tasks.values()]
                done, _ = await asyncio.wait(
                    tasks,
                    timeout=self.poll_wait_time,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if not done:
                    continue

                # Process completed tasks
                for async_task in done:
                    tid = async_task.get_name()
                    task_obj = running_tasks.pop(tid)
                    try:
                        result = await async_task
                    except asyncio.CancelledError:
                        if self.logger:
                            self.logger.warning(
                                f"Task {tid} was cancelled. None will be returned"
                            )
                        result = None
                    except Exception as e:
                        if self.logger:
                            self.logger.error(
                                f"AsyncTaskRunner: Task {tid} "
                                f"failed with exception: {e} ",
                                exc_info=True,
                            )
                        result = None

                    try:
                        # Place result in output queue
                        timed_result: TimedResult[T] = TimedResult(
                            create_time=task_obj.create_time,
                            data=cast(T, result),
                            task_id=task_obj.task_input.task_id,
                        )
                        self.output_queue.put_nowait(timed_result)

                        # Remove task_id from active set now that task is complete
                        with self._active_task_ids_lock:
                            self._active_task_ids.discard(task_obj.task_input.task_id)

                        if self.enable_tracing and self.logger:
                            self.logger.info(
                                f"AsyncTaskRunner: Completed task {tid}. "
                                f"Running: {len(running_tasks)}"
                            )
                    except queue.Full:
                        # This is a critical error that should stop the runner.
                        # Re-add task so it can be cancelled in finally.
                        running_tasks[tid] = task_obj
                        if self.logger:
                            self.logger.critical(
                                f"Output queue is full. Task ID: {tid}. "
                                f"Please increase max_queue_size.",
                                exc_info=True,
                            )
                        raise TaskQueueFullError(
                            "Output queue full. Please increase max_queue_size."
                        )
        finally:
            self._input_event = None
            # Cancel all remaining tasks on shutdown
            pending_tasks = [
                task_obj.task
                for task_obj in running_tasks.values()
                if not task_obj.task.done()
            ]
            if pending_tasks:
                for task in pending_tasks:
                    task.cancel()
                await asyncio.gather(*pending_tasks, return_exceptions=True)

            # Clean up all remaining active task IDs
            with self._active_task_ids_lock:
                self._active_task_ids.clear()

    def _drain_pending_inputs(
        self,
        running_tasks: dict[str, _Task[T]],
    ) -> int:
        tasks_added = 0
        while not self.paused.is_set():
            try:
                task_input = self.input_queue.get_nowait()
            except queue.Empty:
                break

            tid = str(task_input.task_id)

            # Note: Duplicate checking is now done in submit() method
            # This check here is defensive in case of threading issues
            if tid in running_tasks:
                raise ValueError(
                    f"Duplicate task_id: {task_input.task_id}. "
                    f"Task with this ID is already running."
                )

            coroutine: Coroutine[Any, Any, T] = cast(
                Coroutine[Any, Any, T],
                task_input.async_fn(*task_input.args, **task_input.kwargs),
            )
            async_task = asyncio.create_task(coroutine, name=tid)
            running_tasks[tid] = _Task(
                create_time=time.monotonic_ns(),
                task=async_task,
                task_input=task_input,
            )
            if self.enable_tracing and self.logger:
                self.logger.info(
                    f"AsyncTaskRunner: Submitted task {tid}. "
                    f"Running: {len(running_tasks)}"
                )
            tasks_added += 1

        return tasks_added

    async def _wait_for_new_tasks(self) -> None:
        if self._input_event is None:
            await asyncio.sleep(self.poll_sleep_time)
            return

        while not self.exiting.is_set() and not self.paused.is_set():
            # This double-check of the queue size around clearing the event is crucial
            # to prevent a race condition. The race occurs if a producer adds an item
            # and sets the event *after* this thread checks the queue but *before*
            # it clears the event. Without the second check, this thread would clear
            # the event and then wait, potentially missing the wakeup signal.
            if self.input_queue.qsize() > 0:
                return
            self._input_event.clear()
            if self.input_queue.qsize() > 0 or self.exiting.is_set():
                return
            await self._input_event.wait()

    def submit(
        self,
        async_fn: Callable[..., Awaitable[T]],
        *args,
        task_id: int,
        **kwargs,
    ) -> int:
        """Submit an async function for execution.

        The function will be executed in the background thread's event
        loop. Results can be retrieved using wait().

        Parameters
        ----------
        async_fn : Callable[..., Awaitable[T]]
            The async function to execute.
        *args
            Positional arguments to pass to the function.
        task_id : int
            Task ID for tracking. Must be unique among currently running tasks.
        **kwargs
            Keyword arguments to pass to the function.

        Returns
        -------
        int
            The task_id that was provided.

        Raises
        ------
        TaskQueueFullError
            If the input queue is full.
        RuntimeError
            If the background thread has died.
        ValueError
            If task_id is a duplicate of an existing running task.

        Examples
        --------
        >>> async def add(a: int, b: int) -> int:
        ...     return a + b
        >>>
        >>> runner.submit(add, 5, 10, task_id=1)
        1
        >>> runner.submit(add, a=3, b=7, task_id=2)
        2
        """
        # Check if thread is still alive
        self._check_thread_health()

        # Check for duplicate task_id and add to active set
        with self._active_task_ids_lock:
            if task_id in self._active_task_ids:
                raise ValueError(
                    f"Duplicate task_id: {task_id}. "
                    f"Task with this ID is already submitted or running."
                )
            self._active_task_ids.add(task_id)

        # Create task input wrapper
        task_input = _TaskInput(
            async_fn=async_fn, args=args, kwargs=kwargs, task_id=task_id
        )

        # Submit to queue
        try:
            self.input_queue.put_nowait(task_input)
        except queue.Full:
            # Remove from active set if queue is full
            with self._active_task_ids_lock:
                self._active_task_ids.discard(task_id)
            raise TaskQueueFullError(
                "Input queue full. Please increase max_queue_size or "
                "wait for tasks to complete."
            )

        self._signal_new_input()
        return task_id

    def wait(
        self, count: int, timeout: float | None = None, with_timing: bool = False
    ) -> list[TimedResult[T]] | list[T]:
        """Wait for a specified number of task results.

        This method blocks until at least `count` results are available
        or the timeout expires.

        Parameters
        ----------
        count : int
            Number of results to wait for.
        timeout : float | None, optional
            Maximum time in seconds to wait. If None, waits indefinitely
            (up to 7 days). Default is None.
        with_timing : bool, optional
            If True, return TimedResult objects with creation timestamps.
            If False, return only the data values. Default is False.

        Returns
        -------
        list[TimedResult[T]] | list[T]
            If with_timing=True, returns list of TimedResult objects.
            If with_timing=False, returns list of result data.

        Raises
        ------
        TimeoutError
            If timeout expires before `count` results are available.
        RuntimeError
            If the background thread exits before results are ready.

        Examples
        --------
        >>> runner.submit(compute, 1)
        >>> runner.submit(compute, 2)
        >>> runner.submit(compute, 3)
        >>> results = runner.wait(count=3, timeout=10.0)
        >>> len(results)
        3
        """
        start_time = time.perf_counter()
        if timeout is None:
            timeout = float(7 * 24 * 3600)  # 7 days default

        deadline = start_time + timeout
        results_to_return: list[TimedResult[T]] = []

        while len(results_to_return) < count:
            self._check_thread_health()

            if self.exiting.is_set():
                raise RuntimeError(
                    "AsyncTaskRunner is exiting, cannot wait for results."
                )

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for {count} results, only received {len(results_to_return)}."
                )

            try:
                wait_time = min(self.poll_sleep_time, remaining)
                result = self.output_queue.get(timeout=wait_time)
            except queue.Empty:
                continue

            results_to_return.append(result)

        if with_timing:
            return results_to_return
        return [r.data for r in results_to_return]

    def pause(self):
        """Pause submission of new tasks.

        After calling pause(), no new tasks will be started from the
        input queue, but existing running tasks will continue to
        completion.
        """
        self.paused.set()

    def resume(self):
        """Resume submission of new tasks.

        Allows new tasks to be pulled from the input queue and
        started.
        """
        self.paused.clear()
        self._signal_new_input()

    def get_queue_sizes(self) -> tuple[int, int]:
        """Get current sizes of input and output queues.

        Returns
        -------
        tuple[int, int]
            (input_queue_size, output_queue_size)
        """
        return self.input_queue.qsize(), self.output_queue.qsize()

    def get_input_queue_size(self) -> int:
        """Get current size of the input queue.

        Returns
        -------
        int
            Number of tasks waiting in the input queue.
        """
        return self.input_queue.qsize()

    def get_output_queue_size(self) -> int:
        """Get current size of the output queue.

        Returns
        -------
        int
            Number of completed results waiting in the output queue.
        """
        return self.output_queue.qsize()

    def _signal_new_input(self):
        loop = self._loop
        input_event = self._input_event
        if loop is None or input_event is None:
            return

        try:
            loop.call_soon_threadsafe(input_event.set)
        except RuntimeError:
            pass
