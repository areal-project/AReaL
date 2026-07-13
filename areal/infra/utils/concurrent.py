# SPDX-License-Identifier: Apache-2.0

import asyncio
import atexit
import concurrent.futures
import inspect
import threading
import weakref
from functools import partial
from typing import Any

from areal.utils import logging

# Logger for event loop cleanup functionality
logger = logging.getLogger("ConcurrentUtils")

_executors: dict[str, concurrent.futures.ThreadPoolExecutor] = {}
_executors_lock = threading.Lock()
_atexit_registered = False


def get_executor(
    scope: str = "default", *, max_workers: int = 4
) -> concurrent.futures.ThreadPoolExecutor:
    """Get or create a named thread pool executor.

    Executors are lazily created per *scope* and cached for the process
    lifetime.  Using distinct scopes prevents deadlocks when tasks in one
    pool synchronously wait on tasks they submit to the same pool.

    Parameters
    ----------
    scope : str
        Logical namespace for the executor (e.g. ``"default"``,
        ``"ctrl_init"``).  Each scope gets its own independent pool.
    max_workers : int
        Thread count for a *newly created* pool.  Ignored if the pool for
        *scope* already exists.
    """
    global _atexit_registered

    executor = _executors.get(scope)
    if executor is not None:
        return executor

    with _executors_lock:
        executor = _executors.get(scope)
        if executor is not None:
            return executor

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=f"exec_{scope}"
        )
        _executors[scope] = executor

        if not _atexit_registered:
            atexit.register(_shutdown_all_executors)
            _atexit_registered = True

    return executor


def _shutdown_all_executors() -> None:
    with _executors_lock:
        for executor in _executors.values():
            executor.shutdown(wait=False)
        _executors.clear()


def run_async_task(func, *args, **kwargs):
    # Check if we're already in an async context
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # Already in async context, use shared executor to avoid overhead
        future = get_executor().submit(asyncio.run, func(*args, **kwargs))
        return future.result()
    else:
        # Not in async context, use asyncio.run directly
        return asyncio.run(func(*args, **kwargs))


def call_maybe_async(method, *args, **kwargs) -> None:
    """Call a callable and await it if it returns an awaitable.

    This helper lets sync call-sites invoke methods that may be implemented as
    either sync or async without branching at every call site.
    """
    result: Any = method(*args, **kwargs)
    if inspect.isawaitable(result):

        async def _wait_result() -> None:
            await result

        run_async_task(_wait_result)


# ============================================================================
# Event Loop Cleanup Utilities
# ============================================================================
# This section provides atexit-like functionality for asyncio event loops,
# allowing async resources to register cleanup callbacks that run when the
# loop closes. Based on the asyncio-atexit pattern.


# Global registry: event_loop -> _LoopCleanupEntry
# WeakKeyDictionary ensures loops can be garbage collected
_loop_cleanup_registry: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class _LoopCleanupEntry:
    """Registry entry for event loop cleanup callbacks.

    Stores a list of cleanup callbacks for an event loop.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        # Save original close method (avoid double-patching)
        if not hasattr(loop, "_cleanup_orig_close"):
            loop._cleanup_orig_close = loop.close

        self.callbacks: list = []


def register_loop_cleanup(callback, *, loop=None):
    """Register a callback to run when the event loop closes.

    Like atexit.register, but runs when the asyncio loop closes.
    Allows coroutines to cleanup their resources (e.g., closing
    aiohttp sessions, httpx clients, database connections).

    Callbacks are executed in LIFO order (like atexit and context
    managers), meaning the last registered callback runs first.

    Parameters
    ----------
    callback : callable
        A callback (can be sync or async) that takes no arguments.
        To pass arguments, use functools.partial.
    loop : asyncio.AbstractEventLoop, optional
        The loop to attach to. If None, uses current running loop.

    Examples
    --------
    >>> async def cleanup_session():
    ...     await session.close()
    >>> register_loop_cleanup(cleanup_session)

    >>> # With functools.partial for arguments
    >>> from functools import partial
    >>> register_loop_cleanup(partial(close_connection, conn_id=123))
    """
    entry = _get_loop_entry(loop)
    entry.callbacks.append(callback)


def unregister_loop_cleanup(callback, *, loop=None):
    """Unregister a callback registered with register_loop_cleanup.

    Removes all instances of the callback from the registry for the
    specified event loop.

    Parameters
    ----------
    callback : callable
        The callback to remove.
    loop : asyncio.AbstractEventLoop, optional
        The loop to remove from. If None, uses current running loop.
    """
    entry = _get_loop_entry(loop)
    while True:
        try:
            entry.callbacks.remove(callback)
        except ValueError:
            break


def _get_loop_entry(loop=None):
    """Get or create the registry entry for an event loop.

    Parameters
    ----------
    loop : asyncio.AbstractEventLoop, optional
        The loop to get entry for. If None, uses current running loop.

    Returns
    -------
    _LoopCleanupEntry
        The registry entry for the loop.
    """
    if loop is None:
        loop = asyncio.get_running_loop()
    _register_loop(loop)
    return _loop_cleanup_registry[loop]


def _register_loop(loop):
    """Patch an event loop to support cleanup callbacks.

    Patches the loop's close() method if not already patched to run
    cleanup callbacks before actually closing the loop.

    Parameters
    ----------
    loop : asyncio.AbstractEventLoop
        The loop to patch.
    """
    if loop in _loop_cleanup_registry:
        return

    _loop_cleanup_registry[loop] = _LoopCleanupEntry(loop)
    loop.close = partial(_patched_loop_close, loop)


async def _run_cleanup_callbacks(loop, callbacks):
    """Run event loop cleanup callbacks.

    Executes in loop.close() via run_until_complete before closing.
    Callbacks run in LIFO order (like atexit and context managers).

    Parameters
    ----------
    loop : asyncio.AbstractEventLoop
        The event loop being closed.
    callbacks : list
        List of callbacks to execute.
    """
    # Reverse for LIFO execution
    for callback in reversed(callbacks):
        try:
            result = callback()
            if inspect.isawaitable(result):
                await result
        except Exception as e:
            logger.error(
                f"Unhandled exception in loop cleanup callback {callback}: {e}",
                exc_info=True,
            )


def _patched_loop_close(loop):
    """Patched EventLoop.close to run cleanup callbacks first.

    This is the core of the asyncio-atexit pattern. Runs all registered
    cleanup callbacks before calling the original close method.

    Parameters
    ----------
    loop : asyncio.AbstractEventLoop
        The event loop being closed.
    """
    entry = _get_loop_entry(loop)
    if entry.callbacks and not loop.is_closed():
        loop.run_until_complete(_run_cleanup_callbacks(loop, entry.callbacks))
    entry.callbacks[:] = []  # Clear callbacks
    return loop._cleanup_orig_close()
