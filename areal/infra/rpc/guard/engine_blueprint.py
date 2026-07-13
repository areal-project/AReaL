# SPDX-License-Identifier: Apache-2.0

"""Engine Blueprint: engine lifecycle and method invocation.

Provides a Flask Blueprint that manages engine threads, engine creation,
and engine method calls.  Registers hooks on :class:`GuardState` for
``/configure``, ``/health``, and cleanup.

Routes:

- ``POST /set_env``       — set environment variables in the engine thread
- ``POST /create_engine`` — instantiate a TrainEngine or InferenceEngine
- ``POST /call``          — invoke a method on a named engine instance

The engine thread guarantees serial execution of all engine operations,
which is required for NCCL compatibility.
"""

from __future__ import annotations

import os
import traceback
from collections.abc import Callable
from concurrent.futures import Future
from queue import Queue
from threading import Lock, Thread
from typing import Annotated, Any

from flask import Blueprint, jsonify, request
from pydantic import BaseModel, StringConstraints, ValidationError

from areal.api import InferenceEngine, TrainEngine
from areal.infra.platforms import current_platform
from areal.infra.rpc.guard.app import GuardState, get_state
from areal.infra.rpc.rtensor import RTensor
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.utils import logging, perf_tracer, seeding
from areal.utils.data import broadcast_tensor_container, tensor_container_to
from areal.utils.dynamic_import import import_from_string

logger = logging.getLogger("EngineBP")


# ======================================================================================
# Pydantic Models for Engine API
# ======================================================================================

# Define the constraint
NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]


class SetEnvRequest(BaseModel):
    env: dict[str, Any]


class CreateEngineRequest(BaseModel):
    engine: NonEmptyStr
    engine_name: NonEmptyStr
    init_args: list[Any] = []
    init_kwargs: dict[str, Any] = {}


class CallEngineRequest(BaseModel):
    method: NonEmptyStr
    engine_name: NonEmptyStr
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    rpc_meta: dict[str, Any] | None = None


# =======================================================================================


def _should_broadcast_payload(
    engine: TrainEngine | InferenceEngine, rpc_meta: dict[str, Any] | None
) -> bool:
    default_broadcast = isinstance(engine, TrainEngine) and engine.initialized
    if rpc_meta is None:
        return default_broadcast
    if not isinstance(rpc_meta, dict):
        raise ValueError(
            f"Invalid rpc_meta: expected dict or None, got {type(rpc_meta)}"
        )
    broadcast = rpc_meta.get("broadcast", default_broadcast)
    if not isinstance(broadcast, bool):
        raise ValueError(
            f"Invalid rpc_meta.broadcast: expected bool, got {type(broadcast)}"
        )
    return broadcast


engine_bp = Blueprint("engine", __name__)

# ---------------------------------------------------------------------------
# Engine-specific module-level state
# ---------------------------------------------------------------------------

# Global engine instances — keyed by engine_name (e.g., "actor/0", "ref/0")
_engines: dict[str, TrainEngine | InferenceEngine] = {}

# Engine thread for executing all engine-related operations serially.
# This ensures NCCL compatibility by running engine operations in a single
# thread, while allowing /data/ endpoints to be processed concurrently.
_engine_thread: Thread | None = None
_engine_work_queue: Queue | None = None
_engine_thread_lock = Lock()


# ---------------------------------------------------------------------------
# Engine thread management
# ---------------------------------------------------------------------------


def _init_engine_thread() -> None:
    """Lazily initialize the engine worker thread."""
    global _engine_thread, _engine_work_queue

    with _engine_thread_lock:
        if _engine_thread is not None:
            if _engine_thread.is_alive():
                return  # Already initialized
            else:
                raise RuntimeError("Engine thread is dead.")

        _engine_work_queue = Queue()

        def engine_worker():
            logger.info("Engine thread started")
            while True:
                try:
                    work_item = _engine_work_queue.get()
                    if work_item is None:  # Shutdown signal
                        logger.info("Engine thread shutting down")
                        break

                    func, args, kwargs, future, func_name = work_item
                    try:
                        result = func(*args, **kwargs)
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)
                    finally:
                        _engine_work_queue.task_done()
                except Exception as e:
                    logger.error(
                        f"Error in engine thread when "
                        f"running {func_name}: {e}\n{traceback.format_exc()}"
                    )
                    if work_item and len(work_item) > 3:
                        work_item[3].set_exception(e)

        _engine_thread = Thread(target=engine_worker, daemon=True, name="EngineWorker")
        _engine_thread.start()
        logger.info("Engine thread initialized")


def _submit_to_engine_thread(
    func_name: str, func: Callable, *args: Any, **kwargs: Any
) -> Any:
    """Submit work to the engine thread and block until result is available."""
    global _engine_work_queue

    _init_engine_thread()

    future: Future = Future()
    _engine_work_queue.put((func, args, kwargs, future, func_name))
    return future.result()  # Block until result is available


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------


def register_engine_hooks(state: GuardState) -> None:
    """Register engine-specific hooks on the :class:`GuardState`.

    Must be called after creating the Flask app and before starting
    the server.  Registers:

    - health hook → adds ``engine_count`` and ``engines`` to /health
    - configure hook → sets random seeds in the engine thread
    - cleanup hooks → destroy engines and shut down engine thread
    """
    state.register_health_hook(_engine_health_hook)
    state.register_configure_hook(_engine_configure_hook)
    state.register_cleanup_hook(cleanup_engine_thread)
    state.register_cleanup_hook(cleanup_engines)


def _engine_health_hook() -> dict[str, Any]:
    """Contribute engine info to the /health response."""
    return {"engine_count": len(_engines), "engines": list(_engines.keys())}


def _engine_configure_hook(data: dict) -> dict:
    """Handle /configure by setting random seeds in the engine thread.

    Raises
    ------
    ValueError
        If required fields (``config``, ``rank``) are missing.
    """
    config_data = data.get("config")
    if config_data is None:
        raise ValueError("Missing 'config' field in request")

    rank = data.get("rank")
    if rank is None:
        raise ValueError("Missing 'rank' field in request")

    config = deserialize_value(config_data)

    # Capture role from GuardState (we're in a request context)
    state = get_state()
    role = state.role

    def execute_configure():
        seeding.set_random_seed(config.seed, key=f"{role}{rank}")
        return {
            "status": "success",
            "message": "Worker configured successful.",
            "result": None,
        }

    return _submit_to_engine_thread("configure", execute_configure)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup_engines() -> None:
    """Destroy all engine instances."""
    global _engines
    if _engines:
        for engine_name, engine in list(_engines.items()):
            try:
                engine.destroy()
                logger.info(f"Engine '{engine_name}' destroyed successfully")
            except Exception as e:
                logger.error(f"Error destroying engine '{engine_name}': {e}")
        _engines.clear()


def cleanup_engine_thread() -> None:
    """Shut down the engine worker thread."""
    global _engine_thread, _engine_work_queue

    with _engine_thread_lock:
        if _engine_work_queue is not None:
            # Send shutdown signal
            _engine_work_queue.put(None)
            _engine_work_queue = None

        if _engine_thread is not None:
            _engine_thread.join(timeout=5.0)
            if _engine_thread.is_alive():
                logger.warning("Engine thread did not shut down gracefully")
            _engine_thread = None
            logger.info("Engine thread cleaned up")


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------


@engine_bp.route("/set_env", methods=["POST"])
def set_env():
    """Set environment variables for the worker process.

    This endpoint is routed to the engine thread for serial execution.
    """
    try:
        raw_data = request.get_json(silent=True) or {}

        # USE PYDANTIC MODEL FOR VALIDATION
        try:
            payload = SetEnvRequest(**raw_data)
        except ValidationError as e:
            return jsonify({"error": str(e)}), 400

        env_payload = payload.env

        def execute_set_env():
            for key, value in env_payload.items():
                os.environ[key] = str(value)
                logger.info(f"Set {key}={value}")
            return {"status": "success"}

        result = _submit_to_engine_thread("set_env", execute_set_env)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Unexpected error in set_env: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@engine_bp.route("/create_engine", methods=["POST"])
def create_engine():
    """Create and initialize an engine instance on this worker.

    This endpoint is routed to the engine thread for serial execution.
    Supports multiple engines per worker, keyed by ``engine_name``.

    Expected JSON payload::

        {
            "engine": "areal.engine.fsdp_engine.FSDPPPOActor",
            "engine_name": "actor/0",
            "init_args": [...],
            "init_kwargs": {"config": ...}
        }
    """
    global _engines

    try:
        raw_data = request.get_json(silent=True) or {}

        # USE PYDANTIC MODEL FOR VALIDATION
        try:
            payload = CreateEngineRequest(**raw_data)
        except ValidationError as e:
            return jsonify({"error": str(e)}), 400

        engine = payload.engine
        engine_name = payload.engine_name

        # Deserialize init_args and init_kwargs
        init_args = deserialize_value(payload.init_args)
        init_kwargs = deserialize_value(payload.init_kwargs)

        if engine_name in _engines:
            return (
                jsonify(
                    {
                        "error": f"Engine '{engine_name}' already exists. "
                        "Use a different name or delete the existing "
                        "engine first."
                    }
                ),
                400,
            )

        # Dynamic import (can be done in main thread)
        try:
            engine_class = import_from_string(engine)

            # Validate that the class is a TrainEngine or InferenceEngine
            if not issubclass(engine_class, TrainEngine) and not issubclass(
                engine_class, InferenceEngine
            ):
                raise TypeError(
                    "Engine class must be a subclass of TrainEngine or "
                    f"InferenceEngine, got {engine_class}.."
                )
        except (ValueError, ImportError, AttributeError) as e:
            logger.error(f"Failed to import engine '{engine}': {e}")
            return (
                jsonify({"error": (f"Failed to import engine '{engine}': {str(e)}")}),
                400,
            )
        except TypeError as e:
            logger.error(f"Invalid engine type: {e}")
            return jsonify({"error": str(e)}), 400

        # Instantiate engine in engine thread (may involve NCCL init)
        def create_engine_in_engine_thread():
            """Create engine in engine thread."""
            try:
                engine_obj = engine_class(*init_args, **init_kwargs)
                logger.info(
                    f"Engine '{engine_name}' (class: {engine}) "
                    "instantiated successfully"
                )
                return engine_obj
            except Exception as e:
                logger.error(
                    f"Failed to instantiate engine: {e}\n{traceback.format_exc()}"
                )
                raise

        try:
            engine_obj = _submit_to_engine_thread(
                "create_engine", create_engine_in_engine_thread
            )
            _engines[engine_name] = engine_obj
            return jsonify(
                {
                    "status": "success",
                    "message": (f"Engine '{engine_name}' created and initialized"),
                    "engine_name": engine_name,
                    "result": None,
                }
            )
        except Exception as e:
            return (
                jsonify({"error": f"Failed to instantiate engine: {str(e)}"}),
                500,
            )

    except Exception as e:
        logger.error(
            f"Unexpected error in create_engine: {e}\n{traceback.format_exc()}"
        )
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@engine_bp.route("/call", methods=["POST"])
def call_engine_method():
    """Call a method on an engine instance.

    This endpoint is routed to the engine thread to ensure all engine
    operations run serially in the same thread, preventing NCCL conflicts.

    Expected JSON payload::

        {
            "method": "train_batch",
            "engine_name": "actor/0",
            "args": [...],
            "kwargs": {...}
        }
    """
    global _engines

    try:
        raw_data = request.get_json(silent=True) or {}

        # USE PYDANTIC MODEL FOR VALIDATION
        try:
            payload = CallEngineRequest(**raw_data)
        except ValidationError as e:
            return jsonify({"error": str(e)}), 400

        method_name = payload.method
        engine_name = payload.engine_name
        raw_args = payload.args
        raw_kwargs = payload.kwargs
        rpc_meta = payload.rpc_meta

        if engine_name not in _engines:
            return (
                jsonify(
                    {
                        "error": f"Engine '{engine_name}' not found. "
                        f"Available engines: {list(_engines.keys())}"
                    }
                ),
                404,
            )

        # Get the specific engine to call
        engine = _engines[engine_name]

        # Deserialize data
        raw_args = deserialize_value(raw_args)
        raw_kwargs = deserialize_value(raw_kwargs)
        # Fetch remote tensors
        args = RTensor.localize(raw_args)
        kwargs = RTensor.localize(raw_kwargs)

        def execute_in_engine_thread():
            try:
                args_bcast = args
                kwargs_bcast = kwargs
                should_broadcast = _should_broadcast_payload(
                    engine=engine, rpc_meta=rpc_meta
                )
                if should_broadcast:
                    logger.debug(f"Broadcasting RPC payload for method: {method_name}")
                    args_bcast = tensor_container_to(
                        args, current_platform.current_device()
                    )
                    args_bcast = broadcast_tensor_container(
                        args_bcast,
                        src_rank=engine.current_data_parallel_head(),
                        group=engine.context_and_model_parallel_group,
                    )
                    kwargs_bcast = tensor_container_to(
                        kwargs, current_platform.current_device()
                    )
                    kwargs_bcast = broadcast_tensor_container(
                        kwargs_bcast,
                        src_rank=engine.current_data_parallel_head(),
                        group=engine.context_and_model_parallel_group,
                    )
                    logger.debug("Broadcasting RPC payload done.")

                logger.debug(f"Calling engine '{engine_name}' method: {method_name}")

                # Re-establish current device in RPC execution context before
                # calling engine methods that may issue object collectives.
                if (
                    isinstance(engine, TrainEngine)
                    and engine.initialized
                    and current_platform.device_type != "cpu"
                ):
                    current_platform.set_device(current_platform.current_device())

                # Determine trace category based on method name
                category = "misc"  # Default category
                method_lower = method_name.lower()
                if any(keyword in method_lower for keyword in ["submit", "wait"]):
                    category = "scheduler"
                elif any(
                    keyword in method_lower
                    for keyword in ["update_weights", "broadcast"]
                ):
                    category = "comm"
                elif any(keyword in method_lower for keyword in ["save", "load"]):
                    category = "io"
                elif any(
                    keyword in method_lower
                    for keyword in [
                        "train",
                        "eval",
                        "forward",
                        "compute",
                        "step",
                        "update",
                        "optimizer",
                        "zero_grad",
                        "lr_scheduler",
                    ]
                ):
                    category = "compute"

                # Wrap engine method call with perf_tracer
                with perf_tracer.trace_scope(
                    f"rpc.{method_name}",
                    category=category,
                    args={"method": method_name, "engine": engine_name},
                ):
                    method = getattr(engine, method_name)
                    result = method(*args_bcast, **kwargs_bcast)

                    # Handle update weights future
                    if isinstance(result, Future):
                        logger.debug("Waiting for update weights future")
                        result = result.result()
                        logger.debug("Update weights future done")

                return result
            except AttributeError as e:
                traceback.print_exc()
                logger.error(f"Method '{method_name}' not found on engine: {e}")
                raise ValueError(f"Engine does not have method '{method_name}'")
            except Exception as e:
                traceback.print_exc()
                logger.error(
                    f"Engine method '{method_name}' failed: "
                    f"{e}\n{traceback.format_exc()}"
                )
                raise

        try:
            result = _submit_to_engine_thread(
                f"call_{method_name}", execute_in_engine_thread
            )
        except Exception as e:
            error_msg = str(e)
            if "Engine does not have method" in error_msg:
                return (
                    jsonify({"error": error_msg}),
                    400,
                )
            return (
                jsonify(
                    {"error": (f"Engine method '{method_name}' failed: {error_msg}")}
                ),
                500,
            )

        # Convert all tensors to RTensors and store locally — but ONLY on DP
        # heads. The controller's ``_collect_results`` filter discards
        # non-DP-head results (see train_controller.py:595), and
        # ``clear_batches`` only walks ``rollout_batch``/``adv_batch`` which
        # contain shard ids from DP-head results. Remotizing on non-DP-head
        # ranks therefore stores tensors in the local ``_storage`` that NO
        # cleanup path will ever reach — RSS leaks ~per-batch-payload-size
        # per training step on every non-DP-head rank (see #1209 follow-up).
        #
        # Skip remotize only when we are *certain* the result will be
        # discarded — i.e. for an initialized TrainEngine on a non-DP-head
        # rank. Non-TrainEngine workers (e.g. inference backends) and
        # pre-init TrainEngine RPCs (where ``is_data_parallel_head()`` is
        # not yet defined) keep the original remotize path so setup-time
        # calls behave identically to before this gate.
        is_train = isinstance(engine, TrainEngine)
        is_init = is_train and engine.initialized
        if not is_train or not is_init or engine.is_data_parallel_head():
            state = get_state()
            result = RTensor.remotize(result, node_addr=state.node_addr)
            serialized_result = serialize_value(result)
        else:
            # Non-DP-head: result is discarded by controller. Skip remotize
            # (no _storage growth) and return a sentinel.
            serialized_result = serialize_value(None)
        return jsonify({"status": "success", "result": serialized_result})

    except Exception as e:
        logger.error(f"Unexpected error in call: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
