# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import traceback
from collections.abc import Callable
from typing import Any

from areal.api import TrainEngine
from areal.infra.rpc.rtensor import RTensor
from areal.infra.rpc.serialization import deserialize_value
from areal.utils import logging
from areal.utils.dynamic_import import import_from_string

logger = logging.getLogger("TrainWorker")


def create_engine_module(
    *,
    flask_module: Any,
    config: Any,
    get_engine: Callable[[], TrainEngine | None],
    set_engine: Callable[[TrainEngine], None],
    submit_to_engine_thread: Callable[..., Any],
    parse_args_kwargs: Callable[[dict[str, Any] | None], tuple[Any, Any]],
    require_engine: Callable[[], TrainEngine],
    run_endpoint: Callable[[str, Callable[[], Any]], Any],
    execute_compute: Callable[..., Any],
    get_node_addr: Callable[[], str],
) -> Any:
    Blueprint = flask_module.Blueprint
    jsonify = flask_module.jsonify
    request = flask_module.request

    bp = Blueprint("worker_engine", __name__)

    # -- core routes -------------------------------------------------------

    @bp.route("/health", methods=["GET"])
    def health_check():
        rank = int(os.environ.get("RANK", 0))
        role = os.environ.get("ROLE", "train_worker")
        ready = get_engine() is not None
        return jsonify(
            {
                "status": "healthy",
                "rank": rank,
                "role": role,
                "ready": ready,
            }
        )

    @bp.route("/create_engine", methods=["POST"])
    def create_engine():
        try:
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({"error": "Invalid JSON in request body"}), 400

            engine_class_path = data.get("engine_class")
            if engine_class_path is None:
                engine_class_path = data.get("engine")
            init_args = deserialize_value(data.get("init_args", []))
            init_kwargs = deserialize_value(data.get("init_kwargs", {}))

            if not engine_class_path:
                return jsonify(
                    {"error": "Missing 'engine_class' field in request"}
                ), 400
            if get_engine() is not None:
                return jsonify({"error": "Engine already exists on this worker"}), 400

            try:
                engine_class = import_from_string(engine_class_path)
                if not issubclass(engine_class, TrainEngine):
                    raise TypeError(
                        "Engine class must be a subclass of TrainEngine, "
                        f"got {engine_class}."
                    )
            except (ValueError, ImportError, AttributeError) as e:
                return (
                    jsonify(
                        {
                            "error": (
                                f"Failed to import engine class '{engine_class_path}': {str(e)}"
                            )
                        }
                    ),
                    400,
                )
            except TypeError as e:
                return jsonify({"error": str(e)}), 400

            def create_in_engine_thread():
                return engine_class(*init_args, **init_kwargs)

            engine = submit_to_engine_thread("create_engine", create_in_engine_thread)
            set_engine(engine)
            return jsonify(
                {
                    "status": "success",
                    "message": "Engine created and initialized",
                    "result": None,
                }
            )
        except Exception as e:
            logger.error(
                f"Unexpected error in create_engine: {e}\n{traceback.format_exc()}"
            )
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    @bp.route("/configure", methods=["POST"])
    def configure():
        data = request.get_json(silent=True)
        raw_args, raw_kwargs = parse_args_kwargs(data)

        def action():
            engine = require_engine()
            configure_fn = getattr(engine, "configure", None)
            if callable(configure_fn):
                return configure_fn(*raw_args, **raw_kwargs)
            return None

        return run_endpoint(
            "configure",
            lambda: submit_to_engine_thread("configure", action),
        )

    @bp.route("/topology", methods=["GET"])
    def topology():
        try:
            engine = require_engine()
            return jsonify(
                {
                    "rank": int(os.environ.get("RANK", 0)),
                    "world_size": int(os.environ.get("WORLD_SIZE", 1)),
                    "dp_rank": engine.data_parallel_rank,
                    "dp_size": engine.data_parallel_world_size,
                    "is_dp_head": engine.is_data_parallel_head(),
                    "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
                }
            )
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Error in topology: {e}\n{traceback.format_exc()}")
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    @bp.route("/get_param_info", methods=["GET"])
    def get_param_info():
        def action():
            engine = require_engine()
            get_param_info_fn = getattr(engine, "get_param_info", None)
            if callable(get_param_info_fn):
                return get_param_info_fn()
            get_parameter_info_fn = getattr(engine, "get_parameter_info", None)
            if callable(get_parameter_info_fn):
                return get_parameter_info_fn()
            return None

        return run_endpoint(
            "get_param_info",
            lambda: submit_to_engine_thread("get_param_info", action),
        )

    # -- dispatch helpers --------------------------------------------------

    def _register_compute_route(
        path: str, method_name: str, *, endpoint_prefix: str = ""
    ) -> None:
        def handler():
            data = request.get_json(silent=True)
            raw_args, raw_kwargs = parse_args_kwargs(data)
            args = RTensor.localize(raw_args)
            kwargs = RTensor.localize(raw_kwargs)
            result = execute_compute(
                method_name,
                args,
                kwargs,
                require_broadcast=True,
            )
            return RTensor.remotize(result, node_addr=get_node_addr())

        ep_name = (
            f"{endpoint_prefix}{method_name}_endpoint"
            if endpoint_prefix
            else f"{method_name}_compute_endpoint"
        )
        bp.add_url_rule(
            path,
            ep_name,
            lambda: run_endpoint(method_name, handler),
            methods=["POST"],
        )

    def _register_engine_route(
        path: str,
        method_name: str,
        *,
        methods: list[str] | None = None,
        return_result: bool = True,
    ) -> None:
        http_methods = methods or ["POST"]

        def handler():
            if request.method == "GET":
                args, kwargs = [], {}
            else:
                data = request.get_json(silent=True) or {}
                args, kwargs = parse_args_kwargs(data)

            return run_endpoint(
                method_name,
                lambda: submit_to_engine_thread(
                    method_name,
                    lambda: getattr(require_engine(), method_name)(*args, **kwargs),
                ),
                return_result=return_result,
            )

        bp.add_url_rule(path, f"{method_name}_endpoint", handler, methods=http_methods)

    # -- lifecycle routes --------------------------------------------------

    @bp.route("/destroy_engine", methods=["POST"])
    def destroy_engine():
        """Gracefully destroy the engine and release distributed resources.

        Calls ``engine.destroy()`` which runs
        ``dist.destroy_process_group()`` (a local ``ncclCommAbort`` +
        HeartbeatMonitor join).  Must be called on **all** workers before
        any process exits so that rank-0's TCPStore server outlives every
        other rank's HeartbeatMonitor thread.
        """
        engine = get_engine()
        if engine is None:
            return jsonify({"status": "success", "message": "No engine to destroy"})

        def action():
            engine.destroy()

        return run_endpoint(
            "destroy_engine",
            lambda: submit_to_engine_thread("destroy_engine", action),
            return_result=False,
        )

    # -- engine routes -----------------------------------------------------

    _register_engine_route("/train", "train", return_result=False)
    _register_engine_route("/eval", "eval", return_result=False)
    _register_compute_route("/train_batch", "train_batch")
    _register_compute_route("/forward_batch", "forward_batch")
    _register_compute_route("/eval_batch", "eval_batch")
    _register_engine_route("/create_process_group", "create_process_group")
    _register_engine_route("/initialize", "initialize")
    _register_engine_route("/set_version", "set_version")
    _register_engine_route("/get_version", "get_version", methods=["GET"])
    _register_engine_route("/save", "save")
    _register_engine_route("/load", "load")
    _register_engine_route("/offload", "offload")
    _register_engine_route("/onload", "onload")
    _register_engine_route("/optimizer_zero_grad", "optimizer_zero_grad")
    _register_engine_route("/optimizer_step", "optimizer_step")
    _register_engine_route("/step_lr_scheduler", "step_lr_scheduler")
    _register_engine_route("/get_device_stats", "get_device_stats")
    _register_engine_route("/config_perf_tracer", "config_perf_tracer")
    _register_engine_route("/save_perf_tracer", "save_perf_tracer")
    _register_engine_route("/clear_batches", "clear_batches")
    _register_engine_route("/export_stats", "export_stats", methods=["GET"])

    # -- SFT routes --------------------------------------------------------

    _register_compute_route("/sft/train", "train_lm")
    _register_compute_route("/sft/evaluate", "evaluate_lm")

    # -- PPO actor routes --------------------------------------------------

    _register_compute_route(
        "/ppo/actor/compute_logp", "compute_logp", endpoint_prefix="ppo_actor_"
    )
    _register_compute_route(
        "/ppo/actor/compute_advantages",
        "compute_advantages",
        endpoint_prefix="ppo_actor_",
    )
    _register_compute_route(
        "/ppo/actor/update", "ppo_update", endpoint_prefix="ppo_actor_"
    )

    # -- PPO critic routes -------------------------------------------------

    _register_compute_route(
        "/ppo/critic/compute_values",
        "compute_values",
        endpoint_prefix="ppo_critic_",
    )
    _register_compute_route(
        "/ppo/critic/update", "ppo_update", endpoint_prefix="ppo_critic_"
    )

    # -- RW routes ---------------------------------------------------------

    _register_compute_route("/rw/train", "train_rw", endpoint_prefix="rw_")
    _register_compute_route("/rw/evaluate", "evaluate_rw", endpoint_prefix="rw_")

    return bp
