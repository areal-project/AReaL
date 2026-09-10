# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from threading import Lock
from typing import TYPE_CHECKING, Any

from areal.utils import logging

if TYPE_CHECKING:
    from flask import Blueprint

logger = logging.getLogger("AwexBlueprint")


def create_awex_blueprint(
    *,
    flask_module: Any,
    get_engine: Any,
    submit_to_engine_thread: Any,
    run_endpoint: Any,
) -> Blueprint:
    """Create Flask blueprint for awex weight update endpoints.

    Registered alongside the engine blueprint in the training service worker.
    The adapter is lazily created when first needed.

    Follows the same callback injection pattern as create_engine_module().
    """
    bp = flask_module.Blueprint("awex", __name__, url_prefix="/awex")

    _state: dict[str, Any] = {
        "adapter": None,
        "cancelled_operations": set(),
        "operation_lock": Lock(),
    }

    def _operation(data: dict[str, Any]) -> tuple[str, str, float | None]:
        pair_name = data["pair_name"]
        operation_id = data.pop("operation_id", "")
        ttl = data.pop("operation_ttl_s", None)
        expiry = None if ttl is None else time.monotonic() + max(0.0, float(ttl))
        return pair_name, operation_id, expiry

    def _check_operation(
        pair_name: str, operation_id: str, expiry: float | None
    ) -> float | None:
        remaining = None if expiry is None else expiry - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise TimeoutError(f"AWEX init for pair {pair_name!r} expired in queue")
        if operation_id:
            with _state["operation_lock"]:
                cancelled = (pair_name, operation_id) in _state["cancelled_operations"]
            if cancelled:
                raise RuntimeError(
                    f"AWEX operation {operation_id!r} for pair {pair_name!r} "
                    "was cancelled"
                )
        return remaining

    def _apply_remaining_pg_timeout(
        data: dict[str, Any], remaining: float | None
    ) -> None:
        if remaining is not None:
            configured = data.get("process_group_timeout_s")
            data["process_group_timeout_s"] = min(
                remaining, remaining if configured is None else float(configured)
            )

    def _require_adapter():
        if _state["adapter"] is None:
            engine = get_engine()
            if engine is None:
                raise RuntimeError("Engine not initialized")
            _state["adapter"] = _create_training_adapter(engine)
        return _state["adapter"]

    @bp.route("/report_parallelism", methods=["GET"])
    def report_parallelism():
        try:
            adapter = _require_adapter()
            return flask_module.jsonify(adapter.parallelism_strategy)
        except RuntimeError as e:
            return flask_module.jsonify({"error": str(e)}), 400

    @bp.route("/report_weight_meta", methods=["POST"])
    def report_weight_meta():
        def action():
            adapter = _require_adapter()
            return adapter.get_weight_metadata()

        return run_endpoint(
            "report_weight_meta",
            lambda: submit_to_engine_thread("report_weight_meta", action),
        )

    @bp.route("/init_weights_update_group", methods=["POST"])
    def init_weights_update_group():
        data = flask_module.request.get_json(force=True)
        pair_name, operation_id, expiry = _operation(data)

        def action():
            remaining = _check_operation(pair_name, operation_id, expiry)
            _apply_remaining_pg_timeout(data, remaining)
            adapter = _require_adapter()
            adapter.init_weight_update_group(**data)

        return run_endpoint(
            "init_weights_update_group",
            lambda: submit_to_engine_thread("init_weights_update_group", action),
        )

    @bp.route("/update_weights", methods=["POST"])
    def update_weights():
        data = flask_module.request.get_json(force=True)
        pair_name = data["pair_name"]
        version = data.get("version", 0)

        def action():
            adapter = _require_adapter()
            adapter.execute_weight_update(pair_name, version)

        return run_endpoint(
            "update_weights",
            lambda: submit_to_engine_thread("update_weights", action),
        )

    @bp.route("/batch_isend_irecv", methods=["POST"])
    def batch_isend_irecv():
        data = flask_module.request.get_json(force=True)
        pair_name, operation_id, expiry = _operation(data)
        data.pop("pair_name")

        def action():
            _check_operation(pair_name, operation_id, expiry)
            adapter = _require_adapter()
            adapter.batch_isend_irecv(pair_name, **data)

        return run_endpoint(
            "batch_isend_irecv",
            lambda: submit_to_engine_thread("batch_isend_irecv", action),
        )

    @bp.route("/teardown", methods=["POST"])
    def teardown():
        data = flask_module.request.get_json(silent=True) or {}
        pair_name = data.get("pair_name")
        if not pair_name:
            return flask_module.jsonify({"error": "pair_name is required"}), 400
        operation_id = data.get("operation_id", "")
        if operation_id:
            with _state["operation_lock"]:
                _state["cancelled_operations"].add((pair_name, operation_id))

        def action():
            adapter = _state.get("adapter")
            if adapter is not None:
                adapter.teardown_weight_update_group(pair_name)

        return run_endpoint(
            "awex_teardown",
            lambda: submit_to_engine_thread("awex_teardown", action),
            return_result=False,
        )

    @bp.route("/init_colocate_weight_update", methods=["POST"])
    def init_colocate_weight_update():
        data = flask_module.request.get_json(force=True)
        pair_name, operation_id, expiry = _operation(data)

        def action():
            remaining = _check_operation(pair_name, operation_id, expiry)
            _apply_remaining_pg_timeout(data, remaining)
            adapter = _require_adapter()
            adapter.init_colocate_weight_update(**data)

        return run_endpoint(
            "init_colocate_weight_update",
            lambda: submit_to_engine_thread("init_colocate_weight_update", action),
            return_result=False,
        )

    @bp.route("/execute_colocate_weight_update", methods=["POST"])
    def execute_colocate_weight_update():
        data = flask_module.request.get_json(force=True)
        pair_name = data["pair_name"]
        version = data.get("version", 0)

        def action():
            adapter = _require_adapter()
            adapter.execute_colocate_weight_update(pair_name, version)

        return run_endpoint(
            "execute_colocate_weight_update",
            lambda: submit_to_engine_thread("execute_colocate_weight_update", action),
            return_result=False,
        )

    @bp.route("/release_memory", methods=["POST"])
    def release_memory():
        data = flask_module.request.get_json(force=True)
        tags = data.get("tags")

        def action():
            adapter = _require_adapter()
            adapter.release_memory(tags)

        return run_endpoint(
            "release_memory",
            lambda: submit_to_engine_thread("release_memory", action),
            return_result=False,
        )

    @bp.route("/resume_memory", methods=["POST"])
    def resume_memory():
        data = flask_module.request.get_json(force=True)
        tags = data.get("tags")

        def action():
            adapter = _require_adapter()
            adapter.resume_memory(tags)

        return run_endpoint(
            "resume_memory",
            lambda: submit_to_engine_thread("resume_memory", action),
            return_result=False,
        )

    @bp.route("/debug/get_parameters", methods=["POST"])
    def get_parameters():
        """Save local shard parameters to a file for test validation."""
        data = flask_module.request.get_json(force=True)
        save_path = data["save_path"]
        names = data.get("names")

        def action():
            adapter = _require_adapter()
            adapter.save_parameters(save_path, names)

        return run_endpoint(
            "get_parameters",
            lambda: submit_to_engine_thread("get_parameters", action),
            return_result=False,
        )

    return bp


def _create_training_adapter(engine):
    from areal.engine.fsdp_engine import FSDPEngine
    from areal.engine.megatron_engine import MegatronEngine
    from areal.v2.weight_update.awex.fsdp_adapter import AwexFSDPAdapter
    from areal.v2.weight_update.awex.megatron_adapter import (
        AwexMegatronAdapter,
    )

    if isinstance(engine, FSDPEngine):
        return AwexFSDPAdapter(engine)

    if isinstance(engine, MegatronEngine):
        return AwexMegatronAdapter(engine)

    raise TypeError(
        f"Unsupported engine type for weight update: {type(engine).__name__}"
    )
