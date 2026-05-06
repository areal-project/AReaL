# SPDX-License-Identifier: Apache-2.0
"""RDT HTTP endpoints for training worker."""

from __future__ import annotations

import base64
import os
from typing import TYPE_CHECKING, Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from torch.multiprocessing.reductions import reduce_tensor

if TYPE_CHECKING:
    from flask import Blueprint

from areal.utils import logging

logger = logging.getLogger("RDTTWBlueprint")

# Module-level globals for lifecycle (prevent GC)
_rdt_actor: Any = None
_rdt_adapter: Any = None


def create_rdt_blueprint(
    *,
    flask_module: Any,
    get_engine: Any,
    submit_to_engine_thread: Any,
    run_endpoint: Any,
) -> Blueprint:
    """Create Flask blueprint for RDT weight update endpoints."""
    bp = flask_module.Blueprint("rdt", __name__, url_prefix="/rdt")

    def _ensure_actor():
        """Ensure WeightTransportActor is created and return handle."""
        global _rdt_actor
        if _rdt_actor is None:
            if not ray.is_initialized():
                ray.init(address="auto", ignore_reinit_error=True)
            engine = get_engine()
            if engine is None:
                raise RuntimeError("Engine not initialized")

            current_node_id = ray.get_runtime_context().get_node_id()
            current_visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            actor_name = f"weight-transport-{engine.rank}"

            try:
                _rdt_actor = ray.get_actor(actor_name)
                logger.info(f"Reused existing WeightTransportActor: {actor_name}")
            except ValueError:
                from areal.experimental.weight_update.rdt.weight_transport_actor import (
                    WeightTransportActor,
                )

                _rdt_actor = WeightTransportActor.options(
                    name=actor_name,
                    num_gpus=0.0001,
                    max_concurrency=8,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=current_node_id,
                        soft=False,
                    ),
                    runtime_env={
                        "env_vars": {"CUDA_VISIBLE_DEVICES": current_visible_gpus}
                    },
                ).remote()
                logger.info(f"Created new WeightTransportActor: {actor_name}")
        return _rdt_actor

    def _get_adapter():
        """Create RDT adapter based on engine type (cached)."""
        global _rdt_adapter
        if _rdt_adapter is None:
            engine = get_engine()
            if engine is None:
                raise RuntimeError("Engine not initialized")

            from areal.engine.fsdp_engine import FSDPEngine
            from areal.engine.megatron_engine import MegatronEngine
            from areal.experimental.weight_update.rdt.fsdp_adapter import RDTFSDPAdapter
            from areal.experimental.weight_update.rdt.megatron_adapter import (
                RDTMegatronAdapter,
            )

            if isinstance(engine, FSDPEngine):
                _rdt_adapter = RDTFSDPAdapter(engine)
            elif isinstance(engine, MegatronEngine):
                _rdt_adapter = RDTMegatronAdapter(engine)
            else:
                raise TypeError(f"Unsupported engine type: {type(engine).__name__}")
        return _rdt_adapter

    @bp.route("/get_actor_handle", methods=["GET"])
    def get_actor_handle():
        """Return serialized WeightTransportActor handle."""
        try:
            actor = _ensure_actor()
            handle_bytes = ray.cloudpickle.dumps(actor)
            return flask_module.jsonify(
                {"actor_bytes_b64": base64.b64encode(handle_bytes).decode()}
            )
        except RuntimeError as e:
            return flask_module.jsonify({"error": str(e)}), 400

    @bp.route("/report_parallelism", methods=["GET"])
    def report_parallelism():
        """Return parallelism strategy."""
        try:
            adapter = _get_adapter()
            return flask_module.jsonify(adapter.parallelism_strategy)
        except RuntimeError as e:
            return flask_module.jsonify({"error": str(e)}), 400

    @bp.route("/report_weight_meta", methods=["POST"])
    def report_weight_meta():
        """Return parameter metadata."""

        def action():
            adapter = _get_adapter()
            return adapter.get_weight_metadata()

        return run_endpoint(
            "report_weight_meta",
            lambda: submit_to_engine_thread("report_weight_meta", action),
        )

    @bp.route("/init_weight_update_group", methods=["POST"])
    def init_weight_update_group():
        """Initialize TransferPlan."""
        data = flask_module.request.get_json(force=True)

        def action():
            adapter = _get_adapter()
            adapter.init_weight_update_group(**data)

        return run_endpoint(
            "init_weight_update_group",
            lambda: submit_to_engine_thread("init_weight_update_group", action),
        )

    @bp.route("/update_weights", methods=["POST"])
    def update_weights():
        """Slice tensors → create IPC handles → actor.store_ipc_handles."""
        data = flask_module.request.get_json(force=True)
        pair_name = data["pair_name"]
        version = data.get("version", 0)

        def action():
            import time

            t0 = time.monotonic()
            actor = _ensure_actor()
            adapter = _get_adapter()

            plan = adapter._transfer_plans.get(pair_name)
            if not plan:
                raise RuntimeError(f"TransferPlan not found for {pair_name}")

            t1 = time.monotonic()
            local_params = adapter.get_local_shard_parameters()
            t2 = time.monotonic()

            # Build IPC handles dict for each infer_rank
            for send_rank, operations in plan.inter_operations.items():
                infer_rank = operations[0].recv_shard_meta.global_rank
                ipc_handles = {}

                for op in operations:
                    full_tensor = local_params.get(op.send_shard_meta.name)
                    if full_tensor is None:
                        logger.warning(f"Tensor not found: {op.send_shard_meta.name}")
                        continue

                    sliced = full_tensor[op.train_slices]
                    sliced.share_memory_()
                    rebuild_fn, tensor_meta = reduce_tensor(sliced)
                    ipc_handles[op.recv_shard_meta.name] = {
                        "rebuild_fn": rebuild_fn,
                        "tensor_meta": tensor_meta,
                    }

                t3 = time.monotonic()
                ray.get(
                    actor.store_ipc_handles.remote(
                        pair_name, infer_rank, version, ipc_handles
                    )
                )
                t4 = time.monotonic()

                logger.info(
                    f"[RDT-TW-Timing] get_params={1000 * (t2 - t1):.1f}ms | "
                    f"slice_ipc={1000 * (t3 - t2):.1f}ms | "
                    f"store_handles={1000 * (t4 - t3):.1f}ms | "
                    f"total={1000 * (t4 - t0):.1f}ms"
                )

            logger.info(f"[RDT-TW] Prepared weights for pair '{pair_name}' v{version}")

        return run_endpoint(
            "update_weights",
            lambda: submit_to_engine_thread("update_weights", action),
        )

    @bp.route("/teardown", methods=["POST"])
    def teardown():
        """Clear all TransferPlans and cleanup adapter state."""
        global _rdt_adapter
        if _rdt_adapter is None:
            return flask_module.jsonify({"status": "success"})

        def action():
            global _rdt_adapter
            _rdt_adapter.teardown_weight_update_group()
            _rdt_adapter = None

        return run_endpoint(
            "rdt_teardown",
            lambda: submit_to_engine_thread("rdt_teardown", action),
            return_result=False,
        )

    @bp.route("/debug/get_parameters", methods=["POST"])
    def get_parameters():
        """Save local shard parameters to file."""
        data = flask_module.request.get_json(force=True)
        save_path = data["save_path"]
        names = data.get("names")

        def action():
            adapter = _get_adapter()
            adapter.save_parameters(save_path, names)

        return run_endpoint(
            "get_parameters",
            lambda: submit_to_engine_thread("get_parameters", action),
            return_result=False,
        )

    return bp
