# SPDX-License-Identifier: Apache-2.0
"""RDT HTTP endpoints for training worker.

Reference: areal.experimental.training_service.worker.awex
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

import ray

from areal.utils import logging

if TYPE_CHECKING:
    from flask import Blueprint

logger = logging.getLogger("RDTTWBlueprint")

# Module-level global variable for actor handle lifecycle management
# Must be held throughout TW subprocess lifetime to prevent garbage collection
_rdt_actor: Any = None


def create_rdt_blueprint(
    *,
    flask_module: Any,
    get_engine: Any,
    submit_to_engine_thread: Any,
    run_endpoint: Any,
) -> Blueprint:
    """Create Flask blueprint for RDT weight update endpoints.

    Uses the same callback injection pattern as awex blueprint.

    TW subprocess creates internal WeightTransportActor and delegates HTTP endpoints via Ray RPC.

    Args:
        flask_module: Flask module (imported dynamically)
        get_engine: Callable to get current engine
        submit_to_engine_thread: Callable to submit work to engine thread
        run_endpoint: Callable to run endpoint with error handling

    Returns:
        Blueprint: Flask blueprint with /rdt/* endpoints
    """
    bp = flask_module.Blueprint("rdt", __name__, url_prefix="/rdt")

    def _ensure_actor():
        """Ensure WeightTransportActor is created and return handle.

        Uses named actor with ray.get_actor() to guarantee only one actor per TW subprocess.
        """
        global _rdt_actor
        if _rdt_actor is None:
            if not ray.is_initialized():
                ray.init(address="auto", ignore_reinit_error=True)
            engine = get_engine()
            if engine is None:
                raise RuntimeError("Engine not initialized")

            # Use fixed actor name based on engine rank for uniqueness within TW subprocess
            actor_name = f"weight-transport-{engine.rank}"

            # Try to get existing named actor first
            try:
                _rdt_actor = ray.get_actor(actor_name)
                logger.info(f"Reused existing WeightTransportActor: {actor_name}")
            except ValueError:
                # Actor doesn't exist, create new one
                from areal.experimental.weight_update.rdt.weight_transport_actor import (
                    WeightTransportActor,
                )

                _rdt_actor = (
                    ray.remote(WeightTransportActor)
                    .options(num_gpus=1, name=actor_name)
                    .remote(engine)
                )
                logger.info(f"Created new WeightTransportActor: {actor_name}")
        return _rdt_actor

    @bp.route("/get_actor_handle", methods=["GET"])
    def get_actor_handle():
        """Return serialized WeightTransportActor handle for IW storage.

        IW subprocess will deserialize this handle and call TW via Ray RPC.

        Returns:
            JSON with actor_bytes_b64 (Base64-encoded cloudpickle bytes)
        """
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
        """Report parallelism strategy for TransferPlan building.

        Returns:
            JSON with world_size, tp_size, pp_size, dp_size, ep_size
        """
        try:
            actor = _ensure_actor()
            result = ray.get(actor.get_parallelism_strategy.remote())
            return flask_module.jsonify(result)
        except RuntimeError as e:
            return flask_module.jsonify({"error": str(e)}), 400

    @bp.route("/report_weight_meta", methods=["POST"])
    def report_weight_meta():
        """Report weight metadata for TransferPlan building.

        Returns:
            JSON with parameter metadata list
        """

        def action():
            actor = _ensure_actor()
            return ray.get(actor.get_weight_metadata.remote())

        return run_endpoint(
            "report_weight_meta",
            lambda: submit_to_engine_thread("report_weight_meta", action),
        )

    @bp.route("/init_weight_update_group", methods=["POST"])
    def init_weight_update_group():
        """Initialize RDT weight update group for TW.

        Gateway calls this to set up TransferPlan on TW side.

        Request body:
            pair_name: TW-IW pair identifier
            kv_store_url: Gateway KV store URL
            infer_world_size: Total IW world size
            train_world_size: Total TW world size
            num_engines: Number of IW engines
            transfer_rank: TW's transfer rank
        """
        data = flask_module.request.get_json(force=True)

        def action():
            actor = _ensure_actor()
            ray.get(
                actor.init_weight_update_group.remote(
                    pair_name=data["pair_name"],
                    kv_store_url=data["kv_store_url"],
                    infer_world_size=data["infer_world_size"],
                    train_world_size=data["train_world_size"],
                    num_engines=data["num_engines"],
                    transfer_rank=data["transfer_rank"],
                )
            )

        return run_endpoint(
            "init_weight_update_group",
            lambda: submit_to_engine_thread("init_weight_update_group", action),
        )

    @bp.route("/debug/get_parameters", methods=["POST"])
    def get_parameters():
        """Save local shard parameters to a file for test validation."""
        data = flask_module.request.get_json(force=True)
        save_path = data["save_path"]
        names = data.get("names")

        def action():
            engine = get_engine()
            if engine is None:
                raise RuntimeError("Engine not initialized")
            adapter = _create_rdt_adapter(engine)
            adapter.save_parameters(save_path, names)

        return run_endpoint(
            "get_parameters",
            lambda: submit_to_engine_thread("get_parameters", action),
            return_result=False,
        )

    return bp


def _create_rdt_adapter(engine):
    """Create RDT adapter based on engine type.

    Args:
        engine: FSDPEngine or MegatronEngine

    Returns:
        RDTFSDPAdapter or RDTMegatronAdapter
    """
    from areal.engine.fsdp_engine import FSDPEngine
    from areal.engine.megatron_engine import MegatronEngine
    from areal.experimental.weight_update.rdt.fsdp_adapter import RDTFSDPAdapter
    from areal.experimental.weight_update.rdt.megatron_adapter import RDTMegatronAdapter

    if isinstance(engine, FSDPEngine):
        return RDTFSDPAdapter(engine)

    if isinstance(engine, MegatronEngine):
        return RDTMegatronAdapter(engine)

    raise TypeError(
        f"Unsupported engine type for RDT weight update: {type(engine).__name__}"
    )
