# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def register_engine_routes(
    app: FastAPI,
    *,
    _raw_json_response: Callable[[bytes], Any],
) -> None:
    # -- core routes -------------------------------------------------------

    @app.get("/health")
    async def health():
        topology = app.state.topology
        return {
            "status": "ok",
            "worker_count": len(topology.workers),
            "dp_size": topology.dp_size,
            "dp_heads": topology.dp_heads,
        }

    @app.get("/topology")
    async def topology():
        t = app.state.topology
        return {
            "workers": [asdict(w) for w in t.workers],
            "dp_heads": t.dp_heads,
            "dp_size": t.dp_size,
            "dp_groups": t.dp_groups,
            "pp_size": t.pp_size,
            "tp_size": t.tp_size,
            "cp_size": t.cp_size,
            "ep_size": t.ep_size,
        }

    # -- dispatch helpers --------------------------------------------------

    def _dispatch_compute_route(path: str, *, pad_eval_batch: bool = False):
        async def handler(request: Request):
            dispatcher = app.state.dispatcher
            try:
                body = await request.body()
                return _raw_json_response(
                    await dispatcher.dispatch(path, pad_eval_batch=pad_eval_batch).post(
                        body
                    )
                )
            except Exception as exc:
                return JSONResponse({"error": str(exc)}, status_code=502)

        return handler

    def _broadcast_post_route(path: str, *, require_non_empty: bool = False):
        async def handler(request: Request):
            dispatcher = app.state.dispatcher
            try:
                body = await request.body()
                responses = await dispatcher.broadcast(path).post(body)
                if require_non_empty and not responses:
                    raise RuntimeError(f"No worker responses for {path}")
                return _raw_json_response(responses[0])
            except Exception as exc:
                return JSONResponse({"error": str(exc)}, status_code=502)

        return handler

    def _dispatch_get_route(path: str):
        async def handler():
            dispatcher = app.state.dispatcher
            try:
                return _raw_json_response(await dispatcher.dispatch(path).get())
            except Exception as exc:
                return JSONResponse({"error": str(exc)}, status_code=502)

        return handler

    def _broadcast_get_route(path: str, *, require_non_empty: bool = False):
        async def handler():
            dispatcher = app.state.dispatcher
            try:
                responses = await dispatcher.broadcast(path).get()
                if require_non_empty and not responses:
                    raise RuntimeError(f"No worker responses for {path}")
                return _raw_json_response(responses[0])
            except Exception as exc:
                return JSONResponse({"error": str(exc)}, status_code=502)

        return handler

    # -- engine routes -----------------------------------------------------

    app.post("/train_batch")(_dispatch_compute_route("/train_batch"))
    app.post("/forward_batch")(_dispatch_compute_route("/forward_batch"))
    app.post("/eval_batch")(_dispatch_compute_route("/eval_batch", pad_eval_batch=True))

    app.post("/train")(_broadcast_post_route("/train"))
    app.post("/eval")(_broadcast_post_route("/eval"))
    app.post("/offload")(_broadcast_post_route("/offload"))
    app.post("/onload")(_broadcast_post_route("/onload"))
    app.post("/set_version")(_broadcast_post_route("/set_version"))
    app.get("/get_version")(_dispatch_get_route("/get_version"))
    app.post("/save")(_broadcast_post_route("/save", require_non_empty=True))
    app.post("/load")(_broadcast_post_route("/load", require_non_empty=True))
    app.post("/step_lr_scheduler")(_broadcast_post_route("/step_lr_scheduler"))
    app.post("/optimizer_zero_grad")(
        _broadcast_post_route("/optimizer_zero_grad", require_non_empty=True)
    )
    app.post("/optimizer_step")(
        _broadcast_post_route("/optimizer_step", require_non_empty=True)
    )
    app.post("/get_device_stats")(
        _broadcast_post_route("/get_device_stats", require_non_empty=True)
    )
    app.post("/config_perf_tracer")(
        _broadcast_post_route("/config_perf_tracer", require_non_empty=True)
    )
    app.post("/save_perf_tracer")(
        _broadcast_post_route("/save_perf_tracer", require_non_empty=True)
    )
    app.post("/clear_batches")(
        _broadcast_post_route("/clear_batches", require_non_empty=True)
    )
    app.get("/export_stats")(
        _broadcast_get_route("/export_stats", require_non_empty=True)
    )

    # -- SFT routes --------------------------------------------------------

    app.post("/sft/train")(_dispatch_compute_route("/sft/train"))
    app.post("/sft/evaluate")(
        _dispatch_compute_route("/sft/evaluate", pad_eval_batch=True)
    )

    # -- PPO actor routes --------------------------------------------------

    app.post("/ppo/actor/compute_logp")(
        _dispatch_compute_route("/ppo/actor/compute_logp")
    )
    app.post("/ppo/actor/compute_advantages")(
        _dispatch_compute_route("/ppo/actor/compute_advantages")
    )
    app.post("/ppo/actor/update")(_dispatch_compute_route("/ppo/actor/update"))

    # -- PPO critic routes -------------------------------------------------

    app.post("/ppo/critic/compute_values")(
        _dispatch_compute_route("/ppo/critic/compute_values")
    )
    app.post("/ppo/critic/update")(_dispatch_compute_route("/ppo/critic/update"))

    # -- RW routes ---------------------------------------------------------

    app.post("/rw/train")(_dispatch_compute_route("/rw/train"))
    app.post("/rw/evaluate")(
        _dispatch_compute_route("/rw/evaluate", pad_eval_batch=True)
    )
