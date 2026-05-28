# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request

from areal.experimental.training_service.gateway.config import GatewayConfig


def register_engine_routes(
    app: FastAPI,
    config: GatewayConfig,
    *,
    _forward_post: Callable[..., Any],
    _forward_get: Callable[..., Any],
) -> None:
    # -- core routes -------------------------------------------------------

    @app.get("/health")
    async def health():
        return {"status": "ok", "router_addr": config.router_addr}

    @app.post("/train_batch")
    async def train_batch(request: Request):
        return await _forward_post(request, "/train_batch", config)

    @app.post("/forward_batch")
    async def forward_batch(request: Request):
        return await _forward_post(request, "/forward_batch", config)

    @app.post("/eval_batch")
    async def eval_batch(request: Request):
        return await _forward_post(request, "/eval_batch", config)

    @app.post("/train")
    async def train(request: Request):
        return await _forward_post(request, "/train", config)

    @app.post("/eval")
    async def eval_(request: Request):
        return await _forward_post(request, "/eval", config)

    @app.post("/set_version")
    async def set_version(request: Request):
        return await _forward_post(request, "/set_version", config)

    @app.get("/get_version")
    async def get_version(request: Request):
        return await _forward_get(request, "/get_version", config)

    @app.post("/save")
    async def save(request: Request):
        return await _forward_post(request, "/save", config)

    @app.post("/load")
    async def load(request: Request):
        return await _forward_post(request, "/load", config)

    @app.post("/offload")
    async def offload(request: Request):
        return await _forward_post(
            request,
            "/offload",
            config,
            use_admin_auth_for_upstream=True,
        )

    @app.post("/onload")
    async def onload(request: Request):
        return await _forward_post(
            request,
            "/onload",
            config,
            use_admin_auth_for_upstream=True,
        )

    @app.post("/step_lr_scheduler")
    async def step_lr_scheduler(request: Request):
        return await _forward_post(request, "/step_lr_scheduler", config)

    @app.post("/optimizer_zero_grad")
    async def optimizer_zero_grad(request: Request):
        return await _forward_post(request, "/optimizer_zero_grad", config)

    @app.post("/optimizer_step")
    async def optimizer_step(request: Request):
        return await _forward_post(request, "/optimizer_step", config)

    @app.post("/get_device_stats")
    async def get_device_stats(request: Request):
        return await _forward_post(request, "/get_device_stats", config)

    @app.post("/config_perf_tracer")
    async def config_perf_tracer(request: Request):
        return await _forward_post(request, "/config_perf_tracer", config)

    @app.post("/save_perf_tracer")
    async def save_perf_tracer(request: Request):
        return await _forward_post(request, "/save_perf_tracer", config)

    @app.post("/clear_batches")
    async def clear_batches(request: Request):
        return await _forward_post(request, "/clear_batches", config)

    @app.get("/export_stats")
    async def export_stats(request: Request):
        return await _forward_get(request, "/export_stats", config)

    # -- SFT routes --------------------------------------------------------

    @app.post("/sft/train")
    async def train_sft(request: Request):
        return await _forward_post(request, "/sft/train", config)

    @app.post("/sft/evaluate")
    async def evaluate_sft(request: Request):
        return await _forward_post(request, "/sft/evaluate", config)

    # -- PPO actor routes --------------------------------------------------

    @app.post("/ppo/actor/compute_logp")
    async def actor_compute_logp(request: Request):
        return await _forward_post(request, "/ppo/actor/compute_logp", config)

    @app.post("/ppo/actor/compute_advantages")
    async def actor_compute_advantages(request: Request):
        return await _forward_post(request, "/ppo/actor/compute_advantages", config)

    @app.post("/ppo/actor/update")
    async def actor_update(request: Request):
        return await _forward_post(request, "/ppo/actor/update", config)

    # -- PPO critic routes -------------------------------------------------

    @app.post("/ppo/critic/compute_values")
    async def critic_compute_values(request: Request):
        return await _forward_post(request, "/ppo/critic/compute_values", config)

    @app.post("/ppo/critic/update")
    async def critic_update(request: Request):
        return await _forward_post(request, "/ppo/critic/update", config)

    # -- RW routes ---------------------------------------------------------

    @app.post("/rw/train")
    async def rw_train(request: Request):
        return await _forward_post(request, "/rw/train", config)

    @app.post("/rw/evaluate")
    async def rw_evaluate(request: Request):
        return await _forward_post(request, "/rw/evaluate", config)
