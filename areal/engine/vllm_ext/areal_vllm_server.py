# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from http import HTTPStatus

import uvloop
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from vllm.entrypoints.openai.api_server import build_app as _original_build_app
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.completion.api_router import (
    create_completion as original_create_completion,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, OpenAIBaseModel
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import cli_env_setup, load_aware_call, with_cancellation
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.utils.argparse_utils import FlexibleArgumentParser

# AReaL's own router for custom endpoints (replaces vLLM's removed global router)
router = APIRouter()


logger = init_logger("areal_vllm_server")
logger.setLevel(logging.INFO)

# Global event to control generation resume/pause
_generation_run_event = asyncio.Event()
_generation_run_event.set()  # Initially not paused


class UpdateWeightsRequest(OpenAIBaseModel):
    # The model path with the new weights
    model_path: str
    # The format to load the weights
    load_format: str | None = "auto"
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False


class UpdateWeightsRequestLora(OpenAIBaseModel):
    # The model path with the new weights of lora adaptor
    lora_model_path: str
    # The name of lora adaptor
    lora_name: str
    # The id of the lora adaptor in vllm
    lora_int_id: int
    # The name of the base model for lora adaptors
    base_model_name: str
    # The format to load the weights
    load_format: str | None = "auto"
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False


class UpdateGroupRequest(OpenAIBaseModel):
    master_address: str
    master_port: str
    rank_offset: int
    world_size: int
    backend: str
    group_name: str


class UpdateWeightsFromXcclRequest(OpenAIBaseModel):
    names: list[str]
    dtypes: list[str]
    shapes: list[list[int]]
    group_name: str


class UpdateWeightsFromXcclRequestLora(OpenAIBaseModel):
    names: list[str]
    dtypes: list[str]
    shapes: list[list[int]]
    lora_name: str
    lora_int_id: int
    lora_target_modules: list[str] | str
    lora_rank: int
    lora_alpha: int
    lora_bias: str
    base_model_name: str
    group_name: str


def to_json_response(success, message):
    content = {"success": success, "message": message}
    if success:
        return JSONResponse(content, status_code=200)
    else:
        return JSONResponse(content, status_code=400)


def build_response(ret_list):
    success = True
    message = ""
    for rank, ret_value in enumerate(ret_list):
        if_success, msg = ret_value
        success = success if if_success else False
        if if_success:
            message += f"TP rank: {rank} success\n"
        else:
            message += f"TP rank: {rank} failed. reason: {msg}\n"
    return to_json_response(success, message)


def _infer_runtime_lora_path(serving_models, lora_name: str, lora_int_id: int) -> str:
    existing = serving_models.lora_requests.get(lora_name)
    if existing is not None and getattr(existing, "lora_path", ""):
        return existing.lora_path
    for request in serving_models.lora_requests.values():
        if getattr(request, "lora_int_id", None) == lora_int_id and getattr(
            request, "lora_path", ""
        ):
            return request.lora_path
    # Runtime XCCL updates do not come with a filesystem path. Use a stable
    # synthetic path so vLLM can still construct a LoRARequest for routing.
    return f"xccl://{lora_name}"


def _register_runtime_lora_name(
    app,
    *,
    lora_name: str,
    lora_int_id: int,
    base_model_name: str | None,
) -> None:
    serving_models = getattr(app.state, "openai_serving_models", None)
    if serving_models is None:
        logger.warning(
            "openai_serving_models missing; skip runtime LoRA registration for %s",
            lora_name,
        )
        return

    requests = serving_models.lora_requests
    runtime_lora_path = _infer_runtime_lora_path(serving_models, lora_name, lora_int_id)

    # Keep at most one public name per adapter id so /v1/models and request
    # routing reflect the current versioned adapter name.
    for name, request in list(requests.items()):
        if getattr(request, "lora_int_id", None) == lora_int_id and name != lora_name:
            del requests[name]

    lora_request = LoRARequest(
        lora_name=lora_name,
        lora_int_id=lora_int_id,
        lora_path=runtime_lora_path,
    )
    if base_model_name is not None:
        lora_request.base_model_name = base_model_name
    requests[lora_name] = lora_request
    logger.info(
        "Registered runtime LoRA adapter name '%s' for adapter id %s",
        lora_name,
        lora_int_id,
    )


@router.post("/areal_update_weights")
async def areal_update_weight(request: UpdateWeightsRequest, raw_request: Request):
    logger.info(f"API server starts areal_update_weight, {request.model_path}")
    llm = raw_request.app.state.engine_client
    await llm.pause_generation(wait_for_inflight_requests=False, clear_cache=True)
    await llm.reset_mm_cache()
    try:
        ret_list = await llm.collective_rpc(
            "areal_update_weights",
            args=(request.model_path,),
        )
    finally:
        await llm.resume_generation()
    return build_response(ret_list)


@router.post("/areal_update_weights_lora")
async def areal_update_weight_lora(
    request: UpdateWeightsRequestLora, raw_request: Request
):
    logger.info(
        f"API server starts areal_update_weight_lora, lora_model_path-{request.lora_model_path}, lora_name-{request.lora_name}, lora_int_id-{request.lora_int_id}, base_model_name-{request.base_model_name}"
    )
    llm = raw_request.app.state.engine_client
    await llm.pause_generation(wait_for_inflight_requests=False, clear_cache=True)
    await llm.reset_mm_cache()

    try:
        ret_list = await llm.collective_rpc(
            "areal_update_weights_lora",
            args=(
                request.lora_model_path,
                request.lora_name,
                request.lora_int_id,
                request.base_model_name,
            ),
        )
    finally:
        await llm.resume_generation()

    return build_response(ret_list)


@router.post("/areal_update_weights_xccl")
async def areal_update_weight_xccl(raw_request: Request):
    logger.info("API server starts areal_update_weight_xccl")
    llm = raw_request.app.state.engine_client
    await llm.pause_generation(wait_for_inflight_requests=False, clear_cache=True)
    await llm.reset_mm_cache()
    try:
        ret_list = await llm.collective_rpc("areal_update_weight_xccl")
    finally:
        await llm.resume_generation()
    return build_response(ret_list)


@router.post("/areal_update_weights_lora_xccl")
async def areal_update_weight_lora_xccl(
    request: UpdateWeightsFromXcclRequestLora, raw_request: Request
):
    logger.info("API server starts areal_update_weight_lora_xccl")
    llm = raw_request.app.state.engine_client
    await llm.pause_generation(wait_for_inflight_requests=False, clear_cache=True)
    await llm.reset_mm_cache()

    try:
        ret_list = await llm.collective_rpc("areal_update_weight_lora_xccl")
        if all(success for success, _ in ret_list):
            _register_runtime_lora_name(
                raw_request.app,
                lora_name=request.lora_name,
                lora_int_id=request.lora_int_id,
                base_model_name=request.base_model_name,
            )
    finally:
        await llm.resume_generation()

    return build_response(ret_list)


@router.post("/areal_init_weights_update_group")
async def areal_init_weights_update_group(
    request: UpdateGroupRequest, raw_request: Request
):
    logger.info("API server starts areal_init_weights_update_group")
    llm = raw_request.app.state.engine_client
    ret_list = await llm.collective_rpc(
        "areal_init_update_weight_group",
        args=(
            request.master_address,
            request.master_port,
            request.rank_offset,
            request.world_size,
            request.backend,
            request.group_name,
        ),
    )
    return build_response(ret_list)


@router.post("/areal_set_update_weight_meta")
async def areal_set_weight_meta_xccl(
    request: UpdateWeightsFromXcclRequest, raw_request: Request
):
    logger.info("API server starts areal_set_update_weight_meta_xccl")
    llm = raw_request.app.state.engine_client
    ret_list = await llm.collective_rpc(
        "areal_set_weight_meta",
        args=(
            request.names,
            request.dtypes,
            request.shapes,
            request.group_name,
        ),
    )
    return build_response(ret_list)


@router.post("/areal_set_update_weight_meta_lora")
async def areal_set_weight_meta_xccl_lora(
    request: UpdateWeightsFromXcclRequestLora, raw_request: Request
):
    logger.info(
        f"API server starts areal_set_update_weight_meta_lora for {request.lora_name} with id {request.lora_int_id}"
    )
    llm = raw_request.app.state.engine_client
    ret_list = await llm.collective_rpc(
        "areal_set_weight_meta_lora",
        args=(
            request.names,
            request.dtypes,
            request.shapes,
            request.group_name,
            request.lora_name,
            request.lora_int_id,
            request.lora_target_modules,
            request.lora_rank,
            request.lora_alpha,
            request.lora_bias,
            request.base_model_name,
        ),
    )
    return build_response(ret_list)


@router.post("/areal_pause_generation")
async def areal_pause_generation(raw_request: Request):
    logger.info("API server starts areal_pause_generation and aborts all requests")
    llm = raw_request.app.state.engine_client
    # Abort all running and waiting requests
    _generation_run_event.clear()
    await llm.pause_generation(
        wait_for_inflight_requests=False,
        clear_cache=True,
    )
    await llm.reset_mm_cache()

    return to_json_response(True, "Generation paused and all requests aborted")


@router.post("/areal_continue_generation")
async def areal_continue_generation(raw_request: Request):
    logger.info("API server starts areal_continue_generation")
    llm = raw_request.app.state.engine_client
    await llm.resume_generation()
    _generation_run_event.set()
    return to_json_response(True, "Generation continued")


async def _wait_if_paused():
    """Wait if generation is paused."""
    if not _generation_run_event.is_set():
        await _generation_run_event.wait()


@router.post(
    "/v1/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_completion(request: CompletionRequest, raw_request: Request):
    """Wrapped completions endpoint that respects pause state."""

    await _wait_if_paused()

    # Will not use streaming response here.
    response = await original_create_completion(request, raw_request)

    return response


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/entrypoints/cli/main.py for CLI
    # entrypoints.
    import vllm.entrypoints.openai.api_server as _api_server_module

    def _areal_build_app(args, supported_tasks=None, model_config=None):
        """Monkey-patched build_app that replaces vLLM's /v1/completions route
        with AReaL's wrapped version and adds AReaL custom endpoints."""
        app = _original_build_app(
            args, supported_tasks=supported_tasks, model_config=model_config
        )
        # Remove vLLM's /v1/completions POST route so AReaL's takes precedence
        app.router.routes = [
            route
            for route in app.router.routes
            if not (
                hasattr(route, "path")
                and route.path == "/v1/completions"
                and hasattr(route, "methods")
                and "POST" in route.methods
            )
        ]
        # Include AReaL's router with custom endpoints + overridden /v1/completions
        app.include_router(router)
        return app

    # Patch build_app so run_server uses our version
    _api_server_module.build_app = _areal_build_app

    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
