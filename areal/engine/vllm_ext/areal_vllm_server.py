# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import logging
from http import HTTPStatus

import uvloop
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from vllm.entrypoints.openai.api_server import build_app as _original_build_app
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.completion.api_router import (
    create_completion as _decorated_create_completion,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, OpenAIBaseModel
from vllm.entrypoints.serve.disagg.api_router import generate as _decorated_generate
from vllm.entrypoints.serve.disagg.protocol import GenerateRequest
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.utils.argparse_utils import FlexibleArgumentParser

try:
    from vllm.entrypoints.serve.utils.api_utils import (
        cli_env_setup,
        load_aware_call,
        validate_json_request,
        with_cancellation,
    )
except ImportError:
    from vllm.entrypoints.openai.utils import validate_json_request
    from vllm.entrypoints.utils import cli_env_setup, load_aware_call, with_cancellation

# AReaL's own router for custom endpoints (replaces vLLM's removed global router)
router = APIRouter()

# vLLM already decorates its handlers with @with_cancellation and
# @load_aware_call. AReaL's wrappers below re-apply both, so they must delegate
# to the *undecorated* handler. Applying with_cancellation twice starts two
# listen_for_disconnect tasks on one ASGI receive channel -- its own docstring
# calls that unsafe, since each consumes and discards messages the other needs,
# so a disconnect can be seen by the outer listener while the inner generation
# keeps running and the load counter stays elevated.
original_create_completion = inspect.unwrap(_decorated_create_completion)
original_generate = inspect.unwrap(_decorated_generate)


logger = init_logger("areal_vllm_server")
logger.setLevel(logging.INFO)

# Global event to control generation resume/pause
_generation_run_event = asyncio.Event()
_generation_run_event.set()  # Initially not paused


def _apply_ascend_patch_awex():
    try:
        from areal.engine.patch_awex import patch_awex

        patch_awex()
        print("patching awex success for vllm server process.", flush=True)
    except ImportError:
        print("Failed to import awex, skip patching awex.", flush=True)


_apply_ascend_patch_awex()


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
    # Keep previous versioned names routable for requests admitted before the
    # adapter was updated. Versions can be bounded once request draining is tracked.
    requests[lora_request.lora_name] = lora_request
    logger.info(
        "Registered runtime LoRA adapter alias '%s' for adapter id %s",
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


@router.post(
    "/inference/v1/generate",
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
async def generate(request: GenerateRequest, raw_request: Request):
    """Wrapped token-in/token-out endpoint that respects pause state.

    Rollout generation runs here, so without this gate a request admitted
    during a weight update would be served by a model that is mid-update. The
    policy lives in AReaL rather than in the vLLM patches: same path, same
    schema, same handler underneath, so the patches stay deletable.
    """

    await _wait_if_paused()

    return await original_generate(request, raw_request)


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/entrypoints/cli/main.py for CLI
    # entrypoints.
    import vllm.entrypoints.openai.api_server as _api_server_module

    # Generation routes AReaL replaces with pause-aware wrappers. Every route
    # that reaches the engine must wait on the pause event, or a weight update
    # can land underneath an admitted request.
    _OVERRIDDEN_POST_ROUTES = ("/v1/completions", "/inference/v1/generate")

    def _areal_build_app(*args, **kwargs):
        """Monkey-patched build_app: swap in AReaL's generation routes + custom
        endpoints. ``**kwargs`` forwards version-specific params (model_config was
        added in vLLM 0.19) so it works on both 0.18 and 0.19."""
        app = _original_build_app(*args, **kwargs)
        replaced = [
            route
            for route in app.router.routes
            if hasattr(route, "path")
            and route.path in _OVERRIDDEN_POST_ROUTES
            and hasattr(route, "methods")
            and "POST" in route.methods
        ]
        # A silently missing route would drop the pause gate, so fail loudly
        # instead of serving unpaused traffic.
        missing = set(_OVERRIDDEN_POST_ROUTES) - {r.path for r in replaced}
        if missing:
            raise RuntimeError(
                f"vLLM did not register the generation routes AReaL wraps: "
                f"{sorted(missing)}. Without the wrapper these routes bypass "
                f"AReaL's weight-update pause gate."
            )
        app.router.routes = [r for r in app.router.routes if r not in replaced]
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

    if getattr(args, "headless", False):
        from vllm.entrypoints.cli.serve import run_headless

        if getattr(args, "api_server_count", None) is None:
            args.api_server_count = 0
        run_headless(args)
    else:
        uvloop.run(run_server(args))
