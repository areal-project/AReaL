# SPDX-License-Identifier: Apache-2.0

"""Startup probe for the exact-token multimodal generation contract.

Multimodal rollouts only work if the inference server's placeholder expansion
matches the local processor. A version skew between them, or a server without
exact-token support, otherwise fails *every* multimodal request once traffic
starts -- discovered by watching a rollout die wholesale rather than by a clear
startup error.

One synthetic request per server turns that into a failure that names its cause.
It does not replace the per-request check, which still covers media shapes and
prompt histories a single probe cannot reach.
"""

import asyncio
import contextlib
import uuid
from typing import Any
from weakref import WeakKeyDictionary

from areal.utils import logging

logger = logging.getLogger("VisionCanary")

# The server's own rejection text, from the exact-token validation patch. Only
# this specific failure proves the check is installed and enforcing.
# The server's own rejection text, from the exact-token validation patch. Only
# this specific failure proves the check is installed and enforcing.
_MISMATCH_SIGNATURE = "does not match expected_token_ids"


# Errors are quoted into exceptions; a rejected multimodal request carries the
# expanded prompt and a base64 image, so quote only a bounded slice.
_ERROR_CHARS = 400

# Probes recorded per engine, then per model/server identity within it.
#
# The engine is the lifecycle boundary: a replaced engine means replaced
# servers, and a weak key lets its record disappear with it rather than letting
# a new server inherit the old one's success at the same address. Within one
# engine the sub-key is model/processor/addresses, because workflows are
# re-resolved per submitted item and build a fresh processor each time -- an
# identity-keyed probe would re-verify every server for every sample.
#
# Concurrent first requests await the same task, and a failure rejects all of
# them rather than letting the losers proceed against an unverified server.
_probes: WeakKeyDictionary = WeakKeyDictionary()
_probes_lock = asyncio.Lock()


def _resolve_engine(engine: Any) -> Any:
    """Return the object that actually owns the server routing.

    The public engines are thin wrappers holding a ``RemoteInfEngine`` in
    ``_engine`` and do not re-export ``addresses``. Probing the wrapper would
    silently find no servers.
    """
    inner = getattr(engine, "_engine", None)
    if inner is not None and _server_addresses(inner):
        return inner
    return engine


def _server_addresses(engine: Any) -> list[str]:
    """Every server this engine can route to.

    ``RemoteInfEngine`` round-robins over ``addresses``; ``InfBridge`` talks to
    one ``backend_addr``. Both must be probed, so normalise to a list here
    rather than assuming the multi-server shape.
    """
    addresses = list(getattr(engine, "addresses", []) or [])
    if addresses:
        return addresses
    single = getattr(engine, "backend_addr", None)
    return [single] if single else []


@contextlib.contextmanager
def _pinned_to(engine: Any, rid: str, address: str):
    """Route one probe to one specific server, where routing is selectable.

    A round-robin engine would otherwise spread the probes and leave servers
    unverified. A single-backend engine has nothing to choose, so pinning is
    both impossible and unnecessary there.
    """
    routing = getattr(engine, "rid_to_address", None)
    if routing is None:
        yield
        return
    routing[rid] = address
    try:
        yield
    finally:
        routing.pop(rid, None)


def _probe_key(engine: Any, processor: Any) -> str:
    """Identity of what is being verified within one engine: model and servers.

    Deliberately not ``id()``: workflows are re-resolved per submitted item and
    build a fresh processor each time, so an identity-keyed probe would re-run
    an all-server image generation for every sample. Integer ids are also
    reused after garbage collection.
    """
    config = getattr(engine, "config", None)
    model = getattr(config, "tokenizer_path", None) or ""
    addresses = ",".join(sorted(_server_addresses(engine)))
    processor_id = getattr(processor, "name_or_path", None) or type(processor).__name__
    return f"{model}|{processor_id}|{addresses}"


async def run_exact_token_canary(engine: Any, processor: Any, tokenizer: Any) -> None:
    """Probe every configured server once. Raises if any rejects the contract.

    Checks all addresses rather than whichever one round-robin would pick: a
    single unpatched or mismatched server would otherwise stay invisible until
    it happened to be selected.
    """
    from PIL import Image

    from areal.api.cli_args import GenerationHyperparameters
    from areal.api.io_struct import ModelRequest
    from areal.experimental.openai.client import _process_vision_prompt
    from areal.utils.hf_utils import apply_chat_template
    from areal.utils.image import image2base64

    engine = _resolve_engine(engine)
    addresses = _server_addresses(engine)
    if not addresses:
        # Reporting success after probing nothing is worse than not probing:
        # it grants confidence that no server was ever checked.
        raise RuntimeError(
            "Multimodal exact-token canary found no server addresses on "
            f"{type(engine).__name__}, so it verified nothing. The engine is "
            "either not initialized yet or does not expose its routing."
        )

    image_b64 = image2base64(Image.new("RGB", (28, 28), (127, 127, 127)))[0]
    text = apply_chat_template(
        tokenizer,
        [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "."}],
            }
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    vision_prompt = _process_vision_prompt(processor, text, [image_b64])

    gconfig = GenerationHyperparameters(max_new_tokens=1, greedy=True)
    # The probe must select the adapter the deployment actually preloaded.
    # GenerationHyperparameters defaults to "default_lora", which would make the
    # canary fail on a LoRA deployment whose exact-token support is fine.
    configured_lora = getattr(getattr(engine, "config", None), "lora_name", "")
    if configured_lora:
        gconfig.lora_name = configured_lora

    def _build(rid: str, expected: list[int]) -> ModelRequest:
        request = ModelRequest(
            rid=rid,
            input_ids=list(expected),
            collapsed_input_ids=list(vision_prompt.collapsed_input_ids),
            image_data=[image_b64],
            gconfig=gconfig.new(),
            tokenizer=tokenizer,
            processor=processor,
        )
        return request

    # A prompt the server must reject. One token differs from what its own
    # expansion will produce, so a server with the validation installed refuses
    # it. The sentinel is a plain id, not a placeholder, so the media count and
    # therefore the rendered length still match.
    corrupted = list(vision_prompt.input_ids)
    corrupted[-1] = 0 if corrupted[-1] != 0 else 1

    # The negative control carries its own throwaway image. A rejected request
    # is rendered before it is refused, which records the image in the API
    # server's multimodal cache, but it never reaches the engine core -- so a
    # later request for that same hash is sent by-reference and dies on
    # "Expected a cached item for mm_hash=...". Giving the control a unique
    # image keeps its rejection from stranding a hash anything else will use.
    negative_image_b64 = image2base64(
        Image.new("RGB", (28, 28), (uuid.uuid4().int % 200 + 30, 90, 160))
    )[0]

    for index, address in enumerate(addresses):
        # Pin these probes to one specific server; agenerate would otherwise
        # round-robin and leave the others unverified.
        #
        # Positive first: it delivers the media to the engine core the normal
        # way. Leading with the rejected control would leave the server
        # expecting a cached item it never received.
        positive_rid = f"areal-mm-canary-{index}-{uuid.uuid4().hex}"
        with _pinned_to(engine, positive_rid, address):
            try:
                await engine.agenerate(_build(positive_rid, vision_prompt.input_ids))
            except Exception as e:
                raise RuntimeError(
                    f"Multimodal exact-token canary failed against {address}, "
                    "so multimodal rollouts would fail on every request routed "
                    "there. That server either does not support the exact-token "
                    "generation contract, or its media preprocessing disagrees "
                    f"with this worker's processor. Failure was "
                    f"{type(e).__name__}: {str(e)[:_ERROR_CHARS]}"
                ) from e

        negative_rid = f"areal-mm-canary-neg-{index}-{uuid.uuid4().hex}"
        # This request is *meant* to fail. Say so first: the HTTP layer logs a
        # warning per retry, so an operator reading the log would otherwise see
        # a burst of unexplained 400s from a healthy server at startup.
        logger.info(
            "Probing %s with a deliberately invalid prompt; the HTTP 400 and "
            "its retries that follow are the expected result.",
            address,
        )
        with _pinned_to(engine, negative_rid, address):
            try:
                request = _build(negative_rid, corrupted)
                request.image_data = [negative_image_b64]
                await engine.agenerate(request)
            except Exception as e:
                # Only the mismatch rejection proves the validation is
                # installed. A timeout, a connection reset or a 503 would
                # otherwise be read as proof, quietly certifying a server that
                # never checked anything. This requires every transport to
                # carry the response body into the exception text.
                if _MISMATCH_SIGNATURE not in str(e):
                    raise RuntimeError(
                        f"Multimodal exact-token canary could not verify "
                        f"{address}: the deliberately invalid probe failed, but "
                        "not with the prompt-mismatch rejection that proves the "
                        f"check is installed. Failure was {type(e).__name__}: "
                        f"{str(e)[:_ERROR_CHARS]}"
                    ) from e
            else:
                raise RuntimeError(
                    f"Multimodal exact-token canary: {address} accepted a "
                    "prompt whose expected_token_ids deliberately disagree with "
                    "its own expansion. The exact-token validation is therefore "
                    "not installed on that server -- pydantic drops unknown "
                    "request fields silently, so expected_token_ids (and "
                    "content_parts) are being ignored rather than enforced. "
                    "Rollouts would train on prompts the model never saw."
                )

    logger.info("Multimodal exact-token canary passed on %d server(s).", len(addresses))


async def ensure_exact_token_support(
    engine: Any, processor: Any, tokenizer: Any
) -> None:
    """Run :func:`run_exact_token_canary` once per model/server identity.

    For callers that reach the engine directly instead of going through the
    proxy, which probes at initialization. Safe to await on every request: the
    work happens once and later callers observe the cached result.
    """
    if processor is None:
        return  # text-only: nothing expands, nothing to verify

    resolved = _resolve_engine(engine)
    key = _probe_key(resolved, processor)
    async with _probes_lock:
        try:
            per_engine = _probes.setdefault(resolved, {})
        except TypeError:
            # Not weak-referenceable; probe every time rather than risk
            # reporting a stale success.
            per_engine = {}
        task = per_engine.get(key)
        if task is None:
            task = asyncio.create_task(
                run_exact_token_canary(engine, processor, tokenizer)
            )
            per_engine[key] = task

    try:
        # Shielded: a caller that gets cancelled must not cancel the probe out
        # from under every other waiter, nor leave a cancelled task cached.
        await asyncio.shield(task)
    except BaseException:
        if task.done() and not task.cancelled() and task.exception() is not None:
            # Drop the failure so a later attempt against a repaired server can
            # retry rather than replaying the original error forever.
            async with _probes_lock:
                if per_engine.get(key) is task:
                    per_engine.pop(key, None)
        raise


def reset_for_testing() -> None:
    """Forget recorded probes. Tests only."""
    _probes.clear()
