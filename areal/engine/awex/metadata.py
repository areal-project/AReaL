# SPDX-License-Identifier: Apache-2.0

"""Build shared AWEX metadata and coordinate SGLang's GC object scans."""

import threading
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")
_metadata_gc_lock = threading.Lock()


def has_frozen_native_qwen_vision(engine: Any) -> bool:
    """Require a real, fully frozen native visual owner before reusing features.

    CLI defaults do not prove freezing: native providers resolve them during
    model construction. Check the actor's live encoder and projection parameters
    together, and treat missing ownership or unsupported models conservatively.
    This runs on the metadata publisher, never on the inference model (whose
    parameters are universally marked as inference-only).
    """
    if getattr(engine.hf_config, "model_type", None) not in (
        "qwen3_vl",
        "qwen3_vl_moe",
        "qwen3_5",
        "qwen3_5_moe",
    ):
        return False
    if (
        getattr(getattr(engine, "mcore_config", None), "bridge_type", None)
        != "megatron-bridge"
    ):
        return False
    models = getattr(engine, "model", [])
    if not isinstance(models, (list, tuple)):
        models = [models]
    found_visual = False
    for model in models:
        while hasattr(model, "module"):
            model = model.module
        visual = getattr(model, "vision_model", None)
        if visual is None:
            continue
        for parameter in visual.parameters():
            found_visual = True
            if parameter.requires_grad:
                return False
    return found_visual


def build_awex_train_info(engine: Any, world_size: int) -> dict[str, Any]:
    """Publish shared metadata without loading another model's AWEX extension."""
    info: dict[str, Any] = {"train_world_size": world_size}
    if engine.hf_config.architectures == ["Qwen4ExpForConditionalGeneration"]:
        from areal.models.mcore.qwen4_exp_awex_binding import actor_frozen_binder

        binder = actor_frozen_binder(engine)
        info["qwen4_exp_frozen_contract"] = binder.contract.to_dict()
        info["qwen4_exp_frozen_proof"] = engine._qwen4_awex_frozen_proof
        info["frozen_visual_embeddings"] = True
    else:
        info["frozen_visual_embeddings"] = has_frozen_native_qwen_vision(engine)
    return info


def serialize_metadata_gc(func: Callable[_P, _T]) -> Callable[_P, _T]:
    """Keep GC scans from retaining partially constructed metadata tuples.

    CPython's tuple(iterator) may resize an unfinished tuple. gc.get_objects()
    in another thread can retain that tuple, violating the resize refcount
    requirement (CPython issue 15108). SGLang scans objects while freezing GC
    after server startup, concurrently with AWEX's metadata worker.
    """

    @wraps(func)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        with _metadata_gc_lock:
            return func(*args, **kwargs)

    return guarded
