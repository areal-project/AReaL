# SPDX-License-Identifier: Apache-2.0

"""Route distributed MTP weight updates to SGLang's draft model.

SGLang 0.5.10.post1 sends ``update_weights_from_distributed`` only to the
regular TP worker, while NEXTN constructs a separate draft model that owns
the MTP parameters. This bridge receives each distributed bucket once and
applies its tensors to both model loaders, matching the update semantics of
SGLang's ``update_weights_from_tensor`` speculative path.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from areal.utils import logging

logger = logging.getLogger("SGLangMTPBridge")

_SUPPORTED_SGLANG_VERSION = "0.5.10.post1"
_BOUND_MARKER = "_areal_mtp_distributed_update_bound"


def _sglang_version() -> str:
    try:
        return version("sglang").split("+", 1)[0]
    except PackageNotFoundError:
        return "unknown"


def _receive_named_tensors(
    model_runner: Any,
    names: list[str],
    dtypes: list[Any],
    shapes: list[Any],
    group_name: str,
    load_format: str | None,
) -> list[tuple[str, Any]]:
    """Receive one distributed bucket without applying it to a model."""
    if group_name not in model_runner._model_update_group:
        raise AssertionError(
            f"Group {group_name} not in "
            f"{list(model_runner._model_update_group.keys())}. "
            "Please call init_weights_update_group first."
        )
    if load_format not in (None, "flattened_bucket"):
        raise NotImplementedError(
            "AReaL's distributed MTP update bridge supports only the default "
            f"and flattened_bucket load formats, but received {load_format!r}."
        )

    import torch
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

    named_tensors = [
        (
            name,
            torch.empty(
                shape,
                dtype=dtype
                if isinstance(dtype, torch.dtype)
                else getattr(torch, dtype),
                device=model_runner.device,
            ),
        )
        for name, dtype, shape in zip(names, dtypes, shapes)
    ]
    process_group = model_runner._model_update_group[group_name]

    if load_format == "flattened_bucket":
        bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        torch.distributed.broadcast(
            bucket.get_flattened_tensor(),
            src=0,
            group=process_group,
        )
        return list(bucket.reconstruct_tensors())

    handles = [
        torch.distributed.broadcast(
            tensor,
            src=0,
            group=process_group,
            async_op=True,
        )
        for _, tensor in named_tensors
    ]
    for handle in handles:
        handle.wait()
    return named_tensors


def _get_draft_runner(scheduler: Any) -> Any | None:
    """Return the built-in MTP runner from SGLang spec v1 or v2."""
    speculative_worker = getattr(scheduler, "draft_worker", None)

    # Spec v1 uses EAGLEWorker, which subclasses TpModelWorker directly.
    runner = getattr(speculative_worker, "model_runner", None)
    if runner is not None:
        return runner

    # Spec v2 wraps EagleDraftWorker in EAGLEWorkerV2. The public property is
    # named draft_worker in 0.5.10.post1; _draft_worker is retained as a
    # defensive fallback for the same pinned implementation.
    draft_worker = getattr(speculative_worker, "draft_worker", None)
    if draft_worker is None:
        draft_worker = getattr(speculative_worker, "_draft_worker", None)
    return getattr(draft_worker, "draft_runner", None)


class MTPDistributedWeightUpdateBridge:
    """Add built-in MTP routing to SGLang's distributed weight update path.

    This is intentionally a narrow compatibility patch for SGLang
    0.5.10.post1. That version normalizes ``NEXTN`` to ``EAGLE`` before the
    scheduler is created, so a runtime ``EAGLE`` configuration without an
    external draft-model path also identifies the built-in MTP path. External
    EAGLE draft models and other speculative algorithms are left untouched.
    """

    def __init__(self, scheduler: Any, server_args: Any) -> None:
        self._scheduler = scheduler
        self._server_args = server_args

    def bind(self) -> None:
        algorithm = str(
            getattr(self._server_args, "speculative_algorithm", "") or ""
        ).upper()
        draft_model_path = getattr(
            self._server_args, "speculative_draft_model_path", None
        )
        uses_builtin_mtp = algorithm in {"NEXTN", "EAGLE"} and not draft_model_path
        if not uses_builtin_mtp:
            return

        target_worker = getattr(self._scheduler, "tp_worker", None)
        target_runner = getattr(target_worker, "model_runner", None)
        if target_runner is None:
            raise RuntimeError(
                "Built-in MTP speculative decoding is enabled, but SGLang's "
                "target ModelRunner could not be found; refusing to install "
                "the distributed MTP update bridge."
            )

        pp_size = int(getattr(self._server_args, "pp_size", 1))
        if pp_size != 1:
            raise NotImplementedError(
                "AReaL's distributed MTP weight-update bridge currently "
                f"requires SGLang pp_size=1, but found pp_size={pp_size}."
            )

        installed_version = _sglang_version()
        if installed_version != _SUPPORTED_SGLANG_VERSION:
            raise RuntimeError(
                "AReaL's distributed MTP weight-update bridge supports only "
                f"sglang=={_SUPPORTED_SGLANG_VERSION}, but found "
                f"sglang=={installed_version}. Revalidate the SGLang update API "
                "before enabling built-in MTP speculative decoding."
            )

        draft_runner = _get_draft_runner(self._scheduler)
        if draft_runner is None:
            raise RuntimeError(
                "Built-in MTP speculative decoding is enabled, but SGLang's "
                "draft ModelRunner could not be found; refusing to silently "
                "skip MTP updates."
            )

        if getattr(target_runner, _BOUND_MARKER, False):
            return

        self._bind_model_runner(target_runner, draft_runner)
        setattr(target_runner, _BOUND_MARKER, True)
        logger.info(
            "MTPDistributedWeightUpdateBridge bound for SGLang %s.",
            installed_version,
        )

    @staticmethod
    def _bind_model_runner(target_runner: Any, draft_runner: Any) -> None:
        def _update_weights_from_distributed(
            names,
            dtypes,
            shapes,
            group_name,
            load_format=None,
        ):
            try:
                named_tensors = _receive_named_tensors(
                    target_runner,
                    names,
                    dtypes,
                    shapes,
                    group_name,
                    load_format,
                )

                # Reuse the same application layer as SGLang's speculative
                # update_weights_from_tensor path. The tensors are already local,
                # so the transport-specific flattened_bucket format has been
                # consumed and both runners receive regular named tensors.
                success, message = draft_runner.update_weights_from_tensor(
                    named_tensors=named_tensors,
                    load_format=None,
                )
                if not success:
                    return success, message

                success, message = target_runner.update_weights_from_tensor(
                    named_tensors=named_tensors,
                    load_format=None,
                )
                if not success:
                    return success, message

                mtp_tensor_count = sum(
                    "mtp" in name.lower() for name, _ in named_tensors
                )
                return True, (
                    "Succeeded to update target parameters and "
                    f"{mtp_tensor_count} MTP draft tensors online."
                )
            except Exception as exc:
                message = (
                    f"Failed to update target and MTP draft parameters online: {exc}. "
                    "The model weights may be partially updated; discard the whole "
                    "rollout worker."
                )
                logger.error(message, exc_info=True)
                return False, message

        target_runner.update_weights_from_distributed = _update_weights_from_distributed
