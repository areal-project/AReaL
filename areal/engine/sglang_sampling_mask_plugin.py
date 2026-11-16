# SPDX-License-Identifier: Apache-2.0
"""Opt-in SGLang sampler plugin; never rewrites the installed SGLang package."""

import inspect
import os
from collections.abc import Callable
from functools import wraps
from importlib.metadata import entry_points, version
from typing import Any

BACKEND = "areal_sampling_mask"
PLUGIN = "areal_sampling_mask"
LEGACY_VERSION = "0.5.10.post1"
NATIVE_VERSIONS = ("0.5.19", "0.5.20")
_registered = False


def integration_version() -> str:
    installed = version("sglang")
    if installed not in (LEGACY_VERSION, *NATIVE_VERSIONS):
        raise RuntimeError(
            f"sampling_mask_transfer has no validated sampler adapter for SGLang {installed}; "
            f"expected {LEGACY_VERSION}, {', '.join(NATIVE_VERSIONS)}"
        )
    return installed


def prepare_launch(entrypoint: str) -> str:
    """Select native entry-point discovery or the old release's spawn wrapper."""
    installed = integration_version()
    os.environ["NCCL_CUMEM_ENABLE"] = "1"
    if installed == LEGACY_VERSION:
        if entrypoint not in (
            "sglang.launch_server",
            "areal.engine.awex.plugins.sglang_plugin",
        ):
            raise ValueError(
                "Legacy sampling mask launcher supports AReaL v1 HTTP only"
            )
        os.environ["AREAL_SAMPLING_MASK_ENTRYPOINT"] = entrypoint
        return "areal.engine.sglang_sampling_mask_launcher"
    allowed = os.environ.get("SGLANG_PLUGINS", "")
    if allowed and PLUGIN not in {name.strip() for name in allowed.split(",")}:
        raise RuntimeError(
            f"SGLANG_PLUGINS excludes required plugin {PLUGIN!r}; add it to the allowlist"
        )
    expected = "areal.engine.sglang_sampling_mask_plugin:register"
    if not any(
        ep.name == PLUGIN and ep.value == expected
        for ep in entry_points(group="sglang.srt.plugins")
    ):
        raise RuntimeError(
            "AReaL sampling mask entry point is not installed; "
            "reinstall this AReaL checkout in the rollout Python environment"
        )
    return entrypoint


def runtime_settings() -> tuple[str, int, int, Any]:
    """Read effective configuration, including .19/.20 resolution overrides."""
    if integration_version() == LEGACY_VERSION:
        from sglang.srt.server_args import get_global_server_args

        args = get_global_server_args()
        return (
            args.sampling_backend,
            args.pp_size,
            getattr(args, "attn_cp_size", 1),
            args.speculative_algorithm,
        )
    from sglang.srt.runtime_context import get_exec, get_parallel, get_spec

    return (
        get_exec().kernel.sampling_backend,
        get_parallel().pp_size,
        get_parallel().attn_cp_size,
        get_spec().speculative_algorithm,
    )


def preserve_resolved_backend(server_args: Any, previous: Callable) -> Any:
    """Keep deterministic setup, then restore the explicitly requested sampler."""
    from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view

    selected = resolving_view(server_args).sampling_backend
    result = previous(server_args)
    if selected == BACKEND:
        declare_resolution(server_args, PLUGIN, sampling_backend=BACKEND)
    return result


def _around_resolution(previous: Callable, server_args: Any) -> Any:
    return preserve_resolved_backend(server_args, previous)


def _install_legacy_hooks() -> None:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.server_args import ServerArgs

    original = ServerArgs._handle_deterministic_inference

    @wraps(original)
    def deterministic(self, *args, **kwargs):
        selected = self.sampling_backend
        result = original(self, *args, **kwargs)
        if selected == BACKEND:
            self.sampling_backend = BACKEND
        return result

    ServerArgs._handle_deterministic_inference = deterministic
    handle_output = TokenizerManager._handle_batch_output
    # A previous AReaL build may already have installed its source patch. Do
    # not accumulate twice, and do not modify that shared installation here.
    if "accumulate_reference_metadata" in inspect.getsource(handle_output):
        return

    @wraps(handle_output)
    def accumulate(self, recv_obj):
        info = getattr(recv_obj, "customized_info", None)
        if info and "sampling_mask_refs" in info:
            from areal.engine.sglang_sampling_mask import accumulate_reference_metadata

            combined = []
            for rid, increment in zip(
                recv_obj.rids, info["sampling_mask_refs"], strict=True
            ):
                state = self.rid_to_state.get(rid)
                metadata = {}
                if state is not None:
                    accumulate_reference_metadata(
                        state, metadata, "sampling_mask_refs", increment
                    )
                combined.append(metadata.get("sampling_mask_refs", increment))
            recv_obj.customized_info = {**info, "sampling_mask_refs": combined}
        return handle_output(self, recv_obj)

    TokenizerManager._handle_batch_output = accumulate


def register() -> None:
    """SGLang ``sglang.srt.plugins`` entry point, called in each process."""
    global _registered
    if os.environ.get("AREAL_SAMPLING_MASK_TRANSFER") != "1" or _registered:
        return
    try:
        installed = integration_version()
        from areal.engine.sglang_sampling_mask import register_backend

        register_backend()
        if installed == LEGACY_VERSION:
            _install_legacy_hooks()
        elif installed == "0.5.19":
            from sglang.srt.arg_groups.attention_hook import (
                handle_deterministic_inference,
            )
            from sglang.srt.plugins.hook_registry import HookRegistry, HookType

            # Resolve and validate before registering: the plugin loader logs
            # ordinary hook-application failures. Let the loader apply all
            # hooks together so later plugins can compose on this target.
            inspect.signature(handle_deterministic_inference).bind(object())
            HookRegistry.register(
                "sglang.srt.arg_groups.attention_hook.handle_deterministic_inference",
                _around_resolution,
                HookType.AROUND,
            )
        else:
            from sglang.srt.arg_groups.resolution_hooks import register_resolution_hook

            register_resolution_hook("handle_deterministic_inference")(
                preserve_resolved_backend
            )
        _registered = True
    except Exception as exc:
        # SGLang's native plugin loader logs and swallows Exception. This
        # plugin is required for policy correctness: stop instead of falling
        # back to a sampler which cannot publish supports.
        raise SystemExit(
            f"Cannot initialize AReaL sampling mask plugin: {exc}"
        ) from exc
