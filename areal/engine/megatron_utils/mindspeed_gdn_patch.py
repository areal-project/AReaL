# SPDX-License-Identifier: Apache-2.0
"""Runtime fix-ups for MindSpeed's packed-capable GatedDeltaNet."""

import importlib
import inspect

from areal.utils import logging

logger = logging.getLogger("MegatronEngine")


def ensure_mindspeed_gdn_native_kkt() -> bool:
    """Install AscendC KKT in the selected MindSpeed FLA-NPU backend.

    Return False when that backend is unavailable or not selected. An incompatible
    selected backend raises before changing any bindings. Installation is local
    to this process; dependency files, autograd and CP communication are unchanged.
    """
    try:
        backend = importlib.import_module("mindspeed.core.ssm.gdn_backend")
    except ImportError:
        return False
    if backend.load_gdn_operators().name != "fla_npu":
        return False

    from areal.engine.megatron_utils.mindspeed_gdn_native import (
        make_native_kkt_forward,
    )

    try:
        owner = importlib.import_module("mindspeed.core.ssm.npu_chunk_gated_delta_rule")
        if getattr(owner, "_NATIVE_KKT", False):
            logger.info("Using MindSpeed's native KKT integration.")
            return True
        original = owner._chunk_gated_delta_rule_fwd
        if getattr(original, "_areal_native_kkt", False):
            return True
        native = importlib.import_module("fla_npu.ops.ascendc")
        required = (
            (native, "npu_chunk_scaled_dot_kkt"),
            (owner, "chunk_local_cumsum"),
            (owner, "npu_solve_tri"),
            (owner, "npu_recompute_w_u_fwd"),
            (owner, "npu_chunk_gated_delta_rule_fwd_h"),
            (owner, "npu_chunk_fwd_o"),
        )
        for module, name in required:
            if not callable(getattr(module, name, None)):
                raise AttributeError(f"Missing callable {name}")
        forward = make_native_kkt_forward(owner, native)
        expected = inspect.signature(forward).parameters
        actual = inspect.signature(original).parameters
        if tuple(actual) != tuple(expected) or any(
            actual[name].kind != expected[name].kind
            or actual[name].default != expected[name].default
            for name in expected
        ):
            raise TypeError(
                f"Unexpected GDN forward signature: {inspect.signature(original)}"
            )
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Cannot install AReaL AscendC KKT: the selected MindSpeed/FLA-NPU "
            f"backend is incompatible ({exc})."
        ) from exc

    owner._chunk_gated_delta_rule_fwd = forward
    logger.info("Activated AReaL AscendC KKT for MindSpeed GDN (dense and packed).")
    return True


def has_mindspeed_gdn_model_classes(*gdn_classes: type) -> bool:
    """Return whether all captured GDN classes come from MindSpeed."""
    return all(cls.__module__.startswith("mindspeed") for cls in gdn_classes)


def has_mindspeed_gdn_conv1d(mindspeed_gdn: object) -> bool:
    """Return whether MindSpeed's GDN module has a causal conv implementation."""
    return getattr(mindspeed_gdn, "causal_conv1d", None) is not None


def ensure_mindspeed_gdn_conv1d() -> bool:
    """Bind the selected MindSpeed backend's causal conv into MCore GDN.

    Returns True when a varlen conv implementation is available afterwards.
    """
    try:
        mcore_gdn = importlib.import_module("megatron.core.ssm.gated_delta_net")
        ms_gdn = importlib.import_module("mindspeed.core.ssm.gated_delta_net")
    except ImportError:
        return False

    try:
        backend = importlib.import_module("mindspeed.core.ssm.gdn_backend")
        npu_conv = backend.load_gdn_operators().causal_conv1d
    except (AttributeError, ImportError):
        return False

    if (
        getattr(mcore_gdn, "causal_conv1d", None) is npu_conv
        and getattr(ms_gdn, "causal_conv1d", None) is npu_conv
    ):
        return True

    mcore_gdn.causal_conv1d = npu_conv
    ms_gdn.causal_conv1d = npu_conv
    logger.info("Activated the selected MindSpeed GDN causal_conv1d backend.")
    return True
