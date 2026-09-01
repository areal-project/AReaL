# SPDX-License-Identifier: Apache-2.0
"""Runtime fix-ups for MindSpeed's packed-capable GatedDeltaNet."""

import importlib

from areal.utils import logging

logger = logging.getLogger("MegatronEngine")


def has_mindspeed_gdn_model_classes(*gdn_classes: type) -> bool:
    """Return whether all captured GDN classes come from MindSpeed."""
    return all(cls.__module__.startswith("mindspeed") for cls in gdn_classes)


def has_mindspeed_gdn_conv1d(mindspeed_gdn: object) -> bool:
    """Return whether MindSpeed's GDN module has a causal conv implementation."""
    return getattr(mindspeed_gdn, "causal_conv1d", None) is not None


def ensure_mindspeed_gdn_conv1d() -> bool:
    """Bind MindSpeed's NPU causal conv into the inherited MCore GDN.

    Returns True when a varlen conv implementation is available afterwards.
    """
    try:
        mcore_gdn = importlib.import_module("megatron.core.ssm.gated_delta_net")
        ms_gdn = importlib.import_module("mindspeed.core.ssm.gated_delta_net")
    except ImportError:
        return False

    try:
        npu_conv = importlib.import_module(
            "mindspeed.core.ssm.npu_causal_conv1d"
        ).causal_conv1d
    except (AttributeError, ImportError):
        return False

    if (
        getattr(mcore_gdn, "causal_conv1d", None) is npu_conv
        and getattr(ms_gdn, "causal_conv1d", None) is npu_conv
    ):
        return True

    mcore_gdn.causal_conv1d = npu_conv
    ms_gdn.causal_conv1d = npu_conv
    logger.info("Activated MindSpeed's NPU varlen causal_conv1d for GDN.")
    return True
