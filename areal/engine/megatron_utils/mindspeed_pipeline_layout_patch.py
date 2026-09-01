# SPDX-License-Identifier: Apache-2.0
"""Compatibility fixes for MindSpeed pipeline-layout validation."""

import functools


def ensure_mindspeed_pipeline_layout_stage_count() -> bool:
    """Allow MindSpeed to count stages in MCore pipeline-layout objects."""
    try:
        from mindspeed.features_manager.moe.fb_overlap import (
            MoEFwdBwdOverlapFeature,
        )
    except ImportError:
        return False

    current = MoEFwdBwdOverlapFeature._get_pipeline_model_parallel_layout_stage_count
    if getattr(current, "_areal_mcore_layout_patched", False):
        return True

    @functools.wraps(current)
    def wrapper(layout):
        pp_size = getattr(layout, "pipeline_model_parallel_size", None)
        vpp_size = getattr(layout, "virtual_pipeline_model_parallel_size", None)
        if pp_size is not None and vpp_size is not None:
            return int(pp_size) * int(vpp_size)
        return current(layout)

    setattr(wrapper, "_areal_mcore_layout_patched", True)
    MoEFwdBwdOverlapFeature._get_pipeline_model_parallel_layout_stage_count = (
        staticmethod(wrapper)
    )
    return True
