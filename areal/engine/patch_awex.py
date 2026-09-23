# SPDX-License-Identifier: Apache-2.0

from functools import wraps


def _patch_vllm_expert_names():
    from awex.converter.vllm_converter import VLLMToHFWeightConverter

    original = VLLMToHFWeightConverter._normalize_name
    if getattr(original, "_areal_expert_names", False):
        return

    @wraps(original)
    def normalize_name(self, name: str) -> str:
        # vLLM 0.26 nests fused expert parameters under RoutedExperts.
        name = name.replace(".experts.routed_experts.", ".experts.")
        return original(self, name)

    normalize_name._areal_expert_names = True
    VLLMToHFWeightConverter._normalize_name = normalize_name


def patch_awex():
    _patch_vllm_expert_names()
