# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers shared by v1 and v2 SGLang AWEX adapters."""

from __future__ import annotations


def patch_tms_hook_mode() -> None:
    """Ignore unsupported hook-mode assignments after TMS initialization."""
    try:
        import torch_memory_saver as tms
    except Exception:
        return
    instance = getattr(tms, "torch_memory_saver", None)
    if instance is None:
        return
    cls = type(instance)
    prop = cls.hook_mode
    if getattr(prop.fset, "_awex_safe", False):
        return

    def safe_setter(self, value):
        if not hasattr(self, "_impl_ctor_kwargs"):
            return
        prop.fset(self, value)

    safe_setter._awex_safe = True
    cls.hook_mode = property(prop.fget, safe_setter)


# AWEX model-registry imports transitively touch Megatron/TMS, so patch first.
patch_tms_hook_mode()

from awex.meta.meta_resolver import ParamMetaResolver  # noqa: E402
from awex.sharding import get_sharding_strategy_builder  # noqa: E402

from areal.utils import logging  # noqa: E402

logger = logging.getLogger("AwexSGLangCompat")


def ensure_awex_models_registered() -> None:
    """Rebuild a registry cached before the TMS compatibility patch ran."""
    try:
        from awex.models import registry

        registry.import_model_configs.cache_clear()
        registry.ModelRegistry.models = registry.import_model_configs()
        missing = [
            model
            for model in (
                "BailingMoeV2_5ForCausalLM",
                "BailingMoeV2ForCausalLM",
                "Qwen3VLForConditionalGeneration",
                "Qwen3VLMoeForConditionalGeneration",
            )
            if model not in registry.ModelRegistry.models
        ]
        if missing:
            logger.warning("AWEX model registry still missing converters: %s", missing)
    except Exception as exc:  # pragma: no cover - diagnostic guard
        logger.warning("Failed to rebuild AWEX model registry: %s", exc)


ensure_awex_models_registered()


class SingleInstanceMetaResolver(ParamMetaResolver):
    """Aggregate raw metadata for one inference engine instance."""

    def __init__(self, hf_config, engine_name, infer_engine_config, raw_meta_list):
        super().__init__(hf_config)
        self._raw_meta_list = raw_meta_list
        rank0 = self._select_rank0(raw_meta_list)
        self._model_arch_name = rank0["model_arch_name"]
        self._sharding_strategy = get_sharding_strategy_builder(engine_name)(
            self._model_arch_name,
            infer_engine_config,
            rank0["rank_info"],
        )

    @staticmethod
    def _select_rank0(raw_meta_list):
        for info in raw_meta_list:
            if info["rank_info"].global_rank == 0:
                return info
        return raw_meta_list[0]

    def get_model_arch_name(self) -> str:
        return self._model_arch_name

    def get_parameters_meta(self):
        return self._build_params_meta()

    def _get_params_raw_meta(self):
        return self._raw_meta_list

    def _get_sharding_info(self, name, rank_info, param_meta):
        return self._sharding_strategy.get_sharding_strategy(
            name, rank_info=rank_info, param_meta=param_meta
        )
