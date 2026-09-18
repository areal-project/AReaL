# SPDX-License-Identifier: Apache-2.0
"""Explicit Qwen4Exp AWEX mappings under development.

These factories do not register themselves or enable the engine's AWEX guard.
PLE table/buffer residency requires a separate contract;
reject them instead of silently inheriting an unrelated model's conversion.
"""

import re
from collections.abc import Callable, Mapping
from functools import cache, lru_cache

import torch
from torch import nn

from areal.models.mcore.qwen4_exp_awex_contract import Qwen4ExpFrozenContract
from areal.models.mcore.qwen4_exp_awex_layout import (
    Qwen4ExpGDNLayout,
    pack_qwen4_exp_gated_qkv,
)

_HC_WEIGHTS = (
    "hc_norm.weight",
    "input_mix_weight_down.weight",
    "input_mix_weight_up.weight",
    "block_inject_weight.weight",
)
_REPLICATED_LAYER_WEIGHTS = frozenset(
    f"{branch}.{weight}"
    for branch in ("attn_hyper_connection", "mlp_hyper_connection")
    for weight in _HC_WEIGHTS
) | frozenset(
    f"ple.{weight}"
    for weight in (
        "key_proj.weight",
        "value_proj.weight",
        "norm_key.weight",
        "norm_query.weight",
        "norm_conv.weight",
        "conv1d.weight",
    )
)
_QSA_WEIGHTS = frozenset(
    f"self_attn.indexer.{weight}"
    for weight in ("index_qk_proj.weight", "q_layernorm.weight", "k_layernorm.weight")
)
_REPLICATED_LAYER_WEIGHTS |= _QSA_WEIGHTS
_MIXER_WEIGHTS = frozenset(f"hyper_connection_mixer.{w}" for w in _HC_WEIGHTS[:3])


def _replicated_name(name: str) -> bool:
    if name.startswith("model.") and name[len("model.") :] in _MIXER_WEIGHTS:
        return True
    match = re.fullmatch(r"model\.layers\.\d+\.(.+)", name)
    return bool(match and match[1] in _REPLICATED_LAYER_WEIGHTS)


def _reject_pending_contract(name: str) -> None:
    if ".ple_embedding." in name:
        raise NotImplementedError(
            f"Qwen4Exp AWEX requires an explicit table/buffer contract: {name}"
        )
    if "hyper_connection" in name or ".ple." in name or ".indexer." in name:
        if not _replicated_name(name):
            raise NotImplementedError(f"Unknown Qwen4Exp replicated weight: {name}")


def _refresh_frozen_binding(converter, binder: Callable[[object], None] | None) -> None:
    if binder is None:
        raise ValueError("No Qwen4Exp frozen-contract binder was registered")
    # Invalidate first: a failed refresh must never leave old Parameter references
    # usable after model recovery or replacement.
    for attribute in (
        "_qwen4_frozen_contract",
        "_qwen4_original_parameters",
        "_qwen4_local_table_names",
        "_qwen4_preserved_visual_names",
    ):
        converter.__dict__.pop(attribute, None)
    try:
        binder(converter)
        if getattr(converter, "_qwen4_frozen_contract", None) is None:
            raise ValueError("Qwen4Exp binder did not bind a frozen contract")
    except Exception:
        converter.__dict__.pop("_qwen4_frozen_contract", None)
        raise


@cache
def build_mcore_converter(binder: Callable[[object], None] | None = None):
    from awex.converter.mcore_converter import _process_mcore_pp_name
    from awex.models.qwen3_5 import _MCORE_CONVERTER_FACTORY

    base = _MCORE_CONVERTER_FACTORY()

    class McoreToHFWeightConverterQwen4Exp(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if binder is not None:
                self.refresh_frozen_contract()

        def refresh_frozen_contract(self) -> None:
            _refresh_frozen_binding(self, binder)

        def bind_frozen_contract(
            self,
            contract: Qwen4ExpFrozenContract,
            parameters: Mapping[str, nn.Parameter],
            local_table_names: frozenset[str],
        ) -> None:
            contract.validate_actor_parameters(parameters, local_table_names)
            self._qwen4_frozen_contract = contract
            self._qwen4_original_parameters = parameters
            self._qwen4_local_table_names = local_table_names

        def _convert_attention_param(self, name, parameter, layer_number):
            if name in (
                "self_attention.linear_qkv.weight",
                "self_attention.linear_qkv.bias",
            ):
                cfg = self.hf_config
                packed = pack_qwen4_exp_gated_qkv(
                    self._full_tp_tensor(parameter),
                    int(cfg.num_attention_heads),
                    int(cfg.num_key_value_heads),
                    int(cfg.head_dim),
                    int(self.infer_atten_tp_size),
                )
                suffix = name.rsplit(".", 1)[1]
                return [
                    (f"self_attn.qkv_proj.{suffix}", self._take_train_tp_shard(packed))
                ]
            if name == "self_attention.A_log":
                # Native Qwen4Exp SGLang keeps A_log FP32; metadata and payload
                # must describe the same converted dtype on the writer side.
                return [("linear_attn.A_log", parameter.float())]
            if name == "self_attention.out_norm.weight":
                # FlashNext bridge uses ones-style GDN norm: no +1 offset.
                return [("linear_attn.norm.weight", parameter)]
            if name in (
                "self_attention.in_proj.weight",
                "self_attention.conv1d.weight",
                "self_attention.in_proj_qkvz.weight",
                "self_attention.in_proj_ba.weight",
            ):
                cfg = self.hf_config
                layout = Qwen4ExpGDNLayout(
                    cfg.linear_num_key_heads,
                    cfg.linear_num_value_heads,
                    cfg.linear_key_head_dim,
                    cfg.linear_value_head_dim,
                )
                train_tp = int(self.rank_info.attn_tp_size)
                infer_tp = int(self.infer_atten_tp_size)
                full = self._full_tp_tensor(parameter)
                if name in (
                    "self_attention.in_proj_qkvz.weight",
                    "self_attention.in_proj_ba.weight",
                ):
                    component = "qkvz" if "qkvz" in name else "ba"
                    packed = layout.pack_decoupled(full, train_tp, infer_tp, component)
                    return [
                        (
                            f"linear_attn.in_proj_{component}.weight",
                            self._take_train_tp_shard(packed),
                        )
                    ]
                if name == "self_attention.in_proj.weight":
                    qkvz, ba = layout.pack_input(full, train_tp, infer_tp)
                    return [
                        (
                            "linear_attn.in_proj_qkvz.weight",
                            self._take_train_tp_shard(qkvz),
                        ),
                        (
                            "linear_attn.in_proj_ba.weight",
                            self._take_train_tp_shard(ba),
                        ),
                    ]
                packed = layout.pack_conv(full, train_tp, infer_tp)
                return [
                    ("linear_attn.conv1d.weight", self._take_train_tp_shard(packed))
                ]
            return super()._convert_attention_param(name, parameter, layer_number)

        @torch.no_grad()
        def convert_param(self, name, parameter, vp_stage=None):
            clean = name.replace("module.", "")
            if clean.startswith("language_model."):
                clean = clean[len("language_model.") :]
            if clean.startswith("decoder."):
                global_name = _process_mcore_pp_name(
                    clean,
                    self.rank_info,
                    self.hf_config,
                    self.tf_config,
                    vp_stage=vp_stage,
                    pp_stage_layer_id_map=self._pp_stage_layer_id_map,
                )
                canonical = "model." + global_name[len("decoder.") :]
                canonical = canonical.replace(
                    ".self_attention.indexer.", ".self_attn.indexer."
                )
                contract = getattr(self, "_qwen4_frozen_contract", None)
                if contract is not None and contract.excludes(canonical, "actor"):
                    contract.validate_actor_parameters(
                        self._qwen4_original_parameters, self._qwen4_local_table_names
                    )
                    return []
                _reject_pending_contract(canonical)
                if _replicated_name(canonical):
                    return [(canonical, parameter)]
                if canonical == "model.final_layernorm.weight":
                    raise ValueError(
                        "Qwen4Exp has a final HC mixer, not a final layernorm"
                    )
            # Let the parent apply PP numbering once for inherited attention/MLP.
            return super().convert_param(name, parameter, vp_stage=vp_stage)

    return McoreToHFWeightConverterQwen4Exp


@cache
def build_sglang_converter(binder: Callable[[object], None] | None = None):
    from awex.models.qwen3_5 import SGlangToHFWeightConverterQwen3_5

    class SGlangToHFWeightConverterQwen4Exp(SGlangToHFWeightConverterQwen3_5):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if binder is not None:
                self.refresh_frozen_contract()

        def refresh_frozen_contract(self) -> None:
            _refresh_frozen_binding(self, binder)

        def bind_frozen_contract(
            self,
            contract: Qwen4ExpFrozenContract,
            parameters: Mapping[str, nn.Parameter],
            preserved_visual_names: frozenset[str],
        ) -> None:
            contract.validate_inference_parameters(parameters, preserved_visual_names)
            self._qwen4_frozen_contract = contract
            self._qwen4_original_parameters = parameters
            self._qwen4_preserved_visual_names = preserved_visual_names

        @torch.no_grad()
        def convert_param(self, name, parameter):
            canonical = name.replace("model.language_model.", "model.")
            canonical = re.sub(
                r"^(model\.layers\.\d+)\.indexer\.", r"\1.self_attn.indexer.", canonical
            )
            if canonical.startswith("visual."):
                canonical = "model." + canonical
            contract = getattr(self, "_qwen4_frozen_contract", None)
            if contract is not None and contract.excludes(canonical, "inference"):
                contract.validate_inference_parameters(
                    self._qwen4_original_parameters, self._qwen4_preserved_visual_names
                )
                return []
            _reject_pending_contract(canonical)
            if _replicated_name(canonical):
                return [(canonical, parameter)]
            return super().convert_param(name, parameter)

    return SGlangToHFWeightConverterQwen4Exp


@lru_cache(maxsize=1)
def build_sharding_strategy():
    from awex.models.qwen3_5 import Qwen3_5ShardingStrategy
    from awex.sharding.param_sharding import ShardingType

    class Qwen4ExpShardingStrategy(Qwen3_5ShardingStrategy):
        def get_sharding_strategy(self, parameter_name, **kwargs):
            _reject_pending_contract(parameter_name)
            if _replicated_name(parameter_name):
                return ShardingType.NO_SHARDING, 0, 1
            return super().get_sharding_strategy(parameter_name, **kwargs)

    return Qwen4ExpShardingStrategy


def register_qwen4_exp_awex(
    *,
    mcore_binder: Callable[[object], None] | None = None,
    sglang_binder: Callable[[object], None] | None = None,
) -> None:
    """Register explicit factories after AWEX has finished rebuilding its registry.

    Optional process-local, hashable binders run on every native construction,
    including metadata resolvers and payload converters. They must obtain fresh
    original Parameters and call bind_frozen_contract. Reuse the same callback
    when registering again; call refresh_frozen_contract before each transfer
    and after model recovery. Bindings are not transported between processes.

    Registration alone does not enable unsupported frozen-state exclusions or
    remove the engine guard. Refuse to replace an unrelated upstream adapter.
    """
    from awex.models.registry import ModelRegistry

    architecture = "Qwen4ExpForConditionalGeneration"
    entry = {
        "model_name": architecture,
        "mcore_converter": build_mcore_converter
        if mcore_binder is None
        else build_mcore_converter(mcore_binder),
        "sglang_converter": build_sglang_converter
        if sglang_binder is None
        else build_sglang_converter(sglang_binder),
        "sharding_strategy": build_sharding_strategy(),
    }
    existing = ModelRegistry.models.get(architecture)
    if existing is not None and existing != entry:
        raise ValueError("A different Qwen4Exp AWEX adapter is already registered")
    ModelRegistry.models[architecture] = entry
