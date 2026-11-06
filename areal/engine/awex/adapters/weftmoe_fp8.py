# SPDX-License-Identifier: Apache-2.0
"""WeftMoE's ordinary AWEX wire tensors; AWEX itself remains unmodified."""

from __future__ import annotations

import torch
import torch.distributed as dist


def convert_fp8(converter, name, data, scales, **kwargs):
    # Reuse the exact gate/up/expert/PP mapping for both weight bytes and block
    # scales. This is bit reinterpretation, never FP8 -> integer conversion.
    yield from converter.convert_param(name, data.view(torch.uint8), **kwargs)
    for target, value in converter.convert_param(name, scales, **kwargs):
        yield target + "_scale_inv", value


def train_primary(engine):
    adapter = getattr(engine, "_weftmoe_adapter", None)
    return getattr(adapter, "master_weights", None)


def train_parameter(primary, parameter, *, metadata=False):
    if primary is None or id(parameter) not in primary.bindings:
        return None
    if metadata:
        rows, cols = parameter.shape
        return (
            torch.empty((rows, cols), dtype=torch.uint8, device="meta"),
            torch.empty((rows // 128, cols // 128), dtype=torch.float32, device="meta"),
        )
    weight = primary.primary(parameter)
    return weight.data, weight.scales


def train_meta_resolver(engine, shim, infer_conf):
    from awex.converter.mcore_converter import get_mcore_model_parameters
    from awex.meta.train_meta_resolver import (
        McoreParamMetaResolver,
        _build_pp_stage_layer_id_map,
        _canonicalize_pp_layer_names_in_global_meta,
    )
    from awex.models.registry import get_train_weights_converter

    primary = train_primary(engine)

    class Resolver(McoreParamMetaResolver):
        def _collect_model_param_raw_info(self, convert_params=False, **kwargs):
            if not convert_params:
                return super()._collect_model_param_raw_info(convert_params, **kwargs)
            converter = get_train_weights_converter(
                "mcore",
                self._model_arch_name,
                self.hf_config,
                self._rank_info,
                self._infer_conf,
                tf_config=self._tf_config,
            )
            params = []
            for stage, model in enumerate(self._mcore_model):
                for name, parameter in get_mcore_model_parameters(model).items():
                    fp8 = train_parameter(primary, parameter, metadata=True)
                    values = (
                        converter.convert_param(name, parameter, vp_stage=stage)
                        if fp8 is None
                        else convert_fp8(converter, name, *fp8, vp_stage=stage)
                    )
                    for target, tensor in values:
                        params.append(
                            dict(
                                name=target,
                                numel=tensor.numel(),
                                shape=tuple(tensor.shape),
                                dtype=tensor.dtype,
                                vp_stage=stage,
                            )
                        )
            if (
                getattr(self.hf_config, "tie_word_embeddings", False)
                and self._rank_info.pp_rank == self._rank_info.pp_size - 1
                and not any(p["name"] == "lm_head.weight" for p in params)
            ):
                for p in list(params):
                    if p["name"] == "model.embed_tokens.weight":
                        params.append({**p, "name": "lm_head.weight"})
            local = dict(
                rank_info=self._rank_info,
                params_meta=params,
                model_arch_name=self._model_arch_name,
            )
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local)
            self._pp_stage_layer_id_map = _build_pp_stage_layer_id_map(gathered)
            return _canonicalize_pp_layer_names_in_global_meta(
                gathered, self._pp_stage_layer_id_map
            )

    return Resolver(shim, engine.hf_config, infer_conf)


class WireModel:
    """Reader-only view: tensor names already use AWEX's canonical mapping."""

    def __init__(self, model, parameters):
        self.model = model
        self.wire_parameters = parameters

    def __getattr__(self, name):
        return getattr(self.model, name)

    def named_parameters(self):
        return self.wire_parameters.items()


class WireConverter:
    def convert_param(self, name, parameter):
        return [(name, parameter)]
