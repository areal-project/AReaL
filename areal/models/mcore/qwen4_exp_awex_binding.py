# SPDX-License-Identifier: Apache-2.0
"""Bind frozen exclusions to live MCore Parameters and actual PP ownership."""

import re
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from awex.models.qwen4_exp_contract import (
    Qwen4ExpFrozenContract,
    mcore_visual_parameter_name,
)
from torch import nn

from areal.models.mcore.qwen4_exp_awex_contract import (
    FrozenCheckpoint,
    check_frozen_proof,
    visual_segments,
)


class McoreFrozenBinder:
    """Process-local callback; fetch model objects again on every invocation.

    This validates frozen exclusions only, not trainable weight coverage. Native
    metadata conversion retains local layer numbering. The currently validated
    PLE placement has matching local/global IDs; other placements fail explicitly
    until the converter supports a separate global identity for exclusions.
    """

    def __init__(self, engine: Any, contract: Qwen4ExpFrozenContract) -> None:
        self.engine = engine
        self.contract = contract
        self.checkpoint = FrozenCheckpoint(
            Path(engine.config.path), contract.language_model_only
        )
        if self.checkpoint.contract != contract:
            raise ValueError("Actor frozen checkpoint description changed")
        self._proof = None
        self._validated_identity = None
        self._embeddings = {}

    def invalidate(self) -> None:
        """Checkpoint loads may overwrite the same Parameter objects in place."""
        self._validated_identity = None

    def _collect(self, pp_rank: int, stage_map: dict | None):
        config = self.engine.mcore_config
        if (
            config.language_model_only is not self.contract.language_model_only
            or config.freeze_ple_table is not True
        ):
            raise ValueError(
                "Frozen binding requires matching actor mode and frozen PLE"
            )
        if self.engine.hf_config.architectures != ["Qwen4ExpForConditionalGeneration"]:
            raise ValueError("Frozen binding requires the Qwen4Exp architecture")
        models = self.engine.model
        if not isinstance(models, (list, tuple)) or not models:
            raise ValueError("Expected initialized MCore model chunks")
        parameters: dict[str, nn.Parameter] = {}
        global_layers: set[int] = set()
        visual_owners = 0
        local_visual_names: set[str] = set()
        self._embeddings = {}
        for vp_stage, model in enumerate(models):
            unwrapped = model
            while hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module
            owns_visual = False
            if not self.contract.language_model_only:
                if type(getattr(unwrapped, "pre_process", None)) is not bool:
                    raise ValueError(
                        "Vision binding requires explicit chunk pre_process ownership"
                    )
                owns_visual = unwrapped.pre_process
                if owns_visual and (pp_rank != 0 or vp_stage != 0):
                    raise ValueError(
                        "Frozen visual owner must be the first PP/VP stage"
                    )
                visual_owners += int(owns_visual)
            chunk_visual_names: set[str] = set()
            layers: dict[str, tuple[int, int]] = {}
            for path, layer in model.named_modules():
                clean = _clean_name(path)
                match = re.fullmatch(r"decoder\.layers\.(\d+)", clean)
                if match is None:
                    continue
                number = getattr(layer, "layer_number", None)
                if type(number) is not int or number < 1:
                    raise ValueError(f"Missing actual global layer identity: {path}")
                local_id, global_id = int(match[1]), number - 1
                if global_id in global_layers:
                    raise ValueError(
                        f"Duplicate global layer across model chunks: {global_id}"
                    )
                global_layers.add(global_id)
                layers[path] = (local_id, global_id)
                if stage_map:
                    mapped = stage_map.get((pp_rank, vp_stage), {}).get(local_id)
                    if mapped != global_id:
                        raise ValueError(
                            f"AWEX PP map disagrees with actual layer: {path}"
                        )
            if not layers:
                raise ValueError("MCore chunk has no identifiable decoder layers")
            for name, parameter in model.named_parameters():
                clean = _clean_name(name)
                if clean.startswith("visual."):
                    if self.contract.language_model_only or not owns_visual:
                        raise ValueError(
                            "Unexpected actor visual parameters on this chunk"
                        )
                    canonical = mcore_visual_parameter_name(name, self.contract)
                    if canonical in parameters:
                        raise ValueError(
                            f"Duplicate frozen parameter identity: {canonical}"
                        )
                    parameters[canonical] = parameter
                    chunk_visual_names.add(canonical)
                    continue
                if ".ple_embedding." not in clean:
                    continue
                match = re.fullmatch(r"(.+\.layers\.\d+)\.(.+)", name)
                if match is None or match[1] not in layers:
                    raise ValueError(f"PLE parameter has no actual layer owner: {name}")
                local_id, global_id = layers[match[1]]
                canonical = f"model.layers.{global_id}.{match[2]}"
                if stage_map == {} and local_id != global_id:
                    raise ValueError(
                        "Metadata PLE local/global identity differs; unsupported placement"
                    )
                if canonical in parameters:
                    raise ValueError(
                        f"Duplicate frozen parameter identity: {canonical}"
                    )
                parameters[canonical] = parameter
                self._embeddings[canonical] = model.get_submodule(
                    name.rsplit(".", 1)[0]
                )
            if (
                owns_visual
                and chunk_visual_names != self.contract.visual_parameter_names
            ):
                raise ValueError(
                    "Owning actor chunk must contain the complete visual tower"
                )
            local_visual_names.update(chunk_visual_names)
        if not self.contract.language_model_only:
            if visual_owners != int(pp_rank == 0):
                raise ValueError("Missing or duplicate frozen visual PP owner")
        expected = frozenset(
            name
            for name in self.contract.ple_table_names
            if int(name.split(".")[2]) in global_layers
        )
        self.contract.validate_actor_parameters(
            parameters, expected, frozenset(local_visual_names)
        )
        return parameters, expected, frozenset(local_visual_names)

    def _verify(self, parameters: dict) -> dict:
        identity = tuple(
            (n, id(p), tuple(p.shape), p.dtype, p.requires_grad)
            for n, p in parameters.items()
        )
        if identity != self._validated_identity:
            proof = {}
            for name, parameter in parameters.items():
                if name in self._embeddings:
                    proof.update(
                        self.checkpoint.verify_table(
                            name, self._embeddings[name], "actor"
                        )
                    )
                else:
                    shape = self.checkpoint.shape(name)
                    if tuple(parameter.shape) != shape:
                        raise ValueError(
                            f"Frozen actor visual shape differs from checkpoint: {name}"
                        )
                    proof[name] = self.checkpoint.verify_source(
                        name, parameter, [(0, 0, shape[0], 0)]
                    )
            if self._proof is not None:
                check_frozen_proof(proof, self._proof)
                if proof.keys() != self._proof.keys():
                    raise ValueError(
                        "Frozen actor ownership changed after initialization"
                    )
            self._proof = proof
            self._validated_identity = identity
        return self._proof

    def verify_loaded(self) -> dict:
        parameters, _, _ = self._collect(self.engine.pipeline_parallel_rank, None)
        proof = self._verify(parameters)
        ranges = {
            name: [emb.vocab_start_index, emb.vocab_end_index]
            for name, emb in self._embeddings.items()
        }
        return {"contract": self.contract.to_dict(), "proof": proof, "ranges": ranges}

    def __call__(self, converter: Any) -> None:
        parameters, expected, visual = self._collect(
            converter.rank_info.pp_rank, converter._pp_stage_layer_id_map
        )
        self._verify(parameters)
        converter.bind_frozen_contract(self.contract, parameters, expected, visual)


def _clean_name(name: str) -> str:
    while name.startswith("module."):
        name = name[len("module.") :]
    if name.startswith("language_model."):
        name = name[len("language_model.") :]
    return name


def actor_frozen_binder(engine: Any) -> McoreFrozenBinder:
    """Reuse the verification across eager publication and converter binding."""
    binder = getattr(engine, "_qwen4_awex_frozen_binder", None)
    if binder is None:
        if (
            engine.bridge_cls != "mcore-bridge"
            or engine.mcore_config.freeze_ple_table is not True
        ):
            raise ValueError("Qwen4Exp AWEX requires mcore-bridge and frozen PLE")
        checkpoint = FrozenCheckpoint(
            Path(engine.config.path), engine.mcore_config.language_model_only
        )
        binder = McoreFrozenBinder(engine, checkpoint.contract)
        engine._qwen4_awex_frozen_binder = binder
    return binder


def build_awex_train_info(engine: Any, world_size: int) -> dict[str, Any]:
    info: dict[str, Any] = {"train_world_size": world_size}
    if engine.hf_config.architectures == ["Qwen4ExpForConditionalGeneration"]:
        binder = actor_frozen_binder(engine)
        info["qwen4_exp_frozen_contract"] = binder.contract.to_dict()
        info["qwen4_exp_frozen_proof"] = engine._qwen4_awex_frozen_proof
    return info


class SglangFrozenBinder:
    """Bind current original inference Parameters only with preservation installed."""

    def __init__(
        self,
        get_model: Callable[[], nn.Module],
        contract: Qwen4ExpFrozenContract,
        weight_updater: ModuleType,
        checkpoint: FrozenCheckpoint,
        expected_proof: dict,
    ) -> None:
        self.get_model = get_model
        self.contract = contract
        self.weight_updater = weight_updater
        self.checkpoint = checkpoint
        self.expected_proof = expected_proof
        self._validated_identity = None
        self._hooked_model = None

    def invalidate(self, *args, **kwargs) -> None:
        self._validated_identity = None

    def verify_loaded(self):
        model = self.get_model()
        if type(model).__name__ != "Qwen4ExpForConditionalGeneration":
            raise ValueError("Frozen inference binding requires Qwen4Exp")
        if not getattr(self.weight_updater, "_areal_qwen4_exp_static_hooks", False):
            raise ValueError(
                "Qwen4Exp visual preservation must be installed before release"
            )
        parameters = {}
        owners = {}
        visual_names = set()
        for name, parameter in model.named_parameters():
            if name.startswith("visual."):
                canonical = "model." + name
                visual_names.add(canonical)
            elif ".ple_embedding." in name:
                canonical = name.replace("model.language_model.", "model.", 1)
            else:
                continue
            if canonical in parameters:
                raise ValueError(f"Duplicate frozen inference identity: {canonical}")
            parameters[canonical] = parameter
            owners[canonical] = model.get_submodule(name.rsplit(".", 1)[0])
        self.contract.validate_inference_parameters(parameters, frozenset(visual_names))
        self._verify(model, parameters, owners)
        return parameters, frozenset(visual_names)

    def __call__(self, converter: Any) -> None:
        parameters, visual_names = self.verify_loaded()
        converter.bind_frozen_contract(self.contract, parameters, visual_names)

    def _verify(self, model: nn.Module, parameters: dict, owners: dict) -> None:
        identity = tuple(
            (n, id(p), tuple(p.shape), p.dtype, p.requires_grad)
            for n, p in parameters.items()
        )
        if identity == self._validated_identity:
            return
        proof = {}
        for name, parameter in parameters.items():
            if name in self.contract.ple_table_names:
                proof.update(
                    self.checkpoint.verify_table(name, owners[name], "inference")
                )
            else:
                shape = self.checkpoint.shape(name)
                segments, local_shape = visual_segments(owners[name], name, shape)
                if tuple(parameter.shape) != local_shape:
                    raise ValueError(
                        f"Frozen inference visual shape differs from checkpoint: {name}"
                    )
                proof[name] = self.checkpoint.verify_source(name, parameter, segments)
        check_frozen_proof(proof, self.expected_proof)
        if self._hooked_model is not model:
            model.register_load_state_dict_pre_hook(self.invalidate)
            if hasattr(model, "load_weights"):
                original = model.load_weights

                def load_weights(*args, **kwargs):
                    self.invalidate()
                    return original(*args, **kwargs)

                model.load_weights = load_weights
            self._hooked_model = model
        self._validated_identity = identity
