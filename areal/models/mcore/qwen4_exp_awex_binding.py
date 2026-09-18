# SPDX-License-Identifier: Apache-2.0
"""Bind frozen exclusions to live MCore Parameters and actual PP ownership."""

import os
import re
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from torch import nn

from areal.models.mcore.qwen4_exp_awex_contract import Qwen4ExpFrozenContract


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

    def __call__(self, converter: Any) -> None:
        config = self.engine.mcore_config
        if (
            config.language_model_only is not True
            or config.freeze_ple_table is not True
        ):
            raise ValueError(
                "Frozen binding requires language-only actor and frozen PLE"
            )
        if self.engine.hf_config.architectures != ["Qwen4ExpForConditionalGeneration"]:
            raise ValueError("Frozen binding requires the Qwen4Exp architecture")
        models = self.engine.model
        if not isinstance(models, (list, tuple)) or not models:
            raise ValueError("Expected initialized MCore model chunks")
        parameters: dict[str, nn.Parameter] = {}
        global_layers: set[int] = set()
        stage_map = converter._pp_stage_layer_id_map
        for vp_stage, model in enumerate(models):
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
                    mapped = stage_map.get(
                        (converter.rank_info.pp_rank, vp_stage), {}
                    ).get(local_id)
                    if mapped != global_id:
                        raise ValueError(
                            f"AWEX PP map disagrees with actual layer: {path}"
                        )
            if not layers:
                raise ValueError("MCore chunk has no identifiable decoder layers")
            for name, parameter in model.named_parameters():
                clean = _clean_name(name)
                if clean.startswith("visual."):
                    raise ValueError(
                        "Language-only actor unexpectedly contains visual parameters"
                    )
                if ".ple_embedding." not in clean:
                    continue
                match = re.fullmatch(r"(.+\.layers\.\d+)\.(.+)", name)
                if match is None or match[1] not in layers:
                    raise ValueError(f"PLE parameter has no actual layer owner: {name}")
                local_id, global_id = layers[match[1]]
                canonical = f"model.layers.{global_id}.{match[2]}"
                if not stage_map and local_id != global_id:
                    raise ValueError(
                        "Metadata PLE local/global identity differs; unsupported placement"
                    )
                if canonical in parameters:
                    raise ValueError(
                        f"Duplicate frozen parameter identity: {canonical}"
                    )
                parameters[canonical] = parameter
        expected = frozenset(
            name
            for name in self.contract.ple_table_names
            if int(name.split(".")[2]) in global_layers
        )
        converter.bind_frozen_contract(self.contract, parameters, expected)


def _clean_name(name: str) -> str:
    while name.startswith("module."):
        name = name[len("module.") :]
    if name.startswith("language_model."):
        name = name[len("language_model.") :]
    return name


def load_actor_frozen_contract(engine: Any) -> Qwen4ExpFrozenContract:
    """Load explicit local evidence without enabling the engine AWEX guard."""
    from areal.models.mcore.qwen4_exp_awex_contract import load_frozen_contract

    if engine.bridge_cls != "mcore-bridge":
        raise ValueError("Qwen4Exp frozen contract requires mcore-bridge")
    if (
        engine.mcore_config.language_model_only is not True
        or engine.mcore_config.freeze_ple_table is not True
    ):
        raise ValueError("Frozen binding requires language-only actor and frozen PLE")
    manifest = os.environ.get("QWEN_AWEX_FROZEN_CONTRACT")
    if not manifest:
        raise ValueError("Qwen4Exp AWEX requires QWEN_AWEX_FROZEN_CONTRACT")
    return load_frozen_contract(Path(manifest), Path(engine.config.path))


def build_awex_train_info(engine: Any, world_size: int) -> dict[str, Any]:
    """Use one payload for eager publication and adapter initialization."""
    info: dict[str, Any] = {"train_world_size": world_size}
    if engine.hf_config.architectures == ["Qwen4ExpForConditionalGeneration"]:
        info["qwen4_exp_frozen_contract"] = load_actor_frozen_contract(engine).to_dict()
    return info


class SglangFrozenBinder:
    """Bind current original inference Parameters only with preservation installed."""

    def __init__(
        self,
        get_model: Callable[[], nn.Module],
        contract: Qwen4ExpFrozenContract,
        weight_updater: ModuleType,
    ) -> None:
        self.get_model = get_model
        self.contract = contract
        self.weight_updater = weight_updater

    def __call__(self, converter: Any) -> None:
        model = self.get_model()
        if type(model).__name__ != "Qwen4ExpForConditionalGeneration":
            raise ValueError("Frozen inference binding requires Qwen4Exp")
        if not getattr(self.weight_updater, "_areal_qwen4_exp_static_hooks", False):
            raise ValueError(
                "Qwen4Exp visual preservation must be installed before release"
            )
        parameters = {}
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
        converter.bind_frozen_contract(
            self.contract, parameters, frozenset(visual_names)
        )
