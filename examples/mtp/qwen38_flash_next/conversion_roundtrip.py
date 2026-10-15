# SPDX-License-Identifier: Apache-2.0
"""Exact tiny-graph HF↔Megatron round-trip for every mapped Qwen4Exp tensor."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained import PreTrainedCausalLM
from megatron.bridge.models.hf_pretrained.state import StateDict
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoConfig

from examples.mtp.qwen38_flash_next.model_build_smoke import _shrink_provider

from areal.models.mcore.qwen4_exp_megatron_bridge import (
    register_qwen4_exp_megatron_bridge,
)

# This fixture uses BF16 values on a 1/16 grid. Every conversion is a tensor
# permutation, concatenation/split, or zero-centered norm +/-1 on exactly
# representable values. Declare exact tolerance before observing results.
PARAM_RTOL = 0.0
PARAM_ATOL = 0.0


def _fill_deterministically(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter_index, (_, parameter) in enumerate(model.named_parameters()):
            values = torch.arange(
                parameter.numel(), device=parameter.device, dtype=torch.int64
            )
            values = ((values + parameter_index) % 17 - 8).to(torch.float32) / 16
            parameter.copy_(values.reshape(parameter.shape).to(parameter.dtype))


def _max_differences(
    expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]
) -> tuple[float, float, str]:
    max_abs = 0.0
    max_rel = 0.0
    worst = ""
    for name in expected:
        lhs = expected[name].float()
        rhs = actual[name].float()
        abs_diff = (lhs - rhs).abs()
        local_abs = float(abs_diff.max()) if abs_diff.numel() else 0.0
        denominator = torch.maximum(lhs.abs(), rhs.abs()).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        local_rel = float((abs_diff / denominator).max()) if abs_diff.numel() else 0.0
        if local_abs > max_abs:
            max_abs = local_abs
            worst = name
        max_rel = max(max_rel, local_rel)
    return max_abs, max_rel, worst


def _snapshot_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def main() -> None:
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        context_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(17)
    try:
        register_qwen4_exp_megatron_bridge()
        config = AutoConfig.from_pretrained(
            os.environ["MODEL_PATH"], local_files_only=True
        )
        checkpoint_root = Path(os.environ["MODEL_PATH"])
        checkpoint_index = json.loads(
            (checkpoint_root / "model.safetensors.index.json").read_text()
        )["weight_map"]
        # Megatron-Bridge uses source keys to plan export tasks. Preserve the
        # real checkpoint's text/MTP key set without loading any source tensor;
        # fixed PLE tables and vision assets are intentionally outside this tiny
        # normal-mapping round-trip and are validated in the real export test.
        planning_state = {
            name: torch.empty(0)
            for name in checkpoint_index
            if not name.startswith("model.visual.")
            and ".ple.ple_embedding.ngram_embedding." not in name
        }
        full_attention_prefix = "model.language_model.layers.3."
        tiny_full_attention_prefix = "model.language_model.layers.1."
        planning_state.update(
            {
                name.replace(
                    full_attention_prefix, tiny_full_attention_prefix, 1
                ): torch.empty(0)
                for name in checkpoint_index
                if name.startswith(full_attention_prefix)
            }
        )
        wrapper = PreTrainedCausalLM(device="cpu")
        wrapper.config = config
        wrapper._state_dict_accessor = StateDict(planning_state)
        bridge = AutoBridge(wrapper)

        provider = bridge.to_megatron_provider(load_weights=False)
        _shrink_provider(provider)
        provider.finalize()
        with torch.device("cuda"):
            model = provider.provide(pre_process=True, post_process=True)
        _fill_deterministically(model)
        expected_megatron = _snapshot_parameters(model)

        first_hf = {
            name: tensor.detach().cpu().clone()
            for name, tensor in bridge.export_hf_weights(
                [model], cpu=True, show_progress=False
            )
        }
        wrapper._state_dict_accessor = StateDict(first_hf)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(torch.nan)
        bridge.load_hf_weights([model])
        actual_megatron = _snapshot_parameters(model)

        assert expected_megatron.keys() == actual_megatron.keys()
        for name in expected_megatron:
            torch.testing.assert_close(
                actual_megatron[name],
                expected_megatron[name],
                rtol=PARAM_RTOL,
                atol=PARAM_ATOL,
            )
        parameter_abs, parameter_rel, parameter_worst = _max_differences(
            expected_megatron, actual_megatron
        )

        second_hf = {
            name: tensor.detach().cpu().clone()
            for name, tensor in bridge.export_hf_weights(
                [model], cpu=True, show_progress=False
            )
        }
        assert first_hf.keys() == second_hf.keys()
        for name in first_hf:
            torch.testing.assert_close(
                second_hf[name],
                first_hf[name],
                rtol=PARAM_RTOL,
                atol=PARAM_ATOL,
            )
        hf_abs, hf_rel, hf_worst = _max_differences(first_hf, second_hf)

        print(
            json.dumps(
                {
                    "declared_tolerance": {"rtol": PARAM_RTOL, "atol": PARAM_ATOL},
                    "megatron_parameter_count": len(expected_megatron),
                    "hf_tensor_count": len(first_hf),
                    "megatron_max_abs_diff": parameter_abs,
                    "megatron_max_rel_diff": parameter_rel,
                    "megatron_worst_parameter": parameter_worst,
                    "hf_max_abs_diff": hf_abs,
                    "hf_max_rel_diff": hf_rel,
                    "hf_worst_tensor": hf_worst,
                },
                indent=2,
                sort_keys=True,
            )
        )
        print("conversion_roundtrip PASS")
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
