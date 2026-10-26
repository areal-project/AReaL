# SPDX-License-Identifier: Apache-2.0
"""Compare MTP-disabled Megatron-Bridge and mcore-bridge model paths."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoConfig

from examples.mtp.qwen38_flash_next.model_build_smoke import _shrink_provider

from areal.models.mcore.mcore_bridge_adapter import MCoreBridgeAdapter
from areal.models.mcore.qwen4_exp_megatron_bridge import (
    register_qwen4_exp_megatron_bridge,
)

# Declared before observing output. Both paths use BF16 parameters and the same
# deterministic kernels; these bounds allow one BF16 rounding unit around
# typical unit-scale logits/losses while still reporting exact observed maxima.
BF16_FORWARD_RTOL = 2e-2
BF16_FORWARD_ATOL = 2e-2
BF16_LOSS_RTOL = 2e-2
BF16_LOSS_ATOL = 2e-2
SEED = 17
SEQUENCE_LENGTH = 16


def _configure_tiny_no_mtp(config) -> None:
    _shrink_provider(config)
    config.mtp_num_layers = None
    config.mtp_unroll_steps = 0
    config.deterministic_mode = True
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0
    config.moe_router_load_balancing_type = "none"
    config.moe_aux_loss_coeff = 0.0
    config.cross_entropy_loss_fusion = False


def _diff(expected: torch.Tensor, actual: torch.Tensor) -> tuple[float, float]:
    expected = expected.float()
    actual = actual.float()
    absolute = (expected - actual).abs()
    denominator = expected.abs().clamp_min(torch.finfo(torch.float32).tiny)
    return float(absolute.max()), float((absolute / denominator).max())


def _unwrap_float16_module(model: torch.nn.Module) -> torch.nn.Module:
    model = getattr(model, "module", model)
    return getattr(model, "language_model", model)


def _simple_config_differences(left, right) -> dict[str, dict[str, object]]:
    """Return scalar/list config differences useful for graph-parity diagnosis."""

    result = {}
    for name in sorted(set(vars(left)) | set(vars(right))):
        left_value = getattr(left, name, "<missing>")
        right_value = getattr(right, name, "<missing>")
        if isinstance(
            left_value, (str, int, float, bool, type(None), list, tuple)
        ) and isinstance(right_value, (str, int, float, bool, type(None), list, tuple)):
            if left_value != right_value:
                result[name] = {
                    "megatron_bridge": left_value,
                    "mcore_bridge": right_value,
                }
    return result


def main() -> None:
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        context_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(SEED)
    try:
        model_path = os.environ["MODEL_PATH"]
        register_qwen4_exp_megatron_bridge()
        hf_config = AutoConfig.from_pretrained(model_path, local_files_only=True)

        new_bridge = AutoBridge.from_hf_config(hf_config)
        new_provider = new_bridge.to_megatron_provider(load_weights=False)
        _configure_tiny_no_mtp(new_provider)
        new_provider.finalize()
        with torch.device("cuda"):
            new_model = new_provider.provide(pre_process=True, post_process=True)

        control = MCoreBridgeAdapter(
            model_path,
            dtype=torch.bfloat16,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            gradient_checkpointing=False,
            recompute_granularity=None,
            recompute_method=None,
            recompute_num_layers=None,
            distribute_saved_activations=None,
            recompute_modules=None,
            language_model_only=True,
            transformer_config_overrides=None,
        )
        _configure_tiny_no_mtp(control.config)
        control_model = _unwrap_float16_module(
            control.get_model(wrap_with_ddp=False, ddp_config={})[0]
        )
        control_model.cuda()

        assert all(
            parameter.device.type == "cuda" for parameter in new_model.parameters()
        )
        assert all(
            parameter.device.type == "cuda" for parameter in control_model.parameters()
        )

        new_state = new_model.state_dict()
        control_state = control_model.state_dict()
        if new_state.keys() != control_state.keys():
            raise AssertionError(
                "MTP-disabled model state differs: "
                f"new_only={sorted(new_state.keys() - control_state.keys())}, "
                f"control_only={sorted(control_state.keys() - new_state.keys())}"
            )
        control_model.load_state_dict(new_state, strict=True)
        assert getattr(new_model, "mtp", None) is None
        assert getattr(control_model, "mtp", None) is None
        print(
            "simple_config_differences",
            json.dumps(
                _simple_config_differences(new_model.config, control_model.config),
                indent=2,
                sort_keys=True,
            ),
        )

        new_model.eval()
        control_model.eval()
        input_ids = (
            torch.arange(SEQUENCE_LENGTH, device="cuda", dtype=torch.long)
            .remainder(new_provider.padded_vocab_size)
            .unsqueeze(0)
        )
        position_ids = torch.arange(
            SEQUENCE_LENGTH, device="cuda", dtype=torch.long
        ).unsqueeze(0)
        labels = input_ids.roll(shifts=-1, dims=-1)

        with torch.no_grad():
            new_logits = new_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                runtime_gather_output=True,
            )
            control_logits = control_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                runtime_gather_output=True,
            )
            new_loss = new_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                labels=labels,
                runtime_gather_output=True,
            )
            control_loss = control_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                labels=labels,
                runtime_gather_output=True,
            )

        forward_abs, forward_rel = _diff(control_logits, new_logits)
        loss_abs, loss_rel = _diff(control_loss, new_loss)
        print(
            "observed_differences",
            json.dumps(
                {
                    "forward_max_abs_diff": forward_abs,
                    "forward_max_rel_diff": forward_rel,
                    "loss_max_abs_diff": loss_abs,
                    "loss_max_rel_diff": loss_rel,
                },
                sort_keys=True,
            ),
        )
        torch.testing.assert_close(
            new_logits,
            control_logits,
            rtol=BF16_FORWARD_RTOL,
            atol=BF16_FORWARD_ATOL,
        )
        torch.testing.assert_close(
            new_loss,
            control_loss,
            rtol=BF16_LOSS_RTOL,
            atol=BF16_LOSS_ATOL,
        )
        print(
            json.dumps(
                {
                    "seed": SEED,
                    "sequence_length": SEQUENCE_LENGTH,
                    "dtype": "bfloat16",
                    "forward_tolerance": {
                        "rtol": BF16_FORWARD_RTOL,
                        "atol": BF16_FORWARD_ATOL,
                    },
                    "loss_tolerance": {
                        "rtol": BF16_LOSS_RTOL,
                        "atol": BF16_LOSS_ATOL,
                    },
                    "forward_max_abs_diff": forward_abs,
                    "forward_max_rel_diff": forward_rel,
                    "loss_max_abs_diff": loss_abs,
                    "loss_max_rel_diff": loss_rel,
                    "parameter_count": len(new_state),
                },
                indent=2,
                sort_keys=True,
            )
        )
        print("no_mtp_parity PASS")
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
