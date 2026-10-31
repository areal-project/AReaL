# SPDX-License-Identifier: Apache-2.0
"""Native mixed-precision policy for Qwen3.5 Dense training."""

import torch


class Qwen35FP8Training:
    """Preserve the sensitive B/A projections fused into MCore's GDN input."""

    @staticmethod
    def prepare_model(models: list[torch.nn.Module]) -> list[torch.nn.Module]:
        from megatron.core.extensions.transformer_engine import (
            TELayerNormColumnParallelLinear,
        )
        from megatron.core.quantization.quant_config import (
            MatchContext,
            QuantizationConfig,
        )
        from megatron.core.ssm.gated_delta_net.gdn import GatedDeltaNet

        # Bridge rebuilds the VLM decoder after GPTModel's recipe initialization.
        # The pre-wrap hook sees the final modules and uses MCore's native recipe
        # API before DDP construction, without replacing weights or forwards.
        for model in models:
            for name, module in model.named_modules():
                if not isinstance(module, GatedDeltaNet):
                    continue
                if not isinstance(module.in_proj, TELayerNormColumnParallelLinear):
                    raise ValueError(
                        "Qwen3.5 FP8 requires a TransformerEngine GDN input projection"
                    )
                module.in_proj.finish_init(
                    QuantizationConfig(
                        config={
                            "transformer_engine_config_type": "TEQuantizationParams",
                            # Empty recipe explicitly overrides FP8 autocast with
                            # BF16, for both training and log-prob evaluation.
                            "training_recipe": {},
                        },
                        match_input=MatchContext(
                            module_path=f"{name}.in_proj",
                            layer_number=module.layer_number,
                        ),
                        config_key="qwen35_gdn_bf16",
                    )
                )
        return models
