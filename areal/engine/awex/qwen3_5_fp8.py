# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 block-FP8 policy and AWEX converter registration.

Layout conversion remains owned by AWEX. Quantization happens afterwards,
so scales describe the same final-layout shards as the transmitted weights.
"""

from __future__ import annotations

import re
from typing import Any

import torch


class Qwen35FP8Policy:
    """E4M3 weights with dynamic activations and ordinary FP32 128x128 scales."""

    block_size = 128
    architectures = (
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
    )
    _projections = {
        "q_proj",
        "k_proj",
        "v_proj",
        "qkv_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_qkvz",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate_up_proj",
    }
    _fused = {
        "qkv_proj": ("q_proj", "k_proj", "v_proj"),
        "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
        "gate_up_proj": ("gate_proj", "up_proj"),
    }

    def __init__(self, quantization_config: dict[str, Any]):
        if (
            quantization_config.get("quant_method") != "fp8"
            or quantization_config.get("fmt", "e4m3") != "e4m3"
            or quantization_config.get("activation_scheme") != "dynamic"
            or list(quantization_config.get("weight_block_size") or ()) != [128, 128]
            or quantization_config.get("scale_fmt") not in (None, "float32")
        ):
            raise ValueError(
                "Qwen3.5 AWEX requires dynamic E4M3 FP8 with 128x128 blocks "
                "and unpacked FP32 scales."
            )
        ignored = (
            quantization_config.get(
                "ignored_layers", quantization_config.get("modules_to_not_convert", [])
            )
            or []
        )
        self.ignored = tuple(self.normalize(name) for name in ignored)

    @staticmethod
    def value(config: Any, key: str, default: Any = None) -> Any:
        return (
            config.get(key, default)
            if isinstance(config, dict)
            else getattr(config, key, default)
        )

    @staticmethod
    def normalize(name: str) -> str:
        return name.replace("model.language_model.", "model.").replace(
            ".self_attn.", "."
        )

    @classmethod
    def from_config(cls, config: Any) -> Qwen35FP8Policy | None:
        quant = cls.value(config, "quantization_config")
        if quant is None:
            quant = cls.value(
                cls.value(config, "text_config", {}), "quantization_config"
            )
        if not quant or cls.value(quant, "quant_method") != "fp8":
            return None
        return cls(quant)

    @classmethod
    def checkpoint_config(cls, ignored: list[str] | None = None) -> dict[str, Any]:
        return {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            # Suffixes survive SGLang's HF name mapping and cover fused BA.
            "modules_to_not_convert": sorted(
                set(ignored or [])
                | {
                    "visual",
                    "lm_head",
                    "embed_tokens",
                    "norm",
                    "conv1d",
                    "in_proj_a",
                    "in_proj_b",
                    "mtp",
                }
            ),
        }

    def quantizes(self, name: str) -> bool:
        name = self.normalize(name)
        if not name.startswith("model.layers.") or not name.endswith(".weight"):
            return False
        module = name.removesuffix(".weight")
        projection = module.rsplit(".", 1)[-1]
        if projection not in self._projections:
            return False
        pieces = self._fused.get(projection, (projection,))
        enabled = [
            not any(
                ignored in module.rsplit(".", 1)[0] + "." + piece
                for ignored in self.ignored
            )
            for piece in pieces
        ]
        if any(enabled) != all(enabled):
            raise ValueError(
                f"All components of fused FP8 projection {name} must use the same precision"
            )
        return all(enabled)

    def quantize(
        self, name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        if not self.quantizes(name):
            return [(name, tensor)]
        if tensor.ndim != 2 or any(size % self.block_size for size in tensor.shape):
            raise ValueError(
                f"FP8 shard {name} must be a 2D matrix aligned to 128x128 blocks, got {tensor.shape}"
            )
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(
                f"Expected unquantized weights for {name}, got {tensor.dtype}"
            )
        from areal.engine.megatron_utils.fp8.kernels import blockwise_cast_to_fp8_triton

        weight, scale = blockwise_cast_to_fp8_triton(tensor, (128, 128))
        return [(name, weight), (name + "_scale_inv", scale)]

    @classmethod
    def validate_parallelism(cls, config: Any, infer_tp: int) -> None:
        """Check logical fused segments, not just their concatenated size."""
        text = cls.value(config, "text_config", config)
        heads = int(cls.value(text, "num_attention_heads"))
        kv_heads = int(cls.value(text, "num_key_value_heads"))
        hidden = int(cls.value(text, "hidden_size"))
        head_dim = int(cls.value(text, "head_dim", hidden // heads))
        if (
            infer_tp <= 0
            or heads % kv_heads
            or heads % infer_tp
            or max(kv_heads, infer_tp) % min(kv_heads, infer_tp)
        ):
            raise ValueError(
                "Qwen3.5 attention heads, KV heads and inference TP are incompatible"
            )
        components = {
            "gated query": 2 * heads * head_dim,
            # SGLang replicates KV heads when inference TP exceeds KV heads.
            "key/value": max(kv_heads, infer_tp) * head_dim,
            "GDN query/key": int(cls.value(text, "linear_num_key_heads"))
            * int(cls.value(text, "linear_key_head_dim")),
            "GDN value/gate": int(cls.value(text, "linear_num_value_heads"))
            * int(cls.value(text, "linear_value_head_dim")),
        }
        for label, size in components.items():
            if size % (infer_tp * cls.block_size):
                raise ValueError(
                    f"Qwen3.5 FP8 {label} size {size} must be divisible by inference TP ({infer_tp}) * 128"
                )

    def validate_pipeline_exclusions(self, pp_size: int) -> None:
        # AWEX discovers metadata using local layer ids before collecting the
        # PP layer map. Per-layer precision would differ from the payload pass.
        if pp_size <= 1:
            return
        for ignored in self.ignored:
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", ignored)
            if match and any(
                ignored
                in self.normalize(f"model.layers.{match[1]}.{module}.{projection}")
                for module in ("self_attn", "linear_attn", "mlp")
                for projection in self._projections
            ):
                raise ValueError(
                    "Qwen3.5 FP8 AWEX with PP>1 requires uniform projection exclusions across layers"
                )

    def validate_runtime(self, model: torch.nn.Module, server_args: Any) -> None:
        if self.value(server_args, "fp8_gemm_runner_backend") != "triton":
            raise ValueError(
                "Qwen3.5 AWEX FP8 requires sglang.fp8_gemm_runner_backend=triton "
                "to retain writable FP32 scale grids."
            )
        params = dict(model.named_parameters())
        for name, tensor in params.items():
            if tensor.dtype != torch.float8_e4m3fn:
                continue
            if tensor.ndim < 2:
                raise ValueError(
                    f"Unsupported FP8 tensor shape for {name}: {tensor.shape}"
                )
            scale = params.get(name + "_scale_inv")
            expected = (
                *tensor.shape[:-2],
                tensor.shape[-2] // 128,
                tensor.shape[-1] // 128,
            )
            if (
                any(size % 128 for size in tensor.shape[-2:])
                or scale is None
                or scale.dtype != torch.float32
                or tuple(scale.shape) != expected
                or getattr(scale, "format_ue8m0", False)
            ):
                raise ValueError(
                    f"Unsupported FP8 weight/scale storage for {name}; expected an aligned weight and FP32 block grid"
                )

    @classmethod
    def validate_metadata(cls, metadata: list[Any]) -> None:
        """Prove that scale offsets track weight offsets before making a plan."""
        by_name = {param.name: param for param in metadata}
        if len(by_name) != len(metadata):
            raise ValueError("Duplicate parameters in Qwen3.5 FP8 metadata")
        for name, weight in by_name.items():
            if weight.dtype != torch.float8_e4m3fn:
                continue
            scale = by_name.get(name + "_scale_inv")
            if scale is None or scale.dtype != torch.float32:
                raise ValueError(f"Missing FP32 scale metadata for {name}")
            expected = tuple(dim // 128 for dim in weight.global_shape)
            if (
                len(expected) != 2
                or any(dim % 128 for dim in weight.global_shape)
                or tuple(scale.global_shape) != expected
            ):
                raise ValueError(f"Inconsistent FP8 global scale shape for {name}")
            scales = {(s.engine_rank, s.global_rank): s for s in scale.shards}
            for shard in weight.shards:
                scale_shard = scales.get((shard.engine_rank, shard.global_rank))
                if (
                    scale_shard is None
                    or any(dim % 128 for dim in (*shard.shape, *shard.global_offset))
                    or tuple(scale_shard.shape)
                    != tuple(dim // 128 for dim in shard.shape)
                    or tuple(scale_shard.global_offset)
                    != tuple(dim // 128 for dim in shard.global_offset)
                    or scale_shard.sharding_dim != shard.sharding_dim
                    or scale_shard.sharding_type != shard.sharding_type
                ):
                    raise ValueError(
                        f"Inconsistent FP8 scale sharding for {name} on rank {shard.global_rank}"
                    )

    @classmethod
    def register_awex(cls) -> None:
        """Install model registry entries without patching runtime methods."""
        from awex.models.qwen3_5 import CONFIG
        from awex.models.registry import ModelRegistry

        for config in CONFIG:
            if config["model_name"] not in cls.architectures:
                continue
            entry = {
                **config,
                "mcore_converter": _Qwen35FP8TrainConverter,
                "sharding_strategy": _Qwen35FP8Sharding,
            }
            existing = ModelRegistry.models.get(config["model_name"])
            if existing not in (None, config, entry):
                raise ValueError(
                    f"A different Qwen3.5 AWEX adapter is already registered: {config['model_name']}"
                )
            ModelRegistry.models[config["model_name"]] = entry


class _Qwen35FP8TrainConverter:
    def __init__(
        self, hf_config: Any, rank_info: Any, infer_conf: dict, tf_config: Any
    ):
        from awex.models.qwen3_5 import CONFIG

        native = CONFIG[0]["mcore_converter"]()
        self._delegate = native(hf_config, rank_info, infer_conf, tf_config=tf_config)
        self._policy = Qwen35FP8Policy.from_config(infer_conf.get("hf_config", {}))
        if self._policy is not None:
            if infer_conf.get("engine_name") != "sglang":
                raise ValueError("Qwen3.5 AWEX FP8 supports SGLang inference")
            self._policy.validate_parallelism(
                hf_config, int(infer_conf["infer_atten_tp_size"])
            )
            self._policy.validate_pipeline_exclusions(rank_info.pp_size)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    @torch.no_grad()
    def convert_param(
        self, name: str, parameter: torch.Tensor, vp_stage: int | None = None
    ) -> list[tuple[str, torch.Tensor]]:
        converted = self._delegate.convert_param(name, parameter, vp_stage=vp_stage)
        if self._policy is None:
            return converted
        return [
            pair
            for key, tensor in converted
            for pair in self._policy.quantize(key, tensor)
        ]


class _Qwen35FP8Sharding:
    def __init__(self, *args: Any, **kwargs: Any):
        from awex.models.qwen3_5 import Qwen3_5ShardingStrategy

        self._delegate = Qwen3_5ShardingStrategy(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def get_sharding_strategy(self, parameter_name: str, **kwargs: Any):
        # GDN out_proj uses columns for both weights and their block scales.
        return self._delegate.get_sharding_strategy(
            parameter_name.removesuffix("_scale_inv"), **kwargs
        )
