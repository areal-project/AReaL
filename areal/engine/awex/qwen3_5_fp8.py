# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 block-FP8 policy and AWEX converter registration.

Layout conversion remains owned by AWEX. Quantization happens afterwards,
so scales describe the same final-layout shards as the transmitted weights.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
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

    @staticmethod
    def _uses_packed_deepgemm_scale(layer: torch.nn.Module) -> bool:
        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.layers.quantization.fp8_utils import (
            deepgemm_w8a8_block_fp8_linear_with_fallback,
        )

        method = getattr(layer, "quant_method", None)
        return (
            getattr(method, "w8a8_block_fp8_linear", None)
            is deepgemm_w8a8_block_fp8_linear_with_fallback
            and deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
            and getattr(
                getattr(method, "quant_config", None), "weight_block_size", None
            )
            == [128, 128]
            and getattr(layer, "orig_dtype", None) == torch.bfloat16
        )

    def validate_runtime(self, model: torch.nn.Module, server_args: Any) -> None:
        backend = self.value(server_args, "fp8_gemm_runner_backend")
        if backend not in ("triton", "deep_gemm"):
            raise ValueError(
                "Qwen3.5 AWEX FP8 requires an explicit "
                "sglang.fp8_gemm_runner_backend=triton or deep_gemm."
            )
        params = dict(model.named_parameters())
        for name, tensor in params.items():
            if tensor.dtype != torch.float8_e4m3fn:
                continue
            if tensor.ndim != 2 or any(size % 128 for size in tensor.shape):
                raise ValueError(
                    f"FP8 weight {name} must be a 2D matrix aligned to 128x128 blocks"
                )
            scale = params.get(name + "_scale_inv")
            packed = backend == "deep_gemm" and self._uses_packed_deepgemm_scale(
                model.get_submodule(name.rpartition(".")[0])
            )
            expected = (
                (tensor.shape[0], (tensor.shape[1] // 128 + 3) // 4)
                if packed
                else (tensor.shape[0] // 128, tensor.shape[1] // 128)
            )
            if (
                scale is None
                or scale.dtype != (torch.int32 if packed else torch.float32)
                or tuple(scale.shape) != expected
                or bool(getattr(scale, "format_ue8m0", False)) != packed
                or scale.device != tensor.device
            ):
                storage = "packed UE8M0 scales" if packed else "FP32 block grid"
                raise ValueError(
                    f"Unsupported FP8 weight/scale storage for {name}; expected {storage}. "
                    "SGLang must finish processing checkpoint weights before AWEX initialization."
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


class Qwen35FP8Receiver:
    """Receive plain block scales without rebinding DeepGEMM runtime parameters.

    Inference is paused while AWEX writes FP8 weights into their existing buffers.
    Only packed scales need separate FP32 receive storage. Metadata discovery can
    run while weights are offloaded, so allocate these small buffers lazily after
    resume, then keep them alive with the reader's cached tensor views.
    """

    def __init__(self, model: torch.nn.Module, weight_names: set[str]):
        self.model = model
        params = dict(model.named_parameters())
        self._targets = {
            name: (weight, params[name + "_scale_inv"])
            for name, weight in params.items()
            if name in weight_names
            and weight.dtype == torch.float8_e4m3fn
            and params[name + "_scale_inv"].dtype == torch.int32
        }
        self._scales: dict[str, torch.Tensor] = {}
        self._storage = {
            name: (
                parameter,
                parameter.data_ptr(),
                parameter.shape,
                parameter.stride(),
                parameter.dtype,
                parameter.device,
            )
            for name, parameter in params.items()
            if name in self._targets or name.removesuffix("_scale_inv") in self._targets
        }

    @property
    def config(self) -> Any:
        return self.model.config

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        # The device-bound reader must inspect real parameters, never meta scales.
        return self.model.parameters()

    def named_parameters(
        self, *, metadata_only: bool = False
    ) -> Iterator[tuple[str, torch.Tensor]]:
        for name, parameter in self.model.named_parameters():
            target = self._targets.get(name.removesuffix("_scale_inv"))
            if name.endswith("_scale_inv") and target is not None:
                weight, _ = target
                shape = (weight.shape[0] // 128, weight.shape[1] // 128)
                if metadata_only:
                    parameter = torch.empty(shape, dtype=torch.float32, device="meta")
                else:
                    if name not in self._scales:
                        self._scales[name] = torch.empty(
                            shape, dtype=torch.float32, device=weight.device
                        )
                    parameter = self._scales[name]
            yield name, parameter

    def export_parameters(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Export runtime weights with matching block scales, never stale wire scales."""
        self.validate_storage()
        for name, parameter in self.model.named_parameters():
            target = self._targets.get(name.removesuffix("_scale_inv"))
            if name.endswith("_scale_inv") and target is not None:
                from sglang.srt.layers.quantization.fp8_utils import (
                    _unpack_ue8m0_scale_for_triton,
                )

                parameter = _unpack_ue8m0_scale_for_triton(
                    parameter, target[0].shape, [128, 128]
                )
            yield name, parameter

    def validate_storage(self, model: torch.nn.Module | None = None) -> None:
        if model is not None and model is not self.model:
            raise RuntimeError(
                "SGLang model changed; reinitialize AWEX after replacing the model."
            )
        params = dict(self.model.named_parameters())
        for name, (
            parameter,
            pointer,
            shape,
            stride,
            dtype,
            device,
        ) in self._storage.items():
            current = params.get(name)
            if (
                current is not parameter
                or current.data_ptr() != pointer
                or current.shape != shape
                or current.stride() != stride
                or current.dtype != dtype
                or current.device != device
            ):
                raise RuntimeError(
                    f"DeepGEMM parameter storage changed for {name}; "
                    "reinitialize AWEX and recapture CUDA graphs after replacing weights."
                )

    @torch.no_grad()
    def finish_update(self) -> None:
        if not self._targets:
            return
        from sglang.srt.layers.quantization.fp8_utils import requant_weight_ue8m0

        self.validate_storage()
        for name, (weight, packed_scale) in self._targets.items():
            scale = self._scales.get(name + "_scale_inv")
            if scale is None:
                raise RuntimeError(f"No received FP32 scale buffer for {name}")
            # Do not call the *_inplace helper: it rebinds parameter storage and
            # invalidates both the AWEX views and captured CUDA graph addresses.
            converted_weight, converted_scale = requant_weight_ue8m0(
                weight, scale, [128, 128]
            )
            for target, converted in (
                (weight, converted_weight),
                (packed_scale, converted_scale),
            ):
                if (
                    converted.shape != target.shape
                    or converted.dtype != target.dtype
                    or converted.device != target.device
                ):
                    raise RuntimeError(
                        f"DeepGEMM converted storage mismatch for {name}"
                    )
            weight.copy_(converted_weight)
            packed_scale.copy_(converted_scale)
            del converted_weight, converted_scale, converted
        # Some Qwen models have no post_load_weights hook. Completion must still
        # precede the AWEX ready signal and resumption of inference on any stream.
        torch.cuda.synchronize(weight.device)


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
