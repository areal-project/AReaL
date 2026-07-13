# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.distributed as dist


def get_vllm_lora_target_modules(target_modules: list[str]) -> list[str]:
    if not target_modules or "all-linear" in target_modules:
        target_modules = [
            "linear_qkv",
            "linear_proj",
            "linear_fc1",
            "linear_fc2",
        ]

    bridge_to_vllm_targets = {
        "linear_qkv": ["q_proj", "k_proj", "v_proj"],
        "linear_proj": ["o_proj"],
        "linear_fc1": ["gate_proj", "up_proj"],
        "linear_fc2": ["down_proj"],
    }
    targets: list[str] = []
    for module_name in target_modules:
        mapped = bridge_to_vllm_targets.get(module_name)
        if mapped is None:
            raise NotImplementedError(
                f"LoRA target module '{module_name}' is not supported in MegatronEngine yet."
            )
        targets.extend(mapped)
    return sorted(set(targets))


def convert_qwen3_lora_to_hf(
    tf_config,
    name: str,
    tensor: torch.Tensor,
) -> list[tuple[str, torch.Tensor]]:
    pattern = (
        r"(?:^|.*\.)decoder\.layers\.(\d+)\."
        r"(self_attention\.linear_qkv|self_attention\.linear_proj|mlp\.linear_fc1|mlp\.linear_fc2)\."
        r"adapter\.(linear_in|linear_out)\.weight$"
    )
    match = re.match(pattern, name)
    if match is None:
        return []

    layer_idx, module_name, adapter_part = match.groups()
    base_prefix = f"base_model.model.model.layers.{layer_idx}"

    if module_name == "self_attention.linear_proj":
        hf_base = f"{base_prefix}.self_attn.o_proj"
        suffix = (
            "lora_A.default.weight"
            if adapter_part == "linear_in"
            else "lora_B.default.weight"
        )
        return [(f"{hf_base}.{suffix}", tensor)]

    if module_name == "mlp.linear_fc2":
        hf_base = f"{base_prefix}.mlp.down_proj"
        suffix = (
            "lora_A.default.weight"
            if adapter_part == "linear_in"
            else "lora_B.default.weight"
        )
        return [(f"{hf_base}.{suffix}", tensor)]

    if module_name == "mlp.linear_fc1":
        gate_base = f"{base_prefix}.mlp.gate_proj"
        up_base = f"{base_prefix}.mlp.up_proj"
        if adapter_part == "linear_in":
            return [
                (f"{gate_base}.lora_A.default.weight", tensor),
                (f"{up_base}.lora_A.default.weight", tensor),
            ]
        gate_b, up_b = tensor.chunk(2, dim=0)
        return [
            (f"{gate_base}.lora_B.default.weight", gate_b.contiguous()),
            (f"{up_base}.lora_B.default.weight", up_b.contiguous()),
        ]

    if module_name == "self_attention.linear_qkv":
        q_base = f"{base_prefix}.self_attn.q_proj"
        k_base = f"{base_prefix}.self_attn.k_proj"
        v_base = f"{base_prefix}.self_attn.v_proj"
        if adapter_part == "linear_in":
            return [
                (f"{q_base}.lora_A.default.weight", tensor),
                (f"{k_base}.lora_A.default.weight", tensor),
                (f"{v_base}.lora_A.default.weight", tensor),
            ]

        head_dim = (
            tf_config.kv_channels
            if getattr(tf_config, "kv_channels", None) is not None
            else tf_config.hidden_size // tf_config.num_attention_heads
        )
        if getattr(tf_config, "num_query_groups", None) is None:
            return []
        value_num_per_group = (
            tf_config.num_attention_heads // tf_config.num_query_groups
        )

        tensor = tensor.view(tf_config.num_query_groups, -1, head_dim, tensor.shape[1])
        q_b, k_b, v_b = torch.split(tensor, [value_num_per_group, 1, 1], dim=1)

        q_b = q_b.reshape(-1, q_b.shape[-1]).contiguous()
        k_b = k_b.reshape(-1, k_b.shape[-1]).contiguous()
        v_b = v_b.reshape(-1, v_b.shape[-1]).contiguous()

        return [
            (f"{q_base}.lora_B.default.weight", q_b),
            (f"{k_base}.lora_B.default.weight", k_b),
            (f"{v_base}.lora_B.default.weight", v_b),
        ]

    return []


def convert_qwen3_moe_lora_to_hf(
    tf_config,
    name: str,
    tensor: torch.Tensor,
) -> list[tuple[str, torch.Tensor]]:
    # Reuse non-MoE conversion for attention and dense MLP paths.
    converted = convert_qwen3_lora_to_hf(tf_config, name, tensor)
    if converted:
        return converted

    grouped_expert_pattern = (
        r"(?:^|.*\.)decoder\.layers\.(\d+)\.mlp\.experts\."
        r"(linear_fc1|linear_fc2)\.adapter\.(linear_in|linear_out)\.weight$"
    )
    match = re.match(grouped_expert_pattern, name)
    if match is not None:
        layer_idx, module_name, adapter_part = match.groups()
        num_experts = getattr(tf_config, "num_moe_experts", None)
        if num_experts is None:
            num_experts = getattr(tf_config, "num_experts", None)
        if num_experts is None:
            return []

        outputs: list[tuple[str, torch.Tensor]] = []
        for expert_idx in range(num_experts):
            base_prefix = (
                f"base_model.model.model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            )

            if module_name == "linear_fc2":
                hf_base = f"{base_prefix}.down_proj"
                suffix = (
                    "lora_A.default.weight"
                    if adapter_part == "linear_in"
                    else "lora_B.default.weight"
                )
                outputs.append((f"{hf_base}.{suffix}", tensor))
                continue

            gate_base = f"{base_prefix}.gate_proj"
            up_base = f"{base_prefix}.up_proj"
            if adapter_part == "linear_in":
                outputs.extend(
                    [
                        (f"{gate_base}.lora_A.default.weight", tensor),
                        (f"{up_base}.lora_A.default.weight", tensor),
                    ]
                )
                continue

            gate_b, up_b = tensor.chunk(2, dim=0)
            outputs.extend(
                [
                    (f"{gate_base}.lora_B.default.weight", gate_b.contiguous()),
                    (f"{up_base}.lora_B.default.weight", up_b.contiguous()),
                ]
            )

        return outputs

    expert_pattern = (
        r"(?:^|.*\.)decoder\.layers\.(\d+)\.mlp\.experts\."
        r"(linear_fc1|linear_fc2)\.adapter\.(linear_in|linear_out)\.weight(\d+)$"
    )
    match = re.match(expert_pattern, name)
    if match is None:
        return []

    layer_idx, module_name, adapter_part, expert_idx = match.groups()
    base_prefix = f"base_model.model.model.layers.{layer_idx}.mlp.experts.{expert_idx}"

    if module_name == "linear_fc2":
        hf_base = f"{base_prefix}.down_proj"
        suffix = (
            "lora_A.default.weight"
            if adapter_part == "linear_in"
            else "lora_B.default.weight"
        )
        return [(f"{hf_base}.{suffix}", tensor)]

    if module_name == "linear_fc1":
        gate_base = f"{base_prefix}.gate_proj"
        up_base = f"{base_prefix}.up_proj"
        if adapter_part == "linear_in":
            return [
                (f"{gate_base}.lora_A.default.weight", tensor),
                (f"{up_base}.lora_A.default.weight", tensor),
            ]
        gate_b, up_b = tensor.chunk(2, dim=0)
        return [
            (f"{gate_base}.lora_B.default.weight", gate_b.contiguous()),
            (f"{up_base}.lora_B.default.weight", up_b.contiguous()),
        ]

    return []


def _infer_target_modules_from_adapter_weights(weight_keys: Iterable[str]) -> list[str]:
    """
    Infer PEFT target_modules from adapter weight parameter names.

    Extracts module names from HF LoRA weight keys like:
    - base_model.model.layers.0.self_attn.q_proj.lora_A.weight -> q_proj
    - base_model.model.layers.1.mlp.gate_proj.lora_B.weight -> gate_proj
    """
    target_modules = set()

    for key in weight_keys:
        # Remove PEFT prefix
        key = key.replace("base_model.model.", "")

        # Look for .lora_A.weight or .lora_B.weight pattern
        if ".lora_A.weight" in key:
            # Extract module name before .lora_A.weight
            base_name = key.replace(".lora_A.weight", "")
            module_name = base_name.split(".")[-1]
            target_modules.add(module_name)
        elif ".lora_B.weight" in key:
            # Extract module name before .lora_B.weight
            base_name = key.replace(".lora_B.weight", "")
            module_name = base_name.split(".")[-1]
            target_modules.add(module_name)

    return sorted(list(target_modules))


def _build_adapter_config_dict(
    peft_config,
    target_modules: list[str],
    base_model_name_or_path: str,
) -> dict:
    """
    Build PEFT adapter_config.json dictionary.

    Creates a config compatible with HuggingFace PEFT library.
    """
    return {
        "base_model_name_or_path": base_model_name_or_path,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "r": peft_config.dim,
        "lora_alpha": peft_config.alpha,
        "lora_dropout": peft_config.dropout,
        "target_modules": target_modules,
        "bias": "none",
        "fan_in_fan_out": False,
        "modules_to_save": None,
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
    }


def _monkey_patch_save_hf_adapter():
    """Add save_hf_adapter to AutoBridge when megatron-bridge does not provide it."""
    from megatron.bridge import AutoBridge

    if hasattr(AutoBridge, "save_hf_adapter"):
        # Already exists, no need to patch
        return

    def save_hf_adapter(
        self,
        model,
        path: str | Path,
        peft_config,
        base_model_name_or_path: str | None = None,
        show_progress: bool = True,
    ) -> None:
        """
        Save LoRA adapter weights as a HuggingFace PEFT-compatible directory.

        The output directory contains adapter_config.json and adapter_model.safetensors
        and can be loaded directly with peft.PeftModel.from_pretrained(base_model, path).

        Args:
            model: Megatron model instance or list of instances.
            path: Directory path where the adapter files will be saved.
            peft_config: The LoRA config used during training (provides dim, alpha, dropout, etc.).
            base_model_name_or_path: HuggingFace model identifier or local path of the base model.
                If None, inferred from hf_pretrained.model_name_or_path.
            show_progress: Display progress bar during export.

        Example:
            >>> bridge.save_hf_adapter(
            ...     megatron_model,
            ...     "./my-lora-adapter",
            ...     peft_config=lora,
            ...     base_model_name_or_path="Qwen/Qwen3-4B",
            ... )
            >>> # Load with HuggingFace PEFT
            >>> from peft import PeftModel
            >>> from transformers import AutoModelForCausalLM
            >>> base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
            >>> model = PeftModel.from_pretrained(base, "./my-lora-adapter")

        Note:
            This method is collective -- all ranks must call it. Only rank 0 writes files.
        """
        import json

        from safetensors.torch import save_file

        # Synchronize at start
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # Export adapter weights
        adapter_state: dict[str, torch.Tensor] = {}
        for name, tensor in self.export_adapter_weights(
            # cpu=True may reduce memory pressure but hangs for MoE models using slurm
            model,
            cpu=False,
            show_progress=False,
        ):
            adapter_state[f"base_model.model.{name}"] = tensor.clone().float()

        if not adapter_state:
            raise RuntimeError(
                "No adapter weights were found on the model. "
                "Ensure the model has PEFT adapters applied before calling save_hf_adapter()."
            )

        # Only rank 0 writes files
        is_rank0 = (
            not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
        )
        if is_rank0:
            save_dir = Path(path)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Infer base model path if not provided
            if base_model_name_or_path is None:
                base_model_name_or_path = str(
                    getattr(self.hf_pretrained, "model_name_or_path", "")
                    or getattr(self.hf_pretrained, "name_or_path", "")
                )

            # Build adapter config
            target_modules = _infer_target_modules_from_adapter_weights(
                adapter_state.keys()
            )
            adapter_config = _build_adapter_config_dict(
                peft_config,
                target_modules=target_modules,
                base_model_name_or_path=base_model_name_or_path,
            )

            # Save adapter config
            config_path = save_dir / "adapter_config.json"
            with open(config_path, "w") as f:
                json.dump(adapter_config, f, indent=2)

            # Save adapter weights
            weights_path = save_dir / "adapter_model.safetensors"
            save_file(adapter_state, str(weights_path))

            print(f"✓ Saved LoRA adapter to {save_dir}")
            print(f"  - Config: {config_path}")
            print(f"  - Weights: {weights_path} ({len(adapter_state)} parameters)")

        # Synchronize at end
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    # Attach the method to the class
    AutoBridge.save_hf_adapter = save_hf_adapter


# Current: This monkey patch is needed as the current megatron-bridge 0.3.0 does not have a built-in method
# to save LoRA adapters in HuggingFace PEFT format, which is required for our use case.
# Future: This code is however present in main branch of megatron-bridge so this patch is temporary
# and can be removed later when we upgrade the megatron-bridge version.
_monkey_patch_save_hf_adapter()
