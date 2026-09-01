# SPDX-License-Identifier: Apache-2.0


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
        # Megatron-Bridge accepts qualified glob patterns such as
        # ``language_model.*.linear_qkv``.  vLLM only needs the canonical
        # module suffix when constructing the PEFT adapter configuration.
        canonical_name = module_name.rsplit(".", 1)[-1]
        mapped = bridge_to_vllm_targets.get(canonical_name)
        if mapped is None:
            raise NotImplementedError(
                f"LoRA target module '{module_name}' is not supported in MegatronEngine yet."
            )
        targets.extend(mapped)
    return sorted(set(targets))


def normalize_bridge_lora_name(name: str) -> str:
    """Normalize Megatron-Bridge adapter names to AReaL's PEFT convention."""
    if not name.startswith("base_model.model."):
        name = f"base_model.model.{name}"
    for suffix in (".lora_A.weight", ".lora_B.weight"):
        if name.endswith(suffix):
            return f"{name[: -len(suffix)]}{suffix[: -len('.weight')]}.default.weight"
    raise ValueError(f"Unsupported Megatron-Bridge LoRA parameter name: {name}")


def patch_mbridge_name_mapping(bridge):
    """
    Patch mbridge name mapping to handle unfused layernorms in GLM-4 and Qwen models.
    This patch is required for megatron lora where we use unfused layers.

    Handles explicit layernorm names for:
    - input_layernorm -> input_layernorm (not fused with qkv)
    - pre_mlp_layernorm -> post_attention_layernorm (not fused with mlp)
    - q_layernorm -> q_norm (QK layernorm)
    - k_layernorm -> k_norm (QK layernorm)
    """
    import re

    orig = bridge._weight_name_mapping_mcore_to_hf

    def new_mapping(name: str):
        # Handle unfused norms + q/k norms
        m = re.match(r"^decoder\.layers\.(\d+)\.(.+)$", name)
        if m:
            i = m.group(1)
            tail = m.group(2)

            if tail == "input_layernorm.weight":
                return [f"model.layers.{i}.input_layernorm.weight"]

            if tail == "pre_mlp_layernorm.weight":
                return [f"model.layers.{i}.post_attention_layernorm.weight"]

            if tail == "self_attention.q_layernorm.weight":
                return [f"model.layers.{i}.self_attn.q_norm.weight"]

            if tail == "self_attention.k_layernorm.weight":
                return [f"model.layers.{i}.self_attn.k_norm.weight"]

        # Fallback to the original implementation for everything else
        return orig(name)

    bridge._weight_name_mapping_mcore_to_hf = new_mapping
    return bridge
