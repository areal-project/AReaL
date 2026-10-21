import importlib.metadata as md
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    for key in ("NVTE_FUSED_ATTN", "NVTE_FLASH_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(key, None)
    from megatron.bridge import AutoBridge
    from megatron.core.transformer.enums import AttnBackend
    from transformers import AutoConfig

    import areal.engine.megatron_engine as engine
    from areal.engine.core.model import (
        requires_padded_seq,
        resolve_sequence_packing_mode,
        validate_model_packed_seq_dependencies,
    )

    model = os.environ["QWEN_MODEL"]
    hf = AutoConfig.from_pretrained(
        model, local_files_only=True, trust_remote_code=True
    )
    validate_model_packed_seq_dependencies(hf.model_type, "megatron-bridge", 4)
    provider = AutoBridge.from_hf_pretrained(
        model, trust_remote_code=True
    ).to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = 4
    provider.pipeline_model_parallel_size = 4
    provider.context_parallel_size = 4
    provider.expert_model_parallel_size = 1
    provider.sequence_parallel = True
    provider.__post_init__()
    assert provider.attention_backend == AttnBackend.auto
    from megatron.core.models.common.language_module.language_module import (
        LanguageModule,
    )

    LanguageModule._set_attention_backend(SimpleNamespace(config=provider))
    result = {
        "packages": {
            k: md.version(k)
            for k in (
                "torch",
                "transformers",
                "megatron-core",
                "megatron-bridge",
                "awex",
            )
        },
        "model_type": hf.model_type,
        "provider": type(provider).__name__,
        "packing": resolve_sequence_packing_mode(
            hf.model_type, "megatron-bridge"
        ).value,
        "padded": requires_padded_seq(hf.model_type, "megatron-bridge"),
        "parallel": {"tp": 4, "pp": 4, "cp": 4, "ep": 1},
        "linear_num_key_heads": provider.linear_num_key_heads,
        "linear_num_value_heads": provider.linear_num_value_heads,
        "num_layers": provider.num_layers,
        "no_model_or_trainer_instantiated": True,
    }
    cpu_optimizer = engine.MCoreOptimizerConfig(
        optimizer_cpu_offload=True,
        optimizer_offload_fraction=1.0,
        use_torch_optimizer_for_cpu_offload=True,
        overlap_cpu_optimizer_d2h_h2d=True,
        use_precision_aware_optimizer=False,
    )
    assert cpu_optimizer.optimizer_cpu_offload
    assert cpu_optimizer.optimizer_offload_fraction == 1.0
    assert cpu_optimizer.use_torch_optimizer_for_cpu_offload
    assert cpu_optimizer.overlap_cpu_optimizer_d2h_h2d
    assert not cpu_optimizer.use_precision_aware_optimizer
    result["cpu_optimizer_options_accepted"] = True
    result["attention_backend"] = provider.attention_backend.name
    result["mcore_attention_environment_check_passed"] = True
    from qwen27_train_rl import decorate_dense_controller

    captured = {}

    def capture_initialize(**kwargs):
        captured.update(kwargs)

    controller = SimpleNamespace(
        initialize=capture_initialize, start_proxy=lambda: None
    )
    decorate_dense_controller(controller).initialize(
        server_args={"model_path": model, "ple_offload_embedding": True}
    )
    assert "ple_offload_embedding" not in captured["server_args"]
    assert captured["server_args"]["model_path"] == model
    assert "linear_attn_prefill_backend" not in captured["server_args"]
    assert "linear_attn_decode_backend" not in captured["server_args"]
    result["dense_server_options_without_ple_verified"] = True
    result["dense_gdn_uses_native_defaults"] = True
    result["awex_source"] = __import__("awex").__file__
    result["megatron_source"] = __import__("megatron.core", fromlist=["core"]).__file__
    (Path(os.environ["EXPERIMENT_ROOT"]) / "runtime-validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
