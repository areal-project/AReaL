# SPDX-License-Identifier: Apache-2.0
"""Qwen worker: CPU Adam and FlashAttention for the 256K memory budget.

These Megatron options are not exposed by AReaL's config schema yet. Keep the
compatibility overrides scoped to this recipe's worker process.
"""

import os
import runpy


def main():
    os.environ.update(NVTE_FUSED_ATTN="0", NVTE_FLASH_ATTN="1", NVTE_UNFUSED_ATTN="0")

    from examples.swe.qwen38_flash_next.grad_norm_guard import install as install_guard

    install_guard()

    if os.environ.get("QWEN_GDN_CP_COMPAT") == "1":
        from examples.swe.qwen38_flash_next.gdn_cp_compat import install

        install()
    chunk_tokens = int(os.environ.get("QWEN_PLE_CHUNK_TOKENS", "0"))
    if chunk_tokens:
        from examples.swe.qwen38_flash_next.ple_chunked import install

        install(chunk_tokens)

    import torch
    from megatron.core.transformer.enums import AttnBackend

    import areal.engine.megatron_engine as engine

    optimizer_config = engine.MCoreOptimizerConfig
    bridge_adapter = engine.MCoreBridgeAdapter

    def cpu_optimizer_config(*args, **kwargs):
        kwargs.update(
            optimizer_cpu_offload=True,
            optimizer_offload_fraction=1.0,
            use_torch_optimizer_for_cpu_offload=True,
            overlap_cpu_optimizer_d2h_h2d=True,
            use_precision_aware_optimizer=False,
            main_grads_dtype=torch.float32,
            main_params_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.float32,
        )
        config = optimizer_config(*args, **kwargs)
        # The engine has selected the rank-local device before creating Adam.
        torch.cuda.set_per_process_memory_fraction(1.0, torch.cuda.current_device())
        return config

    class FlashBridge(bridge_adapter):
        def __init__(self, *args, **kwargs):
            overrides = dict(kwargs.get("transformer_config_overrides") or {})
            overrides["attention_backend"] = AttnBackend.flash
            kwargs["transformer_config_overrides"] = overrides
            super().__init__(*args, **kwargs)

    engine.MCoreOptimizerConfig = cpu_optimizer_config
    engine.MCoreBridgeAdapter = FlashBridge
    runpy.run_module("areal.infra.rpc.rpc_server", run_name="__main__")


if __name__ == "__main__":
    main()
