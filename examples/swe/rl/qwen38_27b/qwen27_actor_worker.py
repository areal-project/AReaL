"""Keep CPU Adam offload with the historical dense attention selection."""

import os
import runpy


def _install_cpu_optimizer(engine, torch) -> None:
    optimizer_config = engine.MCoreOptimizerConfig

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
        torch.cuda.set_per_process_memory_fraction(1.0, torch.cuda.current_device())
        return config

    engine.MCoreOptimizerConfig = cpu_optimizer_config


def main() -> None:
    # The Flash Next worker forces a different attention backend. The dense
    # reference leaves provider.attention_backend=auto and lets TE choose.
    for key in ("NVTE_FUSED_ATTN", "NVTE_FLASH_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(key, None)

    from examples.swe.qwen38_flash_next.grad_norm_guard import install as install_guard

    install_guard()

    import torch

    import areal.engine.megatron_engine as engine

    _install_cpu_optimizer(engine, torch)
    runpy.run_module("areal.infra.rpc.rpc_server", run_name="__main__")


if __name__ == "__main__":
    main()
