# SPDX-License-Identifier: Apache-2.0
"""Derive all implementation settings from the single public feature switch."""

import importlib.util
import math
import os
from pathlib import Path


def configure(config) -> None:
    if not config.sampling_mask_transfer:
        return
    if config.rollout._version != "v1" or config.actor._version != "v1":
        raise ValueError("sampling_mask_transfer supports AReaL v1 only")
    if not config.rollout.backend.startswith(
        "sglang:"
    ) or not config.actor.backend.startswith("megatron:"):
        raise ValueError("sampling_mask_transfer requires SGLang + Megatron")
    if config.sglang.speculative_algorithm:
        raise ValueError("sampling_mask_transfer does not support speculative decoding")
    if (
        config.actor.use_score_centering
        or config.actor.use_sapo_loss
        or config.actor.use_cispo_loss
    ):
        raise ValueError(
            "sampling_mask_transfer requires the PPO clipped actor objective"
        )
    if config.teacher is not None or config.mopd is not None:
        raise ValueError("sampling_mask_transfer with distillation is not implemented")
    if config.actor.kl_ctl != 0 and config.actor.kl_estimator != "k1":
        raise ValueError(
            "sampling_mask_transfer with a full-vocabulary reference requires "
            "kl_estimator='k1' (KL of the restricted rollout policy against the reference)"
        )
    for sampling in [config.gconfig, config.eval_gconfig]:
        if sampling is None:
            continue
        if (
            sampling.greedy
            or not math.isfinite(sampling.temperature)
            or sampling.temperature < 1e-6
        ):
            raise ValueError(
                "sampling_mask_transfer requires stochastic temperature >= 1e-6"
            )
        if (
            sampling.frequency_penalty != 0
            or sampling.use_beam_search
            or sampling.score_centering_top_k
        ):
            raise ValueError(
                "sampling_mask_transfer does not support penalties, beam search, or score centering"
            )
        if not 0 < sampling.top_p <= 1 or sampling.top_k == 0:
            raise ValueError("Invalid sampling_mask_transfer top_p/top_k")
        sampling._sampling_mask_transfer = True
    from areal.api.alloc_mode import ModelAllocation

    allocation = ModelAllocation.from_str(config.actor.backend, name="actor")
    config.rollout._sampling_mask_sampling = config.gconfig
    config.actor._sampling_mask_transfer = True
    config.actor.temperature = config.gconfig.temperature
    # PPO's denominator is the frozen restricted rollout policy, never a
    # recomputed proximal policy or a full-vocabulary behavior policy.
    config.actor.recompute_logprob = False
    config.actor.use_decoupled_loss = False
    config.actor.prox_logp_method = "recompute"
    config.actor.importance_sampling_level = "token"
    config.sglang.sampling_backend = "areal_sampling_mask"
    config.sglang._sampling_mask_retained_versions = (
        config.rollout.max_head_offpolicyness + 2
    )
    config.sglang._sampling_mask_top_k = config.gconfig.top_k
    config.sglang._sampling_mask_receivers = allocation.parallel.world_size
    # Torch wheels may bundle an older library with the same SONAME. Choose the
    # installed NCCL package before worker startup so training and inference do
    # not initialize different ABIs despite having the same Python environment.
    try:
        nccl = importlib.util.find_spec("nvidia.nccl")
    except ModuleNotFoundError:
        nccl = None
    library_dir = (
        str(Path(next(iter(nccl.submodule_search_locations))) / "lib")
        if nccl is not None and nccl.submodule_search_locations
        else None
    )
    for role in (config.actor, config.rollout):
        for spec in getattr(role, "scheduling_spec", ()):
            spec.env_vars["NCCL_CUMEM_ENABLE"] = "1"
            if library_dir is not None:
                previous = spec.env_vars.get(
                    "LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", "")
                )
                spec.env_vars["LD_LIBRARY_PATH"] = ":".join(
                    [library_dir]
                    + [
                        path
                        for path in previous.split(":")
                        if path and path != library_dir
                    ]
                )


def preflight(config, vocab_size: int) -> dict:
    from areal.api.alloc_mode import ModelAllocation
    from areal.utils.sampling_mask import estimate_mask_memory

    allocation = ModelAllocation.from_str(config.actor.backend, name="actor")
    max_tokens = (
        config.train_dataset.batch_size
        * config.gconfig.n_samples
        * min(config.gconfig.max_tokens, config.gconfig.max_new_tokens)
    )
    # Do not assume a uniform source-owner distribution. One sampler may retain
    # the whole batch. Training storage is sharded by actor DP.
    common = dict(
        max_tokens=max_tokens,
        top_k=config.gconfig.top_k,
        vocab_size=vocab_size,
        chunk_tokens=config.actor.mb_spec.max_tokens_per_mb or max_tokens,
        buffer_bytes=(8 << 20) * allocation.parallel.world_size,
    )
    # Acknowledgements are shard-local: shared prefixes can still be requested
    # by another shard. Source history is reclaimed only after policy expiry.
    retained = config.sglang._sampling_mask_retained_versions + 1
    source = {**common, "max_tokens": max_tokens * retained}
    return {
        "rollout_owner": estimate_mask_memory(data_parallel_size=1, **source),
        "actor_rank": estimate_mask_memory(
            data_parallel_size=allocation.parallel.data_parallel_size, **common
        ),
    }
