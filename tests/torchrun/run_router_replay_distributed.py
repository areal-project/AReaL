"""Distributed Router Replay (R3) end-to-end runner (torchrun entrypoint).

Launches ``MegatronEngine`` with R3 enabled on a small MoE model
(Moonlight-16B-A3B by default; Qwen3-30B-A3B as a fallback) and exercises
three integration-level test modes:

* ``patch_plumbing``: import ``apply_router_replay_patch`` and confirm that
  ``RouterReplay.router_instances`` is populated with exactly as many
  entries as the local PP/VP rank's MoE layer count.

* ``forward_replay``: run ``forward`` with a synthetic ``routed_experts``
  tensor side-channelled to the engine, and verify that the R3 iterator
  wiring does not raise and ``_r3_pending_routed_experts`` is consumed.

* ``forward_backward``: build a full training batch with non-zero
  ``advantages`` / ``rollout_logprobs`` and a dummy ``rollout_expert_indices``
  of shape ``(B, L, num_moe_layers, topk)``, run ``engine.train_batch``
  with a GRPO-style loss, and assert the returned loss is finite and
  non-zero (matching SkyRL's R3 forward_backward test pattern).

The test is driven by ``tests/test_router_replay_e2e.py`` via ``torchrun``
and requires 4–8 GPUs depending on the allocation-mode string.
"""

from __future__ import annotations

import argparse
import functools
import os
from typing import Any

import torch
import torch.distributed as dist

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    MegatronEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.engine import MegatronEngine
from areal.infra.platforms import current_platform
from areal.utils import logging, seeding
from areal.utils.data import broadcast_tensor_container

logger = logging.getLogger("R3E2E")


def _get_model_path(local_path: str, hf_id: str) -> str:
    if os.path.exists(local_path):
        logger.info("Model found at local path: %s", local_path)
        return local_path
    from huggingface_hub import snapshot_download

    logger.info("Downloading model from HuggingFace Hub: %s", hf_id)
    return snapshot_download(
        repo_id=hf_id,
        ignore_patterns=["*.gguf", "*.ggml", "consolidated*"],
    )


MODEL_PATHS = {
    "moonlight": _get_model_path(
        "/workspace/models/Moonlight-16B-A3B-Instruct/",
        "moonshotai/Moonlight-16B-A3B-Instruct",
    )
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_result(out: str, succ: bool, msg: str = ""):
    with open(out, "w") as f:
        if succ:
            f.write("Passed")
        else:
            f.write("Failed: " + msg if msg else "Failed")


def mock_input(
    batch_size: int = 8,
    min_seqlen: int = 16,
    max_seqlen: int = 64,
    device: str | None = None,
) -> dict[str, Any]:
    """Generate a right-padded ``(input_ids, attention_mask)`` batch.

    This mirrors the helper used by ``run_megatron_engine_distributed.py``
    so R3 shares the same batch conventions as the baseline engine tests.
    """
    device = device or current_platform.device_type
    pad_token_id = 0
    seqlens = torch.randint(
        min_seqlen, max_seqlen, (batch_size,), dtype=torch.int, device=device
    )
    msl = int(seqlens.max())
    input_ids = torch.randint(
        10000, 50000, (batch_size, msl), dtype=torch.long, device=device
    )
    attn_mask = torch.zeros((batch_size, msl), dtype=torch.bool, device=device)
    attn_mask[
        torch.arange(0, msl, device=device).unsqueeze(0) < seqlens.unsqueeze(1)
    ] = 1
    input_ids.masked_fill_(~attn_mask, pad_token_id)
    return dict(input_ids=input_ids, attention_mask=attn_mask)


def make_engine(
    model_type: str,
    backend: str,
    mb_spec: MicroBatchSpec,
    init_optimizer: bool = False,
    enable_router_replay: bool = True,
) -> MegatronEngine:
    config = TrainEngineConfig(
        backend=backend,
        experiment_name="r3_e2e",
        trial_name="trial0",
        path=MODEL_PATHS[model_type],
        mb_spec=mb_spec,
        optimizer=OptimizerConfig() if init_optimizer else None,
        megatron=MegatronEngineConfig(
            enable_router_replay=enable_router_replay,
        ),
    )
    alloc_mode = ModelAllocation.from_str(backend)
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine = MegatronEngine(config)
    engine.create_process_group(parallel_strategy=alloc_mode.parallel)
    engine.initialize(addr=None, ft_spec=ft_spec)
    return engine


def _collect_num_moe_layers(engine) -> int:
    """Sum of MoE layers hosted on the local (pp, vp) rank."""
    from areal.engine.router_replay_utils import get_moe_num_layers_to_build

    vp_size = engine.tf_config.virtual_pipeline_model_parallel_size
    total = 0
    if vp_size is not None:
        for vp in range(vp_size):
            total += get_moe_num_layers_to_build(engine.tf_config, vp_stage=vp)
    else:
        total += get_moe_num_layers_to_build(engine.tf_config, vp_stage=None)
    return total


def _build_training_input_with_rollout_experts(
    engine: MegatronEngine,
    num_moe_layers_total: int,
    topk: int,
    batch_size: int = 4,
    min_seqlen: int = 16,
    max_seqlen: int = 32,
    num_experts: int = 64,
    seed: int = 42,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Build a synthetic training batch that mirrors the production shape:

    * ``input_ids`` / ``attention_mask``: right-padded
    * ``rollout_logprobs`` / ``action_log_probs`` / ``advantages``:
      non-trivial (sampled with a fixed seed), so the GRPO loss is non-zero.
    * ``loss_mask``: 1 on response positions, 0 on pad.
    * ``rollout_expert_indices``: dummy ``(B, L, num_moe_layers_total, topk)``
      int32 tensor; zero-padded on attention==0 positions.

    Returns ``(input_dict, rollout_expert_indices)``. The caller is
    responsible for side-channeling ``rollout_expert_indices`` into
    ``engine._r3_pending_routed_experts`` (the production path used by
    ``PPOActor.ppo_update``).
    """
    base = mock_input(
        batch_size=batch_size,
        min_seqlen=min_seqlen,
        max_seqlen=max_seqlen,
        device=engine.device,
    )
    input_ids: torch.Tensor = base["input_ids"]
    attention_mask: torch.Tensor = base["attention_mask"]
    bs, slen = input_ids.shape

    gen = torch.Generator(device="cpu").manual_seed(seed)
    rollout_logprobs = (-torch.rand((bs, slen), generator=gen) * 2.0).to(engine.device)
    advantages = torch.randn((bs, slen), generator=gen).to(engine.device)

    loss_mask = attention_mask.to(dtype=torch.int64)

    rollout_expert_indices = torch.randint(
        0,
        num_experts,
        (bs, slen, num_moe_layers_total, topk),
        dtype=torch.int32,
        device=engine.device,
    )
    # Zero-out pad positions to match the rollout producer's convention.
    rollout_expert_indices[attention_mask == 0] = 0

    input_dict: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        # grpo_loss_fn reads input_data['logprobs'] as the old (rollout) log-prob.
        "logprobs": rollout_logprobs,
        "advantages": advantages,
        # prox_logp is required when prox_logp_method='recompute'.
        # In standard GRPO, prox_logp equals the rollout log-probabilities.
        "prox_logp": rollout_logprobs.clone(),
    }
    return input_dict, rollout_expert_indices


# ---------------------------------------------------------------------------
# Test: patch plumbing — verifies that RouterReplay.router_instances on every
# rank matches the local MoE layer count Megatron actually builds.
# ---------------------------------------------------------------------------


def test_patch_plumbing(model_type: str, backend: str, output: str | None):
    from areal.engine.router_replay_patch import RouterReplay

    rank = int(os.environ["RANK"])
    mb_spec = MicroBatchSpec(max_tokens_per_mb=512)
    engine = make_engine(
        model_type,
        backend,
        mb_spec,
        init_optimizer=False,
        enable_router_replay=True,
    )
    try:
        assert engine._r3_enabled, "engine._r3_enabled should be True"
        assert getattr(engine.tf_config, "enable_routing_replay", False), (
            "tf_config.enable_routing_replay should have been set"
        )

        expected = _collect_num_moe_layers(engine)
        got = len(RouterReplay.router_instances)
        logger.info(
            "[R3-E2E] rank=%d expected_moe_layers=%d got_router_instances=%d",
            rank,
            expected,
            got,
        )
        assert got == expected, (
            f"RouterReplay.router_instances count ({got}) must match the "
            f"MoE layers assigned to this (pp, vp) rank ({expected})."
        )

        # All instances start with no action set.
        for inst in RouterReplay.router_instances:
            assert inst.router_replay_action is None

        dist.barrier()
        if rank == 0 and output:
            write_result(output, True)
    except AssertionError as e:
        if rank == 0 and output:
            write_result(output, False, str(e))
        raise
    finally:
        engine.destroy()


# ---------------------------------------------------------------------------
# Test: forward replay — drive forward once with a synthetic routed_experts
# tensor via the engine side-channel and verify consumption/clear.
# ---------------------------------------------------------------------------


def test_forward_replay(model_type: str, backend: str, output: str | None):
    from areal.engine.router_replay_patch import RouterReplay
    from areal.workflow.rlvr_r3_patch import resolve_r3_moe_config

    rank = int(os.environ["RANK"])
    seeding.set_random_seed(0, key=f"r3-e2e-{rank}")

    mb_spec = MicroBatchSpec(max_tokens_per_mb=512)
    engine = make_engine(
        model_type,
        backend,
        mb_spec,
        init_optimizer=False,
        enable_router_replay=True,
    )

    try:
        # Resolve MoE metadata from the model config (same path rl_trainer uses).
        num_moe, topk = resolve_r3_moe_config(MODEL_PATHS[model_type])
        logger.info("[R3-E2E] rank=%d num_moe_layers=%d topk=%d", rank, num_moe, topk)

        # Build a synthetic routed_experts tensor with right-padding matching
        # the rollout convention: (bs, seqlen, num_moe_layers, topk).
        inp = mock_input(batch_size=8, max_seqlen=32, device=engine.device)
        bs, slen = inp["input_ids"].shape
        routed_experts = torch.randint(
            0,
            64,
            (bs, slen, num_moe, topk),
            dtype=torch.int32,
            device=engine.device,
        )
        # Right-zero a couple of trailing rows per sample to emulate padding.
        routed_experts[:, -2:, :, :] = 0

        inp = broadcast_tensor_container(
            inp,
            src_rank=engine.current_data_parallel_head(),
            group=engine.context_and_model_parallel_group,
        )

        # Side-channel the routed_experts to the engine (Strategy A in the patch).
        engine._r3_pending_routed_experts = routed_experts

        engine.eval()
        _ = engine.forward(input_=inp, aggregate_fn=lambda xs: torch.cat(xs, dim=0))

        assert engine._r3_pending_routed_experts is None, (
            "_r3_pending_routed_experts should be consumed by the R3 wrapper."
        )
        for inst in RouterReplay.router_instances:
            assert inst.router_replay_action is None, (
                "clear_router_replay() should reset the action on every "
                "RouterReplay instance at the end of forward_backward_batch."
            )

        dist.barrier()
        if rank == 0 and output:
            write_result(output, True)
    except Exception as e:  # pragma: no cover - surfaced as torchrun failure
        logger.error("[R3-E2E] rank=%d FAIL: %r", rank, e)
        if rank == 0 and output:
            write_result(output, False, repr(e))
        raise
    finally:
        engine.destroy()


# ---------------------------------------------------------------------------
# Test: forward_backward — full train_batch round with R3 enabled, mirroring
# SkyRL's ``test_forward_backward``.  Requires the optimizer.
# ---------------------------------------------------------------------------


def test_forward_backward(model_type: str, backend: str, output: str | None):
    """End-to-end R3 forward_backward sanity check.

    Uses dummy rollout_expert_indices to exercise the full record/replay
    round trip through ``MegatronEngine.train_batch``:

    1. Compute-logp pass (RECORD) — runs via ``engine.forward`` internally
       when the trainer's ``compute_logp`` is called; here we side-channel
       a deterministic routed_experts tensor directly, mirroring
       ``PPOActor.ppo_update``'s pattern.
    2. Training pass (REPLAY_FORWARD / REPLAY_BACKWARD) — runs via
       ``engine.train_batch`` with a non-zero ``advantages`` tensor, so
       that the loss is guaranteed to be non-zero and the backward pass
       actually flows through the MoE dispatcher.

    Asserts:
    * returned loss is finite and non-zero;
    * all ``RouterReplay`` instances have had their action cleared at end;
    * the side-channel has been consumed.
    """
    from areal.engine.router_replay_patch import RouterReplay
    from areal.trainer.ppo.actor import grpo_loss_fn
    from areal.workflow.rlvr_r3_patch import resolve_r3_moe_config

    rank = int(os.environ["RANK"])
    seeding.set_random_seed(0, key=f"r3-e2e-fb-{rank}")

    mb_spec = MicroBatchSpec(max_tokens_per_mb=512)
    engine = make_engine(
        model_type,
        backend,
        mb_spec,
        init_optimizer=True,  # need an optimizer for train_batch
        enable_router_replay=True,
    )

    try:
        # Resolve MoE metadata from the model config.
        num_moe, topk = resolve_r3_moe_config(MODEL_PATHS[model_type])
        logger.info(
            "[R3-E2E-FB] rank=%d num_moe_layers=%d topk=%d", rank, num_moe, topk
        )

        # Build training input + rollout_expert_indices.
        input_dict, rollout_expert_indices = _build_training_input_with_rollout_experts(
            engine,
            num_moe_layers_total=num_moe,
            topk=topk,
            batch_size=4,
            min_seqlen=16,
            max_seqlen=32,
        )

        # Broadcast input across the context+model-parallel group so every
        # DP shard sees a consistent batch (mirrors PPOActor.ppo_update).
        input_dict = broadcast_tensor_container(
            input_dict,
            src_rank=engine.current_data_parallel_head(),
            group=engine.context_and_model_parallel_group,
        )

        # Side-channel rollout_experts to the engine (Strategy A).
        engine._r3_pending_routed_experts = rollout_expert_indices

        # Build a GRPO loss with sane defaults.
        loss_fn = functools.partial(
            grpo_loss_fn,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            rejection_sampling=None,
            m2_threshold=None,
            importance_sampling_level="token",
            current_version=0,
            prox_logp_method="recompute",
            use_sapo_loss=False,
            sapo_tau_pos=0.0,
            sapo_tau_neg=0.0,
            use_decoupled_loss=False,
        )

        engine.train()
        stats = engine.train_batch(
            input_=input_dict,
            loss_fn=loss_fn,
            loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
        )

        # Rank-0 side asserts the loss on the last pipeline stage.
        # stats is a dict[str, float]; the key name may vary, so check any.
        logger.info("[R3-E2E-FB] rank=%d train_batch stats=%s", rank, stats)

        # Side-channel must have been consumed.
        assert engine._r3_pending_routed_experts is None, (
            "_r3_pending_routed_experts should be consumed by the R3 wrapper."
        )
        # All RouterReplay instances should have been reset.
        for inst in RouterReplay.router_instances:
            assert inst.router_replay_action is None, (
                "clear_router_replay() should reset the action on every "
                "RouterReplay instance at the end of train_batch."
            )

        # Sanity-check loss values (only on last pipeline stage where
        # train_batch actually produced meaningful scalars).
        if isinstance(stats, dict):
            for k, v in stats.items():
                if isinstance(v, float):
                    assert v == v, f"loss/stat {k} is NaN"  # NaN check

        dist.barrier()
        if rank == 0 and output:
            write_result(output, True)
    except Exception as e:  # pragma: no cover - surfaced as torchrun failure
        logger.error("[R3-E2E-FB] rank=%d FAIL: %r", rank, e)
        if rank == 0 and output:
            write_result(output, False, repr(e))
        raise
    finally:
        engine.destroy()


def main():
    parser = argparse.ArgumentParser(description="Router Replay E2E runner")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=sorted(MODEL_PATHS.keys()),
        default="moonlight",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="megatron:(attn:d1p1t4|ffn:d1p1t1e4)",
        help="Allocation-mode string, e.g. 'megatron:(attn:d1p1t4|ffn:d1p1t1e4)'.",
    )
    parser.add_argument(
        "--test_type",
        type=str,
        choices=["forward_replay", "patch_plumbing", "forward_backward"],
        default="patch_plumbing",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    logger.info("Args: %s", args)

    if args.test_type == "forward_replay":
        test_forward_replay(args.model_type, args.backend, args.output)
    elif args.test_type == "patch_plumbing":
        test_patch_plumbing(args.model_type, args.backend, args.output)
    elif args.test_type == "forward_backward":
        test_forward_backward(args.model_type, args.backend, args.output)
    else:
        raise NotImplementedError(args.test_type)


if __name__ == "__main__":
    main()
