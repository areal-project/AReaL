# SPDX-License-Identifier: Apache-2.0

"""CPU model contracts, source packing parity and frozen public-data validation."""

import ast
import bisect
import json
import random
import shutil
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from examples.profile import cost_model_tools
from examples.profile.trace import (
    actor_trace_times,
    parse_microbatch_log,
    stage_metrics,
    trace_events,
)

from areal.elastic.contracts import WorkloadStats
from areal.elastic.reconfigurator import reference as expert
from areal.elastic.reconfigurator.reference.identity import fingerprint

ROOT = Path(__file__).resolve().parents[1]


def qwen4b_components(*, dp=1, tp=1, replicas=1, rollout_tp=1):
    return expert.Catalog().assemble(
        hardware="h20",
        topology="h20-two-node8",
        model="qwen3-4b-instruct-2507",
        actor="megatron-qwen3-4b-ppo",
        rollout="sglang-qwen3-4b",
        actor_parallel=expert.ParallelLayout(dp=dp, tp=tp),
        rollout_tp=rollout_tp,
        rollout_replicas=replicas,
        kv_capacity_tokens=65536,
        resource_pools={
            "actor": tuple(f"node0/gpu{i}" for i in range(8)),
            "rollout": tuple(f"node1/gpu{i}" for i in range(8)),
        },
    )


def test_expert_qwen4b_model_and_megatron_recipe_match_target():
    """Dense Qwen3 dimensions include shared embedding and no router weights."""
    bundle = qwen4b_components(dp=3)
    shape = bundle.actor.model
    assert shape.total_parameters == 4022468096
    assert shape.mlp_type == "dense" and shape.tied_embeddings
    assert shape.router_parameters_per_layer == 0
    parsed = expert.ActorConfig.from_areal(
        {
            "backend": "megatron:d3t1",
            "dtype": "bfloat16",
            "mb_spec": {"n_mbs": 1, "max_tokens_per_mb": 32768},
            "optimizer": {"type": "adam", "weight_decay": 0.017},
            "gradient_checkpointing": True,
            "recompute_logprob": True,
            "use_decoupled_loss": True,
            "megatron": {
                "use_precision_aware_optimizer": True,
                "main_grads_dtype": "bfloat16",
                "main_params_dtype": "float32",
                "exp_avg_dtype": "float32",
                "exp_avg_sq_dtype": "float32",
                "enable_chunked_logits": True,
                "entropy_requires_grad": False,
                "ddp": {
                    "grad_reduce_in_fp32": False,
                    "use_distributed_optimizer": True,
                    "overlap_grad_reduce": True,
                },
            },
        }
    )
    assert parsed == bundle.config.actor
    assert parsed.optimizer_master_storage_bytes is None
    workload = WorkloadStats(3, 3072, sequence_lengths=(1024,) * 3)
    estimate = bundle.actor.estimate(parsed, workload)
    assert dict(estimate.stages)["train_step"].seconds > 0
    assert estimate.total.seconds is None


def test_profile_resource_pools_are_supplied_by_each_assembly():
    """Changing role assignment leaves the component model and compute recipe intact."""
    first = qwen4b_components(dp=2, tp=2, replicas=2, rollout_tp=2)
    swapped = expert.Catalog().assemble(
        hardware="h20",
        topology="h20-two-node8",
        model="qwen3-4b-instruct-2507",
        actor="megatron-qwen3-4b-ppo",
        rollout="sglang-qwen3-4b",
        actor_parallel=expert.ParallelLayout(dp=2, tp=2),
        rollout_tp=2,
        rollout_replicas=2,
        kv_capacity_tokens=65536,
        resource_pools={
            "actor": tuple(f"node1/gpu{i}" for i in range(8)),
            "rollout": tuple(f"node0/gpu{i}" for i in range(8)),
        },
    )
    assert swapped.config.actor_devices == first.config.rollout_devices
    assert swapped.config.rollout_devices == first.config.actor_devices
    assert swapped.config.actor == first.config.actor
    assert swapped.config.rollout == first.config.rollout
    work = WorkloadStats(4, 512, sequence_lengths=(128,) * 4)
    assert swapped.actor.estimate(swapped.config.actor, work) == first.actor.estimate(
        first.config.actor, work
    )
    assert swapped.binding != first.binding


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_expert_dense_tp_partitions_ffn_and_keeps_replicated_norms(tp):
    """Dense weights shard over TP and all gradient traffic uses the true DP group."""
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor import (
        backend_for,
    )
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.operators import (
        Context as ExpertContext,
    )
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.operators import (
        Operators as ExpertOperators,
    )
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.operators import (
        Transformer as ExpertTransformer,
    )

    bundle = qwen4b_components(tp=tp)
    cfg, model = (bundle.config.actor, bundle.actor.model)
    op = ExpertOperators(bundle.actor.hardware)
    ctx = ExpertContext(cfg, model, op, model.layers, True, True)
    dense, experts = ctx.owned_parameters
    replicated = (
        model.layers * (2 * model.hidden_size + 2 * model.head_dim) + model.hidden_size
    )
    assert experts == 0
    assert tp * dense == model.total_parameters + (tp - 1) * replicated
    mb = expert.Microbatch(128, 128, 128**2, 0)
    before = ExpertTransformer().forward(ctx, mb)
    slow_moe = replace(
        op, efficiency=replace(op.efficiency, moe=0.01, moe_imbalance=10)
    )
    assert ExpertTransformer().forward(replace(ctx, operators=slow_moe), mb) == before
    assert before["moe"] == before["raw_ep"] == 0
    dp = 2 if tp < 8 else 1
    cfg = replace(cfg, parallel=expert.ParallelLayout(dp=dp, tp=tp))
    ctx = replace(ctx, config=cfg)
    assert backend_for(ctx.config.backend).gradient_sync(ctx) == pytest.approx(
        op.collective(dense * cfg.grad_bytes, dp, "dense_dp", dp * tp)
    )


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_expert_dense_rollout_reuses_single_replica_calibration_and_kv_width(tp):
    """Per-TP service calibration transfers across replica counts, not phases."""
    bundle = qwen4b_components(rollout_tp=tp)
    model, cfg = (bundle.rollout, bundle.config.rollout)
    multi = replace(cfg, replicas=8 // tp)
    assert model.profile_key(cfg) == model.profile_key(multi)
    batches = {
        "prefill": (expert.ServiceSlice("p", 0, 32, "prefill", True),),
        "decode": (expert.ServiceSlice("d", 128, 1, "decode", True),),
        "mixed": (
            expert.ServiceSlice("p", 0, 32, "prefill", True),
            expert.ServiceSlice("d", 128, 1, "decode", True),
        ),
    }
    for index, (phase, batch) in enumerate(batches.items(), start=2):
        raw = model.estimate(cfg, expert.ServiceState(128), batch)
        assert dict(raw.features)["kv_bytes_per_token_per_rank"] == 144 * 1024 // tp
        calibration = expert.PhaseCalibration(
            "unit-only",
            raw.profile,
            phase,
            index,
            tuple(((key, value, value) for key, value in raw.features)),
            (f"unit-{phase}",),
            timing_boundary=model.timing_boundary,
        )
        fitted = replace(model, calibration=(calibration,))
        estimate = fitted.estimate(multi, expert.ServiceState(128), batch)
        assert estimate.duration.evidence == "calibrated"
        assert estimate.duration.seconds == pytest.approx(raw.duration.seconds * index)


def test_expert_new_dense_adapter_preserves_old_moe_numerics():
    """Pre-change 30B vectors are frozen; no calibration is loaded or refitted."""
    catalog = expert.Catalog()
    shape = catalog.model("qwen3-30b-a3b")
    hardware = catalog.hardware("h20", "h20-node8-r10")
    assert shape.total_parameters == 30532122624
    actor = expert.ExpertActorModel(shape, hardware, "old-numeric-audit")
    cfg = expert.ActorConfig(
        "megatron",
        expert.ParallelLayout(),
        recompute_logprob=True,
        gradient_checkpointing=True,
    )
    estimate = actor.estimate(cfg, WorkloadStats(2, 256, sequence_lengths=(128, 128)))
    assert dict(estimate.stages)["recompute_logp"].seconds == pytest.approx(
        0.038880961748146384
    )
    assert dict(estimate.stages)["train_step"].seconds == pytest.approx(
        0.49711008666829815
    )
    assert estimate.static_memory_gb == pytest.approx((549.578170368,))


@pytest.fixture
def expert_catalog_copy(tmp_path):
    source = expert.Catalog().root
    shutil.copytree(source / "profiles", tmp_path / "profiles")
    return expert.Catalog(tmp_path)


def test_expert_catalog_rejects_model_dimensions_inconsistent_with_hf_config(
    expert_catalog_copy,
):
    path = expert_catalog_copy.root / "profiles/models/qwen3-30b-a3b.json"
    value = json.loads(path.read_text())
    value["values"]["shape"]["layers"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="disagree"):
        expert_catalog_copy.model("qwen3-30b-a3b")


def test_expert_catalog_rejects_unknown_rollout_scheduler(expert_catalog_copy):
    path = (
        expert_catalog_copy.root
        / "profiles/execution/rollout/sglang-qwen3-30b-a3b.json"
    )
    value = json.loads(path.read_text())
    value["values"]["policy"] = "another_batching_rule"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="scheduler"):
        expert_catalog_copy.rollout(
            "sglang-qwen3-30b-a3b", tp=2, replicas=2, kv_capacity_tokens=65536
        )


def _fsdp_cost_config(**changes):
    return expert.ActorConfig(
        backend="fsdp",
        parallel=expert.ParallelLayout(dp=4),
        optimizer_type="adam_bf16",
        optimizer_dtype="bfloat16",
        fsdp_parameter_tensors=17,
        grad_bytes=2,
        offload_params=True,
        max_tokens_per_mb=512,
        **changes,
    )


def test_expert_fsdp_reads_real_offload_and_optimizer_recipe():
    """FSDP CPUOffloadPolicy differs from top-level model swapping."""
    config = expert.ActorConfig.from_areal(
        {
            "enable_offload": False,
            "actor": {
                "backend": "fsdp:d4",
                "dtype": "bfloat16",
                "grad_reduce_dtype": "bfloat16",
                "optimizer_dtype": "bfloat16",
                "optimizer": {"type": "adam_bf16", "weight_decay": 0.01},
                "mb_spec": {"max_tokens_per_mb": 16384},
                "fsdp": {"offload_params": True},
            },
        }
    )
    assert config.backend == "fsdp" and config.parallel.dp == 4
    assert config.offload_params and config.optimizer_type == "adam_bf16"
    assert config.grad_bytes == 2 and config.fsdp_root_reshard_after_forward
    with pytest.raises(ValueError, match="pure DP"):
        replace(config, parallel=expert.ParallelLayout(dp=2, tp=2))
    with pytest.raises(ValueError, match="BF16"):
        replace(config, optimizer_type="adam", optimizer_dtype="float32")


def test_expert_fsdp_packing_and_cpu_work_preserve_microbatch_boundaries(expert_model):
    """PPO calls reset CPU accumulation; checkpointing adds compute, not gathers."""
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.fsdp import (
        fsdp_work,
    )
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.packing import (
        plan_workload as expert_plan,
    )

    workload = WorkloadStats(8, 8 * 129, per_dp_sequence_lengths=((129, 129),) * 4)
    config = _fsdp_cost_config()
    plan = expert_plan(config, workload, expert_model.model.hidden_size)
    mb = plan.train[0][0][0]
    assert (
        mb.real_tokens,
        mb.compute_tokens,
        mb.attention_length_squared,
        mb.padding_tokens,
    ) == (258, 512, 97798, 254)
    shard = expert_model.model.total_parameters / 4
    work = dict(fsdp_work(config, expert_model.model, plan))
    train = dict(work["train_step"])
    assert train["h2d_bytes"] == 4 * shard
    assert train["d2h_bytes"] == 2 * shard
    assert train["cpu_accumulation_bytes"] == 0
    assert train["optimizer_logical_bytes"] == 68 * shard
    two = replace(config, n_mbs=2)
    work_two = dict(
        dict(fsdp_work(two, expert_model.model, expert_plan(two, workload, 128)))[
            "train_step"
        ]
    )
    assert work_two["h2d_bytes"] == 8 * shard
    assert work_two["cpu_accumulation_bytes"] == 6 * shard
    ppo = replace(config, ppo_n_minibatches=2)
    work_ppo = dict(
        dict(fsdp_work(ppo, expert_model.model, expert_plan(ppo, workload, 128)))[
            "train_step"
        ]
    )
    assert work_ppo["cpu_accumulation_bytes"] == 0
    assert work_ppo["optimizer_logical_bytes"] == 136 * shard
    with pytest.raises(ValueError):
        expert_plan(replace(ppo, n_mbs=2), workload, 128)


def test_expert_fsdp_missing_host_rates_remains_unknown(expert_model):
    """A valid backend must not silently fill unmeasured host work with zero."""
    workload = WorkloadStats(8, 8 * 129, per_dp_sequence_lengths=((129, 129),) * 4)
    config = _fsdp_cost_config()
    unknown = expert_model.estimate(
        config, workload, source_profile=expert_model.profile_key(config)
    )
    assert dict(unknown.stages)["train_step"].seconds is None
    assert dict(unknown.stages)["recompute_logp"].seconds == 0
    rates = expert.FSDPHostRates(8, 7, 20, 1e-05, 0.001, ("unit-only",))
    model = replace(expert_model, fsdp_host_rates=rates)
    known = model.estimate(config, workload, source_profile=model.profile_key(config))
    assert dict(known.stages)["train_step"].seconds > 0
    assert known.host_memory_gb[0] == pytest.approx(
        expert_model.model.total_parameters / 4 * 10 / 1000000000.0
    )
    assert known.static_memory_gb == (0,)
    assert model.profile_key(config) != expert_model.profile_key(config)
    assert dict(dict(known.fsdp_work)["train_step"])["gradient_reductions"] == 1


def test_expert_fsdp_checkpoint_does_not_gather_a_third_time(expert_model):
    """PyTorch PRE_BACKWARD suppresses checkpoint forward hooks."""
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.operators import (
        Context as ExpertContext,
    )
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.operators import (
        Operators as ExpertOperators,
    )
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.operators import (
        Transformer as ExpertTransformer,
    )
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.packing import (
        Microbatch,
    )

    config = _fsdp_cost_config()
    ctx = ExpertContext(
        config,
        expert_model.model,
        ExpertOperators(expert_model.hardware),
        4,
        True,
        True,
    )
    assert sum(ctx.owned_parameters) == expert_model.model.total_parameters
    mb = Microbatch(256, 256, 256**2, 0)
    transformer = ExpertTransformer()
    normal = transformer.estimate(ctx, mb, training=True)
    checkpointed = transformer.estimate(
        replace(ctx, config=replace(config, gradient_checkpointing=True)),
        mb,
        training=True,
    )
    assert checkpointed["compute"] > normal["compute"]
    assert checkpointed["raw_fsdp"] == normal["raw_fsdp"]
    assert normal["raw_gradient_sync"] > 0


def test_expert_fsdp_copy_launches_require_tensor_counts(expert_model):
    """Wrap-group counts cannot masquerade as parameter memcpy counts."""
    workload = WorkloadStats(8, 8 * 129, per_dp_sequence_lengths=((129, 129),) * 4)
    rates = expert.FSDPHostRates(8, 7, 20, 0.001, 0)
    model = replace(expert_model, fsdp_host_rates=rates)
    config = _fsdp_cost_config()
    known = model.estimate(config, workload, source_profile=model.profile_key(config))
    assert dict(dict(known.fsdp_work)["train_step"])["transfer_calls"] == 3 * 17
    missing = replace(config, fsdp_parameter_tensors=None)
    unknown = model.estimate(
        missing, workload, source_profile=model.profile_key(missing)
    )
    assert dict(unknown.stages)["train_step"].seconds is None
    assert "tensor count" in " ".join(dict(unknown.stages)["train_step"].notes)


def test_expert_fsdp_publish_stages_are_serial_and_unknowns_stay_unknown():
    """Staging uses current CPU shards; train service never pays AWEX twice."""
    rates = expert.FSDPHostRates(h2d_gbps=10, transfer_launch_seconds=0.001)
    tasks = expert.fsdp_publish_tasks(
        parameter_shard_bytes=10000000000.0,
        staging_tensors=2,
        contiguous_tensor_bytes=1000000000.0,
        rates=rates,
        gpu_copy_gbps=100,
        network=expert.Estimate(3, "assumed"),
        cleanup=expert.Estimate(0.1, "assumed"),
        gpu_copy_launch_seconds=0,
    )
    profile = expert.TransitionProfile("publish", "same", "same", tasks)
    result = expert.ExpertTransitionModel((profile,)).estimate(
        "publish", "same", "same"
    )
    assert result.duration.seconds == pytest.approx(0.1 + 1 + 0.002 + 0.02 + 3)
    unknown_tasks = expert.fsdp_publish_tasks(
        parameter_shard_bytes=10000000000.0,
        staging_tensors=2,
        contiguous_tensor_bytes=1000000000.0,
        rates=rates,
        gpu_copy_gbps=None,
        network=expert.Estimate(3, "assumed"),
        cleanup=expert.Estimate(0.1, "assumed"),
        gpu_copy_launch_seconds=0,
    )
    unknown = expert.ExpertTransitionModel(
        (replace(profile, tasks=unknown_tasks),)
    ).estimate("publish", "same", "same")
    assert unknown.duration.seconds is None
    copy_setup = expert.fsdp_publish_tasks(
        parameter_shard_bytes=10000000000.0,
        staging_tensors=2,
        contiguous_tensor_bytes=1000000000.0,
        contiguous_tensors=3,
        rates=rates,
        gpu_copy_gbps=100,
        gpu_copy_launch_seconds=0.001,
        network=expert.Estimate(3, "assumed"),
        cleanup=expert.Estimate(0.1, "assumed"),
    )
    assert copy_setup[2].estimate().seconds == pytest.approx(0.023)
    assert replace(copy_setup[2], setup_seconds=None).estimate().seconds is None
    no_copy = expert.fsdp_publish_tasks(
        parameter_shard_bytes=10000000000.0,
        staging_tensors=2,
        contiguous_tensor_bytes=0,
        contiguous_tensors=0,
        rates=rates,
        gpu_copy_gbps=None,
        network=expert.Estimate(3, "assumed"),
        cleanup=expert.Estimate(0.1, "assumed"),
    )
    assert no_copy[2].estimate().seconds == 0


def test_expert_fsdp_collectives_use_one_dp_group_and_each_wrap_startup(expert_model):
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor import (
        backend_for,
    )
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.operators import (
        Context as ExpertContext,
    )
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.operators import (
        Operators as ExpertOperators,
    )

    config = _fsdp_cost_config()
    ops = ExpertOperators(
        expert_model.hardware, group_gbps={"dense_dp": 40, "expert_dp": 1}
    )
    ctx = ExpertContext(config, expert_model.model, ops, 4, True, True)
    params = expert_model.model.total_parameters
    groups = expert_model.model.layers + 2
    comm = backend_for("fsdp")
    assert comm.gradient_sync(ctx) == pytest.approx(
        ops.collective(params * 2, 4, "dense_dp", 4)
        + (groups - 1) * ops.collective(0, 4, "dense_dp", 4)
        + ops.memory(params * 4)
    )
    other = replace(
        ctx, operators=replace(ops, group_gbps={"dense_dp": 40, "expert_dp": 100})
    )
    assert comm.gradient_sync(other) == comm.gradient_sync(ctx)
    assert comm.parameter_gather(other) == comm.parameter_gather(ctx)


def test_expert_fsdp_resident_shards_count_foreach_gather_input_copy(expert_model):
    """Torch's resident path packs a BF16 shard before collective copy-in."""
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor import (
        backend_for,
    )
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.fsdp import (
        fsdp_work,
    )
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.operators import (
        Context as ExpertContext,
    )
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.operators import (
        Operators as ExpertOperators,
    )
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.packing import (
        plan_workload as expert_plan,
    )

    config = _fsdp_cost_config(recompute_logprob=True, n_mbs=2)
    resident = replace(config, offload_params=False)
    workload = WorkloadStats(8, 8 * 129, per_dp_sequence_lengths=((129, 129),) * 4)
    plan = expert_plan(config, workload, expert_model.model.hidden_size)
    offload_work = dict(fsdp_work(config, expert_model.model, plan))
    resident_work = dict(fsdp_work(resident, expert_model.model, plan))
    local_copy = 2 * 2 * expert_model.model.total_parameters / config.parallel.dp
    assert dict(resident_work["recompute_logp"])["ag_copy_logical_bytes"] - dict(
        offload_work["recompute_logp"]
    )["ag_copy_logical_bytes"] == pytest.approx(local_copy * len(plan.logprob[0]))
    assert dict(resident_work["train_step"])["ag_copy_logical_bytes"] - dict(
        offload_work["train_step"]
    )["ag_copy_logical_bytes"] == pytest.approx(
        2 * local_copy * sum(len(call[0]) for call in plan.train)
    )
    assert dict(resident_work["train_step"])["cpu_logical_bytes"] == 0
    operators = ExpertOperators(expert_model.hardware)
    context = ExpertContext(
        config, expert_model.model, operators, expert_model.model.layers, True, True
    )
    comm = backend_for("fsdp")
    assert comm.parameter_gather(
        replace(context, config=resident)
    ) - comm.parameter_gather(context) == pytest.approx(operators.memory(local_copy))


def test_expert_fsdp_cpu_offload_keeps_gpu_working_memory_bounds(expert_model):
    workload = WorkloadStats(8, 8 * 129, per_dp_sequence_lengths=((129, 129),) * 4)
    config = _fsdp_cost_config()
    model = replace(expert_model, fsdp_host_rates=expert.FSDPHostRates(8, 7, 20, 0, 0))
    prediction = model.estimate(
        config, workload, source_profile=model.profile_key(config)
    )
    assert prediction.static_memory_gb == (0,)
    bounds = dict(prediction.gpu_working_set_lower_bounds_gb)
    assert (
        bounds["largest_bf16_logits"] == 512 * model.model.vocab_size * 2 / 1000000000.0
    )
    assert all((stage.feasibility == "unknown" for _, stage in prediction.stages))
    tiny = replace(
        model, hardware=replace(model.hardware, memory_gb=min(bounds.values()) / 2)
    )
    impossible = tiny.estimate(
        config, workload, source_profile=tiny.profile_key(config)
    )
    assert impossible.total.feasibility == "infeasible"
    resident_config = replace(config, offload_params=False)
    resident = expert_model.estimate(
        resident_config,
        workload,
        source_profile=expert_model.profile_key(resident_config),
    )
    assert dict(resident.stages)["train_step"].seconds > 0
    assert resident.host_memory_gb == (0.0,)
    assert resident.static_memory_gb[0] > 0


def test_metrics_adapter_does_not_confuse_loss_and_outer_padding():
    """n_tokens is rectangular padding; seq_len/avg supplies attended tokens."""
    stats = WorkloadStats.from_areal_metrics(
        {
            "ppo_actor/n_seqs": 32,
            "ppo_actor/seq_len/avg": 2048,
            "ppo_actor/seq_len/max": 4096,
            "ppo_actor/n_tokens": 32 * 8192,
            "ppo_actor/n_valid_tokens": 10000,
            "ppo_actor/update/num_micro_batches__count": 999,
        }
    )
    assert stats.sequence_tokens == 65536
    assert stats.loss_tokens == 10000 and stats.outer_padded_tokens == 262144
    assert stats.train_microbatches is None
    with pytest.raises(ValueError, match="seq_len/avg"):
        WorkloadStats.from_areal_metrics(
            {"ppo_actor/n_seqs": 32, "ppo_actor/n_tokens": 65536}
        )


@pytest.mark.parametrize(
    "stats",
    [
        WorkloadStats(4, 14, min_seq_len=1, max_seq_len=6),
        WorkloadStats(2, 7, min_seq_len=2, max_seq_len=5),
        WorkloadStats(1, 7, min_seq_len=7, max_seq_len=7),
        WorkloadStats(32, 32768),
    ],
)
def test_summary_surrogates_preserve_token_count_and_extrema(stats):
    """Both nominal and bounded-tail scenarios honor known summary constraints."""
    for tail in (False, True):
        lengths = stats.lengths(tail=tail, capacity=32768)
        assert len(lengths) == stats.n_seqs and sum(lengths) == stats.sequence_tokens
        if stats.min_seq_len is not None:
            assert min(lengths) == stats.min_seq_len
        if stats.max_seq_len is not None:
            assert max(lengths) == stats.max_seq_len


def test_exact_lengths_reject_inconsistent_stats():
    """Optional richer data cannot contradict the normal summary row."""
    with pytest.raises(ValueError, match="max_seq_len"):
        WorkloadStats(2, 7, max_seq_len=6, sequence_lengths=(2, 5))
    with pytest.raises(ValueError, match="attended"):
        WorkloadStats(2, 7, loss_tokens=20)
    with pytest.raises(ValueError, match="positive"):
        WorkloadStats(0, 7)


def test_rank_log_adapter_keeps_scope_ambiguous_until_provided():
    """One line does not identify actor phase or global optimizer iteration."""
    line = "[Megatron Rank 3] #microbatch: 2, microbatch #tokens: [100, 200], aligned to: [100, 200], padded to: [512, 512], padding lengths: [412, 312]."
    row = parse_microbatch_log(line)
    assert row["rank"] == 3 and row["microbatches"] == 2
    assert row["phase"] is None and row["global_step"] is None
    assert (
        parse_microbatch_log("Microbatch #tokens (rank 7): [100], padded to: [512]")[
            "rank"
        ]
        == 7
    )
    assert parse_microbatch_log("unrelated") is None


def test_trace_adapter_avoids_nested_and_cross_rank_double_counting():
    """Trainer/worker scopes overlap and cannot all be summed as GPU time."""
    events = [
        {
            "name": "train.ppo_update",
            "ph": "X",
            "ts": 0,
            "dur": 1000000.0,
            "pid": 0,
            "tid": 1,
            "args": {"global_step": 4},
        },
        {
            "name": "train.ppo_update",
            "ph": "X",
            "ts": 100000.0,
            "dur": 500000.0,
            "pid": 0,
            "tid": 1,
            "args": {"global_step": 4},
        },
        {
            "name": "ppo_actor.ppo_update",
            "ph": "X",
            "ts": 0,
            "dur": 900000.0,
            "pid": 1,
            "tid": 1,
            "args": {"global_step": 4},
        },
        {
            "name": "ppo_actor.ppo_update",
            "ph": "X",
            "ts": 0,
            "dur": 800000.0,
            "pid": 2,
            "tid": 1,
            "args": {"global_step": 4},
        },
    ]
    report = actor_trace_times(trace_events(json.dumps({"traceEvents": events})))
    assert report["steps_seconds"] == {"4": {"train_step": 1}}
    assert (
        actor_trace_times(events, level="worker")["steps_seconds"]["4"]["train_step"]
        == 0.9
    )
    events[2]["args"] = {}
    assert actor_trace_times(events, level="worker")["unassigned_events"] == 1
    assert stage_metrics({"timeperf/train_step": 2.5, "timeperf/rollout": 9}) == {
        "train_step": 2.5
    }


def test_trace_rank_identity_handles_reused_pids_across_nodes():
    """Remote ranks may reuse the same OS pid; union must not merge their steps."""
    events = [
        {
            "name": "ppo_actor.ppo_update",
            "ph": "X",
            "ts": 0,
            "dur": 1000000.0,
            "pid": 10,
            "tid": 1,
            "args": {"global_step": 4, "rank": 0},
        },
        {
            "name": "ppo_actor.ppo_update",
            "ph": "X",
            "ts": 2000000.0,
            "dur": 1000000.0,
            "pid": 10,
            "tid": 1,
            "args": {"global_step": 4, "rank": 8},
        },
    ]
    assert actor_trace_times(events)["steps_seconds"] == {"4": {"train_step": 1}}


def test_jsonl_trace_inherits_parent_step_without_summing_parent():
    """Real JSONL scopes can inherit step metadata from a containing event."""
    parent = {
        "name": "worker.step",
        "ph": "X",
        "ts": 0,
        "dur": 2000000.0,
        "pid": 1,
        "tid": 2,
        "args": {"global_step": 3, "rank": 4},
    }
    child = {
        "name": "ppo_actor.compute_logp",
        "ph": "X",
        "ts": 100000.0,
        "dur": 500000.0,
        "pid": 1,
        "tid": 2,
        "args": {"rank": 4},
    }
    report = actor_trace_times(
        trace_events(json.dumps(parent) + "\n" + json.dumps(child))
    )
    assert report["steps_seconds"] == {"3": {"recompute_logp": 0.5}}
    assert report["unassigned_events"] == 0


@pytest.fixture
def expert_model():
    shape = expert.ModelShape(
        layers=4,
        hidden_size=128,
        intermediate_size=64,
        attention_heads=4,
        kv_heads=2,
        head_dim=32,
        experts=4,
        top_k=2,
        vocab_size=1024,
    )
    hardware = expert.Hardware("unit-only", 100, 1000, 100, 20, 8, 80)
    return expert.ExpertActorModel(
        shape,
        hardware,
        runtime_revision="unit-runtime",
        advantage_model=unit_advantage_model(),
    )


def unit_advantage_model():
    """Explicit arithmetic oracles, never target machine parameters."""
    return expert.AdvantageModel(
        "unit-options",
        "cpu",
        "single",
        "token",
        (
            expert.AdvantageTerm(
                "batching", (("padded_slots", 1e-08),), ("unit-only",)
            ),
            expert.AdvantageTerm(
                "recurrence", (("gae_iterations", 1e-06),), ("unit-only",)
            ),
            expert.AdvantageTerm(
                "normalization", (("constant", 1e-05),), ("unit-only",)
            ),
            expert.AdvantageTerm("transport", (("constant", 0.0001),), ("unit-only",)),
            expert.AdvantageTerm(
                "diagnostics", (("constant", 0.0001),), ("unit-only",)
            ),
        ),
    )


def _expert_actor_input():
    cfg = expert.ActorConfig(
        "archon", expert.ParallelLayout(dp=2), max_tokens_per_mb=256
    )
    return (
        cfg,
        WorkloadStats(
            16,
            1024,
            sequence_lengths=(64,) * 16,
            trajectory_group_sizes=(1,) * 16,
            trajectory_dispatch_weights=(64,) * 16,
        ),
    )


def test_expert_pipeline_uses_dependencies_for_forward_and_one_f_one_b():
    """Two-stage/two-microbatch schedules have independently countable critical paths."""
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.model import (
        _pipeline_time,
    )

    assert _pipeline_time([[1, 1], [1, 1]], None) == 3
    assert _pipeline_time([[1, 1], [1, 1]], [[1, 1], [1, 1]]) == 6
    assert _pipeline_time([[1, 2, 3]], [[3, 2, 1]]) == 12


def test_expert_actor_is_independent_of_basic_prediction_and_preserves_ppo_work(
    expert_model, monkeypatch
):
    """Basic changes cannot alter the reference; PPO splits do not duplicate token work."""
    config, workload = _expert_actor_input()
    before = expert_model.estimate(config, workload)
    from areal.elastic.reconfigurator.basic.cost_model.actor import (
        ExpertActorModel as BasicActorModel,
    )

    monkeypatch.setattr(
        BasicActorModel,
        "estimate",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Basic called")),
    )
    after = expert_model.estimate(config, workload)
    split = expert_model.estimate(replace(config, ppo_n_minibatches=2), workload)
    assert before == after and before.total.seconds > 0
    assert before.total.evidence == "assumed" and before.total.feasibility == "unknown"
    assert len(split.packing.train) == 2
    assert (
        sum(
            m.real_tokens
            for update in split.packing.train
            for rank in update
            for m in rank
        )
        == workload.sequence_tokens
    )
    assert before.profile != split.profile


def test_expert_actor_phase_calibration_has_explicit_scope_and_no_silent_extrapolation(
    expert_model,
):
    """A phase scale is transferable only within its full profile and declared bounds."""
    config, workload = _expert_actor_input()
    raw = expert_model.estimate(config, workload)
    calibrations = tuple(
        (
            cost_model_tools.fit_phase_calibration(
                "unit-fit",
                (
                    expert.CalibrationPoint(
                        f"measured-{name}",
                        raw.profile,
                        name,
                        value.seconds,
                        value.seconds * 2,
                        raw.features_for(name),
                        timing_boundary=raw.timing_boundary,
                    ),
                ),
            )
            for name, value in raw.stages
            if value.seconds
        )
    )
    calibrated_model = replace(expert_model, calibration=calibrations)
    calibrated = calibrated_model.estimate(config, workload)
    assert calibrated.total.evidence == "calibrated"
    assert calibrated.total.seconds == pytest.approx(raw.total.seconds * 2)
    outside = replace(
        workload,
        n_seqs=32,
        sequence_tokens=2048,
        sequence_lengths=(64,) * 32,
        trajectory_group_sizes=(1,) * 32,
        trajectory_dispatch_weights=(64,) * 32,
    )
    assert calibrated_model.estimate(config, outside).total.seconds is None
    other = calibrations[0].apply(
        1, "different-layout", calibrations[0].phase, raw.features
    )
    assert other.evidence == "unsupported" and other.seconds is None


def test_expert_actor_rank_and_observed_packing_cannot_leak_across_profiles(
    expert_model,
):
    """Exact measured packing belongs to its source layout and complete model profile."""
    config, workload = _expert_actor_input()
    raw = expert_model.estimate(config, workload)
    observed = expert.ObservedPacking.for_workload(
        raw.profile, workload, raw.packing, ("rank-trace",)
    )
    same = expert_model.estimate(config, workload, observed_packing=observed)
    assert same.total.seconds == raw.total.seconds
    assert "rank-trace" in same.total.source_ids
    changed = replace(workload, sequence_lengths=(1, 127) + (64,) * 14)
    with pytest.raises(ValueError, match="another workload"):
        expert_model.estimate(config, changed, observed_packing=observed)
    with pytest.raises(ValueError, match="another profile"):
        expert_model.estimate(
            replace(config, n_mbs=2), workload, observed_packing=observed
        )
    rank_workload = replace(workload, per_dp_sequence_lengths=((64,) * 8, (64,) * 8))
    with pytest.raises(ValueError, match="source_profile"):
        expert_model.estimate(config, rank_workload)
    assert expert_model.estimate(
        config, rank_workload, source_profile=raw.profile
    ).packing.exact_dispatch


def test_expert_actor_logprob_auxiliary_stages_and_memory_bound_are_explicit(
    expert_model,
):
    """Metrics logprob still computes a forward; memory lower bounds can reject OOM."""
    config, workload = _expert_actor_input()
    metrics = replace(config, use_decoupled_loss=True, prox_logp_method="metrics")
    result = expert_model.estimate(
        metrics,
        workload,
        auxiliary_stages=(
            ("ref_logp", expert.Estimate(0.5, "measured", source_ids=("ref",))),
        ),
    )
    assert result.stages[0][0] == "ref_logp"
    assert dict(result.stages)["recompute_logp"].seconds > 0
    impossible = replace(
        expert_model, hardware=replace(expert_model.hardware, memory_gb=1e-08)
    ).estimate(config, workload)
    assert impossible.total.feasibility == "infeasible"
    assert (
        dict(expert_model.estimate(config, replace(workload, loss_tokens=0)).stages)[
            "train_step"
        ].seconds
        > 0
    )


def test_expert_actor_snapshot_and_nested_packing_are_immutable(expert_model):
    """Input containers cannot mutate a frozen reference after construction."""
    config, workload = _expert_actor_input()
    result = expert_model.estimate(config, workload)
    rows = [[list(rank) for rank in update] for update in result.packing.train]
    plan = expert.PackingPlan([], rows, True, True, [])
    rows[0][0].clear()
    assert plan.train[0][0]
    with pytest.raises(ValueError, match="Microbatch"):
        expert.Microbatch(8, 8, float("nan"), 0)
    bounds = [["tokens", 1, 2]]
    fit = expert.PhaseCalibration("v", "p", "phase", 1, bounds, ["source"])
    bounds[0][2] = 999
    assert fit.bounds == (("tokens", 1, 2),)


def test_expert_rollout_models_mixed_service_and_rejects_capacity_overflow(
    expert_model,
):
    """Inference query work is separate from queue/external latency and scoped by KV."""
    rollout = expert.ExpertRolloutModel(
        expert_model.model,
        expert_model.hardware,
        expert.InferenceRates(0.5, 0.5, 0.5, 0.5, 1e-06),
        "unit-runtime",
    )
    config = expert.RolloutConfig(2, 2, 4, 128, 4096)
    batch = (
        expert.ServiceSlice("prefill", 16, 32, "prefill", True),
        expert.ServiceSlice("decode", 100, 1, "decode", True),
    )
    result = rollout.estimate(config, expert.ServiceState(116), batch)
    assert result.duration.seconds > 0 and result.duration.evidence == "assumed"
    assert result.kv_tokens_after == 149
    assert dict(result.features)["prefill_tokens"] == 32
    assert dict(result.features)["decode_sequences"] == 1
    overflow = rollout.estimate(
        replace(config, kv_capacity_tokens=120), expert.ServiceState(116), batch
    )
    assert (
        overflow.duration.seconds is None
        and overflow.duration.feasibility == "infeasible"
    )
    with pytest.raises(ValueError, match="distinct"):
        rollout.estimate(config, expert.ServiceState(0), (batch[0], batch[0]))


def test_expert_transition_critical_path_uses_source_target_and_explicit_unknowns():
    """Parallel transfer branches overlap; absent restore evidence has no numeric cost."""
    tasks = (
        expert.TransferTask(
            "stage", measured=expert.Estimate(2, "measured", source_ids=("s",))
        ),
        expert.TransferTask(
            "network",
            ("stage",),
            transferred_bytes=100,
            bandwidth_bytes_s=20,
            setup_seconds=0,
        ),
        expert.TransferTask(
            "disk",
            ("stage",),
            measured=expert.Estimate(3, "measured", source_ids=("d",)),
        ),
    )
    model = expert.ExpertTransitionModel(
        (expert.TransitionProfile("restore", "source", "target", tasks),)
    )
    result = model.estimate("restore", "source", "target")
    assert result.duration.seconds == 7
    assert result.duration.evidence == "assumed"
    assert model.estimate("restore", "source", "different").duration.seconds is None
    unknown = expert.TransferTask("restore").estimate()
    assert unknown.seconds is None and unknown.evidence == "unsupported"
    with pytest.raises(ValueError, match="Cyclic"):
        expert.ExpertTransitionModel(
            (
                expert.TransitionProfile(
                    "restore",
                    "a",
                    "b",
                    (
                        expert.TransferTask(
                            "a", ("b",), measured=expert.Estimate(1, "assumed")
                        ),
                        expert.TransferTask(
                            "b", ("a",), measured=expert.Estimate(1, "assumed")
                        ),
                    ),
                ),
            )
        ).estimate("restore", "a", "b")


def test_expert_pipeline_config_and_transfers_preserve_backend_boundaries(
    expert_model, monkeypatch
):
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor import (
        backend_for,
    )
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.operators import (
        Communications,
        Context,
        Operators,
        Transformer,
    )

    data = {
        "backend": "archon:(attn:d1p4|ffn:d1p4)",
        "mb_spec": {"max_tokens_per_mb": 256},
    }
    with pytest.raises(ValueError, match="1F1B"):
        expert.ActorConfig.from_areal(data)
    data["archon"] = {"pp_schedule": "1F1B"}
    cfg = expert.ActorConfig.from_areal(data)
    assert backend_for(cfg.backend).pipeline_partition(
        cfg, replace(expert_model.model, layers=8)
    ) == (
        2,
        3,
        2,
        1,
    )
    data["archon"]["pp_layers_per_stage"] = 1
    with pytest.raises(ValueError, match="virtual"):
        expert.ActorConfig.from_areal(data)
    monkeypatch.setattr(Communications, "pipeline_transfer", lambda *args: 2.0)
    op = Operators(expert_model.hardware)
    mb = expert.Microbatch(8, 8, 64, 0)
    for first, last, expected in (
        (True, False, 2),
        (False, False, 4),
        (False, True, 2),
    ):
        ctx = Context(cfg, expert_model.model, op, 1, first, last)
        fwd = Transformer().forward(ctx, mb)
        full = Transformer().estimate(ctx, mb, training=True)
        assert fwd["raw_pipeline_transfer"] == (0 if last else 2)
        assert full["raw_pipeline_transfer"] == expected


def test_expert_rejects_known_archon_mesh_constraints_and_invalid_logprob_reuse(
    expert_model,
):
    cfg, work = _expert_actor_input()
    for parallel in (expert.ParallelLayout(tp=4), expert.ParallelLayout(cp=8)):
        result = expert_model.estimate(replace(cfg, parallel=parallel), work)
        assert result.total.feasibility == "infeasible" and result.total.seconds is None
    for decoupled in (False, True):
        with pytest.raises(ValueError, match="ppo_n_minibatches=1"):
            replace(
                cfg,
                use_decoupled_loss=decoupled,
                prox_logp_method="reuse_train_logp",
                ppo_n_minibatches=2,
            )
    assert replace(expert_model, runtime_revision="different-runtime").profile_key(
        cfg
    ) != expert_model.profile_key(cfg)


def test_expert_advantage_missing_boundary_model_stays_unknown(expert_model):
    """Unknown trainer wall costs cannot inherit the old tiny HBM-only estimate."""
    config, workload = _expert_actor_input()
    result = replace(expert_model, advantage_model=None).estimate(config, workload)
    phases = dict(result.stages)
    assert phases["compute_advantage"].seconds is None
    assert phases["train_step"].seconds > 0
    assert result.total.evidence == "unsupported"
    assert "HBM-only" in " ".join(phases["compute_advantage"].notes)


def test_expert_advantage_accounts_padded_width_and_slowest_worker(expert_model):
    """Use max of summed worker components, not sum of independent maxima."""
    config, workload = _expert_actor_input()
    terms = (
        expert.AdvantageTerm("batching", (("padded_slots", 1),), ("synthetic",)),
        expert.AdvantageTerm("recurrence", (("gae_iterations", 1),), ("synthetic",)),
        expert.AdvantageTerm("normalization", (("trajectories", 100),), ("synthetic",)),
        expert.AdvantageTerm("transport", (("input_bytes", 0.5),), ("synthetic",)),
        expert.AdvantageTerm("diagnostics", (("constant", 7),), ("synthetic",)),
    )
    model = replace(
        expert_model, advantage_model=replace(unit_advantage_model(), terms=terms)
    )
    observed = expert.AdvantageWorkload.for_workload(
        model.profile_key(config),
        workload,
        (
            expert.AdvantageRankWork(0, 8, 64, trajectories=8),
            expert.AdvantageRankWork(1, 8, 128, trajectories=1),
        ),
        ("shape-trace",),
        input_bytes=10,
    )
    prediction = model.estimate(config, workload, advantage_workload=observed)
    assert dict(prediction.stages)["compute_advantage"].seconds == 1387
    assert sum((e.seconds for _, e in prediction.advantage_components)) == 1387
    assert (
        dict(prediction.features_for("compute_advantage"))["advantage_padded_slots_max"]
        == 1024
    )
    assert "shape-trace" in prediction.total.source_ids
    with pytest.raises(ValueError, match="another profile"):
        model.estimate(replace(config, n_mbs=2), workload, advantage_workload=observed)


def test_expert_advantage_missing_parameters_and_payload_are_not_zero(expert_model):
    """None remains unsupported while explicit zero is a documented assumption."""
    config, workload = _expert_actor_input()
    model = unit_advantage_model()
    for replacement in (
        expert.AdvantageTerm("transport", (("input_bytes", 1),), ("synthetic",)),
        expert.AdvantageTerm("transport", (("constant", None),), ("unmeasured",)),
    ):
        changed = replace(
            model,
            terms=tuple(
                replacement if t.name == "transport" else t for t in model.terms
            ),
        )
        result = replace(expert_model, advantage_model=changed).estimate(
            config, workload
        )
        assert dict(result.stages)["compute_advantage"].seconds is None
    empty = replace(expert_model, advantage_model=replace(model, terms=()))
    assert "Missing advantage" in " ".join(empty.estimate(config, workload).total.notes)


def test_expert_advantage_turn_iterations_depend_on_device_and_semantic_identity(
    expert_model,
):
    """CPU turn GAE and CUDA turn GAE have different loop bounds."""
    rank = expert.AdvantageRankWork(0, 8, 100, turn_iterations=4)
    assert rank.features("cpu", "token")["gae_iterations"] == 99
    assert rank.features("cpu", "turn")["gae_iterations"] == 4
    assert rank.features("cuda", "turn")["gae_iterations"] == 100
    assert (
        expert.AdvantageRankWork(0, 8, 100).features("cpu", "turn")["gae_iterations"]
        is None
    )
    config, _ = _expert_actor_input()
    for change in (
        {"device": "cuda"},
        {"controller_mode": "multi"},
        {"execution_id": "new-norm"},
        {"gae_mode": "turn"},
    ):
        assert replace(
            expert_model, advantage_model=replace(unit_advantage_model(), **change)
        ).profile_key(config) != expert_model.profile_key(config)
    with pytest.raises(ValueError, match="trainer host wall"):
        replace(unit_advantage_model(), timing_boundary="worker_method_host_wall")


def test_expert_calibration_rejects_mixed_boundaries_and_reports_bounds(expert_model):
    """Coverage failures identify the exact feature without relaxing its range."""
    config, workload = _expert_actor_input()
    raw = expert_model.estimate(config, workload)
    point = expert.CalibrationPoint(
        "first",
        raw.profile,
        "compute_advantage",
        dict(raw.stages)["compute_advantage"].seconds,
        2,
        raw.features_for("compute_advantage"),
        raw.timing_boundary,
    )
    with pytest.raises(ValueError, match="timing boundary"):
        cost_model_tools.fit_phase_calibration(
            "unit",
            (
                point,
                replace(
                    point, sample_id="worker", timing_boundary="worker_method_host_wall"
                ),
            ),
        )
    fitted = cost_model_tools.fit_phase_calibration("unit", (point,))
    wrong = fitted.apply(
        1, raw.profile, point.phase, point.features, timing_boundary="rank_mean"
    )
    assert wrong.seconds is None and "boundary mismatch" in " ".join(wrong.notes)
    features = tuple(
        (
            (name, value * 2 if name == "n_seqs" else value)
            for name, value in point.features
        )
    )
    outside = fitted.apply(
        1, raw.profile, point.phase, features, timing_boundary=raw.timing_boundary
    )
    assert outside.seconds is None and any("n_seqs=" in note for note in outside.notes)
    accepted = fitted.apply(
        point.predicted_seconds,
        raw.profile,
        point.phase,
        point.features,
        timing_boundary=raw.timing_boundary,
    )
    assert accepted.seconds == pytest.approx(2)


def packing_records(config, plan, profile):
    """Synthetic pre-shard logs repeat payload over TP/CP/PP, never over DP."""
    records = []
    calls = [("train_step", i, ranks) for i, ranks in enumerate(plan.train)]
    if plan.logprob:
        calls.append(("recompute_logp", 0, plan.logprob))
    for phase, call, ranks in calls:
        for dp, rank in enumerate(ranks):
            for tp in range(config.parallel.tp):
                for cp in range(config.parallel.cp):
                    for pp in range(config.parallel.pp):
                        global_rank = (
                            (dp * config.parallel.tp + tp) * config.parallel.cp + cp
                        ) * config.parallel.pp + pp
                        batches = tuple(
                            expert.PackingMicrobatch(
                                mb.real_tokens,
                                mb.compute_tokens,
                                mb.compute_tokens - mb.padding_tokens,
                                mb.padding_tokens,
                                mb.attention_length_squared,
                            )
                            for mb in rank
                        )
                        records.append(
                            expert.PackingRecord(
                                "run",
                                "unit-runtime",
                                4,
                                phase,
                                call,
                                global_rank,
                                (dp, tp, cp, pp),
                                batches,
                                f"{phase}-{call}-{global_rank}",
                                profile,
                            )
                        )
    return tuple(records)


def test_expert_observed_packing_deduplicates_mesh_without_losing_dp_or_ppo_work(
    expert_model,
):
    """Multiple PPO calls partition tokens; TP/CP/PP copies do not multiply them."""
    config, workload = _expert_actor_input()
    config = replace(
        config,
        parallel=expert.ParallelLayout(dp=2, tp=2, cp=2, pp=2, ep=2),
        ppo_n_minibatches=2,
        recompute_logprob=True,
    )
    raw = expert_model.estimate(config, workload)
    records = packing_records(config, raw.packing, raw.profile)
    observed = expert_model.observe_packing(config, workload, records)
    result = expert_model.estimate(config, workload, observed_packing=observed)
    assert result.total.seconds == raw.total.seconds
    assert len(observed.plan.train) == 2
    assert (
        sum(
            m.real_tokens for call in observed.plan.train for rank in call for m in rank
        )
        == 1024
    )
    assert sum(m.real_tokens for rank in observed.plan.logprob for m in rank) == 1024
    assert len(observed.source_ids) == len(records)
    assert dict(result.features)["observed_packing"] == 1


@pytest.mark.parametrize(
    "problem",
    [
        "missing_dp",
        "duplicate_rank",
        "conflicting_copy",
        "step",
        "call",
        "mapping",
        "token_scope",
        "profile",
        "runtime",
    ],
)
def test_expert_observed_packing_rejects_ambiguous_log_attribution(
    expert_model, problem
):
    """Conflicting/missing metadata is rejected before constructing an exact plan."""
    config, workload = _expert_actor_input()
    config = replace(config, parallel=expert.ParallelLayout(dp=2, tp=2))
    raw = expert_model.estimate(config, workload)
    records = packing_records(config, raw.packing, raw.profile)
    with pytest.raises(ValueError):
        if problem == "missing_dp":
            records = tuple(r for r in records if r.mesh[0] == 0)
        elif problem == "duplicate_rank":
            records += (records[0],)
        elif problem == "conflicting_copy":
            mb = records[1].microbatches[0]
            changed = replace(
                mb, attention_length_squared=mb.attention_length_squared - 2
            )
            records = (
                records[0],
                replace(
                    records[1], microbatches=(changed,) + records[1].microbatches[1:]
                ),
            ) + records[2:]
        elif problem == "step":
            records = (replace(records[0], global_step=5),) + records[1:]
        elif problem == "call":
            records = tuple(replace(r, call_index=1) for r in records)
        elif problem == "mapping":
            records = (records[0], replace(records[1], global_rank=0)) + records[2:]
        elif problem == "profile":
            records = tuple(replace(r, profile="another-profile") for r in records)
        elif problem == "runtime":
            records = tuple(
                replace(r, source_revision="another-runtime") for r in records
            )
        else:
            records = (replace(records[0], token_scope="post_cp_shard"),) + records[1:]
        expert_model.observe_packing(config, workload, records)


def test_expert_packing_actual_alignment_overrides_requested_padding():
    """Megatron can enlarge the outer padding target after sequence alignment."""
    record = expert.PackingMicrobatch(
        7,
        aligned_tokens=8,
        padding_tokens=8,
        final_sequence_lengths=(4, 4, 8),
        padded_to_tokens=8,
    )
    resolved = record.resolve()
    assert resolved.compute_tokens == 16 and resolved.padding_tokens == 8
    assert resolved.attention_length_squared == 96
    with pytest.raises(ValueError, match="aligned"):
        replace(record, compute_tokens=8).resolve()
    with pytest.raises(ValueError, match="square sum"):
        replace(record, attention_length_squared=49).resolve()


def test_expert_partial_packing_never_invents_attention_work():
    """Token totals alone do not identify sequence-square work or actual padding."""
    partial = expert.PackingMicrobatch(8, padded_to_tokens=16)
    with pytest.raises(ValueError, match="Missing packing"):
        partial.resolve()
    no_attention = replace(partial, aligned_tokens=8, padding_tokens=8)
    with pytest.raises(ValueError, match="attention_length_squared"):
        no_attention.resolve()


@pytest.mark.parametrize("tail,squares", [(8, 16), (0, 9)])
def test_expert_packing_square_sum_respects_tail_and_integer_segments(tail, squares):
    """A tail is a separate attention segment, and n squared has n's parity."""
    record = expert.PackingMicrobatch(
        8, aligned_tokens=8, padding_tokens=tail, attention_length_squared=squares
    )
    with pytest.raises(ValueError, match="square sum"):
        record.resolve()


@pytest.mark.parametrize("per_rank", [False, True])
def test_expert_advantage_shapes_must_hold_observed_lengths(expert_model, per_rank):
    """Excess capacity on one rank cannot hide another rank's impossible width."""
    config, workload = _expert_actor_input()
    if per_rank:
        workload = replace(workload, per_dp_sequence_lengths=((64,) * 8, (64,) * 8))
    profile = expert_model.profile_key(config)
    observed = expert.AdvantageWorkload.for_workload(
        profile,
        workload,
        (expert.AdvantageRankWork(0, 8, 1), expert.AdvantageRankWork(1, 8, 128)),
        ("bad-shapes",),
    )
    with pytest.raises(ValueError, match="Advantage shapes"):
        expert_model.estimate(
            config, workload, source_profile=profile, advantage_workload=observed
        )


def test_expert_advantage_component_priors_remain_explicit_after_phase_fit(
    expert_model,
):
    """A total-phase label does not identify or independently calibrate components."""
    config, workload = _expert_actor_input()
    raw = expert_model.estimate(config, workload)
    total = dict(raw.stages)["compute_advantage"].seconds
    fitted = cost_model_tools.fit_phase_calibration(
        "unit",
        (
            expert.CalibrationPoint(
                "phase-total",
                raw.profile,
                "compute_advantage",
                total,
                total * 2,
                raw.features_for("compute_advantage"),
                raw.timing_boundary,
            ),
        ),
    )
    prediction = replace(expert_model, calibration=(fitted,)).estimate(config, workload)
    assert dict(prediction.stages)["compute_advantage"].seconds == pytest.approx(
        total * 2
    )
    assert sum(
        (e.seconds for _, e in prediction.advantage_components)
    ) == pytest.approx(total)
    assert all(
        (
            e.evidence == "assumed" and "Uncalibrated component prior" in e.notes[0]
            for _, e in prediction.advantage_components
        )
    )


def _aggregate_advantage_fit(expert_model):
    config, _ = _expert_actor_input()
    base = replace(
        expert_model, advantage_model=expert.AggregateAdvantageModel("unit-options")
    )
    observations = tuple(
        expert.AdvantageObservation(
            f"calibration-{n}",
            base.profile_key(config),
            WorkloadStats(16, n, max_seq_len=n / 16),
            1 + n / 128,
            "trainer_host_wall_with_stats",
        )
        for n in (256, 512, 1024)
    )
    fitted = replace(
        base, advantage_model=base.advantage_model.fit("unit-fit", observations)
    )
    return (config, base, fitted, observations)


def test_expert_aggregate_advantage_fits_phase_and_preserves_scope(expert_model):
    """Whole-phase labels predict an unseen batch without fabricated components."""
    config, base, fitted, observations = _aggregate_advantage_fit(expert_model)
    query = WorkloadStats(16, 768, max_seq_len=48)
    assert base.estimate(config, query).total.seconds is None
    prediction = fitted.estimate(config, query)
    stage = dict(prediction.stages)["compute_advantage"]
    assert stage.seconds == pytest.approx(7)
    assert stage.evidence == "calibrated"
    assert stage.source_ids == tuple(p.sample_id for p in observations)
    assert not prediction.advantage_components
    assert prediction.total.seconds == pytest.approx(
        sum((e.seconds for _, e in prediction.stages))
    )
    assert prediction.profile == base.profile_key(config)
    assert fitted.advantage_model.intercept_seconds == pytest.approx(1)
    assert fitted.advantage_model.seconds_per_token == pytest.approx(1 / 128)
    features = prediction.features_for("compute_advantage")
    assert len(dict(features)) == len(features)
    assert all(
        p.workload.sequence_tokens != query.sequence_tokens for p in observations
    )


@pytest.mark.parametrize(
    "labels,expected", [((1, 3, 5), (0, 11 / 7)), ((3, 2, 1), (2, 0))]
)
def test_expert_aggregate_fit_uses_constrained_boundary_optimum(labels, expected):
    """Clipping unconstrained slope/intercept would not minimize squared error."""
    model = expert.AggregateAdvantageModel("unit")
    points = tuple(
        (
            expert.AdvantageObservation(
                str(n), "profile", WorkloadStats(1, n), y, model.timing_boundary
            )
            for n, y in zip((1, 2, 3), labels, strict=True)
        )
    )
    fitted = model.fit("unit", points)
    assert (fitted.intercept_seconds, fitted.seconds_per_token) == pytest.approx(
        expected
    )


def test_expert_aggregate_constant_requires_explicit_selection():
    """Equal token counts cannot identify an affine slope, but can fit a constant."""
    model = expert.AggregateAdvantageModel("unit")
    points = tuple(
        expert.AdvantageObservation(
            str(y), "profile", WorkloadStats(1, 10), y, model.timing_boundary
        )
        for y in (1, 3)
    )
    with pytest.raises(ValueError, match="distinct token counts"):
        model.fit("unit", points)
    fitted = replace(model, form="constant").fit("unit", points)
    assert fitted.estimate(WorkloadStats(1, 10), "profile").total.seconds == 2
    assert fitted.estimate(WorkloadStats(1, 11), "profile").total.seconds is None


@pytest.mark.parametrize(
    "problem", ["duplicate", "profile", "boundary", "features", "few"]
)
def test_expert_aggregate_fit_rejects_incompatible_labels(expert_model, problem):
    """Calibration cannot mix identities, timing aggregation or feature availability."""
    _, base, _, observations = _aggregate_advantage_fit(expert_model)
    first, *rest = observations
    if problem == "duplicate":
        first = replace(first, sample_id=rest[0].sample_id)
    elif problem == "profile":
        first = replace(first, profile="another-profile")
    elif problem == "boundary":
        first = replace(first, timing_boundary="trainer_rank_mean_with_stats")
    elif problem == "features":
        first = replace(first, workload=replace(first.workload, max_seq_len=None))
    else:
        rest = []
    with pytest.raises(ValueError):
        base.advantage_model.fit("unit", (first, *rest))


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_expert_aggregate_rejects_invalid_phase_measurements(value):
    """Invalid/censored timings cannot enter the positive phase-label fit."""
    with pytest.raises(ValueError):
        expert.AdvantageObservation(
            "sample",
            "profile",
            WorkloadStats(1, 10),
            value,
            "trainer_host_wall_with_stats",
        )


@pytest.mark.parametrize(
    "query,reason",
    [
        (WorkloadStats(16, 2048, max_seq_len=128), "advantage_sequence_tokens"),
        (WorkloadStats(8, 512, max_seq_len=64), "advantage_n_seqs"),
        (WorkloadStats(16, 768, max_seq_len=100), "advantage_max_seq_len"),
        (WorkloadStats(16, 768), "Missing calibration feature"),
    ],
)
def test_expert_aggregate_out_of_bounds_does_not_fall_back(expert_model, query, reason):
    """Coverage refusal stays visible even while the analytical train stage works."""
    config, _, fitted, _ = _aggregate_advantage_fit(expert_model)
    result = fitted.estimate(config, query)
    stage = dict(result.stages)["compute_advantage"]
    assert stage.seconds is None and result.total.seconds is None
    assert dict(result.stages)["train_step"].seconds is not None
    assert any(reason in note for note in stage.notes)


@pytest.mark.parametrize("change", ["runtime", "hardware", "execution", "config"])
def test_expert_aggregate_does_not_transfer_across_profiles(expert_model, change):
    """A local interpolation fit has no implicit runtime/layout/hardware transfer."""
    config, _, fitted, _ = _aggregate_advantage_fit(expert_model)
    if change == "runtime":
        fitted = replace(fitted, runtime_revision="new-runtime")
    elif change == "hardware":
        fitted = replace(fitted, hardware=replace(fitted.hardware, name="new-hardware"))
    elif change == "execution":
        fitted = replace(
            fitted,
            advantage_model=replace(fitted.advantage_model, execution_id="new-options"),
        )
    else:
        config = replace(config, max_tokens_per_mb=512)
    stage = dict(
        fitted.estimate(config, WorkloadStats(16, 768, max_seq_len=48)).stages
    )["compute_advantage"]
    assert (
        stage.seconds is None and "Calibration profile or phase mismatch" in stage.notes
    )


def test_expert_aggregate_rejects_double_calibration_and_keeps_fit_in_snapshot(
    expert_model,
):
    """Stable profile identity must not erase fitted versions from result provenance."""
    config, _, fitted, observations = _aggregate_advantage_fit(expert_model)
    guard = fitted.advantage_model.calibration
    with pytest.raises(ValueError, match="already fitted"):
        replace(fitted, calibration=(guard,))
    prediction = fitted.estimate(config, observations[1].workload)
    train = dict(prediction.stages)["train_step"].seconds
    scale = cost_model_tools.fit_phase_calibration(
        "unit-train",
        (
            expert.CalibrationPoint(
                "train-label",
                prediction.profile,
                "train_step",
                train,
                train * 2,
                prediction.features_for("train_step"),
                prediction.timing_boundary,
            ),
        ),
    )
    updated = replace(fitted, calibration=(scale,)).estimate(
        config, observations[1].workload
    )
    assert dict(updated.stages)["train_step"].seconds == pytest.approx(train * 2)
    assert dict(updated.stages)["compute_advantage"].seconds == pytest.approx(5)
    refitted = replace(
        fitted,
        advantage_model=fitted.advantage_model.fit(
            "unit-v2",
            tuple(
                replace(p, measured_seconds=p.measured_seconds * 2)
                for p in observations
            ),
        ),
    )
    assert refitted.profile_key(config) == fitted.profile_key(config)
    assert fingerprint(fitted) != fingerprint(refitted)


def _runtime_ffd():
    """Use current runtime source without importing GPU engines."""
    tree = ast.parse((ROOT / "areal/utils/seqpack.py").read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_ffd_allocate", "ffd_allocate"}
    ]
    namespace = {"np": np, "bisect": bisect}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "runtime FFD", "exec"),
        namespace,
    )
    return namespace["ffd_allocate"]


def test_expert_ffd_matches_runtime_with_ties_and_ppo_dp_synchronization():
    """Unstable NumPy ties affect PPO membership and synchronized MB counts."""
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.packing import (
        ffd as expert_ffd,
    )
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.packing import (
        plan_workload as expert_plan,
    )

    allocate = _runtime_ffd()

    def bins(lengths, capacity, minimum):
        return [
            [lengths[i] for i in group]
            for group in sorted(
                sorted(group) for group in allocate(lengths, capacity, minimum)
            )
        ]

    rng = random.Random(127)
    for _ in range(50):
        lengths = [rng.randrange(1, 20) for _ in range(40)]
        assert expert_ffd(lengths, 40, 2) == bins(lengths, 40, 2)
    shards = [
        [15, 15, 2, 2, 8, 18, 3, 7, 9, 6, 12, 14, 10, 5, 18, 1, 11],
        [10, 17, 12, 5, 2, 11, 14, 10, 15, 9, 16, 4, 6, 9, 5, 5, 3],
    ]
    config = expert.ActorConfig(
        "archon",
        expert.ParallelLayout(dp=2),
        max_tokens_per_mb=20,
        ppo_n_minibatches=2,
        pad_to_maximum=True,
    )
    workload = WorkloadStats(
        sum(map(len, shards)),
        sum(map(sum, shards)),
        per_dp_sequence_lengths=tuple(tuple(row) for row in shards),
    )
    plan = expert_plan(config, workload, 128)
    ppo = [bins(rank, sum(rank), 2) for rank in shards]
    for call, ranks in enumerate(plan.train):
        minimum = 1
        while True:
            groups = [bins(rank[call], 20, minimum) for rank in ppo]
            counts = [len(row) for row in groups]
            if min(counts) == max(counts):
                break
            minimum = max(counts)
        assert [len(rank) for rank in ranks] == counts
        assert [[mb.real_tokens for mb in rank] for rank in ranks] == [
            [sum(group) for group in rank] for rank in groups
        ]


def test_expert_archon_padding_uses_shared_page_then_backend_alignment():
    """A non-power-of-two page exposes the two runtime rounding operations."""
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.packing import (
        _pad as expert_pad,
    )

    config = expert.ActorConfig(
        "archon", expert.ParallelLayout(), max_tokens_per_mb=2048
    )
    result = expert_pad([600], config, 1536)
    assert result.compute_tokens == 1364 and result.padding_tokens == 764
    assert expert_pad([1], config, 2049).compute_tokens == 1022
    assert (
        expert_pad([600], replace(config, pad_to_maximum=True), 1536).compute_tokens
        == 2048
    )


def test_expert_equal_token_summaries_do_not_identify_microbatch_counts():
    """Summary equality is not evidence of equal packing or compute work."""
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.packing import (
        plan_workload as expert_plan,
    )

    sequences = [
        (1, 2, 2, 3, 3, 5, 7, 8, 8, 9, 10, 10),
        (1, 1, 2, 2, 6, 6, 7, 7, 8, 8, 10, 10),
    ]
    config = expert.ActorConfig(
        "archon", expert.ParallelLayout(), max_tokens_per_mb=12, pad_to_maximum=True
    )
    plans = [
        expert_plan(
            config,
            WorkloadStats(12, 68, max_seq_len=10, per_dp_sequence_lengths=(lengths,)),
            128,
        )
        for lengths in sequences
    ]
    assert [len(plan.train[0][0]) for plan in plans] == [6, 7]
    proxy = expert_plan(config, WorkloadStats(12, 68, max_seq_len=10), 128)
    assert not proxy.exact_lengths and (not proxy.exact_dispatch)


def test_expert_partial_packing_retains_known_work_and_independent_phase(expert_model):
    """Missing attention/logprob does not discard measured train tokens/counts."""
    config, workload = _expert_actor_input()
    config = replace(
        config, recompute_logprob=True, parallel=expert.ParallelLayout(dp=2, tp=2)
    )
    raw = expert_model.estimate(config, workload)
    records = tuple(
        replace(
            r,
            microbatches=tuple(
                replace(m, attention_length_squared=None) for m in r.microbatches
            ),
        )
        for r in packing_records(config, raw.packing, raw.profile)
        if r.phase == "train_step"
    )
    work = expert_model.observe_packing_work(config, workload, records)
    assert work.phase_microbatches("train_step") == len(raw.packing.train[0][0])
    assert len(work.calls) == 2
    assert all((s.attention_length_squared is None for _, s in work.calls))
    assert work.missing_calls == (("recompute_logp", 0, 0), ("recompute_logp", 0, 1))
    comparison = work.compare(raw.packing)
    assert comparison.metric("microbatch_count") == (0, 2, 4)
    assert comparison.metric("compute_tokens") == (0, 2, 4)
    assert comparison.metric("attention_length_squared") == (None, 0, 4)
    with pytest.raises(ValueError, match="coverage"):
        work.phase_microbatches("recompute_logp")
    with pytest.raises(ValueError, match="coverage"):
        work.to_exact_plan()
    partial_rank = expert_model.observe_packing_work(config, workload, records[:1])
    assert partial_rank.compare(raw.packing).metric("microbatch_count") == (0, 1, 4)
    with pytest.raises(ValueError, match="coverage"):
        partial_rank.phase_microbatches("train_step")


def test_expert_packing_summary_never_treats_request_as_compute():
    """A padded_to request alone cannot resolve Megatron alignment or attention."""
    value = expert.PackingMicrobatch(8, padded_to_tokens=16).summary()
    assert (
        value.compute_tokens
        is value.padding_tokens
        is value.attention_length_squared
        is None
    )
    assert value.padded_to_tokens == 16
    known = expert.PackingSummary(
        2, real_tokens=16, aligned_tokens=20, padding_tokens=12
    )
    assert known.compute_tokens == 32 and known.attention_length_squared is None


def _packing_timing_point(expert_model, count, step, seconds=None, *, max_length=64):
    config, _ = _expert_actor_input()
    config = replace(
        config, parallel=expert.ParallelLayout(dp=2, tp=2), recompute_logprob=True
    )
    workload = WorkloadStats(16, 1024, max_seq_len=max_length)
    profile = expert_model.profile_key(config)
    records = tuple(
        expert.PackingRecord(
            "synthetic-diagnostic",
            "unit-runtime",
            step,
            "train_step",
            0,
            dp * 2 + tp,
            (dp, tp, 0, 0),
            (),
            f"step{step}-dp{dp}-tp{tp}",
            profile,
            summary=expert.PackingSummary(count, real_tokens=512),
        )
        for dp in range(2)
        for tp in range(2)
    )
    work = expert_model.observe_packing_work(config, workload, records)
    return expert.PackingTimingObservation(
        work,
        "train_step",
        1 + 2 * count if seconds is None else seconds,
        "trainer_host_wall_with_stats",
    )


def test_expert_count_only_diagnostic_separates_packing_from_conditional_timing(
    expert_model,
):
    """Fit calibration only, then freeze the SAME g for observed/predicted counts."""
    points = tuple(
        (_packing_timing_point(expert_model, n, i) for i, n in enumerate((2, 4, 8)))
    )
    diagnostic = cost_model_tools.PackingTimingDiagnostic("train_step").fit(
        "synthetic-fit", points
    )
    heldout = _packing_timing_point(expert_model, 6, 20)
    work = heldout.work
    predicted = expert_model.estimate(work.config, work.workload).packing
    result = diagnostic.evaluate(heldout, predicted)
    assert work.phase_microbatches("train_step") == 6
    assert diagnostic.intercept_seconds == pytest.approx(1)
    assert diagnostic.seconds_per_microbatch == pytest.approx(2)
    assert result.given_observed_packing.seconds == pytest.approx(13)
    assert result.conditional_timing_error == pytest.approx(0)
    predicted_count = len(predicted.train[0][0])
    assert result.given_predicted_packing.seconds == pytest.approx(
        1 + 2 * predicted_count
    )
    assert result.packing_timing_delta == pytest.approx(2 * (predicted_count - 6))
    assert result.packing_error.metric("microbatch_count") == (
        abs(predicted_count - 6) / 6,
        2,
        4,
    )
    assert result.packing_error.metric("attention_length_squared") == (None, 0, 4)
    assert "Empirical" in result.given_observed_packing.notes[0]
    with pytest.raises(ValueError, match="coverage"):
        work.to_exact_plan()
    with pytest.raises(ValueError, match="calibration"):
        diagnostic.evaluate(
            points[0],
            expert_model.estimate(
                points[0].work.config, points[0].work.workload
            ).packing,
        )
    assert len(diagnostic.calibration.sample_ids) == 3


@pytest.mark.parametrize(
    "problem", ["count_range", "length_range", "profile", "missing_dp", "unfitted"]
)
def test_expert_packing_diagnostic_keeps_unsupported_bounds(expert_model, problem):
    """Held-out long sequences and incomplete ranks cannot inflate coverage."""
    points = tuple(
        (_packing_timing_point(expert_model, n, i) for i, n in enumerate((2, 4, 8)))
    )
    diagnostic = cost_model_tools.PackingTimingDiagnostic("train_step").fit(
        "synthetic-fit", points
    )
    point = _packing_timing_point(
        expert_model,
        9 if problem == "count_range" else 6,
        20,
        max_length=128 if problem == "length_range" else 64,
    )
    work = point.work
    if problem == "profile":
        work = replace(
            work,
            profile="other",
            records=tuple(replace(r, profile="other") for r in work.records),
        )
    if problem == "missing_dp":
        work = replace(work, records=work.records[:1])
    if problem == "unfitted":
        diagnostic = cost_model_tools.PackingTimingDiagnostic("train_step")
    result = diagnostic.predict(work)
    assert result.seconds is None and result.evidence == "unsupported"


def test_expert_packing_diagnostic_rejects_mixed_phase_duplicate_fit_and_rank_mean(
    expert_model,
):
    """Calibration keeps run/step provenance, phase and trainer wall-clock scope."""
    points = tuple(
        (_packing_timing_point(expert_model, n, i) for i, n in enumerate((2, 4, 8)))
    )
    with pytest.raises(ValueError, match="wall"):
        replace(points[0], timing_boundary="rank_mean")
    with pytest.raises(ValueError, match="phase"):
        cost_model_tools.PackingTimingDiagnostic("recompute_logp").fit("bad", points)
    with pytest.raises(ValueError, match="Duplicate"):
        cost_model_tools.PackingTimingDiagnostic("train_step").fit(
            "bad", points + (points[0],)
        )
    same = tuple(_packing_timing_point(expert_model, 4, i) for i in range(2))
    with pytest.raises(ValueError, match="distinct"):
        cost_model_tools.PackingTimingDiagnostic("train_step").fit("bad", same)
    constant = cost_model_tools.PackingTimingDiagnostic(
        "train_step", form="constant"
    ).fit("constant", same)
    assert constant.predict(
        _packing_timing_point(expert_model, 4, 50).work
    ).seconds == pytest.approx(9)


@pytest.mark.parametrize(
    "problem",
    [
        "step",
        "runtime",
        "profile",
        "duplicate",
        "rank",
        "call",
        "tokens",
        "counts",
        "copy",
    ],
)
def test_expert_partial_packing_validates_attribution_and_known_conflicts(
    expert_model, problem
):
    """Partial fields relax completeness, never attribution or known invariants."""
    work = _packing_timing_point(expert_model, 4, 1).work
    records = list(work.records)
    with pytest.raises(ValueError):
        if problem == "step":
            records[0] = replace(records[0], global_step=2)
        elif problem == "runtime":
            records[0] = replace(records[0], source_revision="other")
        elif problem == "profile":
            records[0] = replace(records[0], profile="other")
        elif problem == "duplicate":
            records.append(records[0])
        elif problem == "rank":
            records[0] = replace(records[0], global_rank=7)
        elif problem == "call":
            records[0] = replace(records[0], call_index=2)
        elif problem == "tokens":
            records = [
                replace(r, summary=expert.PackingSummary(4, real_tokens=500))
                for r in records
            ]
        elif problem == "counts":
            records = [
                replace(r, summary=expert.PackingSummary(5, real_tokens=512))
                if r.mesh[0] == 0
                else r
                for r in records
            ]
        else:
            records[0] = replace(
                records[0], summary=expert.PackingSummary(5, real_tokens=512)
            )
        replace(work, records=tuple(records))


@pytest.mark.parametrize(
    "metric", ["compute_tokens", "aligned_tokens", "attention_length_squared"]
)
def test_expert_summary_rejects_zero_work_even_without_real_tokens(metric):
    """Every nonempty MB contributes at least one token and one square unit."""
    with pytest.raises(ValueError, match="nonempty microbatch"):
        expert.PackingSummary(2, **{metric: 0})
    assert expert.PackingSummary(2, padding_tokens=0).padding_tokens == 0


@pytest.mark.parametrize("delta", [-1e-05, 1e-05, 2.0])
def test_expert_observed_summary_uses_same_token_rounding_tolerance(
    expert_model, delta
):
    """Measured integer tokens tolerate the same one-token summary error as plans."""
    work = _packing_timing_point(expert_model, 4, 1).work
    workload = replace(work.workload, sequence_tokens=1024 + delta, max_seq_len=128)
    if delta > 1:
        with pytest.raises(ValueError, match="logical workload"):
            replace(work, workload=workload)
    else:
        assert replace(work, workload=workload).phase_microbatches("train_step") == 4


def test_expert_packing_comparison_rejects_observed_count_leakage(expert_model):
    """An observed minimum must never turn an independent packing error into zero."""
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.packing import (
        plan_workload as expert_plan,
    )

    work = _packing_timing_point(expert_model, 8, 20).work
    clean = expert_plan(work.config, work.workload, expert_model.model.hidden_size)
    leaky_workload = replace(work.workload, train_microbatches=8)
    leaked = expert_plan(work.config, leaky_workload, expert_model.model.hidden_size)
    assert len(clean.train[0][0]) < 8 and len(leaked.train[0][0]) == 8
    assert leaked.count_source == "observed"
    with pytest.raises(ValueError, match="independently predicted"):
        work.compare(leaked)
    with pytest.raises(ValueError, match="train_microbatches"):
        replace(work, workload=leaky_workload).compare(clean)
    bound = expert.ObservedPacking.for_workload(
        work.profile, work.workload, clean, ("observed",)
    )
    with pytest.raises(ValueError, match="independently predicted"):
        work.compare(bound.plan)
    assert work.compare(clean).metric("microbatch_count")[0] > 0


def test_expert_partial_counts_sum_ppo_calls_but_not_parallel_copies(expert_model):
    """Train call counts add once; logprob is a separate single forward."""
    config, workload = _expert_actor_input()
    config = replace(
        config,
        ppo_n_minibatches=2,
        recompute_logprob=True,
        parallel=expert.ParallelLayout(dp=2, tp=2),
    )
    raw = expert_model.estimate(config, workload)
    work = expert_model.observe_packing_work(
        config, workload, packing_records(config, raw.packing, raw.profile)
    )
    assert work.phase_microbatches("train_step") == sum(
        len(ranks[0]) for ranks in raw.packing.train
    )
    assert work.phase_microbatches("recompute_logp") == len(raw.packing.logprob[0])
    exact = work.to_exact_plan()
    assert exact.train == raw.packing.train and exact.count_source == "observed"


def _runtime_balanced_partition():
    """Execute the controller's actual pure partition function on CPU."""
    tree = ast.parse((ROOT / "areal/utils/seqpack.py").read_text())
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "balanced_greedy_partition"
    ]
    namespace = {}
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]), "runtime partition", "exec"
        ),
        namespace,
    )
    return namespace["balanced_greedy_partition"]


def test_expert_controller_dispatch_preserves_trajectories_and_rectangular_weights():
    """RTensor numel and attended tokens can assign a different heavy trajectory."""
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.packing import (
        dispatch_workload,
    )
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.packing import (
        plan_workload as expert_plan,
    )

    trajectories = ((100, 1, 1, 1), (99, 99), (80,), (79,))
    weights = tuple(len(t) * max(t) for t in trajectories)
    flat = sum(trajectories, ())
    workload = WorkloadStats(
        len(flat),
        sum(flat),
        sequence_lengths=flat,
        trajectory_group_sizes=tuple(map(len, trajectories)),
        trajectory_dispatch_weights=weights,
    )
    dispatch = expert.DispatchConfig("controller")
    indices = _runtime_balanced_partition()(list(weights), 2)
    expected = [[n for i in rank for n in trajectories[i]] for rank in indices]
    assert dispatch_workload(workload, list(flat), 2, dispatch) == expected
    token_weights = replace(
        workload, trajectory_dispatch_weights=tuple(map(sum, trajectories))
    )
    assert dispatch_workload(token_weights, list(flat), 2, dispatch) != expected
    config = expert.ActorConfig(
        "archon", expert.ParallelLayout(dp=2), max_tokens_per_mb=128
    )
    assert expert_plan(config, workload, 128, dispatch=dispatch).exact_dispatch
    assert not expert_plan(config, workload, 128).exact_dispatch
    assert not expert_plan(
        config,
        replace(workload, trajectory_dispatch_weights=None),
        128,
        dispatch=dispatch,
    ).exact_dispatch


def test_expert_controller_atomic_dispatch_groups_match_runtime():
    """Controller group_size groups consecutive trajectories before DP allocation."""
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.packing import (
        dispatch_workload,
    )

    trajectories = tuple((i + 1, i + 2) for i in range(8))
    flat = sum(trajectories, ())
    weights = tuple(map(sum, trajectories))
    work = WorkloadStats(
        len(flat),
        sum(flat),
        sequence_lengths=flat,
        trajectory_group_sizes=(2,) * 8,
        trajectory_dispatch_weights=weights,
    )
    groups = _runtime_balanced_partition()(
        [sum(weights[i : i + 2]) for i in range(0, 8, 2)], 2
    )
    expected = [
        [n for g in rank for i in range(g * 2, g * 2 + 2) for n in trajectories[i]]
        for rank in groups
    ]
    assert (
        dispatch_workload(work, list(flat), 2, expert.DispatchConfig("controller", 2))
        == expected
    )
    with pytest.raises(ValueError, match="group_size"):
        dispatch_workload(work, list(flat), 2, expert.DispatchConfig("controller", 3))
    with pytest.raises(ValueError, match="atomic groups"):
        expert.DispatchConfig("redistribute", 2)


def test_expert_redistribution_preserves_runtime_ffd_rank_and_row_order():
    """Multi-controller FFD retains raw bin order and may assign unequal local counts."""
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.packing import (
        dispatch_workload,
    )
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.packing import (
        plan_workload as expert_plan,
    )

    rng = random.Random(19)
    allocate = _runtime_ffd()
    dispatch = expert.DispatchConfig("redistribute")
    config = expert.ActorConfig(
        "archon", expert.ParallelLayout(dp=4), max_tokens_per_mb=4096
    )
    for count in (12, 16, 28):
        trajectories = tuple(
            tuple(rng.randrange(1, 70) for _ in range(rng.randrange(1, 5)))
            for _ in range(count)
        )
        flat = sum(trajectories, ())
        work = WorkloadStats(
            len(flat),
            sum(flat),
            sequence_lengths=flat,
            trajectory_group_sizes=tuple(map(len, trajectories)),
        )
        groups = allocate(list(map(sum, trajectories)), int(1000000000000.0), 4)
        expected = [[n for i in rank for n in trajectories[i]] for rank in groups]
        assert dispatch_workload(work, list(flat), 4, dispatch) == expected
        assert expert_plan(config, work, 128, dispatch=dispatch).exact_dispatch
        wrong = replace(
            work, trajectory_dispatch_weights=tuple(sum(t) + 1 for t in trajectories)
        )
        with pytest.raises(ValueError, match="attended tokens"):
            dispatch_workload(wrong, list(flat), 4, dispatch)


def test_expert_atomic_rank_skew_explains_more_mbs_than_global_average():
    """Synthetic S026-scale example: a heavy atomic trajectory survives DP dispatch."""
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.packing import (
        plan_workload as expert_plan,
    )

    heavy = (28280,) * 25 + (1,) * 39
    base, remainder = divmod(9480000 - sum(heavy), 31)
    trajectories = [heavy]
    for rank in range(31):
        target = base + (rank < remainder)
        q, r = divmod(target, 64)
        trajectories.append((q + 1,) * r + (q,) * (64 - r))
    flat = sum(trajectories, ())
    work = WorkloadStats(
        len(flat),
        sum(flat),
        sequence_lengths=flat,
        trajectory_group_sizes=(64,) * 32,
        trajectory_dispatch_weights=tuple(64 * max(t) for t in trajectories),
    )
    config = expert.ActorConfig(
        "archon", expert.ParallelLayout(dp=32), max_tokens_per_mb=32768
    )
    plan = expert_plan(config, work, 2048, dispatch=expert.DispatchConfig("controller"))
    train = [r for r in plan.rank_work if r.phase == "train_step"]
    assert plan.exact_dispatch and plan.exact_lengths
    assert train[0].real_tokens == 707039
    assert train[0].capacity_microbatch_lower_bound == 22
    assert train[0].local_microbatches == 25
    assert {r.synchronized_microbatches for r in train} == {25}
    assert min(r.local_microbatches for r in train) < 25
    assert sum(r.real_tokens for r in train) == 9480000
    proxy = expert_plan(config, WorkloadStats(len(flat), sum(flat)), 2048)
    assert not proxy.exact_dispatch and len(proxy.train[0][0]) <= 11


def test_expert_partial_rank_tokens_expose_capacity_lower_bound_without_fake_lengths(
    expert_model,
):
    """Known heavy-rank tokens prove a bound while local packing remains unknown."""
    config = expert.ActorConfig(
        "archon", expert.ParallelLayout(dp=32), max_tokens_per_mb=32768
    )
    workload = WorkloadStats(2048, 9480000)
    raw = expert_model.estimate(config, workload)
    record = expert.PackingRecord(
        "company-shape-synthetic",
        "unit-runtime",
        63,
        "train_step",
        0,
        0,
        (0, 0, 0, 0),
        (),
        "synthetic-rank0-summary",
        raw.profile,
        summary=expert.PackingSummary(25, real_tokens=707000),
    )
    work = expert_model.observe_packing_work(config, workload, (record,))
    comparison = work.compare(raw.packing)
    rows = {
        r.metric: r
        for r in comparison.rows
        if r.phase == "train_step" and r.dp_rank == 0
    }
    assert rows["capacity_microbatch_lower_bound"].observed == 22
    assert rows["capacity_microbatch_lower_bound"].predicted <= 10
    assert rows["microbatch_count"].observed == 25
    assert rows["local_microbatch_count"].observed is None
    assert comparison.metric("local_microbatch_count")[1] == 0
    assert workload.sequence_lengths is None and workload.train_microbatches is None
    with pytest.raises(ValueError, match="microbatch capacity"):
        expert_model.observe_packing_work(
            config,
            workload,
            (replace(record, summary=expert.PackingSummary(10, real_tokens=707000)),),
        )


def test_expert_dispatch_is_part_of_profile_and_given_rank_remains_conditional(
    expert_model,
):
    """Switching runtime paths invalidates calibration; missing path stays explicit."""
    config, workload = _expert_actor_input()
    controller = replace(expert_model, dispatch=expert.DispatchConfig("controller"))
    assert controller.profile_key(config) != expert_model.profile_key(config)
    assert controller.estimate(config, workload).packing.exact_dispatch
    ranked = replace(workload, per_dp_sequence_lengths=((64,) * 8,) * 2)
    prediction = controller.estimate(
        config, ranked, source_profile=controller.profile_key(config)
    )
    assert any("condition dispatch" in n for n in prediction.packing.assumptions)
    with pytest.raises(ValueError, match="DispatchConfig"):
        replace(expert_model, dispatch="controller")


def test_expert_count_extrapolation_is_explicit_and_does_not_change_other_bounds(
    expert_model,
):
    """Calibration count support is distinct from runtime legality and length support."""
    points = tuple(
        (_packing_timing_point(expert_model, n, i) for i, n in enumerate((2, 4)))
    )
    diagnostic = cost_model_tools.PackingTimingDiagnostic("train_step").fit(
        "frozen", points
    )
    heldout = _packing_timing_point(expert_model, 6, 63)
    predicted = expert_model.estimate(
        heldout.work.config, heldout.work.workload
    ).packing
    original = diagnostic.calibration
    assert (
        diagnostic.evaluate(heldout, predicted).given_observed_packing.seconds is None
    )
    result = diagnostic.evaluate(heldout, predicted, extrapolate_count=True)
    assert result.given_observed_packing.seconds == pytest.approx(13)
    assert result.given_observed_packing.evidence == "assumed"
    assert any(
        "Extrapolated microbatch_count=6" in n
        for n in result.given_observed_packing.notes
    )
    assert diagnostic.calibration == original
    for bad in (
        _packing_timing_point(expert_model, 6, 64, max_length=128).work,
        replace(heldout.work, records=heldout.work.records[:1]),
    ):
        assert diagnostic.predict(bad, extrapolate_count=True).seconds is None
    with pytest.raises(ValueError, match="calibration"):
        diagnostic.evaluate(points[0], predicted, extrapolate_count=True)
    with pytest.raises(ValueError, match="boolean"):
        diagnostic.predict(heldout.work, extrapolate_count=1)


def test_expert_redistribution_checks_runtime_gather_shape_before_assignment():
    """Exact redistribution requires equal input-list sizes on original DP heads."""
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.packing import (
        dispatch_workload,
    )

    work = WorkloadStats(
        13, 130, sequence_lengths=(10,) * 13, trajectory_group_sizes=(1,) * 13
    )
    with pytest.raises(ValueError, match="pre-gather list sizes"):
        dispatch_workload(
            work, list(work.sequence_lengths), 4, expert.DispatchConfig("redistribute")
        )


def test_expert_local_count_must_fit_observed_rank_tokens(expert_model):
    """Synchronized capacity cannot mask an impossible smaller local count."""
    config, workload = _expert_actor_input()
    profile = expert_model.profile_key(config)
    record = expert.PackingRecord(
        "synthetic-local-count",
        "unit-runtime",
        1,
        "train_step",
        0,
        0,
        (0, 0, 0, 0),
        (),
        "synthetic",
        profile,
        summary=expert.PackingSummary(4, real_tokens=512, local_microbatch_count=1),
    )
    with pytest.raises(ValueError, match="microbatch capacity"):
        expert_model.observe_packing_work(config, workload, (record,))
    record = replace(
        record,
        summary=expert.PackingSummary(4, real_tokens=512, local_microbatch_count=2),
    )
    work = expert_model.observe_packing_work(config, workload, (record,))
    raw = expert_model.estimate(config, workload)
    rows = [
        r
        for r in work.compare(raw.packing).rows
        if r.phase == "train_step" and r.dp_rank == 0
    ]
    assert next(r for r in rows if r.metric == "local_microbatch_count").observed == 2


def test_expert_advantage_and_packing_share_the_audited_dispatch(expert_model):
    """Rank tensor-shape priors must follow the same allocation as logp/train."""
    trajectories = ((100, 1, 1, 1), (99, 99), (80,), (79,))
    flat = sum(trajectories, ())
    work = WorkloadStats(
        len(flat),
        sum(flat),
        sequence_lengths=flat,
        trajectory_group_sizes=tuple(map(len, trajectories)),
    )
    model = replace(
        expert_model,
        dispatch=expert.DispatchConfig("redistribute"),
        advantage_model=replace(expert_model.advantage_model, controller_mode="multi"),
    )
    config = expert.ActorConfig(
        "archon", expert.ParallelLayout(dp=2), max_tokens_per_mb=128
    )
    result = model.estimate(config, work)
    expected = [
        [n for i in group for n in trajectories[i]]
        for group in _runtime_ffd()(
            list(map(sum, trajectories)), int(1000000000000.0), 2
        )
    ]
    features = dict(result.features_for("compute_advantage"))
    assert features["advantage_padded_slots_sum"] == sum(
        len(r) * max(r) for r in expected
    )
    trace = [r for r in result.packing.rank_work if r.phase == "train_step"]
    assert [r.sequences for r in trace] == list(map(len, expected))
    assert [r.real_tokens for r in trace] == list(map(sum, expected))
    with pytest.raises(ValueError, match="controller mode"):
        replace(expert_model, dispatch=expert.DispatchConfig("redistribute"))


@pytest.fixture
def publication_case(expert_model):
    """Two senders, one receiver and explicit zero-work counters for arithmetic."""
    hardware = replace(
        expert_model.hardware,
        intra_node_gbps=1,
        inter_node_gbps=10,
        intra_latency_us=1,
        inter_latency_us=1,
    )
    raw = replace(expert_model, hardware=hardware)
    actor = expert.ActorConfig(backend="megatron", parallel=expert.ParallelLayout(dp=2))
    rollout = expert.RolloutConfig(1, 1, 4, 100, 1000)
    zeros = {f.name: 0 for f in fields(expert.PublicationRank)[3:]}
    ranks = (
        expert.PublicationRank("a0", "actor", "node0", **zeros),
        expert.PublicationRank("a1", "actor", "node0", **zeros),
        expert.PublicationRank(
            "r0", "rollout", "node0", **zeros | {"expected_receive_bytes": 2000000000}
        ),
    )
    work = expert.PublicationWork(
        expert.publication_binding(raw, actor, rollout),
        "megatron",
        "awex_dense",
        "unit-traffic-oracle",
        ranks,
        (
            expert.PublicationFlow("a0", "r0", 1000000000, 1),
            expert.PublicationFlow("a1", "r0", 1000000000, 1),
        ),
        complete=True,
    )
    rates = expert.PublicationRates(
        h2d_gbps=1,
        host_h2d_gbps=1,
        h2d_launch_seconds=0,
        gather_gbps=1,
        gather_launch_seconds=0,
        copy_gbps=1,
        copy_launch_seconds=0,
        network_efficiency=1,
        node_fabric_gbps=1000,
        message_launch_seconds=0,
        pause_seconds=0,
        cleanup_seconds=0,
        barrier_seconds=0,
        resume_seconds=0,
    )
    model = expert.ExpertPublicationModel(hardware, rates)
    return (model, raw, actor, rollout, work)


def test_publication_shared_receiver_is_a_bottleneck(publication_case):
    model, raw, actor, rollout, work = publication_case
    result = model.estimate(raw, actor, rollout, work)
    assert result.duration.seconds == pytest.approx(2.000002)
    assert dict(result.features) == {"wire_bytes": 2000000000.0, "messages": 2}
    assert result.rounds[0]["resource_bytes"]["receive:r0"] == 2000000000.0
    transition = expert.ExpertTransitionModel((result.transition("a", "a"),))
    assert (
        transition.estimate("publish", "a", "a").duration.seconds
        == result.duration.seconds
    )


def test_publication_all_rank_prepare_work_counts_even_without_sending(
    publication_case,
):
    model, raw, actor, rollout, work = publication_case
    actor = replace(actor, backend="megatron")
    ranks = (
        work.ranks[0],
        replace(work.ranks[1], gather_bytes=3000000000, gather_calls=1),
        replace(work.ranks[2], expected_receive_bytes=1000000000),
    )
    work = replace(
        work,
        backend="megatron",
        ranks=ranks,
        flows=work.flows[:1],
        binding=expert.publication_binding(raw, actor, rollout),
    )
    result = model.estimate(raw, actor, rollout, work)
    assert dict(result.stages)["prepare"].seconds == 3
    assert result.duration.seconds == pytest.approx(4.000001)


def test_publication_shared_host_capacity_and_serial_rank_components(publication_case):
    model, raw, actor, rollout, work = publication_case
    ranks = tuple(
        replace(
            r,
            h2d_bytes=1000000000,
            h2d_tensors=1,
            send_copy_bytes=1000000000,
            send_copy_tensors=1,
        )
        if r.role == "actor"
        else r
        for r in work.ranks
    )
    result = model.estimate(raw, actor, rollout, replace(work, ranks=ranks))
    assert dict(result.stages)["prepare"].seconds == 3
    assert result.duration.seconds == pytest.approx(5.000002)


def test_publication_pre_receive_and_writeback_copies_are_separate(publication_case):
    model, raw, actor, rollout, work = publication_case
    ranks = work.ranks[:2] + (
        replace(
            work.ranks[2],
            receive_prepare_bytes=1000000000,
            receive_prepare_tensors=1,
            receive_copy_bytes=2000000000,
            receive_copy_tensors=1,
        ),
    )
    result = model.estimate(raw, actor, rollout, replace(work, ranks=ranks))
    assert dict(result.stages)["prepare"].seconds == 1
    assert dict(result.stages)["writeback"].seconds == 2
    assert result.duration.seconds == pytest.approx(5.000002)


def test_publication_shared_node_nic_is_not_per_flow_bandwidth(publication_case):
    model, raw, actor, rollout, work = publication_case
    work = replace(work, ranks=work.ranks[:2] + (replace(work.ranks[2], node="node1"),))
    assert model.estimate(raw, actor, rollout, work).duration.seconds is None
    model = replace(model, rates=replace(model.rates, node_network_gbps=1))
    result = model.estimate(raw, actor, rollout, work)
    assert result.rounds[0]["resource_bytes"]["nic-out:node0"] == 2000000000.0
    assert result.duration.seconds == 2


@pytest.mark.parametrize(
    "missing", ["copy_gbps", "message_launch_seconds", "pause_seconds"]
)
def test_publication_active_missing_rates_never_become_zero(publication_case, missing):
    model, raw, actor, rollout, work = publication_case
    ranks = (
        replace(work.ranks[0], send_copy_bytes=100, send_copy_tensors=1),
        *work.ranks[1:],
    )
    result = replace(model, rates=replace(model.rates, **{missing: None})).estimate(
        raw, actor, rollout, replace(work, ranks=ranks)
    )
    assert result.duration.seconds is None
    assert missing in " ".join(result.duration.notes)
    transition = expert.ExpertTransitionModel((result.transition("a", "a"),))
    assert transition.estimate("publish", "a", "a").duration.seconds is None


@pytest.mark.parametrize(
    "change", ["binding", "partial", "counter", "coverage", "delta"]
)
def test_publication_rejects_unbound_or_partial_geometry(publication_case, change):
    model, raw, actor, rollout, work = publication_case
    if change == "binding":
        work = replace(work, binding="another-config")
    if change == "partial":
        work = replace(work, complete=False)
    if change == "counter":
        work = replace(
            work, ranks=(replace(work.ranks[0], gather_bytes=None), *work.ranks[1:])
        )
    if change == "coverage":
        work = replace(work, flows=work.flows[:1])
    if change == "delta":
        work = replace(work, transport="dte_delta")
    assert model.estimate(raw, actor, rollout, work).duration.seconds is None


def test_publication_calibration_replaces_whole_dag_and_is_bound(publication_case):
    model, raw, actor, rollout, work = publication_case
    prior = model.estimate(raw, actor, rollout, work)
    fit = cost_model_tools.fit_phase_calibration(
        "unit",
        (
            expert.CalibrationPoint(
                "prefix-step",
                prior.profile,
                "update_weights",
                prior.duration.seconds,
                prior.duration.seconds * 2,
                prior.features,
                "trainer.pause-through-version-confirmation",
            ),
        ),
    )
    scaled = prior.calibrated(fit)
    assert scaled.duration.seconds == 2 * prior.duration.seconds
    assert scaled.stages == prior.stages
    transition = expert.ExpertTransitionModel((scaled.transition("a", "a"),))
    assert (
        transition.estimate("publish", "a", "a").duration.seconds
        == scaled.duration.seconds
    )
    assert len(scaled.transition("a", "a").tasks) == 1
    changed = replace(model, rates=replace(model.rates, copy_gbps=2)).estimate(
        raw, actor, rollout, work
    )
    assert changed.calibrated(fit).duration.seconds is None


def test_publication_archon_uses_legacy_bucket_ack_not_v2_awex(publication_case):
    model, raw, actor, rollout, work = publication_case
    actor = replace(actor, backend="archon")
    work = replace(
        work, backend="archon", binding=expert.publication_binding(raw, actor, rollout)
    )
    assert model.estimate(raw, actor, rollout, work).duration.seconds is None
    work = replace(
        work,
        transport="xccl_broadcast",
        flows=(work.flows[0], replace(work.flows[1], round=1)),
    )
    assert model.estimate(raw, actor, rollout, work).duration.seconds is None
    model = replace(model, rates=replace(model.rates, bucket_ack_seconds=0.5))
    result = model.estimate(raw, actor, rollout, work)
    assert len(result.rounds) == 2
    assert result.duration.seconds == pytest.approx(3.000002)
    assert "not an inferred NCCL broadcast tree" in " ".join(result.duration.notes)


def test_publication_disjoint_endpoints_overlap_but_explicit_rounds_serialize(
    publication_case,
):
    model, raw, actor, rollout, work = publication_case
    rollout = replace(rollout, replicas=2)
    zero = {f.name: 0 for f in fields(expert.PublicationRank)[3:]}
    ranks = work.ranks[:2] + (
        replace(work.ranks[2], expected_receive_bytes=1000000000),
        expert.PublicationRank(
            "r1", "rollout", "node0", **zero | {"expected_receive_bytes": 1000000000}
        ),
    )
    work = replace(
        work,
        binding=expert.publication_binding(raw, actor, rollout),
        ranks=ranks,
        flows=(work.flows[0], replace(work.flows[1], target="r1")),
    )
    parallel = model.estimate(raw, actor, rollout, work)
    serial = model.estimate(
        raw,
        actor,
        rollout,
        replace(work, flows=(work.flows[0], replace(work.flows[1], round=1))),
    )
    assert parallel.duration.seconds == pytest.approx(1.000001)
    assert serial.duration.seconds == pytest.approx(2.000002)


def test_publication_shared_local_fabric_caps_disjoint_peers(publication_case):
    model, raw, actor, rollout, work = publication_case
    model = replace(model, rates=replace(model.rates, node_fabric_gbps=0.5))
    result = model.estimate(raw, actor, rollout, work)
    assert result.rounds[0]["resource_bytes"]["fabric:node0"] == 2000000000.0
    assert result.duration.seconds == 4
    missing = replace(
        model, rates=replace(model.rates, node_fabric_gbps=None)
    ).estimate(raw, actor, rollout, work)
    assert missing.duration.seconds is None
    assert all(v is None for v in missing.rounds[0]["resource_seconds"].values())
