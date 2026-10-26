# SPDX-License-Identifier: Apache-2.0

"""CPU model contracts, source packing parity and frozen public-data validation."""

import ast
import bisect
import json
import math
import os
import random
import shutil
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from areal.autopilot.domains.elastic.model_adapters import (
    ACTOR_BOUNDARY,
    BasicHistoryPredictor,
)
from areal.autopilot.domains.elastic.model_evaluator import HistoricalInput
from areal.elastic.monitor.trace import (
    actor_trace_times,
    parse_microbatch_log,
    stage_metrics,
    trace_events,
)
from areal.elastic.policy import reference as expert
from areal.elastic.policy.basic.cost_model import (
    ActorComponents,
    ActorConfig,
    ActorCostModel,
    Efficiency,
    Parallel,
    ParallelLayout,
    Scale,
    Serial,
    WorkloadStats,
    hardware_profile,
    qwen3_30b_a3b,
    trial_config,
)
from areal.elastic.policy.basic.cost_model.calibration import (
    Calibration,
    bundled_calibration,
    point_inputs,
    public_validation,
)
from areal.elastic.policy.basic.cost_model.catalog import DATA, read_data
from areal.elastic.policy.basic.cost_model.components import (
    Communications,
    Constant,
    Context,
    Operators,
    Optimizer,
    Transformer,
)
from areal.elastic.policy.basic.cost_model.packing import (
    _pad,
    balanced_partition,
    ffd,
    plan_workload,
)

ROOT = Path(__file__).resolve().parents[1]


def qwen4b_components(*, dp=1, tp=1, replicas=1, rollout_tp=1):
    return expert.Catalog().assemble(
        hardware="h20",
        system="h20-actor8-rollout8",
        model="qwen3-4b-instruct-2507",
        actor="megatron-qwen3-4b-ppo",
        rollout="sglang-qwen3-4b",
        actor_parallel=expert.ParallelLayout(dp=dp, tp=tp),
        rollout_tp=rollout_tp,
        rollout_replicas=replicas,
        kv_capacity_tokens=65536,
    )


def test_expert_qwen4b_model_and_megatron_recipe_match_target():
    """Dense Qwen3 dimensions include shared embedding and no router weights."""
    bundle = qwen4b_components(dp=3)
    shape = bundle.actor.model
    assert shape.total_parameters == 4_022_468_096
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
    assert estimate.total.seconds is None  # Unmeasured advantage is not a zero cost.


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_expert_dense_tp_partitions_ffn_and_keeps_replicated_norms(tp):
    """Dense weights shard over TP and all gradient traffic uses the true DP group."""
    from areal.elastic.policy.reference.cost_model._actor.components import (
        Communications as ExpertCommunications,
    )
    from areal.elastic.policy.reference.cost_model._actor.components import (
        Context as ExpertContext,
    )
    from areal.elastic.policy.reference.cost_model._actor.components import (
        Operators as ExpertOperators,
    )
    from areal.elastic.policy.reference.cost_model._actor.components import (
        Transformer as ExpertTransformer,
    )

    bundle = qwen4b_components(tp=tp)
    cfg, model = bundle.config.actor, bundle.actor.model
    op = ExpertOperators(bundle.actor.hardware)
    ctx = ExpertContext(cfg, model, op, model.layers, True, True)
    dense, experts = ctx.owned_parameters
    replicated = (
        model.layers * (2 * model.hidden_size + 2 * model.head_dim) + model.hidden_size
    )
    assert experts == 0
    assert tp * dense == model.total_parameters + (tp - 1) * replicated
    # Changing MoE efficiency cannot change the dense FFN's roofline.
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
    assert ExpertCommunications().gradient_sync(ctx) == pytest.approx(
        op.collective(dense * cfg.grad_bytes, dp, "dense_dp", dp * tp)
    )


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_expert_dense_rollout_reuses_single_replica_calibration_and_kv_width(tp):
    """Per-TP service calibration transfers across replica counts, not phases."""
    bundle = qwen4b_components(rollout_tp=tp)
    model, cfg = bundle.rollout, bundle.config.rollout
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
            tuple((key, value, value) for key, value in raw.features),
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
    hardware = catalog.hardware("h20", "h20-8gpu-r10")
    assert shape.total_parameters == 30_532_122_624
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
    source = Path(expert.__file__).parent
    shutil.copytree(source / "profiles", tmp_path / "profiles")
    return expert.Catalog(tmp_path)


def r10_catalog():
    """Historical inputs are optional; an explicitly selected archive must exist."""
    selected = os.environ.get("AREAL_TEST_EXPERT_CATALOG")
    catalog = (
        expert.Catalog(Path(selected).expanduser()) if selected else expert.Catalog()
    )
    if "r10" not in catalog.names("trials"):
        if selected:
            pytest.fail(
                "AREAL_TEST_EXPERT_CATALOG does not contain trials/r10/manifest.json"
            )
        pytest.skip(
            "Optional R10 archive: set AREAL_TEST_EXPERT_CATALOG to run this test"
        )
    return catalog


@pytest.fixture
def expert_trial_copy(expert_catalog_copy):
    shutil.copytree(r10_catalog().root / "trials", expert_catalog_copy.root / "trials")
    return expert_catalog_copy


def test_expert_catalog_archives_resolved_r10_inputs_and_observations():
    catalog = r10_catalog()
    assert catalog.names("hardware") == ("h20", "h200-sxm-reference")
    assert "r10" in catalog.names("trials")
    trial = catalog.trial("r10")
    assert trial.verify()["data_files"] == 14
    assembled = trial.models()
    assert assembled.config.actor.parallel.dp == 4
    assert assembled.config.rollout.world_size == 4
    assert assembled.raw_actor.hardware.memory_gb == 97871 * 2**20 / 1e9
    assert assembled.raw_actor.model.experts == 128
    assert assembled.raw_actor.fsdp_host_rates.source_ids == (
        "explicit-unmeasured-host-prior",
    )
    assert len(trial.read("sample_observations")) == 324
    requests = trial.read("request_observations")
    assert len(requests) == 2286
    assert sum(row["output_tokens"] for row in requests) == 1_112_248
    assert len({(row["episode_id"], row["index"]) for row in requests}) == len(requests)
    groups = trial.read("group_observations")
    assert len(groups) == 81
    assert sum(row["completed"] is not None for row in groups) == 77
    assert sum(row["export"] is not None for row in groups) == 69


def test_expert_catalog_loads_frozen_coefficients_without_fitting(monkeypatch):
    def no_fit(*args, **kwargs):
        raise AssertionError("Archive loading must not fit labels")

    monkeypatch.setattr(expert.PhaseCalibration, "fit", no_fit)
    monkeypatch.setattr(expert.AggregateAdvantageModel, "fit", no_fit)
    trial = r10_catalog().trial("r10")
    models = trial.models()
    archived = trial.read("calibration")
    for phase, stored in zip(
        models.actor.calibration, archived["actor"]["phase_scales"], strict=True
    ):
        assert phase.scale == stored["scale"]
        assert phase.bounds == tuple(tuple(v) for v in stored["bounds"])
        assert phase.sample_ids == tuple(stored["sample_ids"])
        assert phase.profile == models.actor.profile_key(models.config.actor)
    assert models.rpc_timing.service_scale == archived["rpc"]["service_scale"]
    assert models.rpc_timing.response_seconds == archived["rpc"]["response_seconds"]
    assert (
        models.rpc_timing.response_delay(models.rpc_timing.profile, 791, 976).evidence
        == "assumed"
    )
    assert r10_catalog().trial("r10").models(calibrated=False).rpc_timing is None


def test_expert_catalog_preserves_strict_coverage_and_explicit_diagnostic_fallback():
    trial = r10_catalog().trial("r10")
    assembled = trial.models()
    unsupported = []
    for row in trial.read("batch_observations"):
        lengths = tuple(row["sequence_lengths"])
        work = WorkloadStats(
            16,
            sum(lengths),
            min_seq_len=min(lengths),
            max_seq_len=max(lengths),
            loss_tokens=row["loss_tokens"],
            sequence_lengths=lengths,
            trajectory_group_sizes=(4,) * 4,
            trajectory_dispatch_weights=tuple(row["trajectory_dispatch_weights"]),
        )
        if assembled.actor.estimate(assembled.config.actor, work).total.seconds is None:
            unsupported.append(row["step"])
    assert unsupported == [9, 11, 16]
    strict_models, scenario, plan = trial.simulation()
    diagnostic, zero, zero_plan = trial.simulation(
        cleanup_seconds=0, diagnostic_fallback=True
    )
    assert (
        strict_models.actor is not None
        and strict_models.actor == diagnostic.actor.bounded
    )
    assert len(scenario.groups) == len(zero.groups) == 77
    assert len(plan) == len(zero_plan) == 16
    assert plan[-1].checkpoint and not any(p.checkpoint for p in plan[:-1])
    assert scenario.staleness == 2 and scenario.workflow == "v2_inline"
    assert (
        sum(isinstance(n, expert.Request) for g in scenario.groups for n in g.nodes)
        == 2286
    )
    cleanups = [
        n
        for g in scenario.groups
        for n in g.nodes
        if isinstance(n, expert.ExternalCall) and n.pool == "cleanup"
    ]
    assert len(cleanups) == 308 and all(n.duration.seconds == 30 for n in cleanups)
    assert all(
        n.duration.seconds == 0
        for g in zero.groups
        for n in g.nodes
        if isinstance(n, expert.ExternalCall) and n.pool == "cleanup"
    )
    with pytest.raises(ValueError, match="nonnegative"):
        trial.simulation(cleanup_seconds=-1)


@pytest.mark.parametrize("key", ["request_observations", "groups"])
def test_expert_catalog_rejects_changed_numeric_or_compressed_data(
    expert_trial_copy, key
):
    trial = expert_trial_copy.trial("r10")
    path = trial.root / trial.manifest()["data"][key]["path"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="content hash mismatch"):
        trial.read(key)


def test_expert_catalog_rejects_profile_changes_and_unreviewed_formulas(
    expert_trial_copy, monkeypatch
):
    import areal.elastic.policy.reference.catalog as catalog_module

    trial = expert_trial_copy.trial("r10")
    monkeypatch.setattr(catalog_module, "code_revision", lambda: "unreviewed-formula")
    with pytest.raises(ValueError, match="compatibility audit"):
        trial.models()
    assert trial.models(calibrated=False).rpc_timing is None
    path = expert_trial_copy.root / "profiles/hardware/h20.json"
    value = json.loads(path.read_text())
    value["values"]["bf16_tflops"] *= 2
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="content hash mismatch"):
        trial.configuration()


def test_expert_catalog_accepts_new_model_profile_but_rejects_old_calibration(
    expert_trial_copy,
):
    import hashlib

    path = expert_trial_copy.root / "profiles/models/qwen3-30b-a3b.json"
    value = json.loads(path.read_text())
    value["id"] = "another-qwen3-moe"
    value["values"]["shape"]["layers"] += 1
    value["values"]["hf_config"]["num_hidden_layers"] += 1
    other = path.with_name("another-qwen3-moe.json")
    other.write_text(json.dumps(value))
    assert expert_trial_copy.model("another-qwen3-moe").layers == 49
    assert expert_trial_copy.model("qwen3-30b-a3b").layers == 48
    trial = expert_trial_copy.trial("r10")
    manifest = trial.manifest()
    manifest["profiles"]["model"] = {
        "name": value["id"],
        "sha256": hashlib.sha256(other.read_bytes()).hexdigest(),
    }
    (trial.root / "manifest.json").write_text(json.dumps(manifest))
    assert trial.models(calibrated=False).raw_actor.model.layers == 49
    with pytest.raises(ValueError, match="another resolved trial profile"):
        trial.models()


@pytest.mark.parametrize("name", ["../r10", "/tmp/r10", "r10/name", ""])
def test_expert_catalog_rejects_nonlocal_trial_names(name):
    with pytest.raises(ValueError, match="simple identifiers"):
        expert.Catalog().trial(name)


def test_expert_catalog_rejects_escaped_data_reference(expert_trial_copy):
    trial = expert_trial_copy.trial("r10")
    manifest = trial.manifest()
    manifest["data"]["workload"]["path"] = "../manifest.json"
    (trial.root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inside"):
        trial.read("workload")


def test_expert_catalog_assembles_only_the_hash_verified_profile_snapshot(monkeypatch):
    original = expert.Catalog.profile

    def changing_profile(self, kind, name, *, sha256=None):
        value = original(self, kind, name, sha256=sha256)
        if sha256 is None:
            if kind == "hardware":
                value["values"]["bf16_tflops"] *= 2
            elif kind == "rollout":
                value["values"]["config"]["max_batch_sequences"] *= 2
        return value

    monkeypatch.setattr(expert.Catalog, "profile", changing_profile)
    trial = r10_catalog().trial("r10")
    assembled = trial.models()
    assert assembled.raw_actor.hardware.bf16_tflops == 148
    assert assembled.config.rollout.max_batch_sequences == 4
    assert trial.configuration() == assembled.config


def test_expert_catalog_rejects_unreviewed_numpy_environment(monkeypatch):
    import areal.elastic.policy.reference.catalog as catalog_module

    monkeypatch.setattr(
        catalog_module, "numerical_environment", lambda: {"numpy": "unreviewed"}
    )
    trial = r10_catalog().trial("r10")
    with pytest.raises(ValueError, match="compatibility audit"):
        trial.models()
    assert trial.models(calibrated=False).actor.calibration == ()


def test_expert_catalog_rejects_model_dimensions_inconsistent_with_hf_config(
    expert_catalog_copy,
):
    path = expert_catalog_copy.root / "profiles/models/qwen3-30b-a3b.json"
    value = json.loads(path.read_text())
    value["values"]["shape"]["layers"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="disagree"):
        expert_catalog_copy.model("qwen3-30b-a3b")


def test_expert_catalog_rejects_unknown_rollout_policy(expert_catalog_copy):
    path = expert_catalog_copy.root / "profiles/rollout/sglang-dp2-tp2.json"
    value = json.loads(path.read_text())
    value["values"]["policy"] = "another_batching_rule"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="scheduling policy"):
        expert_catalog_copy.rollout("sglang-dp2-tp2")


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
    from areal.elastic.policy.reference.cost_model._actor.packing import (
        plan_workload as expert_plan,
    )
    from areal.elastic.policy.reference.cost_model.fsdp import fsdp_work

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
    assert dict(unknown.stages)["recompute_logp"].seconds == 0  # Disabled in config.
    rates = expert.FSDPHostRates(8, 7, 20, 0.00001, 0.001, ("unit-only",))
    model = replace(expert_model, fsdp_host_rates=rates)
    known = model.estimate(config, workload, source_profile=model.profile_key(config))
    assert dict(known.stages)["train_step"].seconds > 0
    assert known.host_memory_gb[0] == pytest.approx(
        expert_model.model.total_parameters / 4 * 10 / 1e9
    )
    assert known.static_memory_gb == (0,)
    assert model.profile_key(config) != expert_model.profile_key(config)
    assert dict(dict(known.fsdp_work)["train_step"])["gradient_reductions"] == 1


def test_expert_fsdp_checkpoint_does_not_gather_a_third_time(expert_model):
    """PyTorch PRE_BACKWARD suppresses checkpoint forward hooks."""
    from areal.elastic.policy.reference.cost_model._actor.components import (
        Context as ExpertContext,
    )
    from areal.elastic.policy.reference.cost_model._actor.components import (
        Operators as ExpertOperators,
    )
    from areal.elastic.policy.reference.cost_model._actor.components import (
        Transformer as ExpertTransformer,
    )
    from areal.elastic.policy.reference.cost_model._actor.packing import (
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
        parameter_shard_bytes=10e9,
        staging_tensors=2,
        contiguous_tensor_bytes=1e9,
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
        parameter_shard_bytes=10e9,
        staging_tensors=2,
        contiguous_tensor_bytes=1e9,
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
        parameter_shard_bytes=10e9,
        staging_tensors=2,
        contiguous_tensor_bytes=1e9,
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
        parameter_shard_bytes=10e9,
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
    from areal.elastic.policy.reference.cost_model._actor.components import (
        Communications as ExpertCommunications,
    )
    from areal.elastic.policy.reference.cost_model._actor.components import (
        Context as ExpertContext,
    )
    from areal.elastic.policy.reference.cost_model._actor.components import (
        Operators as ExpertOperators,
    )

    config = _fsdp_cost_config()
    ops = ExpertOperators(
        expert_model.hardware, group_gbps={"dense_dp": 40, "expert_dp": 1}
    )
    ctx = ExpertContext(config, expert_model.model, ops, 4, True, True)
    params = expert_model.model.total_parameters
    groups = expert_model.model.layers + 2
    comm = ExpertCommunications()
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
    from areal.elastic.policy.reference.cost_model._actor.components import (
        Communications as ExpertCommunications,
    )
    from areal.elastic.policy.reference.cost_model._actor.components import (
        Context as ExpertContext,
    )
    from areal.elastic.policy.reference.cost_model._actor.components import (
        Operators as ExpertOperators,
    )
    from areal.elastic.policy.reference.cost_model._actor.packing import (
        plan_workload as expert_plan,
    )
    from areal.elastic.policy.reference.cost_model.fsdp import fsdp_work

    config = _fsdp_cost_config(recompute_logprob=True, n_mbs=2)
    resident = replace(config, offload_params=False)
    workload = WorkloadStats(8, 8 * 129, per_dp_sequence_lengths=((129, 129),) * 4)
    plan = expert_plan(config, workload, expert_model.model.hidden_size)
    offload_work = dict(fsdp_work(config, expert_model.model, plan))
    resident_work = dict(fsdp_work(resident, expert_model.model, plan))
    # One local BF16 read and write per gather, twice per train microbatch.
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
    comm = ExpertCommunications()
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
    assert bounds["largest_bf16_logits"] == 512 * model.model.vocab_size * 2 / 1e9
    assert all(stage.feasibility == "unknown" for _, stage in prediction.stages)
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


@pytest.fixture
def model():
    return ActorCostModel(qwen3_30b_a3b(), hardware_profile("h200_sxm_reference"))


@pytest.fixture
def workload():
    return WorkloadStats(128, 128 * 4096, min_seq_len=1024, max_seq_len=8192)


def test_model_shape_matches_frozen_hf_config():
    """Active MoE compute and full weight storage must use different dimensions."""
    shape = qwen3_30b_a3b()
    assert shape.attention_parameters_per_layer == 18_874_368
    assert shape.expert_parameters_per_layer == 603_979_776
    assert shape.layers == 48 and shape.top_k == 8 and shape.experts == 128
    assert 30e9 < shape.total_parameters < 31e9
    assert len(read_data("qwen3_30b_a3b.json")["source_sha256"]) == 64


@pytest.mark.parametrize(
    "trial,expected",
    [
        ("S031", (4, 2, 4, 1, 4, 2)),
        ("S029", (4, 4, 1, 2, 8, 4)),
        ("S026", (32, 1, 1, 1, 1, 32)),
        ("T0390", (8, 4, 1, 1, 4, 8)),
    ],
)
def test_trial_geometry_matches_screenshot(trial, expected):
    """EP replaces other dimensions; it is not multiplied into world_size."""
    config = trial_config(trial)
    p = config.parallel
    assert (p.dp, p.tp, p.pp, p.cp, p.ep, p.expert_dp) == expected
    assert p.world_size == 32
    assert config.pad_to_maximum and config.max_tokens_per_mb == 32768
    assert config.n_mbs == 1


def test_nine_trials_include_duplicates_without_deleted_trial(model, workload):
    """The screenshot contains nine names but four distinct actor layouts."""
    rows = read_data("trials.json")["trials"]
    assert {row["id"] for row in rows} == {
        "S031",
        "S029",
        "S026",
        "S030",
        "T0390",
        "S028",
        "S025",
        "S013",
        "S024",
    }
    times = {
        row["id"]: model.predict(trial_config(row["id"]), workload) for row in rows
    }
    assert times["S031"] == times["S030"] == times["S013"]
    assert times["S029"] == times["S028"]
    assert times["T0390"] == times["S025"] == times["S024"]


@pytest.mark.parametrize(
    "allocation",
    [
        "megatron:(attn:d4p4t2|ffn:d3p4e4)",
        "archon:(attn:d4t4|ffn:e3)",
        "archon:(attn:d4|ffn:e4",
        "archon:attn:d4|ffn:e4)",
        "archon:(attn:d4d4|ffn:e4)",
    ],
)
def test_bad_layout_is_rejected(allocation):
    """Malformed allocations cannot silently turn into another GPU count."""
    with pytest.raises(ValueError):
        ParallelLayout.from_allocation(allocation)


def test_resolved_config_maps_source_defaults_and_checkpoint_mode():
    """Unknown selective checkpoint work is a declared prior, not full recompute."""
    data = {
        "actor": {
            "backend": "archon:(attn:d8t4|ffn:d8e4)",
            "mb_spec": {"max_tokens_per_mb": 32768},
            "gradient_checkpointing": True,
            "archon": {"ac_mode": "selective", "selective_ac_option": "2"},
        }
    }
    cfg = ActorConfig.from_areal(data)
    assert cfg.checkpoint_recompute_fraction == 0.5
    assert cfg.pad_to_maximum is False  # Actual TrainEngineConfig default.
    assert cfg.expert_dispatcher == "alltoall"
    data["actor"]["archon"]["fp8_config"] = {"mode": "blockwise"}
    with pytest.raises(ValueError, match="FP8"):
        ActorConfig.from_areal(data)


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


def test_value_packing_matches_actual_areal_source():
    """Execute only the CPU allocation functions from current source as an oracle."""
    tree = ast.parse((ROOT / "areal/utils/seqpack.py").read_text())
    names = {"_ffd_allocate", "ffd_allocate", "balanced_greedy_partition"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"np": np, "bisect": bisect}
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]),
            "AReaL seqpack allocation",
            "exec",
        ),
        namespace,
    )
    rng = random.Random(127)
    for _ in range(50):
        lengths = rng.sample(range(1, 500), 32)
        actual = namespace["ffd_allocate"](lengths, 1200, 4, 2)
        expected_values = sorted(sorted(lengths[i] for i in group) for group in actual)
        assert (
            sorted(sorted(group) for group in ffd(lengths, 1200, 4, 2))
            == expected_values
        )
        assert ffd(lengths, 1200, 4, 2) == [
            [lengths[i] for i in group]
            for group in sorted(sorted(group) for group in actual)
        ]
        groups = namespace["balanced_greedy_partition"](lengths, 4)
        assert balanced_partition(lengths, 4) == [
            [lengths[i] for i in group] for group in groups
        ]


def test_padding_includes_one_quadratic_dummy_segment():
    """Packed padding is computed by attention, even though loss strips it."""
    cfg = ActorConfig(
        "archon", ParallelLayout(), max_tokens_per_mb=32768, pad_to_maximum=True
    )
    mb = _pad([2048, 4096], cfg, 2048)
    assert mb.compute_tokens == 32768
    assert mb.attention_length_squared == 2048**2 + 4096**2 + (32768 - 6144) ** 2
    assert _pad([513], replace(cfg, pad_to_maximum=False), 2048).compute_tokens == 1024


def test_megatron_pipeline_overrides_n_mbs_one(workload):
    """PP4 enforces eight microbatches and PP divisibility for both passes."""
    cfg = replace(trial_config("S031"), recompute_logprob=True)
    plan = plan_workload(cfg, workload, 2048)
    assert len(plan.train[0][0]) == 8 and len(plan.logprob[0]) == 8
    assert plan.train_compute_tokens == 4 * 8 * 32768
    with pytest.raises(ValueError, match="too few"):
        plan_workload(cfg, WorkloadStats(16, 16 * 4096), 2048)


def test_prompt_batch_is_not_inferred_as_actor_sequences():
    """A prompt batch of 16 is not enough to infer PP4 actor microbatch count."""
    data = {
        "actor": {
            "backend": "megatron:(attn:d4p4t2|ffn:d2p4e4)",
            "mb_spec": {"max_tokens_per_mb": 32768},
        },
        "train_dataset": {"batch_size": 16},
    }
    cfg = ActorConfig.from_areal(data)
    assert cfg.parallel.dp == 4 and not hasattr(cfg, "n_seqs")


def test_observed_microbatch_count_is_optional_and_checked(workload):
    """An observed per-rank count refines packing but cannot override capacity."""
    cfg = trial_config("S026")
    plan = plan_workload(cfg, replace(workload, train_microbatches=2), 2048)
    assert len(plan.train[0][0]) == 2
    with pytest.raises(ValueError, match="incompatible"):
        plan_workload(
            replace(cfg, max_tokens_per_mb=8192),
            replace(workload, train_microbatches=1),
            2048,
        )


@pytest.mark.parametrize(
    "decoupled,method,recompute,enabled",
    [
        (False, "recompute", False, False),
        (False, "loglinear", True, True),
        (True, "recompute", False, True),
        (True, "metrics", False, True),
        (True, "loglinear", True, False),
        (True, "reuse_train_logp", False, False),
    ],
)
def test_logprob_phase_matches_proximal_config(
    model, workload, decoupled, method, recompute, enabled
):
    """Conditional logprob timing follows AReaL should_compute_prox_logp."""
    cfg = replace(
        trial_config("S026"),
        use_decoupled_loss=decoupled,
        prox_logp_method=method,
        recompute_logprob=recompute,
    )
    assert (model.predict(cfg, workload)["recompute_logp"] > 0) == enabled


def test_full_checkpoint_only_changes_training_phase(model, workload):
    """Logprob executes eval forward and does not pay checkpoint recomputation."""
    cfg = replace(trial_config("S026"), recompute_logprob=True)
    before = model.predict(cfg, workload)
    after = model.predict(replace(cfg, gradient_checkpointing=True), workload)
    assert after["recompute_logp"] == before["recompute_logp"]
    assert after["train_step"] > before["train_step"]


def test_ppo_minibatches_count_optimizer_steps_and_respect_padding(model, workload):
    """Each PPO minibatch causes one update; no invented PPO epoch multiplier."""
    cfg = replace(trial_config("S026"), ppo_n_minibatches=2)
    report = model.estimate(cfg, workload, sensitivity=False)
    assert len(report["packing"]["train_microbatches_per_rank"]) == 2
    assert report["total_actor_seconds"] == sum(report["stages_seconds"].values())


def test_components_can_be_replaced_independently(model, workload):
    """Replacing one phase preserves the other component estimates."""

    class FixedAdvantages:
        def estimate(self, config, tokens, op):
            return 7.0

    cfg = trial_config("S026")
    before = model.predict(cfg, workload)
    custom = replace(model, components=ActorComponents(advantages=FixedAdvantages()))
    after = custom.predict(cfg, workload)
    assert after["compute_advantage"] == 7
    assert after["train_step"] == before["train_step"]
    assert Serial((Constant(2), Parallel((Constant(3), Constant(4))))).seconds() == 6
    assert Scale(Constant(2), 3).seconds() == 6


def test_fsdp_uses_all_expert_weights_and_optimizer_shards():
    """FSDP gathers full experts, while Adam state shards use expert DP."""
    cfg = trial_config("S029")
    shape = qwen3_30b_a3b()
    ctx = Context(
        cfg,
        shape,
        Operators(hardware_profile("h200_sxm_reference")),
        shape.layers,
        True,
        True,
    )
    dense, expert = ctx.owned_parameters
    assert expert == shape.layers * shape.expert_parameters_per_layer / 8
    assert dense > 0 and expert > 3e9
    opt = Optimizer().estimate(ctx)
    assert opt["optimizer"] > 0 and opt["gradient_sync_raw"] == 0
    mb = _pad([4096], cfg, shape.hidden_size)
    assert Transformer().forward(ctx, mb)["raw_fsdp"] > 0
    assert Transformer().estimate(ctx, mb, training=True)["raw_gradient_sync"] > 0


def test_hierarchical_ep_is_not_applied_to_regular_alltoall():
    """Public HybridEP optimizations must not leak into target backend defaults."""
    shape = qwen3_30b_a3b()
    cfg = ActorConfig("megatron", ParallelLayout(dp=16, ep=16))
    ctx = Context(
        cfg, shape, Operators(hardware_profile("h100_sxm_reference")), 48, True, True
    )
    regular = Communications().expert_exchange(ctx, 4096)
    hierarchical = Communications().expert_exchange(
        replace(ctx, config=replace(cfg, expert_dispatcher="hierarchical")), 4096
    )
    assert regular > hierarchical > 0


def test_estimate_reports_uncertainty_without_claiming_confidence(model, workload):
    """Summary-only scenarios and backend transfer have visible limitations."""
    report = model.estimate(trial_config("S029"), workload)
    assert "NOT a confidence interval" in report["sensitivity"]["kind"]
    assert "bounded_tail_lengths" in report["sensitivity"]["scenarios_seconds"]
    assert report["packing"]["exact_lengths"] is False
    assert report["calibration"]["independent_stage_validation"] is False
    assert any("Archon" in item for item in report["assumptions"])


def test_public_fit_excludes_holdout_and_requires_transfer_opt_in(model):
    """Independent evaluation is reproducible and target-domain claims stay honest."""
    report = public_validation()
    assert report["holdout_count"] == 1
    heldout = next(row for row in report["points"] if row["split"] == "holdout")
    assert heldout["used_for_fit"] is False
    assert (
        0 < report["holdout_mape"] < 1
    )  # Broad sanity, not an accuracy acceptance gate.
    points = read_data("public_training.json")["points"]
    with pytest.raises(ValueError, match="Holdout"):
        Calibration.fit(points)
    cfg = trial_config("S029")
    with pytest.raises(ValueError, match="opt into transfer"):
        bundled_calibration().apply(model, cfg)
    transferred = bundled_calibration().apply(model, cfg, allow_transfer=True)
    assert "UNVALIDATED" in transferred.calibration_note
    config, workload = point_inputs(points[0])
    fitted = bundled_calibration().apply(
        ActorCostModel(qwen3_30b_a3b(), hardware_profile("h100_sxm_reference")), config
    )
    assert fitted.predict(config, workload)["train_step"] == pytest.approx(
        points[0]["step_seconds"]
    )


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
            "dur": 1e6,
            "pid": 0,
            "tid": 1,
            "args": {"global_step": 4},
        },
        {
            "name": "train.ppo_update",
            "ph": "X",
            "ts": 1e5,
            "dur": 5e5,
            "pid": 0,
            "tid": 1,
            "args": {"global_step": 4},
        },
        {
            "name": "ppo_actor.ppo_update",
            "ph": "X",
            "ts": 0,
            "dur": 9e5,
            "pid": 1,
            "tid": 1,
            "args": {"global_step": 4},
        },
        {
            "name": "ppo_actor.ppo_update",
            "ph": "X",
            "ts": 0,
            "dur": 8e5,
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


@pytest.mark.parametrize(
    "kwargs", [{"moe": 0}, {"network": math.nan}, {"fsdp_overlap": 2}]
)
def test_efficiency_rejects_invalid_physical_rates(kwargs):
    """Invalid priors cannot silently produce nan/negative predictions."""
    with pytest.raises(ValueError):
        Efficiency(**kwargs)


def test_bundled_data_all_have_valid_json():
    """The model ships no network dependency for loading facts or trial templates."""
    assert len(list(DATA.glob("*.json"))) >= 5
    for path in DATA.glob("*.json"):
        assert isinstance(json.loads(path.read_text()), dict)


def test_trace_rank_identity_handles_reused_pids_across_nodes():
    """Remote ranks may reuse the same OS pid; union must not merge their steps."""
    events = [
        {
            "name": "ppo_actor.ppo_update",
            "ph": "X",
            "ts": 0,
            "dur": 1e6,
            "pid": 10,
            "tid": 1,
            "args": {"global_step": 4, "rank": 0},
        },
        {
            "name": "ppo_actor.ppo_update",
            "ph": "X",
            "ts": 2e6,
            "dur": 1e6,
            "pid": 10,
            "tid": 1,
            "args": {"global_step": 4, "rank": 8},
        },
    ]
    assert actor_trace_times(events)["steps_seconds"] == {"4": {"train_step": 1}}


def test_calibration_rejects_changed_hardware_with_same_profile_name():
    """A name alone cannot prove matching bandwidth and layout assumptions."""
    point = read_data("public_training.json")["points"][0]
    cfg, _ = point_inputs(point)
    changed = replace(hardware_profile("h100_sxm_reference"), inter_node_gbps=10)
    with pytest.raises(ValueError, match="opt into transfer"):
        bundled_calibration().apply(ActorCostModel(qwen3_30b_a3b(), changed), cfg)


def test_config_rejects_string_booleans():
    """A JSON string 'false' must not enable a computational phase."""
    with pytest.raises(ValueError, match="boolean"):
        replace(trial_config("S026"), recompute_logprob="false")


def test_archon_gradient_sync_is_charged_per_microbatch(model, workload):
    """Current SequentialRunner has no FSDP no_sync across microbatches."""

    class NoGradientSync(Communications):
        def gradient_sync(self, ctx):
            return 0.0

    cfg = replace(trial_config("S026"), n_mbs=2)
    synced = replace(model, efficiency=replace(model.efficiency, fsdp_overlap=0))
    unsynced = replace(
        synced, components=ActorComponents(transformer=Transformer(NoGradientSync()))
    )
    ctx = Context(
        cfg, model.model, Operators(model.hardware, synced.efficiency), 48, True, True
    )
    difference = (
        synced.predict(cfg, workload)["train_step"]
        - unsynced.predict(cfg, workload)["train_step"]
    )
    assert difference == pytest.approx(2 * Communications().gradient_sync(ctx))


def test_archon_final_head_gather_is_cached_in_eval(model):
    """Default final norm/head is not resharded between eval microbatches."""
    cfg = trial_config("S029")
    ctx = Context(cfg, model.model, Operators(model.hardware), 48, True, True)
    comm = Communications()
    first = comm.parameter_gather(ctx)
    later = comm.parameter_gather(replace(ctx, microbatch_index=1))
    assert first > later > 0
    assert comm.parameter_gather(ctx, backward=True) == later
    no_reshard = replace(
        ctx, config=replace(cfg, reshard_after_forward=False), microbatch_index=1
    )
    assert comm.parameter_gather(no_reshard) == 0


def test_archon_cp_repeats_kv_after_tp_sharding(model, monkeypatch):
    """TP4/CP2 expands local KV1 to KV2, equivalent to global KV8."""
    recorded = []
    original = Operators.collective

    def capture(self, payload_bytes, size, group, world, **kwargs):
        if group == "cp":
            recorded.append(payload_bytes)
        return original(self, payload_bytes, size, group, world, **kwargs)

    monkeypatch.setattr(Operators, "collective", capture)
    cfg = trial_config("S029")
    ctx = Context(cfg, model.model, Operators(model.hardware), 48, True, True)
    Communications().forward(ctx, _pad([4096], cfg, 2048))
    assert recorded == [32768 / (4 * 2) * 128 * (2 * 32 + 2 * 8) * 2]


def test_megatron_pp_partition_matches_current_source():
    """The backend balances local expert params, so EP4 splits 11/13/13/11."""
    from types import SimpleNamespace

    from areal.elastic.policy.basic.cost_model.actor import layer_partition

    path = ROOT / "areal/engine/megatron_utils/pipeline_parallel.py"
    tree = ast.parse(path.read_text())
    names = {"estimate_stage_parameter_buckets", "_compute_stage_layer_lengths"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"math": math, "PretrainedConfig": object, "TransformerConfig": object}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    hf = SimpleNamespace(**read_data("qwen3_30b_a3b.json")["config"])
    tf = SimpleNamespace(
        num_layers=48,
        expert_model_parallel_size=4,
        num_moe_experts=128,
        gated_linear_unit=True,
    )
    weights, embedding, output = namespace["estimate_stage_parameter_buckets"](hf, tf)
    expected = tuple(
        namespace["_compute_stage_layer_lengths"](weights, embedding, output, 4)
    )
    assert (
        layer_partition(trial_config("S031"), qwen3_30b_a3b())
        == expected
        == (11, 13, 13, 11)
    )


def test_megatron_dp_groups_fit_inside_pipeline_stage_node(model):
    """PP4 splits the 32 ranks into four 8-rank blocks; DP need not use IB."""
    first = Operators(model.hardware, stage_world_size=8)
    second = Operators(replace(model.hardware, inter_node_gbps=1), stage_world_size=8)
    assert first.collective(1e9, 4, "dense_dp", 32) == second.collective(
        1e9, 4, "dense_dp", 32
    )
    assert first.collective(1e9, 2, "pp", 32) < second.collective(1e9, 2, "pp", 32)


def test_megatron_alignment_expands_capacity_after_ffd():
    """Backend grows the padded target when per-sequence alignment exceeds cap."""
    cfg = ActorConfig(
        "megatron", ParallelLayout(tp=2), max_tokens_per_mb=32768, pad_to_maximum=True
    )
    mb = _pad([16383, 16385], cfg, 2048)
    assert mb.compute_tokens == 33024
    assert _pad([513], replace(cfg, pad_to_maximum=False), 2048).compute_tokens == 768


def test_multiturn_trajectories_are_atomic_dispatch_items():
    """Sequence count need not divide DP when one trajectory contains many rows."""
    from areal.elastic.policy.basic.cost_model.packing import (
        dispatch_workload,
    )

    stats = WorkloadStats(
        5,
        50,
        sequence_lengths=(10, 10, 10, 10, 10),
        trajectory_group_sizes=(1, 2, 1, 1),
    )
    shards = dispatch_workload(stats, stats.lengths(), 2)
    assert sorted(len(rank) for rank in shards) == [2, 3]
    cfg = ActorConfig("archon", ParallelLayout(dp=2), max_tokens_per_mb=32768)
    plan = plan_workload(cfg, stats, 2048)
    assert plan.exact_lengths and not plan.exact_dispatch
    assert plan.exact_dispatch is False  # RTensor weights are not supplied.
    with pytest.raises(ValueError, match="Atomic dispatch"):
        plan_workload(cfg, replace(stats, trajectory_group_sizes=(5,)), 2048)


def test_summary_only_dispatch_does_not_assert_sequence_divisibility():
    """Without trajectory metadata, nondivisible row counts remain an approximation."""
    stats = WorkloadStats(5, 50)
    cfg = ActorConfig("archon", ParallelLayout(dp=2))
    plan = plan_workload(cfg, stats, 2048)
    assert not plan.exact_dispatch
    assert any("trajectory grouping" in item for item in plan.assumptions)
    observed = replace(stats, per_dp_sequence_lengths=((10, 10), (10, 10, 10)))
    assert plan_workload(cfg, observed, 2048).exact_dispatch


@pytest.mark.parametrize("optimizer", [None, {"type": "sgd"}, {"type": "adam_bf16"}])
def test_resolved_config_rejects_optimizer_outside_traffic_model(optimizer):
    """Adam FP32-state traffic cannot be silently reused for another optimizer."""
    actor = {
        "backend": "archon:(attn:d32|ffn:d32)",
        "mb_spec": {"max_tokens_per_mb": 32768},
        "optimizer": optimizer,
    }
    with pytest.raises(ValueError, match="Adam"):
        ActorConfig.from_areal(actor)


def test_jsonl_trace_inherits_parent_step_without_summing_parent():
    """Real JSONL scopes can inherit step metadata from a containing event."""
    parent = {
        "name": "worker.step",
        "ph": "X",
        "ts": 0,
        "dur": 2e6,
        "pid": 1,
        "tid": 2,
        "args": {"global_step": 3, "rank": 4},
    }
    child = {
        "name": "ppo_actor.compute_logp",
        "ph": "X",
        "ts": 1e5,
        "dur": 5e5,
        "pid": 1,
        "tid": 2,
        "args": {"rank": 4},
    }
    report = actor_trace_times(
        trace_events(json.dumps(parent) + "\n" + json.dumps(child))
    )
    assert report["steps_seconds"] == {"3": {"recompute_logp": 0.5}}
    assert report["unassigned_events"] == 0


def test_history_actor_adapter_preserves_phase_and_runtime_identity(model, workload):
    """Phase predictions use the recorded configuration and matching timing scope."""
    config = trial_config("S026")
    inputs = HistoricalInput("actor", config, workload, "synthetic", ACTOR_BOUNDARY)
    predictor = BasicHistoryPredictor("synthetic", actor=model)
    values = dict(predictor.predict(inputs).values)
    assert values["train_step_seconds"] > 0
    assert values["actor_total_seconds"] == pytest.approx(
        sum(
            values[name + "_seconds"]
            for name in ("recompute_logp", "compute_advantage", "train_step")
        )
    )
    assert not predictor.predict(replace(inputs, execution_id="other-runtime")).values
    assert not predictor.predict(
        replace(inputs, timing_boundary="gpu-kernel-only")
    ).values
    assert not BasicHistoryPredictor("synthetic").predict(inputs).values


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
            expert.AdvantageTerm("batching", (("padded_slots", 1e-8),), ("unit-only",)),
            expert.AdvantageTerm(
                "recurrence", (("gae_iterations", 1e-6),), ("unit-only",)
            ),
            expert.AdvantageTerm(
                "normalization", (("constant", 1e-5),), ("unit-only",)
            ),
            expert.AdvantageTerm("transport", (("constant", 1e-4),), ("unit-only",)),
            expert.AdvantageTerm("diagnostics", (("constant", 1e-4),), ("unit-only",)),
        ),
    )


def _expert_actor_input():
    cfg = expert.ActorConfig(
        "archon", expert.ParallelLayout(dp=2), max_tokens_per_mb=256
    )
    return cfg, WorkloadStats(
        16,
        1024,
        sequence_lengths=(64,) * 16,
        trajectory_group_sizes=(1,) * 16,
        trajectory_dispatch_weights=(64,) * 16,
    )


def test_expert_pipeline_uses_dependencies_for_forward_and_one_f_one_b():
    """Two-stage/two-microbatch schedules have independently countable critical paths."""
    from areal.elastic.policy.reference.cost_model.actor import (
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
    monkeypatch.setattr(
        ActorCostModel,
        "predict",
        lambda *args: (_ for _ in ()).throw(AssertionError("Basic called")),
    )
    monkeypatch.setattr(
        ActorCostModel,
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
        expert.PhaseCalibration.fit(
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
        expert_model, hardware=replace(expert_model.hardware, memory_gb=1e-8)
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
        expert.InferenceRates(0.5, 0.5, 0.5, 0.5, 1e-6),
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
    from areal.elastic.policy.reference.cost_model._actor.components import (
        Communications,
        Context,
        Operators,
        Transformer,
    )
    from areal.elastic.policy.reference.cost_model.actor import (
        _pipeline_partition,
    )

    data = {
        "backend": "archon:(attn:d1p4|ffn:d1p4)",
        "mb_spec": {"max_tokens_per_mb": 256},
    }
    with pytest.raises(ValueError, match="1F1B"):
        expert.ActorConfig.from_areal(data)
    data["archon"] = {"pp_schedule": "1F1B"}
    cfg = expert.ActorConfig.from_areal(data)
    assert _pipeline_partition(cfg, replace(expert_model.model, layers=8)) == (
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
    # Rank 0 = 512+63+800=1375; rank 1 = 1024+127+100=1251.
    assert dict(prediction.stages)["compute_advantage"].seconds == 1387
    assert sum(e.seconds for _, e in prediction.advantage_components) == 1387
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
        expert.PhaseCalibration.fit(
            "unit",
            (
                point,
                replace(
                    point, sample_id="worker", timing_boundary="worker_method_host_wall"
                ),
            ),
        )
    fitted = expert.PhaseCalibration.fit("unit", (point,))
    wrong = fitted.apply(
        1, raw.profile, point.phase, point.features, timing_boundary="rank_mean"
    )
    assert wrong.seconds is None and "boundary mismatch" in " ".join(wrong.notes)
    features = tuple(
        (name, value * 2 if name == "n_seqs" else value)
        for name, value in point.features
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
    fitted = expert.PhaseCalibration.fit(
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
    assert sum(e.seconds for _, e in prediction.advantage_components) == pytest.approx(
        total
    )
    assert all(
        e.evidence == "assumed" and "Uncalibrated component prior" in e.notes[0]
        for _, e in prediction.advantage_components
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
    return config, base, fitted, observations


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
        sum(e.seconds for _, e in prediction.stages)
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
        expert.AdvantageObservation(
            str(n), "profile", WorkloadStats(1, n), y, model.timing_boundary
        )
        for n, y in zip((1, 2, 3), labels, strict=True)
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
    # The other stages retain their independent phase scales.
    prediction = fitted.estimate(config, observations[1].workload)
    train = dict(prediction.stages)["train_step"].seconds
    scale = expert.PhaseCalibration.fit(
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
    # ExpertModels hashes the entire immutable fit, including labels and coefficients.
    assert (
        expert.ExpertModels(fitted, None, None).snapshot_id
        != expert.ExpertModels(refitted, None, None).snapshot_id
    )


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
    from areal.elastic.policy.reference.cost_model._actor.packing import (
        ffd as expert_ffd,
    )
    from areal.elastic.policy.reference.cost_model._actor.packing import (
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
    from areal.elastic.policy.reference.cost_model._actor.packing import (
        _pad as expert_pad,
    )

    config = expert.ActorConfig(
        "archon", expert.ParallelLayout(), max_tokens_per_mb=2048
    )
    # utils/data.py: round 600 -> 768 (256 page), then Archon: page=682 -> 1364.
    result = expert_pad([600], config, 1536)
    assert result.compute_tokens == 1364 and result.padding_tokens == 764
    # seq_len_divisor=2*TP*CP matters even when a page is odd.
    assert expert_pad([1], config, 2049).compute_tokens == 1022
    assert (
        expert_pad([600], replace(config, pad_to_maximum=True), 1536).compute_tokens
        == 2048
    )


def test_expert_equal_token_summaries_do_not_identify_microbatch_counts():
    """Summary equality is not evidence of equal packing or compute work."""
    from areal.elastic.policy.reference.cost_model._actor.packing import (
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
    assert not proxy.exact_lengths and not proxy.exact_dispatch


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
    assert len(work.calls) == 2  # TP copies removed; DP retained.
    assert all(s.attention_length_squared is None for _, s in work.calls)
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
        _packing_timing_point(expert_model, n, i) for i, n in enumerate((2, 4, 8))
    )
    diagnostic = expert.PackingTimingDiagnostic("train_step").fit(
        "synthetic-fit", points
    )
    heldout = _packing_timing_point(expert_model, 6, 20)
    work = heldout.work
    predicted = expert_model.estimate(work.config, work.workload).packing
    result = diagnostic.evaluate(heldout, predicted)
    assert work.phase_microbatches("train_step") == 6  # Not 24 (DP2/TP2).
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
        _packing_timing_point(expert_model, n, i) for i, n in enumerate((2, 4, 8))
    )
    diagnostic = expert.PackingTimingDiagnostic("train_step").fit(
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
        diagnostic = expert.PackingTimingDiagnostic("train_step")
    result = diagnostic.predict(work)
    assert result.seconds is None and result.evidence == "unsupported"


def test_expert_packing_diagnostic_rejects_mixed_phase_duplicate_fit_and_rank_mean(
    expert_model,
):
    """Calibration keeps run/step provenance, phase and trainer wall-clock scope."""
    points = tuple(
        _packing_timing_point(expert_model, n, i) for i, n in enumerate((2, 4, 8))
    )
    with pytest.raises(ValueError, match="wall"):
        replace(points[0], timing_boundary="rank_mean")
    with pytest.raises(ValueError, match="phase"):
        expert.PackingTimingDiagnostic("recompute_logp").fit("bad", points)
    with pytest.raises(ValueError, match="Duplicate"):
        expert.PackingTimingDiagnostic("train_step").fit("bad", points + (points[0],))
    same = tuple(_packing_timing_point(expert_model, 4, i) for i in range(2))
    with pytest.raises(ValueError, match="distinct"):
        expert.PackingTimingDiagnostic("train_step").fit("bad", same)
    constant = expert.PackingTimingDiagnostic("train_step", form="constant").fit(
        "constant", same
    )
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


@pytest.mark.parametrize("delta", [-0.00001, 0.00001, 2.0])
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
    from areal.elastic.policy.reference.cost_model._actor.packing import (
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
    from areal.elastic.policy.reference.cost_model._actor.packing import (
        dispatch_workload,
    )
    from areal.elastic.policy.reference.cost_model._actor.packing import (
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
    from areal.elastic.policy.reference.cost_model._actor.packing import (
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
    from areal.elastic.policy.reference.cost_model._actor.packing import (
        dispatch_workload,
    )
    from areal.elastic.policy.reference.cost_model._actor.packing import (
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
        groups = allocate(list(map(sum, trajectories)), int(1e12), 4)
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
    from areal.elastic.policy.reference.cost_model._actor.packing import (
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
        _packing_timing_point(expert_model, n, i) for i, n in enumerate((2, 4))
    )
    diagnostic = expert.PackingTimingDiagnostic("train_step").fit("frozen", points)
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
    from areal.elastic.policy.reference.cost_model._actor.packing import (
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
    # The actual runtime FFD gives DP0 (99,99,79), DP1 (100,1,1,1,80).
    expected = [
        [n for i in group for n in trajectories[i]]
        for group in _runtime_ffd()(list(map(sum, trajectories)), int(1e12), 2)
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
            "r0",
            "rollout",
            "node0",
            **(zeros | {"expected_receive_bytes": 2_000_000_000}),
        ),
    )
    work = expert.PublicationWork(
        expert.publication_binding(raw, actor, rollout),
        "megatron",
        "awex_dense",
        "unit-traffic-oracle",
        ranks,
        (
            expert.PublicationFlow("a0", "r0", 1_000_000_000, 1),
            expert.PublicationFlow("a1", "r0", 1_000_000_000, 1),
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
    return model, raw, actor, rollout, work


def test_publication_shared_receiver_is_a_bottleneck(publication_case):
    model, raw, actor, rollout, work = publication_case
    result = model.estimate(raw, actor, rollout, work)
    assert result.duration.seconds == pytest.approx(2.000002)
    assert dict(result.features) == {"wire_bytes": 2e9, "messages": 2}
    assert result.rounds[0]["resource_bytes"]["receive:r0"] == 2e9
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
        replace(work.ranks[1], gather_bytes=3_000_000_000, gather_calls=1),
        replace(work.ranks[2], expected_receive_bytes=1_000_000_000),
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
            h2d_bytes=1_000_000_000,
            h2d_tensors=1,
            send_copy_bytes=1_000_000_000,
            send_copy_tensors=1,
        )
        if r.role == "actor"
        else r
        for r in work.ranks
    )
    result = model.estimate(raw, actor, rollout, replace(work, ranks=ranks))
    assert dict(result.stages)["prepare"].seconds == 3  # 2s shared staging + 1s copy
    assert result.duration.seconds == pytest.approx(5.000002)


def test_publication_pre_receive_and_writeback_copies_are_separate(publication_case):
    model, raw, actor, rollout, work = publication_case
    ranks = work.ranks[:2] + (
        replace(
            work.ranks[2],
            receive_prepare_bytes=1_000_000_000,
            receive_prepare_tensors=1,
            receive_copy_bytes=2_000_000_000,
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
    assert result.rounds[0]["resource_bytes"]["nic-out:node0"] == 2e9
    assert result.duration.seconds == 2  # shared NIC dominates per-GPU links


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
    fit = expert.PhaseCalibration.fit(
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
    assert scaled.stages == prior.stages  # raw components were not independently fitted
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


def test_r10_publication_archives_actual_peer_geometry_without_timing_fit():
    assembled = r10_catalog().trial("r10").models(calibrated=False)
    document = json.loads(
        (
            ROOT
            / "examples/autopilot/elastic/profiles/r10-publication-structural-v1.json"
        ).read_text()
    )
    model, work = expert.ExpertPublicationModel.from_dict(
        document, assembled.raw_actor.hardware
    )
    result = model.estimate(
        assembled.raw_actor, assembled.config.actor, assembled.config.rollout, work
    )
    assert result.duration.evidence == "assumed"
    assert dict(result.features) == {"wire_bytes": 122_179_665_920, "messages": 79_520}
    for rank in work.ranks[:4]:
        assert rank.h2d_bytes == 15_266_061_312 and rank.h2d_tensors == 5043
        assert rank.send_copy_bytes == 10_066_329_600 and rank.send_copy_tensors == 3168
    changed = replace(
        assembled.raw_actor,
        hardware=expert.Catalog().hardware("h200-sxm-reference", "h200-8gpu-reference"),
    )
    assert (
        replace(model, hardware=changed.hardware)
        .estimate(changed, assembled.config.actor, assembled.config.rollout, work)
        .duration.seconds
        is None
    )


def test_publication_disjoint_endpoints_overlap_but_explicit_rounds_serialize(
    publication_case,
):
    model, raw, actor, rollout, work = publication_case
    rollout = replace(rollout, replicas=2)
    zero = {f.name: 0 for f in fields(expert.PublicationRank)[3:]}
    ranks = work.ranks[:2] + (
        replace(work.ranks[2], expected_receive_bytes=1_000_000_000),
        expert.PublicationRank(
            "r1",
            "rollout",
            "node0",
            **(zero | {"expected_receive_bytes": 1_000_000_000}),
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
    assert result.rounds[0]["resource_bytes"]["fabric:node0"] == 2e9
    assert (
        result.duration.seconds == 4
    )  # shared fabric, rather than each 1 GB/s endpoint
    missing = replace(
        model, rates=replace(model.rates, node_fabric_gbps=None)
    ).estimate(raw, actor, rollout, work)
    assert missing.duration.seconds is None
    assert all(v is None for v in missing.rounds[0]["resource_seconds"].values())
