# SPDX-License-Identifier: Apache-2.0

"""Real-NCCL MCore torch_dist checkpoint test for GPU-staged AdamW."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

from areal.engine.megatron_utils import checkpoint_snapshot as snapshot_module
from areal.engine.megatron_utils.checkpoint_snapshot import (
    discover_orphaned_snapshot_directories,
    validate_shared_snapshot_capacity,
)
from areal.engine.megatron_utils.checkpointer import load_dist_checkpointing
from areal.engine.megatron_utils.gpu_staged_optimizer import (
    GPUStagedAdamW,
    GPUStagedAdamWConfig,
    bind_gpu_staged_adamw,
)
from areal.engine.megatron_utils.gpu_staged_optimizer_checkpoint import (
    abort_managed_checkpoint_load,
    apply_begin_managed_checkpoint_load,
    attach_managed_optimizer_identities,
    build_managed_optimizer_identities,
    build_managed_optimizer_outer_template,
    build_managed_optimizer_tensor_manifest,
    create_managed_checkpoint_load_transaction,
    decide_managed_checkpoint_commit,
    merge_managed_optimizer_tensor_manifests,
    poison_managed_checkpoint_transaction,
    preflight_managed_checkpoint_snapshots,
    prepare_managed_checkpoint_commit,
    prepare_managed_checkpoint_load,
    prepare_managed_checkpoint_recovery,
    prepare_managed_checkpoint_save,
    retry_managed_checkpoint_cleanup,
    validate_managed_optimizer_outer_state,
    validate_managed_optimizer_source_tensor_metadata,
    vote_managed_checkpoint_phase,
)


class _ModelChunk:
    def __init__(self, ddp_config, model_param: torch.nn.Parameter):
        self.ddp_config = ddp_config
        self.model_param = model_param
        self.owned_shard: torch.Tensor | None = None

    def start_param_sync(self) -> None:
        assert self.owned_shard is not None
        dist.all_gather_into_tensor(self.model_param.data, self.owned_shard)


def _build_optimizer(numel: int):
    world_size = dist.get_world_size()
    torch.manual_seed(20260815)
    initial = torch.randn(numel, device="cuda", dtype=torch.bfloat16)
    dist.broadcast(initial, src=0)
    model_param = torch.nn.Parameter(initial.clone())
    kwargs = {
        "lr": 3e-3,
        "betas": (0.8, 0.95),
        "eps": 1e-6,
        "weight_decay": 0.07,
    }
    inner = GPUStagedAdamW(
        [
            {
                "params": [model_param],
                "lr_mult": 1.0,
                "wd_mult": 1.0,
                "is_expert_parallel": False,
                "is_decoupled_lr": False,
            }
        ],
        staged_config=GPUStagedAdamWConfig(
            buffer_count=2, bucket_size_mb=7 * 4 / (1024 * 1024)
        ),
        **kwargs,
    )
    ddp_config = SimpleNamespace(
        use_megatron_fsdp=False,
        overlap_param_gather=False,
        use_distributed_optimizer=True,
        num_distributed_optimizer_instances=1,
        reduce_scatter_with_fp32_accumulation=False,
    )
    bucket = SimpleNamespace(
        grad_data=torch.empty_like(model_param),
        param_data=model_param.detach(),
        offset=0,
        numel_unpadded=numel,
        params_list=[model_param],
    )
    buffer = SimpleNamespace(
        param_dtype=torch.bfloat16,
        grad_dtype=torch.bfloat16,
        buckets=[bucket],
        param_index_map={model_param: (0, numel, 0)},
        data_parallel_group=dist.group.WORLD,
        data_parallel_world_size=world_size,
        ddp_config=ddp_config,
        params=[model_param],
        numel_unpadded=numel,
    )
    model_chunk = _ModelChunk(ddp_config, model_param)
    config = OptimizerConfig(
        optimizer="adam",
        lr=kwargs["lr"],
        min_lr=0.0,
        weight_decay=kwargs["weight_decay"],
        adam_beta1=kwargs["betas"][0],
        adam_beta2=kwargs["betas"][1],
        adam_eps=kwargs["eps"],
        bf16=True,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        main_grads_dtype=torch.bfloat16,
        main_params_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        clip_grad=0.0,
    )
    optimizer = DistributedOptimizer(
        inner,
        config,
        grad_scaler=None,
        init_state_fn=None,
        model_chunks=[model_chunk],
        per_model_buffers={0: [buffer]},
        data_parallel_group=dist.group.WORLD,
        data_parallel_group_gloo=None,
        data_parallel_group_idx=0,
        distributed_optimizer_instance_id=0,
    )
    assert bind_gpu_staged_adamw(optimizer) == 1
    owned_shard = inner.param_groups[0]["params"][0]
    model_chunk.owned_shard = owned_shard
    scheduler = torch.optim.lr_scheduler.StepLR(inner, step_size=1, gamma=0.9)
    return (
        initial,
        model_param,
        optimizer,
        inner,
        owned_shard,
        model_chunk,
        scheduler,
        kwargs,
    )


def _step(model_param, optimizer, scheduler, step: int, numel: int) -> None:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_numel = numel // world_size
    full_grad = torch.linspace(-1, 1, numel, device="cuda", dtype=torch.float32)
    full_grad.add_(step * 0.125)
    model_param.main_grad = torch.zeros_like(model_param)
    start = rank * local_numel
    model_param.main_grad[start : start + local_numel].copy_(
        full_grad[start : start + local_numel]
    )
    success, _, _ = optimizer.step()
    assert success
    scheduler.step()


def _baseline(initial, steps: int, kwargs):
    param = torch.nn.Parameter(initial.float().clone())
    optimizer = torch.optim.AdamW([param], **kwargs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    for step in range(steps):
        param.grad = (
            torch.linspace(-1, 1, param.numel(), device="cuda")
            .add_(step * 0.125)
            .bfloat16()
            .float()
        )
        optimizer.step()
        scheduler.step()
    return param, optimizer.state[param], scheduler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("save", "load"), required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--numel", type=int, default=96)
    parser.add_argument("--inject-prepare-rank", type=int, default=-1)
    parser.add_argument("--inject-local-validate-rank", type=int, default=-1)
    parser.add_argument("--inject-preflight-rank", type=int, default=-1)
    parser.add_argument("--inject-abort-rank", type=int, default=-1)
    parser.add_argument("--inject-pending-action-rank", type=int, default=-1)
    parser.add_argument("--inject-cleanup-rank", type=int, default=-1)
    parser.add_argument("--inject-transition-rank", type=int, default=-1)
    parser.add_argument("--inject-snapshot-preflight-rank", type=int, default=-1)
    parser.add_argument("--inject-snapshot-write-rank", type=int, default=-1)
    parser.add_argument("--inject-snapshot-rename-rank", type=int, default=-1)
    parser.add_argument("--inject-snapshot-read-rank", type=int, default=-1)
    parser.add_argument("--inject-restore-fd-preclose-rank", type=int, default=-1)
    parser.add_argument("--inject-snapshot-fd-close-rank", type=int, default=-1)
    parser.add_argument(
        "--inject-snapshot-fd-preclose-reuse-rank", type=int, default=-1
    )
    parser.add_argument("--inject-shared-capacity", action="store_true")
    parser.add_argument("--inject-filesystem-identity-conflict", action="store_true")
    parser.add_argument(
        "--inject-snapshot-directory-replacement-rank", type=int, default=-1
    )
    parser.add_argument("--cleanup-request-mismatch", action="store_true")
    parser.add_argument("--check-template-no-mutation", action="store_true")
    parser.add_argument("--expect-source-metadata-error", action="store_true")
    parser.add_argument("--recover-after-abort", action="store_true")
    parser.add_argument("--continuation-steps", type=int, default=1)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    checkpoint_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=60))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    parallel_state.initialize_model_parallel()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert args.numel % world_size == 0
    (
        initial,
        model_param,
        optimizer,
        inner,
        owned_shard,
        model_chunk,
        scheduler,
        kwargs,
    ) = _build_optimizer(args.numel)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.check_template_no_mutation:
        slabs = inner.cpu_slabs
        before_versions = tuple(
            slab._version for slab in (slabs.master, slabs.exp_avg, slabs.exp_avg_sq)
        )
        before_pointers = tuple(
            slab.untyped_storage().data_ptr()
            for slab in (slabs.master, slabs.exp_avg, slabs.exp_avg_sq)
        )
        before_values = tuple(
            slab.clone() for slab in (slabs.master, slabs.exp_avg, slabs.exp_avg_sq)
        )
        before_groups = tuple(
            {key: value for key, value in group.items() if key != "params"}
            for group in inner.param_groups
        )
        before_scheduler = scheduler.state_dict()
        before_rng = torch.get_rng_state().clone()
        identities = build_managed_optimizer_identities(
            optimizer, {model_param: "model.parameter"}
        )
        template = build_managed_optimizer_outer_template(optimizer, identities)
        assert template["optimizer"].data is None
        after_versions = tuple(
            slab._version for slab in (slabs.master, slabs.exp_avg, slabs.exp_avg_sq)
        )
        for actual, expected in zip(
            (slabs.master, slabs.exp_avg, slabs.exp_avg_sq), before_values
        ):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        assert after_versions == before_versions, (
            "outer preflight template construction mutated CPU slabs: "
            f"before={before_versions}, after={after_versions}"
        )
        assert before_pointers == tuple(
            slab.untyped_storage().data_ptr()
            for slab in (slabs.master, slabs.exp_avg, slabs.exp_avg_sq)
        )
        assert before_groups == tuple(
            {key: value for key, value in group.items() if key != "params"}
            for group in inner.param_groups
        )
        assert before_scheduler == scheduler.state_dict()
        torch.testing.assert_close(
            torch.get_rng_state(), before_rng, rtol=0.0, atol=0.0
        )
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
        return

    if args.mode == "save":
        for step in range(3):
            _step(model_param, optimizer, scheduler, step, args.numel)
        inner.drain()
        prepare_managed_checkpoint_save(optimizer, async_save=False)
        state = optimizer.sharded_state_dict(
            {},
            is_loading=False,
            metadata={"distrib_optim_sharding_type": "dp_reshardable"},
        )
        identities = build_managed_optimizer_identities(
            optimizer, {model_param: "model.parameter"}
        )
        attach_managed_optimizer_identities(optimizer, state, identities)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        dist_checkpointing.save(
            sharded_state_dict={
                "optimizer": state,
                "lr_scheduler": scheduler.state_dict(),
            },
            checkpoint_dir=str(checkpoint_dir),
        )
        expected_param, expected_state, expected_scheduler = _baseline(
            initial, 3, kwargs
        )
    else:
        slabs = inner.cpu_slabs
        slab_ptrs = tuple(
            slab.untyped_storage().data_ptr()
            for slab in (
                slabs.master,
                slabs.exp_avg,
                slabs.exp_avg_sq,
            )
        )
        preflight_versions = tuple(
            slab._version for slab in (slabs.master, slabs.exp_avg, slabs.exp_avg_sq)
        )
        preflight_values = tuple(
            slab.clone() for slab in (slabs.master, slabs.exp_avg, slabs.exp_avg_sq)
        )
        preflight_groups = tuple(
            {key: value for key, value in group.items() if key != "params"}
            for group in inner.param_groups
        )
        preflight_scheduler = scheduler.state_dict()
        preflight_rng = torch.get_rng_state().clone()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        preflight_error = (
            RuntimeError("injected single-rank preflight failure")
            if rank == args.inject_preflight_rank
            else None
        )
        preflight_phase_error = vote_managed_checkpoint_phase(
            checkpoint_group, "preflight", preflight_error
        )
        if preflight_phase_error is not None:
            gathered_results = [None] * world_size
            dist.all_gather_object(
                gathered_results,
                {
                    "rank": rank,
                    "success": False,
                    "error": repr(preflight_phase_error),
                    "committed": False,
                    "lifecycle": inner.checkpoint_lifecycle,
                    "rollback_directories": 0,
                },
            )
            assert len({result["success"] for result in gathered_results}) == 1
            probe = torch.ones(1, device="cuda")
            dist.all_reduce(probe)
            assert probe.item() == world_size
            (output_dir / f"failure_dp{world_size}_rank{rank}.json").write_text(
                json.dumps(gathered_results, indent=2) + "\n"
            )
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
            return

        identities = None
        outer_template = None
        local_template_error = None
        try:
            identities = build_managed_optimizer_identities(
                optimizer, {model_param: "model.parameter"}
            )
            outer_template = build_managed_optimizer_outer_template(
                optimizer, identities
            )
        except BaseException as error:
            local_template_error = error
        preflight_phase_error = vote_managed_checkpoint_phase(
            checkpoint_group, "preflight_template", local_template_error
        )
        if preflight_phase_error is None:
            local_preflight_error = None
            outer_loaded = None
            try:
                outer_loaded = load_dist_checkpointing(
                    {"optimizer": outer_template},
                    str(checkpoint_dir),
                    strict_managed=True,
                    allowed_unrequested_prefixes=("lr_scheduler",),
                    allow_managed_optimizer_tensor_state=True,
                )
            except BaseException as error:
                local_preflight_error = error
            preflight_phase_error = vote_managed_checkpoint_phase(
                checkpoint_group, "preflight_load", local_preflight_error
            )
        if preflight_phase_error is None:
            local_preflight_error = None
            try:
                validate_managed_optimizer_outer_state(
                    optimizer, outer_loaded["optimizer"], identities
                )
            except BaseException as error:
                local_preflight_error = error
            preflight_phase_error = vote_managed_checkpoint_phase(
                checkpoint_group, "preflight_validate", local_preflight_error
            )
        if preflight_phase_error is not None:
            raise preflight_phase_error
        tensor_template = optimizer.sharded_state_dict(
            {},
            is_loading=False,
            metadata={"distrib_optim_sharding_type": "dp_reshardable"},
        )
        attach_managed_optimizer_identities(optimizer, tensor_template, identities)
        local_manifest = build_managed_optimizer_tensor_manifest(tensor_template)
        manifests = [None] * world_size
        dist.all_gather_object(manifests, local_manifest, group=checkpoint_group)
        tensor_manifest = merge_managed_optimizer_tensor_manifests(manifests)
        source_metadata_error = None
        try:
            validate_managed_optimizer_source_tensor_metadata(
                str(checkpoint_dir), tensor_manifest
            )
        except BaseException as error:
            source_metadata_error = error
        source_phase_error = vote_managed_checkpoint_phase(
            checkpoint_group, "source_metadata_validate", source_metadata_error
        )
        if args.expect_source_metadata_error:
            assert source_phase_error is not None
            assert preflight_versions == tuple(
                slab._version
                for slab in (slabs.master, slabs.exp_avg, slabs.exp_avg_sq)
            )
            for actual, expected in zip(
                (slabs.master, slabs.exp_avg, slabs.exp_avg_sq), preflight_values
            ):
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            result = {
                "rank": rank,
                "success": False,
                "error": repr(source_phase_error),
                "lifecycle": inner.checkpoint_lifecycle,
                "versions_unchanged": True,
            }
            gathered_results = [None] * world_size
            dist.all_gather_object(gathered_results, result)
            (output_dir / f"metadata_dp{world_size}_rank{rank}.json").write_text(
                json.dumps(gathered_results, indent=2) + "\n"
            )
            probe = torch.ones(1, device="cuda")
            dist.all_reduce(probe)
            assert probe.item() == world_size
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
            return
        if source_phase_error is not None:
            raise source_phase_error
        assert preflight_versions == tuple(
            slab._version for slab in (slabs.master, slabs.exp_avg, slabs.exp_avg_sq)
        )
        assert slab_ptrs == tuple(
            slab.untyped_storage().data_ptr()
            for slab in (slabs.master, slabs.exp_avg, slabs.exp_avg_sq)
        )
        for actual, expected in zip(
            (slabs.master, slabs.exp_avg, slabs.exp_avg_sq), preflight_values
        ):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        assert preflight_groups == tuple(
            {key: value for key, value in group.items() if key != "params"}
            for group in inner.param_groups
        )
        assert preflight_scheduler == scheduler.state_dict()
        torch.testing.assert_close(
            torch.get_rng_state(), preflight_rng, rtol=0.0, atol=0.0
        )

        rollback_root = output_dir / f"rollback-rank-{rank}"
        rollback_root.mkdir(mode=0o700, exist_ok=True)
        inner.configure_checkpoint_snapshot(
            parent=str(rollback_root),
            leaf_identity=next(iter(identities.values())),
        )
        managed = create_managed_checkpoint_load_transaction(optimizer)
        snapshot_error: BaseException | None = None
        try:
            if rank == args.inject_snapshot_preflight_rank:
                raise OSError("injected rollback snapshot capacity failure")
            capacity_reports = preflight_managed_checkpoint_snapshots(managed)
        except BaseException as error:
            snapshot_error = error
        snapshot_phase_error = vote_managed_checkpoint_phase(
            checkpoint_group, "snapshot_preflight", snapshot_error
        )
        if snapshot_phase_error is None:
            if args.inject_shared_capacity:
                local_required = sum(
                    report.required_bytes for report in capacity_reports
                )
                capacity_reports = tuple(
                    replace(
                        report,
                        free_bytes=local_required + 64 * 1024 * 1024 + 1,
                    )
                    for report in capacity_reports
                )
            if args.inject_filesystem_identity_conflict:
                capacity_reports = tuple(
                    replace(
                        report,
                        filesystem_id=7 + rank,
                        root_path="/injected-shared-root",
                        root_device=1,
                        root_inode=10,
                    )
                    for report in capacity_reports
                )
            try:
                validate_shared_snapshot_capacity(capacity_reports, checkpoint_group)
            except BaseException as error:
                snapshot_error = error
            snapshot_phase_error = vote_managed_checkpoint_phase(
                checkpoint_group, "snapshot_capacity", snapshot_error
            )
        if snapshot_phase_error is None:
            original_write = snapshot_module._write_all
            original_replace = snapshot_module.os.replace
            replace_count = 0

            def inject_snapshot_write(fd: int, payload) -> None:
                fd_path = os.readlink(f"/proc/self/fd/{fd}")
                if rank == args.inject_snapshot_write_rank and fd_path.endswith(
                    ".master.data.partial"
                ):
                    os.write(fd, memoryview(payload)[:4])
                    raise OSError("injected rollback snapshot partial write")
                original_write(fd, payload)

            def inject_snapshot_rename(src, dst, *rename_args, **kwargs) -> None:
                nonlocal replace_count
                replace_count += 1
                if rank == args.inject_snapshot_rename_rank and replace_count == 2:
                    raise OSError("injected rollback snapshot partial rename failure")
                original_replace(src, dst, *rename_args, **kwargs)

            snapshot_module._write_all = inject_snapshot_write
            snapshot_module.os.replace = inject_snapshot_rename
            try:
                apply_begin_managed_checkpoint_load(managed)
            except BaseException as error:
                snapshot_error = error
            finally:
                snapshot_module._write_all = original_write
                snapshot_module.os.replace = original_replace
            snapshot_phase_error = vote_managed_checkpoint_phase(
                checkpoint_group, "snapshot_materialize", snapshot_error
            )
        if snapshot_phase_error is not None:
            rollback_error = None
            try:
                abort_managed_checkpoint_load(
                    managed, snapshot_phase_error, poison=False
                )
            except BaseException as error:
                rollback_error = error
            abort_error = vote_managed_checkpoint_phase(
                checkpoint_group, "snapshot_abort", rollback_error
            )
            if abort_error is not None:
                poison_managed_checkpoint_transaction(managed, abort_error)
            local_result = {
                "rank": rank,
                "success": False,
                "error": repr(snapshot_phase_error),
                "committed": False,
                "lifecycle": inner.checkpoint_lifecycle,
                "rollback_directories": len(
                    discover_orphaned_snapshot_directories(rollback_root)
                ),
            }
            gathered_results = [None] * world_size
            dist.all_gather_object(gathered_results, local_result)
            (output_dir / f"failure_dp{world_size}_rank{rank}.json").write_text(
                json.dumps(gathered_results, indent=2) + "\n"
            )
            probe = torch.ones(1, device="cuda")
            dist.all_reduce(probe)
            assert probe.item() == world_size
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
            return
        pending_action_round = 1
        pending_action_calls: list[str] = []
        pending_action_names: list[str] = []
        request_mismatch_seen = False
        committed_state_preserved = False
        post_cleanup_step = False
        post_commit_step = False
        load_request_mismatch_seen = False
        save_request_mismatch_seen = False
        replacement_fd = -1
        replacement_source_fd = -1
        restore_preclose_retained = False
        restore_preclose_finalized = False
        restore_snapshot = None
        local_load_error: BaseException | None = None
        try:
            loaded = dist_checkpointing.load(
                {
                    "optimizer": tensor_template,
                    "lr_scheduler": scheduler.state_dict(),
                },
                str(checkpoint_dir),
            )
            optimizer.load_state_dict(loaded["optimizer"])
            scheduler.load_state_dict(loaded["lr_scheduler"])
        except BaseException as error:
            local_load_error = error
        phase_error = vote_managed_checkpoint_phase(
            checkpoint_group, "dcp_apply", local_load_error
        )
        if phase_error is None:
            local_load_error = None
            original_prepare = inner.prepare_checkpoint_load
            try:
                if rank == args.inject_local_validate_rank:
                    inner.prepare_checkpoint_load = lambda: (_ for _ in ()).throw(
                        RuntimeError("injected single-rank local-validate failure")
                    )
                prepare_managed_checkpoint_load(managed)
            except BaseException as error:
                local_load_error = error
            finally:
                inner.prepare_checkpoint_load = original_prepare
            phase_error = vote_managed_checkpoint_phase(
                checkpoint_group, "local_validate", local_load_error
            )
        if phase_error is None:
            local_load_error = None
            original_prepare_commit = inner.prepare_checkpoint_commit
            try:
                if rank == args.inject_prepare_rank:
                    inner.prepare_checkpoint_commit = lambda: (_ for _ in ()).throw(
                        RuntimeError("injected single-rank prepare-commit failure")
                    )
                prepare_managed_checkpoint_commit(managed)
            except BaseException as error:
                local_load_error = error
            finally:
                inner.prepare_checkpoint_commit = original_prepare_commit
            phase_error = vote_managed_checkpoint_phase(
                checkpoint_group, "prepare_commit", local_load_error
            )
        if phase_error is not None:
            rollback_error = None
            corrupted_snapshot_path = None
            corrupted_snapshot_payload = None
            if rank == args.inject_snapshot_read_rank:
                snapshot_action = next(
                    action
                    for action in inner._checkpoint_rollback.actions
                    if action.name == "slab.exp_avg"
                )
                corrupted_snapshot_path = snapshot_action.snapshot.data_path
                corrupted_snapshot_payload = corrupted_snapshot_path.read_bytes()
                corrupted_snapshot_path.write_bytes(
                    corrupted_snapshot_payload[
                        : max(0, len(corrupted_snapshot_payload) // 2)
                    ]
                )
            original_abort = inner.abort_checkpoint_load
            original_fd_prepare = snapshot_module._prepare_fd_close
            restore_preclose_injected = False
            if rank == args.inject_restore_fd_preclose_rank:
                restore_snapshot = next(
                    action.snapshot
                    for action in inner._checkpoint_rollback.actions
                    if action.name == "slab.master"
                )

                def fail_restore_fd_preclose_once(owner) -> None:
                    nonlocal restore_preclose_injected
                    fd_path = os.readlink(f"/proc/self/fd/{owner.fd}")
                    if (
                        fd_path.endswith("master.data")
                        and not restore_preclose_injected
                    ):
                        restore_preclose_injected = True
                        raise OSError("injected restore FD pre-close failure")
                    original_fd_prepare(owner)

                snapshot_module._prepare_fd_close = fail_restore_fd_preclose_once
            if rank == args.inject_pending_action_rank:
                for action in inner._checkpoint_rollback.actions:
                    if not action.name.startswith("slab."):
                        continue
                    original_restore = action.restore

                    def tracked_restore(
                        target,
                        snapshot,
                        *,
                        action_name=action.name,
                        restore=original_restore,
                    ):
                        pending_action_calls.append(
                            f"{pending_action_round}:{action_name}"
                        )
                        if pending_action_round == 1 and action_name == "slab.exp_avg":
                            raise RuntimeError("injected pending exp_avg rollback")
                        if pending_action_round > 1 and action_name != "slab.exp_avg":
                            raise RuntimeError(
                                f"replayed completed action {action_name}"
                            )
                        restore(target, snapshot)

                    action.restore = tracked_restore
            if rank == args.inject_abort_rank:
                inner.abort_checkpoint_load = lambda *args, **kwargs: (
                    _ for _ in ()
                ).throw(RuntimeError("injected single-rank abort failure"))
            try:
                abort_managed_checkpoint_load(managed, phase_error, poison=False)
            except BaseException as error:
                rollback_error = error
            finally:
                inner.abort_checkpoint_load = original_abort
                snapshot_module._prepare_fd_close = original_fd_prepare
            if restore_snapshot is not None:
                restore_preclose_retained = (
                    restore_snapshot._restore_fd_owner is not None
                    and not restore_snapshot.restore_complete
                )
            if rank == args.inject_pending_action_rank:
                pending_action_names = [
                    action.name
                    for action in inner._checkpoint_rollback.actions
                    if action.pending
                ]
            if managed.poisoned and rollback_error is None:
                rollback_error = RuntimeError(
                    "rollback did not restore every local optimizer leaf"
                )
            abort_error = vote_managed_checkpoint_phase(
                checkpoint_group, "abort", rollback_error
            )
            if abort_error is not None:
                poison_managed_checkpoint_transaction(managed, abort_error)
                phase_error.add_note(str(abort_error))
            local_load_error = phase_error
            if args.recover_after_abort:
                pending_action_round = 2
                if corrupted_snapshot_path is not None:
                    corrupted_snapshot_path.write_bytes(corrupted_snapshot_payload)
                recovery_error = None
                try:
                    prepare_managed_checkpoint_recovery(managed)
                except BaseException as error:
                    recovery_error = error
                if restore_snapshot is not None:
                    restore_preclose_finalized = (
                        restore_snapshot._restore_fd_owner is None
                        and restore_snapshot.restore_complete
                    )
                recovery_phase_error = vote_managed_checkpoint_phase(
                    checkpoint_group, "recovery_rollback", recovery_error
                )
                if recovery_phase_error is None:
                    assert inner.checkpoint_lifecycle == "RELOAD_REQUIRED"
                    recovery = create_managed_checkpoint_load_transaction(optimizer)
                    try:
                        recovery_capacity = preflight_managed_checkpoint_snapshots(
                            recovery
                        )
                    except BaseException as error:
                        recovery_error = error
                    recovery_phase_error = vote_managed_checkpoint_phase(
                        checkpoint_group,
                        "recovery_snapshot_preflight",
                        recovery_error,
                    )
                if recovery_phase_error is None:
                    recovery_error = None
                    try:
                        validate_shared_snapshot_capacity(
                            recovery_capacity, checkpoint_group
                        )
                    except BaseException as error:
                        recovery_error = error
                    recovery_phase_error = vote_managed_checkpoint_phase(
                        checkpoint_group,
                        "recovery_snapshot_capacity",
                        recovery_error,
                    )
                if recovery_phase_error is None:
                    recovery_error = None
                    try:
                        apply_begin_managed_checkpoint_load(recovery)
                    except BaseException as error:
                        recovery_error = error
                    recovery_phase_error = vote_managed_checkpoint_phase(
                        checkpoint_group,
                        "recovery_snapshot_materialize",
                        recovery_error,
                    )
                if recovery_phase_error is not None:
                    recovery_abort_error = None
                    if "recovery" in locals():
                        try:
                            abort_managed_checkpoint_load(
                                recovery, recovery_phase_error, poison=True
                            )
                        except BaseException as error:
                            recovery_abort_error = error
                    voted_abort_error = vote_managed_checkpoint_phase(
                        checkpoint_group,
                        "recovery_snapshot_abort",
                        recovery_abort_error,
                    )
                    if voted_abort_error is not None and "recovery" in locals():
                        poison_managed_checkpoint_transaction(
                            recovery, voted_abort_error
                        )
                    local_load_error = recovery_phase_error
                else:
                    try:
                        loaded = dist_checkpointing.load(
                            {
                                "optimizer": tensor_template,
                                "lr_scheduler": scheduler.state_dict(),
                            },
                            str(checkpoint_dir),
                        )
                        optimizer.load_state_dict(loaded["optimizer"])
                        scheduler.load_state_dict(loaded["lr_scheduler"])
                        prepare_managed_checkpoint_load(recovery)
                        prepare_managed_checkpoint_commit(recovery)
                    except BaseException as error:
                        recovery_error = error
                    recovery_phase_error = vote_managed_checkpoint_phase(
                        checkpoint_group, "recovery_prepare_commit", recovery_error
                    )
                    if recovery_phase_error is None:
                        decide_managed_checkpoint_commit(recovery)
                        retry_managed_checkpoint_cleanup(recovery)
                        assert inner.checkpoint_lifecycle == "CLEAN"
                        _step(model_param, optimizer, scheduler, 3, args.numel)
                        inner.drain()
                        local_load_error = None
                    else:
                        abort_managed_checkpoint_load(
                            recovery, recovery_phase_error, poison=True
                        )
                        local_load_error = recovery_phase_error
        else:
            decide_managed_checkpoint_commit(managed)
            committed_slabs = tuple(
                slab.clone()
                for slab in (
                    inner.cpu_slabs.master,
                    inner.cpu_slabs.exp_avg,
                    inner.cpu_slabs.exp_avg_sq,
                )
            )
            cleanup_error = None
            original_discard = inner.discard_checkpoint_snapshot
            original_decide = inner.decide_checkpoint_commit
            original_close_fd = snapshot_module._close_fd
            original_prepare_fd_close = snapshot_module._prepare_fd_close
            replaced_cleanup = None
            moved_cleanup_directory = None
            fd_close_target = None
            fd_close_failed = False
            replacement_fd = -1
            replacement_source_fd = -1
            if rank == args.inject_cleanup_rank:
                inner.discard_checkpoint_snapshot = lambda: (_ for _ in ()).throw(
                    RuntimeError("injected single-rank cleanup failure")
                )
            if rank == args.inject_transition_rank:
                inner.decide_checkpoint_commit = lambda: (_ for _ in ()).throw(
                    RuntimeError("injected single-rank leaf transition failure")
                )
            if rank == args.inject_snapshot_directory_replacement_rank:
                replaced_cleanup = next(
                    reference
                    for reference in inner._checkpoint_prepared_cleanup.references
                    if isinstance(reference, snapshot_module.DiskSnapshotCleanup)
                )
                moved_cleanup_directory = (
                    replaced_cleanup.parent
                    / f"{replaced_cleanup.directory.name}-moved-by-test"
                )
                replaced_cleanup.directory.rename(moved_cleanup_directory)
                replaced_cleanup.directory.mkdir(mode=0o700)
                shutil.copy2(
                    moved_cleanup_directory / "owner.json",
                    replaced_cleanup.directory / "owner.json",
                )
            if rank == args.inject_snapshot_fd_close_rank:
                fd_cleanup = next(
                    reference
                    for reference in inner._checkpoint_prepared_cleanup.references
                    if isinstance(reference, snapshot_module.DiskSnapshotCleanup)
                )
                fd_close_target = fd_cleanup._directory_fd
                replacement_source_fd = os.open("/dev/null", os.O_RDONLY)

                def fail_snapshot_fd_close_once(fd: int) -> None:
                    nonlocal fd_close_failed, replacement_fd, replacement_source_fd
                    if fd == fd_close_target and not fd_close_failed:
                        fd_close_failed = True
                        original_close_fd(fd)
                        os.dup2(replacement_source_fd, fd)
                        original_close_fd(replacement_source_fd)
                        replacement_source_fd = -1
                        replacement_fd = fd
                        raise OSError("injected snapshot directory FD close failure")
                    original_close_fd(fd)

                snapshot_module._close_fd = fail_snapshot_fd_close_once
            if rank == args.inject_snapshot_fd_preclose_reuse_rank:
                fd_cleanup = next(
                    reference
                    for reference in inner._checkpoint_prepared_cleanup.references
                    if isinstance(reference, snapshot_module.DiskSnapshotCleanup)
                )
                fd_close_target = fd_cleanup._directory_fd
                replacement_source_fd = os.open("/dev/null", os.O_RDONLY)

                def fail_snapshot_fd_prepare_once(owner) -> None:
                    nonlocal fd_close_failed, replacement_fd, replacement_source_fd
                    if owner.fd == fd_close_target and not fd_close_failed:
                        fd_close_failed = True
                        original_close_fd(owner.fd)
                        os.dup2(replacement_source_fd, owner.fd)
                        original_close_fd(replacement_source_fd)
                        replacement_source_fd = -1
                        replacement_fd = fd_close_target
                        raise OSError("injected snapshot pre-close FD reuse failure")
                    original_prepare_fd_close(owner)

                snapshot_module._prepare_fd_close = fail_snapshot_fd_prepare_once
            try:
                retry_managed_checkpoint_cleanup(managed)
            except BaseException as error:
                cleanup_error = error
            finally:
                inner.discard_checkpoint_snapshot = original_discard
                inner.decide_checkpoint_commit = original_decide
                snapshot_module._close_fd = original_close_fd
                snapshot_module._prepare_fd_close = original_prepare_fd_close
            local_load_error = vote_managed_checkpoint_phase(
                checkpoint_group, "cleanup", cleanup_error
            )
            if local_load_error is not None and args.inject_transition_rank >= 0:
                before_step = inner.cpu_slabs.master.clone()
                _step(model_param, optimizer, scheduler, 3, args.numel)
                inner.drain()
                post_commit_step = not torch.equal(inner.cpu_slabs.master, before_step)
                retry_error = None
                try:
                    retry_managed_checkpoint_cleanup(managed)
                except BaseException as error:
                    retry_error = error
                retry_phase_error = vote_managed_checkpoint_phase(
                    checkpoint_group, "transition_cleanup_retry", retry_error
                )
                if retry_phase_error is None:
                    committed_after_step = tuple(
                        slab.clone()
                        for slab in (
                            inner.cpu_slabs.master,
                            inner.cpu_slabs.exp_avg,
                            inner.cpu_slabs.exp_avg_sq,
                        )
                    )
                    for phase in (
                        "load_request_after_cleanup",
                        "save_request_after_cleanup",
                    ):
                        mismatch_error = vote_managed_checkpoint_phase(
                            checkpoint_group,
                            phase,
                            None,
                            details={"path": f"rank-{rank}"},
                            require_consistent_details=True,
                        )
                        assert mismatch_error is not None
                        if phase.startswith("load"):
                            load_request_mismatch_seen = True
                        else:
                            save_request_mismatch_seen = True
                    committed_state_preserved = all(
                        torch.equal(actual, expected)
                        for actual, expected in zip(
                            (
                                inner.cpu_slabs.master,
                                inner.cpu_slabs.exp_avg,
                                inner.cpu_slabs.exp_avg_sq,
                            ),
                            committed_after_step,
                        )
                    )
                    local_load_error = None
                else:
                    local_load_error = retry_phase_error
            elif local_load_error is not None and (
                args.inject_cleanup_rank >= 0
                or args.inject_snapshot_fd_close_rank >= 0
                or args.inject_snapshot_fd_preclose_reuse_rank >= 0
                or args.inject_snapshot_directory_replacement_rank >= 0
            ):
                if replaced_cleanup is not None:
                    shutil.rmtree(replaced_cleanup.directory)
                    assert moved_cleanup_directory is not None
                    moved_cleanup_directory.rename(replaced_cleanup.directory)
                retry_error = None
                try:
                    retry_managed_checkpoint_cleanup(managed)
                except BaseException as error:
                    retry_error = error
                retry_phase_error = vote_managed_checkpoint_phase(
                    checkpoint_group, "cleanup_retry", retry_error
                )
                if retry_phase_error is None:
                    if args.cleanup_request_mismatch:
                        request_error = vote_managed_checkpoint_phase(
                            checkpoint_group,
                            "request_after_cleanup",
                            None,
                            details={"path": f"rank-{rank}"},
                            require_consistent_details=True,
                        )
                        assert request_error is not None
                        request_mismatch_seen = True
                        committed_state_preserved = all(
                            torch.equal(actual, expected)
                            for actual, expected in zip(
                                (
                                    inner.cpu_slabs.master,
                                    inner.cpu_slabs.exp_avg,
                                    inner.cpu_slabs.exp_avg_sq,
                                ),
                                committed_slabs,
                            )
                        )
                        assert inner._checkpoint_rollback is None
                        assert inner._checkpoint_cleanup is None
                        assert inner.checkpoint_lifecycle == "CLEAN"
                        _step(model_param, optimizer, scheduler, 3, args.numel)
                        inner.drain()
                        post_cleanup_step = True
                    local_load_error = None
        if any(
            rank_value >= 0
            for rank_value in (
                args.inject_prepare_rank,
                args.inject_local_validate_rank,
                args.inject_abort_rank,
                args.inject_pending_action_rank,
                args.inject_restore_fd_preclose_rank,
                args.inject_cleanup_rank,
                args.inject_transition_rank,
                args.inject_snapshot_fd_close_rank,
                args.inject_snapshot_fd_preclose_reuse_rank,
                args.inject_snapshot_directory_replacement_rank,
            )
        ):
            if managed.committed and not (post_cleanup_step or post_commit_step):
                committed_state_preserved = all(
                    torch.equal(actual, expected)
                    for actual, expected in zip(
                        (
                            inner.cpu_slabs.master,
                            inner.cpu_slabs.exp_avg,
                            inner.cpu_slabs.exp_avg_sq,
                        ),
                        committed_slabs,
                    )
                )
            replacement_fd_alive = False
            if replacement_fd >= 0:
                os.fstat(replacement_fd)
                replacement_fd_alive = True
            local_result = {
                "rank": rank,
                "success": local_load_error is None,
                "error": None if local_load_error is None else repr(local_load_error),
                "committed": managed.committed,
                "lifecycle": inner.checkpoint_lifecycle,
                "pending_action_calls": pending_action_calls,
                "pending_action_names": pending_action_names,
                "post_recovery_step": (
                    args.recover_after_abort and local_load_error is None
                ),
                "request_mismatch_seen": request_mismatch_seen,
                "committed_state_preserved": committed_state_preserved,
                "post_cleanup_step": post_cleanup_step,
                "post_commit_step": post_commit_step,
                "load_request_mismatch_seen": load_request_mismatch_seen,
                "save_request_mismatch_seen": save_request_mismatch_seen,
                "rollback_directories": len(
                    discover_orphaned_snapshot_directories(rollback_root)
                ),
                "replacement_fd_alive": replacement_fd_alive,
                "restore_preclose_retained": restore_preclose_retained,
                "restore_preclose_finalized": restore_preclose_finalized,
            }
            gathered_results = [None] * world_size
            dist.all_gather_object(gathered_results, local_result)
            (output_dir / f"failure_dp{world_size}_rank{rank}.json").write_text(
                json.dumps(gathered_results, indent=2) + "\n"
            )
            assert len({result["success"] for result in gathered_results}) == 1, (
                f"checkpoint outcome diverged across ranks: {gathered_results}"
            )
            probe = torch.ones(1, device="cuda")
            dist.all_reduce(probe)
            assert probe.item() == world_size
            if replacement_fd >= 0:
                original_close_fd(replacement_fd)
            if replacement_source_fd >= 0:
                original_close_fd(replacement_source_fd)
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
            return
        if local_load_error is not None:
            raise local_load_error
        allocated_peak = torch.cuda.max_memory_allocated()
        assert slab_ptrs == tuple(
            slab.untyped_storage().data_ptr()
            for slab in (
                inner.cpu_slabs.master,
                inner.cpu_slabs.exp_avg,
                inner.cpu_slabs.exp_avg_sq,
            )
        )
        with torch.no_grad():
            owned_shard.copy_(inner.state[owned_shard]["master_param"])
        model_chunk.start_param_sync()
        for step in range(3, 3 + args.continuation_steps):
            _step(model_param, optimizer, scheduler, step, args.numel)
        inner.drain()
        expected_param, expected_state, expected_scheduler = _baseline(
            initial, 3 + args.continuation_steps, kwargs
        )

    assert scheduler.last_epoch == expected_scheduler.last_epoch
    assert scheduler.get_last_lr() == expected_scheduler.get_last_lr()

    local_numel = args.numel // world_size
    local_slice = slice(rank * local_numel, (rank + 1) * local_numel)
    expected = {
        "master_param": expected_param.detach()[local_slice].cpu(),
        "exp_avg": expected_state["exp_avg"][local_slice].cpu(),
        "exp_avg_sq": expected_state["exp_avg_sq"][local_slice].cpu(),
    }
    errors = {}
    for key, expected_value in expected.items():
        actual = inner.state[owned_shard][key]
        errors[key] = float((actual - expected_value).abs().max())
        torch.testing.assert_close(actual, expected_value, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        model_param, expected_param.detach().bfloat16(), rtol=0.0, atol=0.0
    )
    payload = {
        "mode": args.mode,
        "rank": rank,
        "world_size": world_size,
        "owned_numel": inner.cpu_slabs.master.numel(),
        "step": inner.param_groups[0]["step"],
        "scheduler_last_epoch": scheduler.last_epoch,
        "scheduler_lr": scheduler.get_last_lr(),
        "residency": inner.residency,
        "cuda_state_numel": inner.cuda_state_numel,
        "errors": errors,
    }
    if args.mode == "load":
        payload.update(
            allocated_before=allocated_before,
            allocated_peak=allocated_peak,
            checkpoint_bytes=sum(
                path.stat().st_size
                for path in checkpoint_dir.rglob("*")
                if path.is_file()
            ),
            rollback_directories=len(
                discover_orphaned_snapshot_directories(rollback_root)
            ),
        )
    (output_dir / f"{args.mode}_dp{world_size}_rank{rank}.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
