# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from areal.engine.megatron_utils.checkpointer import MegatronCheckpointManager
from areal.engine.megatron_utils.gpu_staged_muon import (
    GPUStagedEmptyOptimizer,
    GPUStagedMuon,
    GPUStagedMuonConfig,
    _make_staged_layerwise_class,
    merge_muon_checkpoint_metadata,
)
from areal.engine.megatron_utils.gpu_staged_optimizer import (
    GPUStagedAdamW,
    GPUStagedAdamWConfig,
)
from areal.engine.megatron_utils.gpu_staged_optimizer_checkpoint import (
    abort_managed_checkpoint_load,
    apply_begin_managed_checkpoint_load,
    begin_managed_checkpoint_load,
    begin_managed_checkpoint_replacement,
    build_managed_optimizer_tensor_manifest,
    cancel_managed_checkpoint_replacement_configuration,
    commit_managed_checkpoint_load,
    create_managed_checkpoint_load_transaction,
    decide_managed_checkpoint_commit,
    merge_managed_optimizer_tensor_manifests,
    poison_managed_checkpoint_transaction,
    prepare_managed_checkpoint_commit,
    prepare_managed_checkpoint_load,
    prepare_managed_checkpoint_recovery,
    prepare_managed_checkpoint_save,
    retry_managed_checkpoint_cleanup,
    validate_managed_checkpoint_load_request,
    validate_managed_optimizer_source_tensor_metadata,
)
from areal.utils.network import find_free_ports


def _identity_orthogonalize(
    param: torch.Tensor, update: torch.Tensor, **kwargs
) -> torch.Tensor:
    del param, kwargs
    return update


def _checkpoint_algorithm() -> dict[str, object]:
    return {
        "momentum": 0.95,
        "use_nesterov": False,
        "fp32_matmul_prec": "highest",
        "coefficient_type": "quintic",
        "num_ns_steps": 5,
        "scale_mode": "spectral",
        "split_qkv": False,
        "tp_mode": "duplicated",
        "extra_scale_factor": 1.0,
    }


def _fake_group(*members: int) -> SimpleNamespace:
    return SimpleNamespace(
        size=lambda: len(members), rank=lambda: 0, members=list(members)
    )


def _empty_layerwise_checkpoint_root(
    monkeypatch: pytest.MonkeyPatch,
) -> object:
    real_version = __import__("importlib.metadata", fromlist=["version"]).version
    monkeypatch.setattr(
        "importlib.metadata.version",
        lambda package: "0.3.0"
        if package == "emerging-optimizers"
        else real_version(package),
    )
    real_get_process_group_ranks = torch.distributed.get_process_group_ranks

    def get_process_group_ranks(group):
        if hasattr(group, "members"):
            return list(group.members)
        return real_get_process_group_ranks(group)

    monkeypatch.setattr(
        torch.distributed,
        "get_process_group_ranks",
        get_process_group_ranks,
    )
    group = _fake_group(0)
    pg_collection = SimpleNamespace(
        tp=group,
        expt_tp=group,
        dp=group,
        dp_cp=group,
        expt_dp=group,
        pp=group,
        cp=group,
        ep=group,
    )
    config = SimpleNamespace(log_num_zeros_in_grad=False)
    bases = [
        GPUStagedEmptyOptimizer("muon"),
        GPUStagedEmptyOptimizer("scalar_adamw"),
    ]
    leaves = [
        SimpleNamespace(optimizer=base, config=config, param_groups=[])
        for base in bases
    ]
    official = SimpleNamespace(
        pg_collection=pg_collection,
        dp_cp_params_list=None,
        expt_dp_params_list=None,
    )
    root = _make_staged_layerwise_class()(official, leaves)
    root.configure_managed_checkpoint_schema(
        {},
        algorithm=_checkpoint_algorithm(),
    )
    return root


_MUON_CHECKPOINT_TEST_GROUPS = (
    "dp",
    "dp_cp",
    "tp",
    "cp",
    "ep",
    "expt_tp",
    "expt_dp",
    "pp",
)


def _set_checkpoint_participant_topology(
    metadata: dict[str, object],
    *,
    rank: int,
    world_size: int,
    partitions: dict[str, list[list[int]]] | None = None,
) -> None:
    """Install one structurally valid participant topology for merge tests."""
    partitions = partitions or {}
    topology = metadata["topology"]
    topology["world_size"] = world_size
    topology["global_rank"] = rank
    groups = topology["groups"]
    for group_name in _MUON_CHECKPOINT_TEST_GROUPS:
        default = (
            [list(range(world_size))]
            if group_name in {"dp", "dp_cp", "expt_dp"}
            else [[participant] for participant in range(world_size)]
        )
        candidates = partitions.get(group_name, default)
        matching = [members for members in candidates if rank in members]
        assert len(matching) == 1
        members = list(matching[0])
        groups[group_name] = {
            "size": len(members),
            "rank": members.index(rank),
            "members": members,
        }


def _dp2_checkpoint_participants(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object]]:
    root = _empty_layerwise_checkpoint_root(monkeypatch)
    participants = (
        copy.deepcopy(root._checkpoint_local_metadata()),
        copy.deepcopy(root._checkpoint_local_metadata()),
    )
    for rank, metadata in enumerate(participants):
        _set_checkpoint_participant_topology(metadata, rank=rank, world_size=2)
    return participants


def test_muon_checkpoint_identity_preserves_empty_leaf_tree_and_model_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fixed-topology identity keeps both empty chain positions without IDs."""
    root = _empty_layerwise_checkpoint_root(monkeypatch)

    metadata = root._checkpoint_metadata()
    root.validate_managed_checkpoint_outer_state({"metadata": copy.deepcopy(metadata)})

    assert [leaf["kind"] for leaf in metadata["leaf_tree"]] == [
        "muon",
        "scalar_adamw",
    ]
    assert [leaf["tree_path"] for leaf in metadata["leaf_tree"]] == [[0], [1]]
    assert "id(" not in repr(metadata)
    participants = _dp2_checkpoint_participants(monkeypatch)
    dp_changed = merge_muon_checkpoint_metadata(
        participants, trusted_global_ranks=[0, 1]
    )
    root.validate_managed_checkpoint_outer_state({"metadata": dp_changed})

    changed = copy.deepcopy(metadata)
    changed["topology"]["invariant_group_sizes"]["tp"] = 2
    with pytest.raises(ValueError, match="invariant topology mismatch"):
        root.validate_managed_checkpoint_outer_state({"metadata": changed})


def test_muon_checkpoint_rejects_async_and_partial_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported async and partial-load modes fail before a DCP operation."""
    root = _empty_layerwise_checkpoint_root(monkeypatch)

    with pytest.raises(RuntimeError, match="asynchronous checkpoint.*staged Muon"):
        prepare_managed_checkpoint_save(root, async_save=True)
    with pytest.raises(RuntimeError, match="model-only checkpoint load"):
        validate_managed_checkpoint_load_request(
            root, with_model=True, with_optimizer=False
        )
    with pytest.raises(RuntimeError, match="optimizer-only checkpoint load"):
        validate_managed_checkpoint_load_request(
            root, with_model=False, with_optimizer=True
        )
    with pytest.raises(RuntimeError, match="asynchronous checkpoint.*staged Muon"):
        MegatronCheckpointManager(
            model=[],
            optimizer=root,
            lr_scheduler=None,
            use_distributed_optimizer=False,
            async_save=True,
            managed_checkpoint_enabled=True,
        )


@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
def test_muon_checkpoint_requires_managed_protocol(
    monkeypatch: pytest.MonkeyPatch, use_distributed_optimizer: bool
) -> None:
    """Fixed-topology Muon must not fall through the ordinary load path."""
    root = _empty_layerwise_checkpoint_root(monkeypatch)

    with pytest.raises(RuntimeError, match="managed checkpoint"):
        MegatronCheckpointManager(
            model=[],
            optimizer=root,
            lr_scheduler=None,
            use_distributed_optimizer=use_distributed_optimizer,
            managed_checkpoint_enabled=False,
        )


def test_ordinary_optimizer_may_keep_managed_checkpoint_disabled() -> None:
    """The Muon capability gate does not alter the ordinary manager path."""
    optimizer = SimpleNamespace()

    manager = MegatronCheckpointManager(
        model=[],
        optimizer=optimizer,
        lr_scheduler=None,
        use_distributed_optimizer=True,
        managed_checkpoint_enabled=False,
    )

    assert manager.optimizer is optimizer
    assert manager.managed_checkpoint_enabled is False


def test_muon_checkpoint_local_empty_owner_manifest_is_mergeable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rank with no owned tensors contributes an empty manifest to the union."""
    root = _empty_layerwise_checkpoint_root(monkeypatch)

    assert build_managed_optimizer_tensor_manifest(root.sharded_state_dict({})) == {}
    metadata = copy.deepcopy(root.state_dict()["metadata"])
    transaction = begin_managed_checkpoint_load(root)
    root.load_state_dict({"metadata": metadata})
    abort_managed_checkpoint_load(transaction, RuntimeError("empty owner probe"))


def test_muon_checkpoint_failed_begin_preserves_preexisting_empty_leaf_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected global begin must not abort another transaction's empty leaf."""
    root = _empty_layerwise_checkpoint_root(monkeypatch)
    existing_leaf = root.chained_optimizers[1].optimizer
    existing_leaf.begin_checkpoint_load()

    with pytest.raises(RuntimeError, match="already active"):
        begin_managed_checkpoint_load(root)

    assert existing_leaf._checkpoint_active is True
    existing_leaf.abort_checkpoint_load(RuntimeError("test cleanup"))


@pytest.mark.parametrize("existing_index", [0, 1, 2])
def test_managed_begin_attempt_only_aborts_authorized_prefix(
    existing_index: int,
) -> None:
    """A rejected leaf's older token never grants the new attempt authority."""

    class TokenLeaf:
        manages_cpu_residency = True

        def __init__(self, name: str) -> None:
            self.name = name
            self.checkpoint_load_attempt_token = None
            self.abort_calls = 0

        def begin_checkpoint_load(self, *, attempt_token=None) -> None:
            if self.checkpoint_load_attempt_token is not None:
                raise RuntimeError(f"{self.name} already active")
            self.checkpoint_load_attempt_token = attempt_token

        def abort_checkpoint_load(self, error, *, attempt_token=None) -> None:
            del error
            if self.checkpoint_load_attempt_token is not attempt_token:
                raise RuntimeError("attempt token mismatch")
            self.abort_calls += 1
            self.checkpoint_load_attempt_token = None

    leaves = [TokenLeaf(str(index)) for index in range(3)]
    old_token = object()
    leaves[existing_index].begin_checkpoint_load(attempt_token=old_token)
    root = SimpleNamespace(
        chained_optimizers=[SimpleNamespace(optimizer=leaf) for leaf in leaves]
    )
    transaction = create_managed_checkpoint_load_transaction(root)

    with pytest.raises(RuntimeError, match="already active") as exc_info:
        apply_begin_managed_checkpoint_load(transaction)
    abort_managed_checkpoint_load(transaction, exc_info.value, poison=False)

    assert leaves[existing_index].checkpoint_load_attempt_token is old_token
    assert leaves[existing_index].abort_calls == 0
    assert [leaf.abort_calls for leaf in leaves[:existing_index]] == [
        1
    ] * existing_index
    assert all(
        leaf.checkpoint_load_attempt_token is None for leaf in leaves[:existing_index]
    )
    assert all(
        leaf.checkpoint_load_attempt_token is None
        for leaf in leaves[existing_index + 1 :]
    )
    leaves[existing_index].abort_checkpoint_load(
        RuntimeError("old transaction cleanup"), attempt_token=old_token
    )


def test_managed_begin_attempt_aborts_partial_token_owner_once() -> None:
    """A begin failure after token publication remains recoverable and idempotent."""

    class PartialLeaf:
        manages_cpu_residency = True

        def __init__(self) -> None:
            self.checkpoint_load_attempt_token = None
            self.value = "old"
            self.abort_calls = 0

        def begin_checkpoint_load(self, *, attempt_token=None) -> None:
            self.checkpoint_load_attempt_token = attempt_token
            self.value = "partial"
            raise RuntimeError("partial begin")

        def abort_checkpoint_load(self, error, *, attempt_token=None) -> None:
            del error
            if self.checkpoint_load_attempt_token is not attempt_token:
                raise RuntimeError("attempt token mismatch")
            self.abort_calls += 1
            self.value = "old"
            self.checkpoint_load_attempt_token = None

    leaf = PartialLeaf()
    transaction = create_managed_checkpoint_load_transaction(
        SimpleNamespace(optimizer=leaf)
    )

    with pytest.raises(RuntimeError, match="partial begin") as exc_info:
        apply_begin_managed_checkpoint_load(transaction)
    assert len(transaction.begun) == 1
    assert transaction.begun[0].leaf is leaf
    assert transaction.begun[0].attempt_token is transaction.attempt_token

    abort_managed_checkpoint_load(transaction, exc_info.value, poison=False)
    abort_managed_checkpoint_load(transaction, exc_info.value, poison=False)

    assert leaf.value == "old"
    assert leaf.abort_calls == 1
    assert leaf.checkpoint_load_attempt_token is None
    assert transaction.begun == []


@pytest.mark.parametrize("optimizer_kind", ["muon", "scalar_adamw"])
def test_empty_leaf_poison_recovery_releases_begin_attempt_authority(
    optimizer_kind: str,
) -> None:
    """Recovered empty leaves must allow the required replacement full load."""
    leaf = GPUStagedEmptyOptimizer(optimizer_kind)
    root = SimpleNamespace(optimizer=leaf)
    transaction = create_managed_checkpoint_load_transaction(root)
    apply_begin_managed_checkpoint_load(transaction)
    poison_managed_checkpoint_transaction(transaction, RuntimeError("injected poison"))

    recovery = create_managed_checkpoint_load_transaction(root)
    prepare_managed_checkpoint_recovery(recovery)

    assert leaf.checkpoint_load_attempt_token is None
    assert leaf.checkpoint_lifecycle == "RELOAD_REQUIRED"
    assert leaf._checkpoint_recovery_journal is None
    assert leaf._checkpoint_error is None
    replacement = begin_managed_checkpoint_load(root)
    abort_managed_checkpoint_load(
        replacement, RuntimeError("replacement cleanup"), poison=False
    )


@pytest.mark.parametrize("optimizer_kind", ["muon", "scalar_adamw"])
def test_empty_leaf_reload_requires_matching_replacement_authority(
    optimizer_kind: str,
) -> None:
    """RELOAD_REQUIRED is train-closed and only a fresh manager attempt may load."""
    leaf = GPUStagedEmptyOptimizer(optimizer_kind)
    root = SimpleNamespace(optimizer=leaf)
    failed = create_managed_checkpoint_load_transaction(root)
    apply_begin_managed_checkpoint_load(failed)
    poison_managed_checkpoint_transaction(failed, RuntimeError("poison"))
    recovery = create_managed_checkpoint_load_transaction(root)
    prepare_managed_checkpoint_recovery(recovery)
    generation = leaf.checkpoint_reload_generation

    assert generation is recovery.reload_generation
    assert leaf.checkpoint_lifecycle == "RELOAD_REQUIRED"
    with pytest.raises(RuntimeError, match="RELOAD_REQUIRED"):
        leaf.step()
    with pytest.raises(RuntimeError, match="not clean"):
        leaf.prepare_checkpoint_save()
    with pytest.raises(RuntimeError, match="replacement authority"):
        leaf.configure_checkpoint_snapshot(leaf_identity={"leaf": optimizer_kind})

    replacement = create_managed_checkpoint_load_transaction(root)
    begin_managed_checkpoint_replacement(replacement)
    with pytest.raises(RuntimeError, match="replacement authority"):
        leaf.configure_checkpoint_snapshot(
            leaf_identity={"leaf": optimizer_kind},
            replacement_generation=object(),
            attempt_token=replacement.attempt_token,
        )
    with pytest.raises(RuntimeError, match="replacement authority"):
        leaf.configure_checkpoint_snapshot(
            leaf_identity={"leaf": optimizer_kind},
            replacement_generation=generation,
            attempt_token=object(),
        )
    leaf.configure_checkpoint_snapshot(
        leaf_identity={"leaf": optimizer_kind},
        replacement_generation=generation,
        attempt_token=replacement.attempt_token,
    )
    replacement.snapshot_configured.append(leaf)
    apply_begin_managed_checkpoint_load(replacement)
    first_attempt = replacement.attempt_token
    abort_managed_checkpoint_load(
        replacement, RuntimeError("replacement failed"), poison=False
    )

    assert leaf.checkpoint_lifecycle == "RELOAD_REQUIRED"
    assert leaf.checkpoint_reload_generation is generation
    assert leaf.checkpoint_load_attempt_token is None
    retry = create_managed_checkpoint_load_transaction(root)
    begin_managed_checkpoint_replacement(retry)
    assert retry.attempt_token is not first_attempt
    cancel_managed_checkpoint_replacement_configuration(retry)
    assert generation.active_attempt is None


@pytest.mark.parametrize("optimizer_kind", ["muon", "scalar_adamw"])
def test_empty_leaf_cleanup_reconciles_discard_after_effect_failure(
    monkeypatch: pytest.MonkeyPatch, optimizer_kind: str
) -> None:
    """A completed empty cleanup is not replayed after its caller observes failure."""
    leaf = GPUStagedEmptyOptimizer(optimizer_kind)
    transaction = create_managed_checkpoint_load_transaction(
        SimpleNamespace(optimizer=leaf)
    )
    apply_begin_managed_checkpoint_load(transaction)
    prepare_managed_checkpoint_load(transaction)
    prepare_managed_checkpoint_commit(transaction)
    decide_managed_checkpoint_commit(transaction)
    original_discard = leaf.discard_checkpoint_snapshot
    failures_remaining = 3

    def discard_then_fail() -> None:
        nonlocal failures_remaining
        original_discard()
        if failures_remaining:
            failures_remaining -= 1
            raise RuntimeError("discard after-effect")

    monkeypatch.setattr(leaf, "discard_checkpoint_snapshot", discard_then_fail)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="discard after-effect"):
            retry_managed_checkpoint_cleanup(transaction)
        assert leaf.checkpoint_lifecycle == "CLEANUP_PENDING"
        assert leaf._checkpoint_cleanup_receipt is not None
        assert leaf._checkpoint_cleanup_receipt.completed
        assert leaf.step() is None
        with pytest.raises(RuntimeError, match="commit decision"):
            leaf.abort_checkpoint_load(RuntimeError("late abort"))
    monkeypatch.setattr(leaf, "discard_checkpoint_snapshot", original_discard)

    retry_managed_checkpoint_cleanup(transaction)
    assert transaction.cleanup_journal is None
    assert transaction.commit_token is None
    assert leaf.checkpoint_lifecycle == "CLEAN"
    assert leaf._checkpoint_cleanup_receipt is None
    assert leaf._checkpoint_cleanup_error is None


def test_empty_leaf_recovery_retries_only_pending_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recovered prefix is not replayed while a later empty leaf remains pending."""
    leaves = [
        GPUStagedEmptyOptimizer("muon"),
        GPUStagedEmptyOptimizer("scalar_adamw"),
    ]
    root = SimpleNamespace(
        chained_optimizers=[SimpleNamespace(optimizer=leaf) for leaf in leaves]
    )
    transaction = create_managed_checkpoint_load_transaction(root)
    apply_begin_managed_checkpoint_load(transaction)
    poison_managed_checkpoint_transaction(transaction, RuntimeError("poison"))

    attempts = [0, 0]
    first_recover = leaves[0].prepare_checkpoint_recovery
    second_recover = leaves[1].prepare_checkpoint_recovery

    def recover_first(*, attempt_token=None, reload_generation=None) -> None:
        attempts[0] += 1
        first_recover(
            attempt_token=attempt_token,
            reload_generation=reload_generation,
        )

    failures_remaining = 2

    def recover_second(*, attempt_token=None, reload_generation=None) -> None:
        nonlocal failures_remaining
        attempts[1] += 1
        if failures_remaining:
            failures_remaining -= 1
            raise RuntimeError("pending recovery")
        second_recover(
            attempt_token=attempt_token,
            reload_generation=reload_generation,
        )

    monkeypatch.setattr(leaves[0], "prepare_checkpoint_recovery", recover_first)
    monkeypatch.setattr(leaves[1], "prepare_checkpoint_recovery", recover_second)
    recovery = create_managed_checkpoint_load_transaction(root)
    retained_journal = leaves[1]._checkpoint_recovery_journal
    retained_token = leaves[1].checkpoint_load_attempt_token

    for _ in range(2):
        with pytest.raises(RuntimeError, match="pending recovery"):
            prepare_managed_checkpoint_recovery(recovery)
        assert attempts[0] == 1
        assert leaves[0].checkpoint_lifecycle == "RELOAD_REQUIRED"
        assert leaves[1].checkpoint_lifecycle == "POISONED"
        assert leaves[1]._checkpoint_recovery_journal is retained_journal
        assert leaves[1].checkpoint_load_attempt_token is retained_token

    prepare_managed_checkpoint_recovery(recovery)

    assert attempts == [1, 3]
    assert recovery.recovery_completed == []
    assert all(leaf.checkpoint_lifecycle == "RELOAD_REQUIRED" for leaf in leaves)
    assert all(
        leaf.checkpoint_reload_generation is recovery.reload_generation
        for leaf in leaves
    )
    assert all(leaf._checkpoint_recovery_journal is None for leaf in leaves)


@pytest.mark.parametrize("optimizer_kind", ["muon", "scalar_adamw"])
def test_empty_leaf_recovery_reconciles_after_effect_failure(
    monkeypatch: pytest.MonkeyPatch, optimizer_kind: str
) -> None:
    """A completed recovery remains identifiable until manager acknowledgement."""
    leaf = GPUStagedEmptyOptimizer(optimizer_kind)
    root = SimpleNamespace(optimizer=leaf)
    transaction = create_managed_checkpoint_load_transaction(root)
    apply_begin_managed_checkpoint_load(transaction)
    poison_managed_checkpoint_transaction(transaction, RuntimeError("poison"))
    recovery = create_managed_checkpoint_load_transaction(root)
    original_recover = leaf.prepare_checkpoint_recovery
    failures_remaining = 1

    def recover_then_fail(*, attempt_token=None, reload_generation=None) -> None:
        nonlocal failures_remaining
        original_recover(
            attempt_token=attempt_token,
            reload_generation=reload_generation,
        )
        if failures_remaining:
            failures_remaining -= 1
            raise RuntimeError("recovery after-effect")

    monkeypatch.setattr(leaf, "prepare_checkpoint_recovery", recover_then_fail)
    with pytest.raises(RuntimeError, match="recovery after-effect"):
        prepare_managed_checkpoint_recovery(recovery)
    monkeypatch.setattr(leaf, "prepare_checkpoint_recovery", original_recover)

    prepare_managed_checkpoint_recovery(recovery)
    assert leaf.checkpoint_lifecycle == "RELOAD_REQUIRED"
    assert leaf.checkpoint_reload_generation is recovery.reload_generation
    assert leaf.checkpoint_load_attempt_token is None
    assert leaf._checkpoint_recovery_journal is None
    assert leaf._checkpoint_error is None
    assert leaf._checkpoint_recovery_terminal_receipt is not None
    assert leaf._checkpoint_recovery_terminal_receipt.manager_confirmed


@pytest.mark.parametrize("optimizer_kind", ["muon", "scalar_adamw"])
@pytest.mark.parametrize("failure_timing", ["pre_effect", "post_effect"])
def test_manager_reuses_empty_leaf_recovery_authority_across_public_loads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    optimizer_kind: str,
    failure_timing: str,
) -> None:
    """A public load retry must retain the recovery action it already published."""
    leaf = GPUStagedEmptyOptimizer(optimizer_kind)
    root = SimpleNamespace(optimizer=leaf)
    transaction = create_managed_checkpoint_load_transaction(root)
    apply_begin_managed_checkpoint_load(transaction)
    poison = RuntimeError("injected poison")
    poison_managed_checkpoint_transaction(transaction, poison)
    manager = MegatronCheckpointManager(
        model=[],
        optimizer=root,
        lr_scheduler=None,
        use_distributed_optimizer=True,
        managed_checkpoint_enabled=True,
    )
    manager._managed_checkpoint_poisoned_error = poison

    class LocalPhaseError(RuntimeError):
        def __init__(self, phase: str, local_error: BaseException) -> None:
            super().__init__(f"{phase}: {local_error}")
            self.local_error = local_error

    def local_vote(
        phase: str,
        local_error: BaseException | None,
        transaction,
        **kwargs,
    ) -> LocalPhaseError | None:
        del transaction, kwargs
        return None if local_error is None else LocalPhaseError(phase, local_error)

    monkeypatch.setattr(manager, "_vote_managed_phase", local_vote)
    monkeypatch.setattr(manager, "_require_managed_checkpoint_group", lambda: None)
    original_recover = leaf.prepare_checkpoint_recovery
    failures_remaining = 1

    def fail_recovery_once(
        *,
        attempt_token=None,
        recovery_action_token=None,
        reload_generation=None,
    ) -> None:
        nonlocal failures_remaining
        if failure_timing == "pre_effect" and failures_remaining:
            failures_remaining -= 1
            raise RuntimeError("injected recovery pre-effect")
        original_recover(
            attempt_token=attempt_token,
            recovery_action_token=recovery_action_token,
            reload_generation=reload_generation,
        )
        if failure_timing == "post_effect" and failures_remaining:
            failures_remaining -= 1
            raise RuntimeError("injected recovery post-effect")

    monkeypatch.setattr(leaf, "prepare_checkpoint_recovery", fail_recovery_once)
    expected_failure = failure_timing.replace("_", "-")
    with pytest.raises(LocalPhaseError, match=f"injected recovery {expected_failure}"):
        manager.load_checkpoint(str(tmp_path))
    monkeypatch.setattr(leaf, "prepare_checkpoint_recovery", original_recover)
    retained = manager._managed_checkpoint_recovery_transaction
    assert retained is not None
    retained_attempt = retained.attempt_token
    retained_actions = tuple(
        authority.recovery_action_token for authority in retained.recovery_authorities
    )

    with pytest.raises(LocalPhaseError, match="save_ready"):
        manager.save_checkpoint(str(tmp_path))
    assert manager._managed_checkpoint_recovery_transaction is retained
    assert retained.attempt_token is retained_attempt

    with pytest.raises(LocalPhaseError, match=r"full model\+optimizer\+RNG"):
        manager.load_checkpoint(str(tmp_path), with_rng=False)
    assert manager._managed_checkpoint_recovery_transaction is retained
    assert retained.attempt_token is retained_attempt
    terminal = leaf._checkpoint_recovery_terminal_receipt
    assert terminal is not None
    assert terminal.manager_confirmed
    assert any(terminal.action_token is action for action in retained_actions)

    monkeypatch.setattr(manager, "_managed_optimizer_identities", lambda: {(): {}})
    monkeypatch.setattr(manager, "_build_checkpoint_load_template", lambda **kwargs: {})
    monkeypatch.setattr(manager, "_load_checkpoint_data", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        manager,
        "_apply_checkpoint_state",
        lambda *args, **kwargs: leaf.load_state_dict({"state": {}, "param_groups": []}),
    )
    monkeypatch.setattr(manager, "get_rng_state", lambda **kwargs: {"rng": "old"})
    monkeypatch.setattr(manager, "load_rng_states", lambda state: None)

    manager.load_checkpoint(str(tmp_path))
    assert manager._managed_checkpoint_recovery_transaction is None
    assert manager._managed_checkpoint_poisoned_error is None
    assert manager._managed_checkpoint_cleanup_recovery is None
    assert leaf.checkpoint_lifecycle == "CLEAN"
    assert leaf._checkpoint_attempt_token is None
    assert leaf._checkpoint_recovery_journal is None
    assert leaf._checkpoint_recovery_terminal_receipt is None
    assert leaf._checkpoint_cleanup_receipt is None
    assert leaf._checkpoint_cleanup_terminal_receipt is None

    manager.load_checkpoint(str(tmp_path))
    assert manager._managed_checkpoint_recovery_transaction is None
    assert manager._managed_checkpoint_poisoned_error is None
    assert manager._managed_checkpoint_cleanup_recovery is None
    assert leaf.checkpoint_lifecycle == "CLEAN"
    assert leaf._checkpoint_attempt_token is None
    assert leaf._checkpoint_recovery_journal is None
    assert leaf._checkpoint_recovery_terminal_receipt is None
    assert leaf._checkpoint_cleanup_receipt is None
    assert leaf._checkpoint_cleanup_terminal_receipt is None


@pytest.mark.parametrize("optimizer_kind", ["muon", "scalar_adamw"])
def test_empty_leaf_cleanup_retries_pre_effect_and_releases_receipt(
    monkeypatch: pytest.MonkeyPatch, optimizer_kind: str
) -> None:
    """Pre-effect cleanup failures retain one action until a later success."""
    leaf = GPUStagedEmptyOptimizer(optimizer_kind)
    transaction = create_managed_checkpoint_load_transaction(
        SimpleNamespace(optimizer=leaf)
    )
    apply_begin_managed_checkpoint_load(transaction)
    prepare_managed_checkpoint_load(transaction)
    prepare_managed_checkpoint_commit(transaction)
    decide_managed_checkpoint_commit(transaction)
    original_discard = leaf.discard_checkpoint_snapshot
    attempts = 0

    def fail_before_discard() -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("discard pre-effect")
        original_discard()

    monkeypatch.setattr(leaf, "discard_checkpoint_snapshot", fail_before_discard)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="discard pre-effect"):
            retry_managed_checkpoint_cleanup(transaction)
        assert leaf._checkpoint_cleanup_receipt is not None
        assert not leaf._checkpoint_cleanup_receipt.completed
        assert leaf.step() is None

    retry_managed_checkpoint_cleanup(transaction)

    assert attempts == 3
    assert leaf.checkpoint_lifecycle == "CLEAN"
    assert leaf._checkpoint_cleanup_receipt is None
    assert leaf._checkpoint_token is None


@pytest.mark.parametrize("optimizer_kind", ["muon", "scalar_adamw"])
def test_empty_leaf_cleanup_reconciles_acknowledgement_after_effect_failure(
    monkeypatch: pytest.MonkeyPatch, optimizer_kind: str
) -> None:
    """A completed acknowledgement remains identifiable to a manager retry."""
    leaf = GPUStagedEmptyOptimizer(optimizer_kind)
    transaction = create_managed_checkpoint_load_transaction(
        SimpleNamespace(optimizer=leaf)
    )
    apply_begin_managed_checkpoint_load(transaction)
    prepare_managed_checkpoint_load(transaction)
    prepare_managed_checkpoint_commit(transaction)
    decide_managed_checkpoint_commit(transaction)
    original_acknowledge = leaf.acknowledge_checkpoint_cleanup
    failures_remaining = 1

    def acknowledge_then_fail(commit_token, action_token) -> None:
        nonlocal failures_remaining
        original_acknowledge(commit_token, action_token)
        if failures_remaining:
            failures_remaining -= 1
            raise RuntimeError("acknowledgement after-effect")

    monkeypatch.setattr(leaf, "acknowledge_checkpoint_cleanup", acknowledge_then_fail)
    with pytest.raises(RuntimeError, match="acknowledgement after-effect"):
        retry_managed_checkpoint_cleanup(transaction)
    monkeypatch.setattr(leaf, "acknowledge_checkpoint_cleanup", original_acknowledge)

    retry_managed_checkpoint_cleanup(transaction)
    assert transaction.cleanup_journal is None
    assert transaction.commit_token is None
    assert leaf.checkpoint_lifecycle == "CLEAN"
    assert leaf._checkpoint_cleanup_receipt is None
    assert leaf._checkpoint_cleanup_error is None
    assert leaf._checkpoint_cleanup_terminal_receipt is None


@pytest.mark.parametrize("optimizer_kind", ["muon", "scalar_adamw"])
def test_empty_leaf_cleanup_receipt_release_reconciles_after_effect(
    monkeypatch: pytest.MonkeyPatch, optimizer_kind: str
) -> None:
    """A released terminal receipt is reconciled without replaying its effect."""
    leaf = GPUStagedEmptyOptimizer(optimizer_kind)
    transaction = create_managed_checkpoint_load_transaction(
        SimpleNamespace(optimizer=leaf)
    )
    apply_begin_managed_checkpoint_load(transaction)
    prepare_managed_checkpoint_load(transaction)
    prepare_managed_checkpoint_commit(transaction)
    decide_managed_checkpoint_commit(transaction)
    original_release = leaf.release_checkpoint_cleanup_receipt
    release_calls = 0

    def release_then_fail(commit_token, action_token) -> None:
        nonlocal release_calls
        release_calls += 1
        original_release(commit_token, action_token)
        raise RuntimeError("receipt release after-effect")

    monkeypatch.setattr(leaf, "release_checkpoint_cleanup_receipt", release_then_fail)
    with pytest.raises(RuntimeError, match="receipt release after-effect"):
        retry_managed_checkpoint_cleanup(transaction)

    assert transaction.cleanup_pending == [leaf]
    assert leaf._checkpoint_cleanup_terminal_receipt is None
    assert leaf.step() is None
    monkeypatch.setattr(
        leaf,
        "release_checkpoint_cleanup_receipt",
        lambda *_: pytest.fail("completed receipt release was replayed"),
    )
    retry_managed_checkpoint_cleanup(transaction)

    assert release_calls == 1
    assert transaction.cleanup_pending == []
    assert transaction.commit_token is None
    assert leaf.checkpoint_lifecycle == "CLEAN"


@pytest.mark.parametrize("optimizer_kind", ["muon", "scalar_adamw"])
def test_empty_leaf_cleanup_identity_and_repeated_cycles(
    optimizer_kind: str,
) -> None:
    """Cleanup identity is strict while successful cycles release all receipts."""
    leaf = GPUStagedEmptyOptimizer(optimizer_kind)
    for _ in range(3):
        transaction = create_managed_checkpoint_load_transaction(
            SimpleNamespace(optimizer=leaf)
        )
        apply_begin_managed_checkpoint_load(transaction)
        prepare_managed_checkpoint_load(transaction)
        prepare_managed_checkpoint_commit(transaction)
        decide_managed_checkpoint_commit(transaction)
        leaf.decide_checkpoint_commit()
        entry = transaction.cleanup_journal.entries[0]
        with pytest.raises(RuntimeError, match="commit token mismatch"):
            leaf.bind_checkpoint_cleanup_action(object(), entry.action_token)
        leaf.bind_checkpoint_cleanup_action(entry.commit_token, entry.action_token)
        with pytest.raises(RuntimeError, match="action token mismatch"):
            leaf.bind_checkpoint_cleanup_action(entry.commit_token, object())
        leaf.discard_checkpoint_snapshot()
        leaf.discard_checkpoint_snapshot()
        with pytest.raises(RuntimeError, match="acknowledgement mismatch"):
            leaf.acknowledge_checkpoint_cleanup(entry.commit_token, object())
        retry_managed_checkpoint_cleanup(transaction)

        assert leaf.checkpoint_lifecycle == "CLEAN"
        assert leaf.checkpoint_load_attempt_token is None
        assert leaf._checkpoint_token is None
        assert leaf._checkpoint_cleanup_receipt is None
        assert leaf._checkpoint_recovery_journal is None
        assert leaf._checkpoint_error is None
        assert leaf._checkpoint_cleanup_error is None


def test_muon_checkpoint_manifest_merge_accepts_symmetric_empty_owners() -> None:
    """Either rank may own no state without changing the global manifest."""
    key = "optimizer.gpu_staged_muon.v2.leaf_0.dense.coord_x.param_y.master_param"
    owned = {key: ((16,), "torch.float32")}

    assert merge_managed_optimizer_tensor_manifests([{}, owned]) == owned
    assert merge_managed_optimizer_tensor_manifests([owned, {}]) == owned


def test_muon_checkpoint_manifest_merge_rejects_global_empty_union() -> None:
    """A globally state-free checkpoint is unsupported despite local emptiness."""
    with pytest.raises(ValueError, match="manifest union is empty"):
        merge_managed_optimizer_tensor_manifests([{}, {}])


@pytest.mark.parametrize(
    "conflicting",
    [
        ((15,), "torch.float32"),
        ((16,), "torch.bfloat16"),
    ],
)
def test_muon_checkpoint_manifest_merge_rejects_cross_rank_metadata_conflict(
    conflicting: tuple[tuple[int, ...], str],
) -> None:
    """Duplicate stable tensor identities must agree before DCP mutation."""
    key = "optimizer.gpu_staged_muon.v2.leaf_0.dense.coord_x.param_y.master_param"
    expected = {key: ((16,), "torch.float32")}

    with pytest.raises(ValueError, match="differs across checkpoint participants"):
        merge_managed_optimizer_tensor_manifests([expected, {key: conflicting}])


def test_muon_checkpoint_manifest_merge_rejects_duplicate_logical_owner() -> None:
    """A logical Muon state has one authority even if descriptors agree."""
    key = "optimizer.gpu_staged_muon.v2.leaf_0.dense.coord_x.param_y.master_param"
    owned = {key: ((16,), "torch.float32")}

    with pytest.raises(ValueError, match="duplicate owners"):
        merge_managed_optimizer_tensor_manifests([owned, owned])


def test_muon_checkpoint_wire_key_is_independent_of_dp_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source/target DP ownership diagnostics never enter a wire tensor key."""
    root = _empty_layerwise_checkpoint_root(monkeypatch)
    coordinate = {"pp": 0, "tp": 0}

    source_key = root._checkpoint_tensor_key(
        leaf_index=0,
        parameter_name="model.weight",
        domain="dense",
        coordinate=coordinate,
        state_kind="master_param",
    )
    target_key = root._checkpoint_tensor_key(
        leaf_index=0,
        parameter_name="model.weight",
        domain="dense",
        coordinate=coordinate,
        state_kind="master_param",
    )

    assert source_key == target_key
    assert "rank_" not in source_key
    assert "owner" not in source_key


def test_muon_checkpoint_global_metadata_allows_rank_local_empty_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The global schema is the union of official owner records, not rank 0."""
    root = _empty_layerwise_checkpoint_root(monkeypatch)
    empty = root._checkpoint_local_metadata()
    owned = copy.deepcopy(empty)
    _set_checkpoint_participant_topology(empty, rank=0, world_size=2)
    _set_checkpoint_participant_topology(owned, rank=1, world_size=2)
    parameter = {
        "name": "model.weight",
        "domain": "dense",
        "coordinate": {"pp": 0, "tp": 0},
        "shape": [8, 8],
        "dtype": "torch.bfloat16",
        "state_kinds": ["master_param", "momentum_buffer"],
        "source_owner": {
            "global_rank": 1,
            "owner_rank": 1,
            "owner_ordinal": 0,
            "group_index": 0,
            "parameter_index": 0,
            "unit_order": 0,
        },
    }
    owned["leaf_tree"][0]["parameters"] = [parameter]

    merged = merge_muon_checkpoint_metadata([empty, owned], trusted_global_ranks=[0, 1])

    assert merged["leaf_tree"][0]["parameters"] == [parameter]
    assert merged["topology"]["source_world_size"] == 2
    assert merged["topology"]["source_ownership_group_sizes"]["dp_cp"] == 2


def test_muon_checkpoint_global_metadata_rejects_duplicate_participant_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two participants cannot publish the same WORLD/DP coordinate."""
    first, second = _dp2_checkpoint_participants(monkeypatch)
    second["topology"]["global_rank"] = 0

    with pytest.raises(ValueError, match="global rank|participant"):
        merge_muon_checkpoint_metadata([first, second], trusted_global_ranks=[0, 1])


@pytest.mark.parametrize(
    "case",
    [
        "swapped_declarations",
        "bool_declaration",
        "bool_world_size",
        "out_of_range_declaration",
        "world_size_mismatch",
        "duplicate_trusted_rank",
        "out_of_range_trusted_rank",
        "participant_count_mismatch",
    ],
)
def test_muon_checkpoint_metadata_rejects_untrusted_participant_authority(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Receive-slot authority cannot be replaced by self-reported rank metadata."""
    first, second = _dp2_checkpoint_participants(monkeypatch)
    metadata = [first, second]
    trusted = [0, 1]
    if case == "swapped_declarations":
        metadata.reverse()
    elif case == "bool_declaration":
        second["topology"]["global_rank"] = True
    elif case == "bool_world_size":
        second["topology"]["world_size"] = True
    elif case == "out_of_range_declaration":
        second["topology"]["global_rank"] = 2
    elif case == "world_size_mismatch":
        second["topology"]["world_size"] = 3
    elif case == "duplicate_trusted_rank":
        trusted = [0, 0]
    elif case == "out_of_range_trusted_rank":
        trusted = [0, 2]
    elif case == "participant_count_mismatch":
        trusted = [0]
    else:  # pragma: no cover - guards parametrization drift.
        raise AssertionError(case)

    with pytest.raises(ValueError, match="participant|rank|world_size|count"):
        merge_muon_checkpoint_metadata(metadata, trusted_global_ranks=trusted)


@pytest.mark.parametrize(
    "case",
    [
        "rank_out_of_range",
        "rank_bool",
        "size_bool",
        "duplicate_members",
        "member_bool",
        "missing_self",
        "wrong_member_index",
        "membership_inconsistent",
        "group_size_conflict",
    ],
)
def test_muon_checkpoint_metadata_rejects_invalid_group_partition(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Every topology dimension must be a complete, consistent WORLD partition."""
    first, second = _dp2_checkpoint_participants(monkeypatch)
    identity = second["topology"]["groups"]["dp"]
    if case == "rank_out_of_range":
        identity["rank"] = 2
    elif case == "rank_bool":
        identity["rank"] = True
    elif case == "size_bool":
        identity["size"] = True
    elif case == "duplicate_members":
        identity["members"] = [1, 1]
    elif case == "member_bool":
        identity["members"] = [0, True]
    elif case == "missing_self":
        identity.update({"size": 1, "rank": 0, "members": [0]})
    elif case == "wrong_member_index":
        identity["rank"] = 0
    elif case == "membership_inconsistent":
        identity.update({"size": 1, "rank": 0, "members": [1]})
    elif case == "group_size_conflict":
        first["topology"]["groups"]["dp"].update({"size": 1, "rank": 0, "members": [0]})
    else:  # pragma: no cover - guards parametrization drift.
        raise AssertionError(case)

    with pytest.raises(ValueError, match="dp|member|rank|size|partition"):
        merge_muon_checkpoint_metadata([first, second], trusted_global_ranks=[0, 1])


@pytest.mark.parametrize("group_name", _MUON_CHECKPOINT_TEST_GROUPS)
def test_muon_checkpoint_metadata_validates_every_group_dimension(
    monkeypatch: pytest.MonkeyPatch, group_name: str
) -> None:
    """Every declared group dimension independently binds the participant rank."""
    first, second = _dp2_checkpoint_participants(monkeypatch)
    identity = second["topology"]["groups"][group_name]
    identity.update({"size": 1, "rank": 0, "members": [0]})

    with pytest.raises(ValueError, match=group_name):
        merge_muon_checkpoint_metadata([first, second], trusted_global_ranks=[0, 1])


@pytest.mark.parametrize("domain", ["dense", "expert"])
@pytest.mark.parametrize("fault", ["contributor", "owner_coordinate"])
def test_muon_checkpoint_metadata_binds_source_owner_to_contributor(
    monkeypatch: pytest.MonkeyPatch, domain: str, fault: str
) -> None:
    """A logical payload is authoritative only on its official source owner."""
    empty, owned = _dp2_checkpoint_participants(monkeypatch)
    owner_group = "dp_cp" if domain == "dense" else "expt_dp"
    coordinate = (
        {"pp": 0, "tp": 0} if domain == "dense" else {"pp": 0, "ep": 0, "expt_tp": 0}
    )
    parameter = {
        "name": f"model.{domain}_weight",
        "domain": domain,
        "coordinate": coordinate,
        "shape": [8, 8],
        "dtype": "torch.bfloat16",
        "state_kinds": ["master_param", "momentum_buffer"],
        "source_owner": {
            "global_rank": 1,
            "owner_rank": owned["topology"]["groups"][owner_group]["rank"],
            "owner_ordinal": 0,
            "group_index": 0,
            "parameter_index": 0,
            "unit_order": 0,
        },
    }
    owned["leaf_tree"][0]["parameters"] = [parameter]
    if fault == "contributor":
        parameter["source_owner"]["global_rank"] = 0
    else:
        parameter["source_owner"]["owner_rank"] = 0

    with pytest.raises(ValueError, match="contributor|coordinate"):
        merge_muon_checkpoint_metadata([empty, owned], trusted_global_ranks=[0, 1])


@pytest.mark.parametrize("topology", ["dp2", "tp2_dp2", "ep2", "tp2_ep2"])
def test_muon_checkpoint_metadata_accepts_valid_world_partitions(
    monkeypatch: pytest.MonkeyPatch, topology: str
) -> None:
    """DP, TP/DP, and EP group partitions retain trusted participant authority."""
    world_size = 4 if topology in {"tp2_dp2", "tp2_ep2"} else 2
    root = _empty_layerwise_checkpoint_root(monkeypatch)
    participants = [
        copy.deepcopy(root._checkpoint_local_metadata()) for _ in range(world_size)
    ]
    partitions: dict[str, list[list[int]]] = {}
    if topology == "tp2_dp2":
        partitions = {
            "tp": [[0, 1], [2, 3]],
            "expt_tp": [[0, 1], [2, 3]],
            "dp": [[0, 2], [1, 3]],
            "dp_cp": [[0, 2], [1, 3]],
            "expt_dp": [[0, 2], [1, 3]],
        }
    elif topology == "ep2":
        partitions = {
            "ep": [[0, 1]],
            "expt_dp": [[0], [1]],
        }
    elif topology == "tp2_ep2":
        partitions = {
            "tp": [[0, 1], [2, 3]],
            "dp": [[0, 2], [1, 3]],
            "dp_cp": [[0, 2], [1, 3]],
            "expt_tp": [[0, 1], [2, 3]],
            "ep": [[0, 2], [1, 3]],
            "expt_dp": [[0], [1], [2], [3]],
        }
    for rank, metadata in enumerate(participants):
        _set_checkpoint_participant_topology(
            metadata,
            rank=rank,
            world_size=world_size,
            partitions=partitions,
        )

    merged = merge_muon_checkpoint_metadata(
        participants, trusted_global_ranks=list(range(world_size))
    )

    source_groups = merged["topology"]["source_participant_groups"]
    assert len(source_groups) == world_size
    assert [groups["tp"]["members"] for groups in source_groups] == [
        participants[rank]["topology"]["groups"]["tp"]["members"]
        for rank in range(world_size)
    ]


@pytest.mark.parametrize(
    ("topology", "dense_sizes", "expert_sizes"),
    [
        ("dp2", (1, 1, 2, 1), (1, 1, 2, 1)),
        ("tp2_dp2", (2, 1, 2, 1), (2, 1, 2, 1)),
        ("ep2", (1, 1, 2, 1), (1, 2, 1, 1)),
        ("cp2_dp2", (1, 2, 2, 1), (1, 1, 4, 1)),
    ],
)
def test_muon_checkpoint_metadata_accepts_mcore_rank_generator_memberships(
    monkeypatch: pytest.MonkeyPatch,
    topology: str,
    dense_sizes: tuple[int, int, int, int],
    expert_sizes: tuple[int, int, int, int],
) -> None:
    """MCore 0.17 RankGenerator is the authority for valid group memberships."""
    del topology
    from megatron.core.parallel_state import RankGenerator

    dense_tp, dense_cp, dense_dp, pp = dense_sizes
    expert_tp, expert_ep, expert_dp, expert_pp = expert_sizes
    assert pp == expert_pp
    order = "tp-cp-ep-dp-pp"
    dense = RankGenerator(
        tp=dense_tp,
        ep=1,
        dp=dense_dp,
        pp=pp,
        cp=dense_cp,
        order=order,
    )
    expert = RankGenerator(
        tp=expert_tp,
        ep=expert_ep,
        dp=expert_dp,
        pp=expert_pp,
        cp=1,
        order=order,
    )
    assert dense.world_size == expert.world_size
    assert dense.get_ranks("pp") == expert.get_ranks("pp")
    partitions = {
        "tp": dense.get_ranks("tp"),
        "cp": dense.get_ranks("cp"),
        "dp": dense.get_ranks("dp"),
        "dp_cp": dense.get_ranks("dp-cp"),
        "pp": dense.get_ranks("pp"),
        "expt_tp": expert.get_ranks("tp"),
        "ep": expert.get_ranks("ep"),
        "expt_dp": expert.get_ranks("dp"),
    }
    root = _empty_layerwise_checkpoint_root(monkeypatch)
    participants = [
        copy.deepcopy(root._checkpoint_local_metadata())
        for _ in range(dense.world_size)
    ]
    for rank, metadata in enumerate(participants):
        _set_checkpoint_participant_topology(
            metadata,
            rank=rank,
            world_size=dense.world_size,
            partitions=partitions,
        )

    merged = merge_muon_checkpoint_metadata(
        participants, trusted_global_ranks=list(range(dense.world_size))
    )

    source_groups = merged["topology"]["source_participant_groups"]
    assert {
        name: sorted({tuple(groups[name]["members"]) for groups in source_groups})
        for name in partitions
    } == {
        name: sorted(tuple(group) for group in groups)
        for name, groups in partitions.items()
    }


@pytest.mark.parametrize(
    "case",
    ["expert_size_product", "nonorthogonal_dense_coordinates"],
)
def test_muon_checkpoint_metadata_rejects_noncartesian_topology(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Individually valid group partitions must form one MCore WORLD topology."""
    world_size = 2 if case == "expert_size_product" else 4
    root = _empty_layerwise_checkpoint_root(monkeypatch)
    participants = [
        copy.deepcopy(root._checkpoint_local_metadata()) for _ in range(world_size)
    ]
    if case == "expert_size_product":
        partitions = {
            "ep": [[0, 1]],
            "expt_dp": [[0, 1]],
        }
    else:
        partitions = {
            "tp": [[0, 1], [2, 3]],
            "expt_tp": [[0, 1], [2, 3]],
            "dp": [[0, 1], [2, 3]],
            "dp_cp": [[0, 1], [2, 3]],
            "expt_dp": [[0, 2], [1, 3]],
        }
    for rank, metadata in enumerate(participants):
        _set_checkpoint_participant_topology(
            metadata,
            rank=rank,
            world_size=world_size,
            partitions=partitions,
        )

    with pytest.raises(ValueError, match="topology|Cartesian|coordinate|WORLD"):
        merge_muon_checkpoint_metadata(
            participants, trusted_global_ranks=list(range(world_size))
        )


@pytest.mark.parametrize(
    ("case", "world_size", "partitions"),
    [
        (
            "dense_product_smaller",
            4,
            {
                "dp": [[0, 1], [2, 3]],
                "dp_cp": [[0, 1], [2, 3]],
            },
        ),
        (
            "dense_product_larger",
            2,
            {
                "tp": [[0, 1]],
                "dp": [[0, 1]],
                "dp_cp": [[0, 1]],
                "expt_tp": [[0, 1]],
                "expt_dp": [[0], [1]],
            },
        ),
        (
            "expert_product_smaller",
            4,
            {"expt_dp": [[0, 1], [2, 3]]},
        ),
        (
            "nonorthogonal_expert",
            4,
            {
                "ep": [[0, 1], [2, 3]],
                "expt_dp": [[0, 1], [2, 3]],
            },
        ),
        (
            "dp_cp_not_derived",
            4,
            {
                "tp": [[0, 1], [2, 3]],
                "dp": [[0, 2], [1, 3]],
                "dp_cp": [[0, 1], [2, 3]],
                "expt_tp": [[0, 1], [2, 3]],
                "expt_dp": [[0, 2], [1, 3]],
            },
        ),
    ],
)
def test_muon_checkpoint_metadata_rejects_cartesian_closure_failures(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    world_size: int,
    partitions: dict[str, list[list[int]]],
) -> None:
    """Cross-family topology errors fail even when each family partitions WORLD."""
    del case
    root = _empty_layerwise_checkpoint_root(monkeypatch)
    participants = [
        copy.deepcopy(root._checkpoint_local_metadata()) for _ in range(world_size)
    ]
    for rank, metadata in enumerate(participants):
        _set_checkpoint_participant_topology(
            metadata,
            rank=rank,
            world_size=world_size,
            partitions=partitions,
        )

    with pytest.raises(ValueError, match="Cartesian|WORLD|dp_cp|topology"):
        merge_muon_checkpoint_metadata(
            participants, trusted_global_ranks=list(range(world_size))
        )


@pytest.mark.parametrize(
    ("domain", "coordinate_field"),
    [
        ("dense", "pp"),
        ("dense", "tp"),
        ("expert", "pp"),
        ("expert", "ep"),
        ("expert", "expt_tp"),
    ],
)
def test_muon_checkpoint_metadata_rejects_boolean_parameter_coordinate(
    monkeypatch: pytest.MonkeyPatch, domain: str, coordinate_field: str
) -> None:
    """Boolean values cannot impersonate integer TP/EP coordinates."""
    first, second = _dp2_checkpoint_participants(monkeypatch)
    if domain == "dense":
        partitions = {
            "tp": [[0, 1]],
            "expt_tp": [[0, 1]],
            "dp": [[0], [1]],
            "dp_cp": [[0], [1]],
            "expt_dp": [[0], [1]],
        }
        coordinate = {"pp": 0, "tp": 1}
        owner_group = "dp_cp"
    else:
        partitions = {
            "ep": [[0, 1]],
            "expt_dp": [[0], [1]],
        }
        coordinate = {"pp": 0, "ep": 1, "expt_tp": 0}
        owner_group = "expt_dp"
    coordinate[coordinate_field] = True
    for rank, metadata in enumerate((first, second)):
        _set_checkpoint_participant_topology(
            metadata,
            rank=rank,
            world_size=2,
            partitions=partitions,
        )
    second["leaf_tree"][0]["parameters"] = [
        {
            "name": f"model.{domain}_weight",
            "domain": domain,
            "coordinate": coordinate,
            "shape": [8, 8],
            "dtype": "torch.bfloat16",
            "state_kinds": ["master_param", "momentum_buffer"],
            "source_owner": {
                "global_rank": 1,
                "owner_rank": second["topology"]["groups"][owner_group]["rank"],
                "owner_ordinal": 0,
                "group_index": 0,
                "parameter_index": 0,
                "unit_order": 0,
            },
        }
    ]

    with pytest.raises(ValueError, match="coordinate|integer|bool"):
        merge_muon_checkpoint_metadata([first, second], trusted_global_ranks=[0, 1])


@pytest.mark.parametrize(
    "field",
    [
        "global_rank",
        "owner_rank",
        "owner_ordinal",
        "group_index",
        "parameter_index",
        "unit_order",
    ],
)
def test_muon_checkpoint_metadata_rejects_boolean_source_owner_field(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Every source-owner integer uses exact-int rather than bool-compatible checks."""
    first, second = _dp2_checkpoint_participants(monkeypatch)
    owner = {
        "global_rank": 1,
        "owner_rank": 1,
        "owner_ordinal": 0,
        "group_index": 0,
        "parameter_index": 0,
        "unit_order": 0,
    }
    owner[field] = True
    second["leaf_tree"][0]["parameters"] = [
        {
            "name": "model.weight",
            "domain": "dense",
            "coordinate": {"pp": 0, "tp": 0},
            "shape": [8, 8],
            "dtype": "torch.bfloat16",
            "state_kinds": ["master_param", "momentum_buffer"],
            "source_owner": owner,
        }
    ]

    with pytest.raises(ValueError, match=f"{field}|integer"):
        merge_muon_checkpoint_metadata([first, second], trusted_global_ranks=[0, 1])


def test_muon_checkpoint_metadata_rejects_boolean_parameter_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bool cannot impersonate a parameter shape dimension."""
    first, second = _dp2_checkpoint_participants(monkeypatch)
    second["leaf_tree"][0]["parameters"] = [
        {
            "name": "model.weight",
            "domain": "dense",
            "coordinate": {"pp": 0, "tp": 0},
            "shape": [True, 8],
            "dtype": "torch.bfloat16",
            "state_kinds": ["master_param", "momentum_buffer"],
            "source_owner": {
                "global_rank": 1,
                "owner_rank": 1,
                "owner_ordinal": 0,
                "group_index": 0,
                "parameter_index": 0,
                "unit_order": 0,
            },
        }
    ]

    with pytest.raises(ValueError, match="shape|integer"):
        merge_muon_checkpoint_metadata([first, second], trusted_global_ranks=[0, 1])


def test_muon_checkpoint_metadata_rejects_boolean_leaf_tree_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leaf paths use exact integers, so True cannot alias leaf index one."""
    first, second = _dp2_checkpoint_participants(monkeypatch)
    second["leaf_tree"][1]["tree_path"] = [True]

    with pytest.raises(ValueError, match="tree_path|integer"):
        merge_muon_checkpoint_metadata([first, second], trusted_global_ranks=[0, 1])


def test_muon_checkpoint_outer_rejects_boolean_invariant_group_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python dict equality cannot let True impersonate topology size one."""
    root = _empty_layerwise_checkpoint_root(monkeypatch)
    checkpoint = root._checkpoint_metadata()
    checkpoint["topology"]["invariant_group_sizes"]["tp"] = True

    with pytest.raises(ValueError, match="integer|topology"):
        root.validate_managed_checkpoint_outer_state({"metadata": checkpoint})


@pytest.mark.parametrize("field", ["global_shape", "chunk_offset", "chunk_size"])
def test_muon_checkpoint_source_manifest_rejects_boolean_integer_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str
) -> None:
    """DCP shape, offset, and chunk-size controls reject bool before slab copy."""
    from torch.distributed.checkpoint.metadata import (
        ChunkStorageMetadata,
        Metadata,
        TensorProperties,
        TensorStorageMetadata,
    )

    prefix = "optimizer.gpu_staged_muon.v2.leaf_0.dense.coord_x.param_y"
    manifest = {
        f"{prefix}.master_param": ((16,), "torch.float32"),
        f"{prefix}.momentum_buffer": ((16,), "torch.float32"),
    }
    entries = {}
    for key in manifest:
        chunk = ChunkStorageMetadata(torch.Size([0]), torch.Size([16]))
        entry = TensorStorageMetadata(
            TensorProperties(dtype=torch.float32), torch.Size([16]), [chunk]
        )
        entries[key] = entry
    target = entries[f"{prefix}.master_param"]
    if field == "global_shape":
        target.size = (True,)
    elif field == "chunk_offset":
        target.chunks[0].offsets = (True,)
    else:
        target.chunks[0].sizes = (True,)
    monkeypatch.setattr(
        "torch.distributed.checkpoint.FileSystemReader.read_metadata",
        lambda _reader: Metadata(entries),
    )

    with pytest.raises(ValueError, match="integer"):
        validate_managed_optimizer_source_tensor_metadata(str(tmp_path), manifest)


def test_muon_checkpoint_source_manifest_rejects_split_matrix_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """DP reshard accepts one complete matrix payload, never matrix chunks."""
    from torch.distributed.checkpoint.metadata import (
        ChunkStorageMetadata,
        Metadata,
        TensorProperties,
        TensorStorageMetadata,
    )

    prefix = "optimizer.gpu_staged_muon.v2.leaf_0.dense.coord_x.param_y"
    manifest = {
        f"{prefix}.master_param": ((16,), "torch.float32"),
        f"{prefix}.momentum_buffer": ((16,), "torch.float32"),
    }
    entries = {
        key: TensorStorageMetadata(
            TensorProperties(dtype=torch.float32),
            torch.Size([16]),
            [
                ChunkStorageMetadata(torch.Size([0]), torch.Size([8])),
                ChunkStorageMetadata(torch.Size([8]), torch.Size([8])),
            ],
        )
        for key in manifest
    }
    monkeypatch.setattr(
        "torch.distributed.checkpoint.FileSystemReader.read_metadata",
        lambda _reader: Metadata(entries),
    )

    with pytest.raises(ValueError, match="exactly one complete payload"):
        validate_managed_optimizer_source_tensor_metadata(str(tmp_path), manifest)


@pytest.mark.parametrize("corruption", ["v1", "missing", "extra"])
def test_muon_checkpoint_source_manifest_rejects_namespace_corruption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, corruption: str
) -> None:
    """Legacy, missing, and extra Muon states fail before a slab is exposed."""
    from torch.distributed.checkpoint.metadata import (
        ChunkStorageMetadata,
        Metadata,
        TensorProperties,
        TensorStorageMetadata,
    )

    v2_key = "optimizer.gpu_staged_muon.v2.leaf_0.dense.coord_x.param_y.master_param"
    manifest = {v2_key: ((16,), "torch.float32")}

    def tensor_metadata() -> TensorStorageMetadata:
        return TensorStorageMetadata(
            TensorProperties(dtype=torch.float32),
            torch.Size([16]),
            [ChunkStorageMetadata(torch.Size([0]), torch.Size([16]))],
        )

    if corruption == "v1":
        entries = {
            "optimizer.gpu_staged_muon.rank_0.leaf_0.master_param": tensor_metadata()
        }
    elif corruption == "missing":
        entries = {}
    else:
        entries = {
            v2_key: tensor_metadata(),
            v2_key.replace("master_param", "momentum_buffer"): tensor_metadata(),
        }
    monkeypatch.setattr(
        "torch.distributed.checkpoint.FileSystemReader.read_metadata",
        lambda _reader: Metadata(entries),
    )

    with pytest.raises(KeyError, match="source tensor key mismatch"):
        validate_managed_optimizer_source_tensor_metadata(str(tmp_path), manifest)


@pytest.mark.parametrize(
    "chunks",
    [
        [(0, 8), (8, 8)],
        [(0, 7), (8, 8)],
        [(0, 9), (8, 8)],
    ],
    ids=["split", "gap", "overlap"],
)
def test_muon_checkpoint_source_manifest_rejects_noncanonical_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    chunks: list[tuple[int, int]],
) -> None:
    """A logical Muon matrix is one complete payload, never DP chunks."""
    from torch.distributed.checkpoint.metadata import (
        ChunkStorageMetadata,
        Metadata,
        TensorProperties,
        TensorStorageMetadata,
    )

    key = "optimizer.gpu_staged_muon.v2.leaf_0.dense.coord_x.param_y.master_param"
    entry = TensorStorageMetadata(
        TensorProperties(dtype=torch.float32),
        torch.Size([16]),
        [
            ChunkStorageMetadata(torch.Size([offset]), torch.Size([size]))
            for offset, size in chunks
        ],
    )
    monkeypatch.setattr(
        "torch.distributed.checkpoint.FileSystemReader.read_metadata",
        lambda _reader: Metadata({key: entry}),
    )

    with pytest.raises(ValueError, match="exactly one complete payload"):
        validate_managed_optimizer_source_tensor_metadata(
            str(tmp_path), {key: ((16,), "torch.float32")}
        )


@pytest.mark.parametrize("field", ["tp", "cp", "ep", "expt_tp", "pp"])
def test_muon_checkpoint_invariant_topology_rejects_before_leaf_mutation(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Every non-ownership parallel dimension remains fixed across DP reshard."""
    root = _empty_layerwise_checkpoint_root(monkeypatch)
    leaf_states = tuple(
        (
            leaf.optimizer.checkpoint_lifecycle,
            leaf.optimizer.checkpoint_load_attempt_token,
        )
        for leaf in root.chained_optimizers
    )
    checkpoint = root._checkpoint_metadata()
    checkpoint["topology"]["invariant_group_sizes"][field] += 1

    with pytest.raises(ValueError, match="invariant topology mismatch"):
        root.validate_managed_checkpoint_outer_state({"metadata": checkpoint})

    assert (
        tuple(
            (
                leaf.optimizer.checkpoint_lifecycle,
                leaf.optimizer.checkpoint_load_attempt_token,
            )
            for leaf in root.chained_optimizers
        )
        == leaf_states
    )


@pytest.mark.parametrize("corruption", ["v1", "algorithm", "leaf_tree"])
def test_muon_checkpoint_global_schema_rejects_before_leaf_mutation(
    monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    """Schema, algorithm, and leaf-tree drift are mutation-free failures."""
    root = _empty_layerwise_checkpoint_root(monkeypatch)
    leaf_states = tuple(
        (
            leaf.optimizer.checkpoint_lifecycle,
            leaf.optimizer.checkpoint_load_attempt_token,
        )
        for leaf in root.chained_optimizers
    )
    checkpoint = root._checkpoint_metadata()
    if corruption == "v1":
        checkpoint["schema_version"] = 1
    elif corruption == "algorithm":
        checkpoint["algorithm"]["momentum"] = 0.5
    else:
        checkpoint["leaf_tree"][0]["kind"] = "scalar_adamw"

    with pytest.raises(ValueError, match="mismatch"):
        root.validate_managed_checkpoint_outer_state({"metadata": checkpoint})

    assert (
        tuple(
            (
                leaf.optimizer.checkpoint_lifecycle,
                leaf.optimizer.checkpoint_load_attempt_token,
            )
            for leaf in root.chained_optimizers
        )
        == leaf_states
    )


def _gpu_muon(tmp_path) -> tuple[torch.nn.Parameter, GPUStagedMuon]:
    param = torch.nn.Parameter(
        torch.arange(16, device="cuda", dtype=torch.bfloat16).view(4, 4)
    )
    optimizer = GPUStagedMuon(
        [{"params": [param], "lr": 0.1, "momentum": 0.9, "weight_decay": 0.01}],
        staged_config=GPUStagedMuonConfig(
            buffer_count=1,
            slot_size_mb=1,
            checkpoint_snapshot_root=str(tmp_path),
            checkpoint_snapshot_chunk_mb=1,
        ),
        orthogonalize=_identity_orthogonalize,
        matmul_precision=nullcontext,
        nesterov=False,
        weight_decay_method="decoupled",
    )
    optimizer.bind_owned_params(optimizer.param_groups)
    return param, optimizer


def _mixed_layerwise_checkpoint_root(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> tuple[object, GPUStagedMuon, GPUStagedAdamW]:
    real_version = __import__("importlib.metadata", fromlist=["version"]).version
    monkeypatch.setattr(
        "importlib.metadata.version",
        lambda package: "0.3.0"
        if package == "emerging-optimizers"
        else real_version(package),
    )
    real_get_process_group_ranks = torch.distributed.get_process_group_ranks

    def get_process_group_ranks(group):
        if hasattr(group, "members"):
            return list(group.members)
        return real_get_process_group_ranks(group)

    monkeypatch.setattr(
        torch.distributed,
        "get_process_group_ranks",
        get_process_group_ranks,
    )
    matrix, muon = _gpu_muon(tmp_path)
    scalar = torch.nn.Parameter(torch.arange(4, device="cuda", dtype=torch.bfloat16))
    adam = GPUStagedAdamW(
        [{"params": [scalar], "lr": 0.02, "weight_decay": 0.03}],
        staged_config=GPUStagedAdamWConfig(
            buffer_count=1,
            bucket_size_mb=1,
            checkpoint_snapshot_root=str(tmp_path),
            checkpoint_snapshot_chunk_mb=1,
        ),
    )
    adam.bind_owned_params(adam.param_groups)
    adam.optimizer_kind = "scalar_adamw"
    group = _fake_group(0)
    config = SimpleNamespace(log_num_zeros_in_grad=False)
    leaves = [
        SimpleNamespace(optimizer=base, config=config, param_groups=base.param_groups)
        for base in (muon, adam)
    ]
    official = SimpleNamespace(
        pg_collection=SimpleNamespace(
            tp=group,
            expt_tp=group,
            dp=group,
            dp_cp=group,
            expt_dp=group,
            pp=group,
            cp=group,
            ep=group,
        ),
        dp_cp_params_list=[[matrix, scalar]],
        expt_dp_params_list=None,
    )
    root = _make_staged_layerwise_class()(official, leaves)
    root.configure_managed_checkpoint_schema(
        {matrix: "model.matrix", scalar: "model.scalar"},
        algorithm=_checkpoint_algorithm(),
    )
    return root, muon, adam


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("corruption", ["missing", "extra", "dtype", "shape"])
def test_muon_checkpoint_schema_rejects_before_slab_or_group_mutation(
    tmp_path, corruption: str
) -> None:
    """Every corrupt Muon state form fails before authoritative CPU mutation."""
    _, optimizer = _gpu_muon(tmp_path)
    state = optimizer.state_dict()
    state = {
        "state": {
            state_id: {key: value.clone() for key, value in values.items()}
            for state_id, values in state["state"].items()
        },
        "param_groups": [dict(group) for group in state["param_groups"]],
    }
    parameter_state = state["state"][0]
    if corruption == "missing":
        parameter_state.pop("momentum_buffer")
    elif corruption == "extra":
        parameter_state["unexpected"] = torch.zeros(1)
    elif corruption == "dtype":
        parameter_state["momentum_buffer"] = parameter_state["momentum_buffer"].to(
            torch.float64
        )
    else:
        parameter_state["momentum_buffer"] = torch.zeros(15, dtype=torch.float32)

    optimizer.cpu_slabs.master.fill_(17.0)
    optimizer.cpu_slabs.momentum.fill_(19.0)
    before_master = optimizer.cpu_slabs.master.clone()
    before_momentum = optimizer.cpu_slabs.momentum.clone()
    before_groups = [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]
    optimizer.begin_checkpoint_load()
    with pytest.raises((TypeError, ValueError)):
        optimizer.load_state_dict(state)
    torch.testing.assert_close(
        optimizer.cpu_slabs.master, before_master, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        optimizer.cpu_slabs.momentum, before_momentum, rtol=0.0, atol=0.0
    )
    assert [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ] == before_groups
    optimizer.abort_checkpoint_load(RuntimeError("corrupt checkpoint"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_muon_checkpoint_disk_rollback_restores_slabs_and_metadata(tmp_path) -> None:
    """Muon rollback restores master, momentum, and group metadata from disk."""
    _, optimizer = _gpu_muon(tmp_path)
    before_master = optimizer.cpu_slabs.master.clone()
    before_momentum = optimizer.cpu_slabs.momentum.clone()
    before_lr = optimizer.param_groups[0]["lr"]

    optimizer.begin_checkpoint_load()
    optimizer.cpu_slabs.master.fill_(23.0)
    optimizer.cpu_slabs.momentum.fill_(29.0)
    optimizer.param_groups[0]["lr"] = 0.75
    optimizer.abort_checkpoint_load(RuntimeError("injected DCP failure"))

    torch.testing.assert_close(
        optimizer.cpu_slabs.master, before_master, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        optimizer.cpu_slabs.momentum, before_momentum, rtol=0.0, atol=0.0
    )
    assert optimizer.param_groups[0]["lr"] == before_lr
    assert optimizer.checkpoint_lifecycle == "CLEAN"
    assert optimizer.residency == "CPU_RESIDENT"
    assert optimizer.cuda_state_numel == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_muon_mixed_leaf_checkpoint_uses_one_global_transaction(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Muon and scalar AdamW leaves validate before one coordinated commit."""
    root, muon, adam = _mixed_layerwise_checkpoint_root(tmp_path, monkeypatch)
    saved = root.state_dict()
    saved = {
        "metadata": copy.deepcopy(saved["metadata"]),
        "state": {key: value.clone() for key, value in saved["state"].items()},
    }
    assert muon.cpu_slabs is not None
    assert adam.cpu_slabs is not None
    expected_muon = muon.cpu_slabs.master.clone()
    expected_momentum = muon.cpu_slabs.momentum.clone()
    expected_adam = adam.cpu_slabs.master.clone()
    expected_exp_avg = adam.cpu_slabs.exp_avg.clone()
    expected_exp_avg_sq = adam.cpu_slabs.exp_avg_sq.clone()

    muon.cpu_slabs.master.fill_(31.0)
    muon.cpu_slabs.momentum.fill_(37.0)
    adam.cpu_slabs.master.fill_(41.0)
    adam.cpu_slabs.exp_avg.fill_(43.0)
    adam.cpu_slabs.exp_avg_sq.fill_(47.0)
    transaction = begin_managed_checkpoint_load(root)
    root.load_state_dict(saved)
    prepare_managed_checkpoint_load(transaction)
    commit_managed_checkpoint_load(transaction)

    for actual, expected in (
        (muon.cpu_slabs.master, expected_muon),
        (muon.cpu_slabs.momentum, expected_momentum),
        (adam.cpu_slabs.master, expected_adam),
        (adam.cpu_slabs.exp_avg, expected_exp_avg),
        (adam.cpu_slabs.exp_avg_sq, expected_exp_avg_sq),
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert muon.checkpoint_lifecycle == "CLEAN"
    assert adam.checkpoint_lifecycle == "CLEAN"
    assert root.residency == "CPU_RESIDENT"
    assert root.cuda_state_numel == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "corruption", ["missing", "missing_container", "extra", "dtype", "shape"]
)
def test_muon_root_tensor_schema_rejects_before_any_leaf_mutation(
    tmp_path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    """The flat multi-leaf payload is validated completely before apply."""
    root, muon, adam = _mixed_layerwise_checkpoint_root(tmp_path, monkeypatch)
    saved = root.state_dict()
    loaded = {
        "metadata": copy.deepcopy(saved["metadata"]),
        "state": {key: value.clone() for key, value in saved["state"].items()},
    }
    target_key = next(
        key for key in loaded["state"] if key.endswith(".momentum_buffer")
    )
    if corruption == "missing":
        loaded["state"].pop(target_key)
    elif corruption == "missing_container":
        loaded.pop("state")
    elif corruption == "extra":
        loaded["state"][f"{target_key}.unexpected"] = torch.zeros(
            1, dtype=torch.float32
        )
    elif corruption == "dtype":
        loaded["state"][target_key] = loaded["state"][target_key].to(torch.float64)
    else:
        loaded["state"][target_key] = torch.zeros(3, dtype=torch.float32)

    slabs = (
        muon.cpu_slabs.master,
        muon.cpu_slabs.momentum,
        adam.cpu_slabs.master,
        adam.cpu_slabs.exp_avg,
        adam.cpu_slabs.exp_avg_sq,
    )
    before = tuple(slab.clone() for slab in slabs)
    groups = [
        {key: value for key, value in group.items() if key != "params"}
        for leaf in root.chained_optimizers
        for group in leaf.optimizer.param_groups
    ]
    transaction = begin_managed_checkpoint_load(root)
    with pytest.raises((KeyError, TypeError, ValueError)):
        root.load_state_dict(loaded)
    for slab, expected in zip(slabs, before, strict=True):
        torch.testing.assert_close(slab, expected, rtol=0.0, atol=0.0)
    assert [
        {key: value for key, value in group.items() if key != "params"}
        for leaf in root.chained_optimizers
        for group in leaf.optimizer.param_groups
    ] == groups
    abort_managed_checkpoint_load(transaction, RuntimeError("schema rejection"))


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_muon_fixed_topology_real_torch_dist_dcp_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real torch_dist save/load writes directly from and into CPU slabs."""
    import torch.distributed as dist
    from megatron.core import dist_checkpointing

    if dist.is_initialized():
        pytest.skip("test requires ownership of the process-global world=1 group")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'muon_checkpoint_pg'}",
        rank=0,
        world_size=1,
    )
    try:
        root, muon, adam = _mixed_layerwise_checkpoint_root(tmp_path, monkeypatch)
        root.bind_managed_checkpoint_process_group(dist.group.WORLD)
        assert muon.cpu_slabs is not None
        assert adam.cpu_slabs is not None
        slabs = (
            muon.cpu_slabs.master,
            muon.cpu_slabs.momentum,
            adam.cpu_slabs.master,
            adam.cpu_slabs.exp_avg,
            adam.cpu_slabs.exp_avg_sq,
        )
        expected = tuple(slab.clone() for slab in slabs)
        pointers = tuple(slab.data_ptr() for slab in slabs)
        checkpoint_dir = tmp_path / "muon_torch_dist"
        checkpoint_dir.mkdir()
        sharded = root.sharded_state_dict({})
        dist_checkpointing.save(
            {"optimizer": sharded},
            str(checkpoint_dir),
            async_sharded_save=False,
        )
        manifest = build_managed_optimizer_tensor_manifest(sharded)
        validate_managed_optimizer_source_tensor_metadata(str(checkpoint_dir), manifest)

        for index, slab in enumerate(slabs):
            slab.fill_(101.0 + index)
        transaction = begin_managed_checkpoint_load(root)
        template = {"optimizer": root.sharded_state_dict({})}
        loaded = dist_checkpointing.load(template, str(checkpoint_dir))
        root.load_state_dict(loaded["optimizer"])
        prepare_managed_checkpoint_load(transaction)
        commit_managed_checkpoint_load(transaction)

        for slab, expected_value in zip(slabs, expected, strict=True):
            torch.testing.assert_close(slab, expected_value, rtol=0.0, atol=0.0)
        assert tuple(slab.data_ptr() for slab in slabs) == pointers
        assert all(slab.is_pinned() for slab in slabs)
        assert root.residency == "CPU_RESIDENT"
        assert root.cuda_state_numel == 0
        assert sum(path.stat().st_size for path in checkpoint_dir.rglob("*")) > 0
    finally:
        dist.destroy_process_group()


def _run_muon_checkpoint_topology(
    tmp_path: Path, *, topology: str, world_size: int
) -> list[dict[str, object]]:
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"real staged Muon {topology} checkpoint requires {world_size} CUDA devices"
        )
    if importlib.util.find_spec("emerging_optimizers") is None:
        pytest.skip("MCore's optional emerging-optimizers backend is unavailable")

    output_dir = tmp_path / f"muon-checkpoint-{topology}"
    env = os.environ.copy()
    env["ACCEPTANCE_OUTPUT_DIR"] = str(output_dir)
    env["MUON_CHECKPOINT_TOPOLOGY"] = topology
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={world_size}",
        "--nnodes=1",
        "--master_addr=localhost",
        f"--master_port={find_free_ports(1)[0]}",
        "tests/torchrun/run_gpu_staged_muon_checkpoint.py",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    results = [
        json.loads((output_dir / f"rank_{rank}.json").read_text())
        for rank in range(world_size)
    ]
    assert {result["world_size"] for result in results} == {world_size}
    assert {result["topology"] for result in results} == {topology}
    assert {result["residency"] for result in results} == {"CPU_RESIDENT"}
    assert {result["cuda_state_numel"] for result in results} == {0}
    assert {result["continued_steps"] for result in results} == {3}
    assert all(result["checkpoint_bytes"] > 0 for result in results)
    assert all(result["load_cuda_peak_bytes"] >= 0 for result in results)
    return results


def _run_muon_checkpoint_reshard(
    tmp_path: Path,
    *,
    source_dp: int,
    target_dp: int,
    model_kind: str = "mixed",
    fault_cycles: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    required_gpus = max(source_dp, target_dp)
    if torch.cuda.device_count() < required_gpus:
        pytest.skip(
            f"real Muon DP={source_dp}->{target_dp} reshard requires "
            f"{required_gpus} CUDA devices"
        )
    if importlib.util.find_spec("emerging_optimizers") is None:
        pytest.skip("isolated emerging-optimizers 0.3.0 target is unavailable")

    output_dir = tmp_path / f"muon-reshard-dp{source_dp}-dp{target_dp}"
    base_env = os.environ.copy()
    base_env["ACCEPTANCE_OUTPUT_DIR"] = str(output_dir)
    base_env["MUON_CHECKPOINT_TOPOLOGY"] = "dp_reshard"
    base_env["MUON_CHECKPOINT_MODEL"] = model_kind
    if fault_cycles:
        base_env["MUON_RESHARD_FAULT_CYCLES"] = "1"

    def run_phase(phase: str, world_size: int) -> list[dict[str, object]]:
        env = dict(base_env)
        env["MUON_CHECKPOINT_PHASE"] = phase
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={world_size}",
            "--nnodes=1",
            "--master_addr=localhost",
            f"--master_port={find_free_ports(1)[0]}",
            "tests/torchrun/run_gpu_staged_muon_checkpoint.py",
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return [
            json.loads((output_dir / f"{phase}_rank_{rank}.json").read_text())
            for rank in range(world_size)
        ]

    source = run_phase("save", source_dp)
    target = run_phase("load", target_dp)
    for phase, results, world_size in (
        ("save", source, source_dp),
        ("load", target, target_dp),
    ):
        assert {result["phase"] for result in results} == {phase}
        assert {result["world_size"] for result in results} == {world_size}
        assert {result["model_kind"] for result in results} == {model_kind}
        assert {result["residency"] for result in results} == {"CPU_RESIDENT"}
        assert {result["cuda_state_numel"] for result in results} == {0}
        assert all(result["checkpoint_bytes"] > 0 for result in results)
    assert sum(len(result["owned_parameters"]) for result in source) == sum(
        len(result["owned_parameters"]) for result in target
    )
    assert all(result["load_cuda_peak_bytes"] >= 0 for result in target)
    if fault_cycles:
        traces = []
        for rank in range(target_dp):
            phases = (output_dir / f"rank_{rank}.phases").read_text().splitlines()
            start = phases.index("reshard_load_manifest")
            traces.append(phases[start:])
        assert all(trace == traces[0] for trace in traces[1:])
        assert traces[0][1:4] == [
            "fault_local_validate_rolled_back",
            "fault_rollback_recovered",
            "fault_cleanup_after_effect_reconciled",
        ]
    return source, target


def _run_muon_manager_checkpoint_reshard(
    tmp_path: Path, *, inject_partial_dcp_fault: bool, fault_rank: int = 1
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if torch.cuda.device_count() < 2:
        pytest.skip("public manager Muon DP=1->2 reshard requires two CUDA devices")
    if importlib.util.find_spec("emerging_optimizers") is None:
        pytest.skip("isolated emerging-optimizers 0.3.0 target is unavailable")

    suffix = f"fault-rank{fault_rank}" if inject_partial_dcp_fault else "roundtrip"
    output_dir = tmp_path / f"muon-manager-reshard-{suffix}"
    base_env = os.environ.copy()
    base_env["ACCEPTANCE_OUTPUT_DIR"] = str(output_dir)
    base_env["MUON_CHECKPOINT_TOPOLOGY"] = "manager_reshard"
    if inject_partial_dcp_fault:
        base_env["MUON_MANAGER_DCP_FAULT"] = "1"
        base_env["MUON_MANAGER_DCP_FAULT_RANK"] = str(fault_rank)

    def run_phase(phase: str, world_size: int) -> list[dict[str, object]]:
        env = dict(base_env)
        env["MUON_CHECKPOINT_PHASE"] = phase
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={world_size}",
            "--nnodes=1",
            "--master_addr=localhost",
            f"--master_port={find_free_ports(1)[0]}",
            "tests/torchrun/run_gpu_staged_muon_checkpoint.py",
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=420,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return [
            json.loads((output_dir / f"manager_{phase}_rank_{rank}.json").read_text())
            for rank in range(world_size)
        ]

    source = run_phase("save", 1)
    target = run_phase("load", 2)
    for phase, results, world_size in (
        ("save", source, 1),
        ("load", target, 2),
    ):
        assert {result["phase"] for result in results} == {phase}
        assert {result["world_size"] for result in results} == {world_size}
        assert {result["residency"] for result in results} == {"CPU_RESIDENT"}
        assert {result["cuda_state_numel"] for result in results} == {0}
        assert {result["rollback_disk_final_bytes"] for result in results} == {0}
        assert all(result["checkpoint_bytes"] > 0 for result in results)
    assert {result["max_model_error"] for result in target} == {0.0}
    assert all(
        all(error == 0.0 for error in result["max_optimizer_errors"].values())
        for result in target
    )
    if inject_partial_dcp_fault:
        assert {result["first_load_failed"] for result in target} == {True}
        assert {result["fault_rank"] for result in target} == {fault_rank}
        assert {result["replacement_configure_failed"] for result in target} == {True}
        assert {result["second_cycle"] for result in target} == {True}
        assert target[fault_rank]["partial_changed"]
        assert target[fault_rank]["partial_unchanged"]
        assert {result["recovery_prefix_verified"] for result in target} == {True}
        assert target[0]["phase_trace"] == target[1]["phase_trace"]
        phases = target[0]["phase_trace"]
        assert "dcp_load" in phases
        assert "abort" in phases
        assert "recovery_required" in phases
        assert "recovery_rollback" in phases
        assert "recovery_acknowledge" in phases
    else:
        assert {result["first_load_failed"] for result in target} == {False}
    return source, target


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_muon_real_dcp_reshard_dp1_to_dp2(tmp_path: Path) -> None:
    """A complete Muon/AdamW payload migrates to official DP=2 ownership."""
    source, target = _run_muon_checkpoint_reshard(tmp_path, source_dp=1, target_dp=2)
    assert len(source[0]["owned_parameters"]) > 1
    source_owner = {
        parameter: result["rank"]
        for result in source
        for parameter in result["owned_parameters"]
    }
    target_owner = {
        parameter: result["rank"]
        for result in target
        for parameter in result["owned_parameters"]
    }
    assert set(source_owner) == set(target_owner)
    assert any(source_owner[name] != target_owner[name] for name in source_owner)


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_muon_public_manager_reshard_dp1_to_dp2(tmp_path: Path) -> None:
    """Separate manager processes restore full state and continue exactly."""
    _run_muon_manager_checkpoint_reshard(tmp_path, inject_partial_dcp_fault=False)


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.parametrize("fault_rank", [0, 1])
def test_muon_public_manager_partial_dcp_write_recovers(
    tmp_path: Path, fault_rank: int
) -> None:
    """A one-rank DCP after-effect reuses only pending rollback authority."""
    _run_muon_manager_checkpoint_reshard(
        tmp_path, inject_partial_dcp_fault=True, fault_rank=fault_rank
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_muon_real_dcp_reshard_dp2_to_dp4(tmp_path: Path) -> None:
    """DP=4 empty-owner ranks load no phantom tensors or slabs."""
    _, target = _run_muon_checkpoint_reshard(
        tmp_path, source_dp=2, target_dp=4, model_kind="pure_muon"
    )
    assert any(not result["owned_parameters"] for result in target)


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_muon_real_dcp_reshard_dp2_to_dp1(tmp_path: Path) -> None:
    """DP=2 owners consolidate into a fresh DP=1 mixed optimizer instance."""
    source, target = _run_muon_checkpoint_reshard(tmp_path, source_dp=2, target_dp=1)
    source_owner = {
        parameter: result["rank"]
        for result in source
        for parameter in result["owned_parameters"]
    }
    target_owner = {
        parameter: result["rank"]
        for result in target
        for parameter in result["owned_parameters"]
    }
    assert set(source_owner) == set(target_owner)
    assert set(target_owner.values()) == {0}
    assert any(source_owner[name] != target_owner[name] for name in source_owner)


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_muon_real_dcp_reshard_ep2_expert_dp1_to_dp2(tmp_path: Path) -> None:
    """Dense and expert payloads migrate to new DP and expert-DP owners."""
    source, target = _run_muon_checkpoint_reshard(
        tmp_path,
        source_dp=2,
        target_dp=4,
        model_kind="dense_expert",
    )
    assert {len(result["groups"]["ep"]) for result in source + target} == {2}
    assert {len(result["groups"]["expt_dp"]) for result in source} == {1}
    assert {len(result["groups"]["expt_dp"]) for result in target} == {2}
    source_owner = {
        parameter: result["rank"]
        for result in source
        for parameter in result["owned_parameters"]
    }
    target_owner = {
        parameter: result["rank"]
        for result in target
        for parameter in result["owned_parameters"]
    }
    assert set(source_owner) == set(target_owner)
    for domain in ("dense", "expert"):
        identities = [name for name in source_owner if name.startswith(f"{domain}|")]
        assert identities
        assert any(source_owner[name] != target_owner[name] for name in identities)
        assert any(target_owner[name] >= 2 for name in identities)


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_muon_real_dcp_reshard_new_owner_fault_cycles(tmp_path: Path) -> None:
    """A new DP=2 owner rolls back, recovers, cleans, and reloads consistently."""
    source, target = _run_muon_checkpoint_reshard(
        tmp_path,
        source_dp=1,
        target_dp=2,
        fault_cycles=True,
    )
    source_owner = {
        parameter: result["rank"]
        for result in source
        for parameter in result["owned_parameters"]
    }
    target_owner = {
        parameter: result["rank"]
        for result in target
        for parameter in result["owned_parameters"]
    }
    assert any(
        source_owner[parameter] == 0 and target_owner[parameter] == 1
        for parameter in source_owner
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_muon_manager_recovery_fault_matrix_has_dp2_consensus(
    tmp_path: Path,
) -> None:
    """Public manager recovery retains authority across rank-local empty faults."""
    if torch.cuda.device_count() < 2:
        pytest.skip("real staged Muon manager recovery requires two CUDA devices")

    output_dir = tmp_path / "muon-manager-recovery-dp2"
    env = os.environ.copy()
    env["ACCEPTANCE_OUTPUT_DIR"] = str(output_dir)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--nnodes=1",
        "--master_addr=localhost",
        f"--master_port={find_free_ports(1)[0]}",
        "tests/torchrun/run_gpu_staged_muon_manager_recovery.py",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    results = [
        json.loads((output_dir / f"rank_{rank}.json").read_text()) for rank in range(2)
    ]
    assert [result["rank"] for result in results] == [0, 1]
    assert all(len(result["cases"]) == 20 for result in results)
    for left, right in zip(results[0]["cases"], results[1]["cases"], strict=True):
        assert left["case"] == right["case"]
        assert left["phase_trace"] == right["phase_trace"]
        assert left["lifecycle"] == right["lifecycle"] == "CLEAN"


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_muon_checkpoint_metadata_authority_fault_has_dp2_consensus(
    tmp_path: Path,
) -> None:
    """A rank-local forged participant is rejected before DCP or slab mutation."""
    if torch.cuda.device_count() < 2:
        pytest.skip("real staged Muon metadata authority requires two CUDA devices")
    if importlib.util.find_spec("emerging_optimizers") is None:
        pytest.skip("isolated emerging-optimizers 0.3.0 target is unavailable")

    output_dir = tmp_path / "muon-metadata-authority-dp2"
    env = os.environ.copy()
    env["ACCEPTANCE_OUTPUT_DIR"] = str(output_dir)
    env["MUON_CHECKPOINT_TOPOLOGY"] = "dp2"
    env["MUON_CHECKPOINT_PHASE"] = "metadata_fault"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--nnodes=1",
        "--master_addr=localhost",
        f"--master_port={find_free_ports(1)[0]}",
        "tests/torchrun/run_gpu_staged_muon_checkpoint.py",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    results = [
        json.loads((output_dir / f"metadata_fault_rank_{rank}.json").read_text())
        for rank in range(2)
    ]
    assert len({result["error"] for result in results}) == 1
    assert "coordinate" in results[0]["error"]
    assert {result["residency"] for result in results} == {"CPU_RESIDENT"}
    assert {result["cuda_state_numel"] for result in results} == {0}
    assert {result["snapshot_files"] for result in results} == {0}
    assert {result["dcp_files"] for result in results} == {0}
    assert {result["nccl_health"] for result in results} == {3.0}
    assert {result["gloo_health"] for result in results} == {3}


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_muon_fixed_topology_real_dp2_checkpoint_resume(tmp_path: Path) -> None:
    """Real DP=2 owners resume in a new instance and continue identically."""
    _run_muon_checkpoint_topology(tmp_path, topology="dp2", world_size=2)


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_muon_fixed_topology_dp2_empty_owner_checkpoint_resume(
    tmp_path: Path,
) -> None:
    """The rank-local empty owner participates in every global checkpoint phase."""
    results = _run_muon_checkpoint_topology(
        tmp_path, topology="dp2_empty_owner", world_size=2
    )
    assert sorted(len(result["owned_parameters"]) for result in results) == [0, 1]


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_muon_fixed_topology_real_tp2_dp2_checkpoint_resume(tmp_path: Path) -> None:
    """A fixed TP=2/DP=2 topology restores owner-local state without reshard."""
    _run_muon_checkpoint_topology(tmp_path, topology="tp2_dp2", world_size=4)
