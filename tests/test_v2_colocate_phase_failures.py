# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from areal.trainer.rl_trainer import PPOTrainer
from areal.v2.training_service.controller.controller import GatewayTrainController


def _controller(
    events,
    *,
    fail_onload_tag=None,
    fail_continue=False,
    fail_actor_release=False,
):
    controller = object.__new__(GatewayTrainController)
    controller._colocate = True

    def onload(tags):
        tag = tags[0]
        events.append(("onload", tag))
        if tag == fail_onload_tag:
            raise RuntimeError(f"cannot restore {tag}")

    def continue_generation():
        events.append("continue")
        if fail_continue:
            raise RuntimeError("cannot continue")

    controller.rollout = SimpleNamespace(
        pause=lambda: events.append("pause"),
        pause_generation_sync=lambda: events.append("drain"),
        offload=lambda tags: events.append(("offload", tags[0])),
        onload=onload,
        continue_generation=continue_generation,
        resume=lambda: events.append("resume"),
        pause_generation=lambda: events.append("pause_generation"),
    )

    def actor_memory_op(op, tags):
        events.append(("actor", op, tuple(tags)))
        if fail_actor_release and op == "release_memory":
            raise RuntimeError("cannot release actor")

    controller._broadcast_awex_memory_op = actor_memory_op
    return controller


def test_failure_before_weight_update_restores_all_rollout_memory():
    events = []
    controller = _controller(events)

    with pytest.raises(RuntimeError, match="training failed"):
        with controller.train_phase():
            raise RuntimeError("training failed")

    assert ("onload", "weights") in events
    assert ("onload", "cuda_graph") in events
    assert ("onload", "kv_cache") in events
    assert "continue" not in events
    assert "resume" not in events


def test_exit_attempts_remaining_cleanup_after_one_restore_fails():
    events = []
    controller = _controller(events, fail_onload_tag="cuda_graph")
    controller.enter_train_phase()
    controller._colocate_update_started = True
    controller._colocate_update_succeeded = True

    with pytest.raises(ExceptionGroup, match="cleanup"):
        controller.exit_train_phase()

    assert ("onload", "cuda_graph") in events
    assert ("onload", "kv_cache") in events
    assert "continue" not in events
    assert "resume" not in events


def test_resume_is_gated_on_generation_continue_success():
    events = []
    controller = _controller(events, fail_continue=True)
    controller.enter_train_phase()
    controller._colocate_update_started = True
    controller._colocate_transfer_succeeded = True
    controller._colocate_update_succeeded = True

    with pytest.raises(RuntimeError, match="cannot continue"):
        controller.exit_train_phase()

    assert "continue" in events
    assert "resume" not in events


def test_inference_restore_is_gated_on_actor_release_success():
    events = []
    controller = _controller(events, fail_actor_release=True)
    controller.enter_train_phase()
    controller._colocate_update_started = True
    controller._colocate_transfer_succeeded = True
    controller._colocate_update_succeeded = True

    with pytest.raises(ExceptionGroup, match="cleanup"):
        controller.exit_train_phase()

    assert ("actor", "release_memory", ("weights", "optimizer")) in events
    assert not any(event[0] == "onload" for event in events if isinstance(event, tuple))
    assert "continue" not in events
    assert "resume" not in events


def test_failed_weight_update_keeps_generation_closed():
    events = []
    controller = _controller(events)
    controller._weight_update_ctrl = SimpleNamespace(
        update_weights=lambda version: (_ for _ in ()).throw(
            RuntimeError(f"update {version} failed")
        )
    )

    with pytest.raises(RuntimeError, match="update 1 failed"):
        with controller.train_phase():
            controller.update_weights(SimpleNamespace(version=1))

    assert controller._colocate_update_started
    assert not controller._colocate_transfer_succeeded
    assert not controller._colocate_update_succeeded
    assert "continue" not in events
    assert "resume" not in events


def test_error_weight_update_result_keeps_generation_closed():
    events = []
    controller = _controller(events)
    controller._weight_update_ctrl = SimpleNamespace(
        update_weights=lambda version: SimpleNamespace(
            status="error", error=f"update {version} failed"
        )
    )

    with pytest.raises(RuntimeError, match="update 1 failed"):
        with controller.train_phase():
            controller.update_weights(SimpleNamespace(version=1))

    assert controller._colocate_update_started
    assert not controller._colocate_transfer_succeeded
    assert not controller._colocate_update_succeeded
    assert "continue" not in events
    assert "resume" not in events


def test_transfer_without_version_commit_keeps_generation_closed():
    events = []
    controller = _controller(events)
    controller._weight_update_ctrl = SimpleNamespace(
        update_weights=lambda version: SimpleNamespace(
            status="ok", version=version, duration_ms=1.0
        )
    )

    with pytest.raises(RuntimeError, match="version commit failed"):
        with controller.train_phase():
            controller.update_weights(SimpleNamespace(version=1))
            raise RuntimeError("version commit failed")

    assert controller._colocate_update_started
    assert controller._colocate_transfer_succeeded
    assert not controller._colocate_update_succeeded
    assert "continue" not in events
    assert "resume" not in events


def test_mark_committed_allows_generation_to_resume():
    events = []
    controller = _controller(events)
    controller._weight_update_ctrl = SimpleNamespace(
        update_weights=lambda version: SimpleNamespace(
            status="ok", version=version, duration_ms=1.0
        )
    )

    with controller.train_phase():
        controller.update_weights(SimpleNamespace(version=1))
        controller.mark_colocate_update_committed()

    assert controller._colocate_update_succeeded
    assert controller._colocate_transfer_succeeded
    assert events == [
        "pause",
        "drain",
        ("offload", "kv_cache"),
        ("offload", "cuda_graph"),
        ("offload", "weights"),
        ("actor", "resume_memory", ("weights", "optimizer")),
        "pause_generation",
        ("actor", "release_memory", ("weights", "optimizer")),
        ("onload", "cuda_graph"),
        ("onload", "kv_cache"),
        "continue",
        "resume",
    ]


def test_phase_auxiliary_failure_keeps_rollout_and_actor_offloaded():
    """Unknown auxiliary residency must not overlap restored actor or rollout memory."""
    events = []
    controller = _controller(events)

    def fail_auxiliary():
        assert ("offload", "weights") in events
        raise RuntimeError("reference scoring failed")

    with pytest.raises(RuntimeError, match="reference scoring failed"):
        with controller.train_phase(before_actor_onload=fail_auxiliary):
            pytest.fail("actor training must not start")

    assert not any(event[0] in ("actor", "onload") for event in events)
    assert "continue" not in events
    assert "resume" not in events


def test_phase_actor_onload_failure_after_auxiliary_scoring_rolls_back():
    """Splitting entry must retain rollback after a partial actor restore."""
    events = []
    controller = _controller(events)

    def actor_memory(op, tags):
        events.append(("actor", op, tuple(tags)))
        if op == "resume_memory":
            assert "auxiliary_done" in events
            raise RuntimeError("actor onload failed")

    controller._broadcast_awex_memory_op = actor_memory

    with pytest.raises(RuntimeError, match="actor onload failed"):
        with controller.train_phase(
            before_actor_onload=lambda: events.append("auxiliary_done")
        ):
            pytest.fail("actor training must not start")

    release = ("actor", "release_memory", ("weights", "optimizer"))
    assert events.index(release) < events.index(("onload", "weights"))
    assert events[-2:] == ["continue", "resume"]


def test_final_train_phase_keeps_workflows_paused_but_allows_evaluation():
    events = []
    controller = _controller(events)

    with controller.train_phase(resume_rollout=False):
        pass

    assert ("onload", "weights") in events
    assert "continue" in events
    assert "resume" not in events


def test_close_destroys_training_children_before_rollout_guards(monkeypatch):
    from areal.trainer.rl_trainer import PPOTrainer

    events = []
    trainer = object.__new__(PPOTrainer)
    trainer._is_v2_awex_colocate = True
    for role in ("eval_rollout", "rollout", "teacher", "ref", "critic", "actor"):
        setattr(
            trainer,
            role,
            SimpleNamespace(destroy=lambda role=role: events.append(role)),
        )
    monkeypatch.setattr(
        "areal.trainer.rl_trainer.perf_tracer.save", lambda **kwargs: None
    )

    trainer.close()

    assert events.index("ref") < events.index("actor") < events.index("rollout")
    assert events.index("teacher") < events.index("actor")
    assert events.index("critic") < events.index("actor")


@pytest.mark.parametrize("colocate", [False, True])
def test_update_weights_leaves_resume_to_colocated_phase_only(colocate):
    events = []
    controller = _controller(events)
    controller._colocate = colocate
    controller._weight_update_ctrl = SimpleNamespace(
        update_weights=lambda version: SimpleNamespace(status="ok", duration_ms=1.0)
    )
    controller.update_weights(SimpleNamespace(version=1))
    assert events == (
        ["pause_generation"] if colocate else ["pause_generation", "continue"]
    )


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_awex_colocation_accepts_megatron_and_sglang(version):
    PPOTrainer._validate_awex_colocate_backends(
        "megatron", "sglang", enable_memory_saver=True, version=version
    )


def test_v2_awex_colocation_rejects_fsdp_actor():
    with pytest.raises(ValueError, match="requires Megatron actor"):
        PPOTrainer._validate_awex_colocate_backends(
            "fsdp", "sglang", enable_memory_saver=True, version="v2"
        )


def test_v2_awex_colocation_rejects_vllm_rollout():
    with pytest.raises(ValueError, match="requires SGLang rollout"):
        PPOTrainer._validate_awex_colocate_backends(
            "megatron", "vllm", enable_memory_saver=True, version="v2"
        )


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_awex_colocation_requires_sglang_memory_saver(version):
    with pytest.raises(ValueError, match="enable_memory_saver=True"):
        PPOTrainer._validate_awex_colocate_backends(
            "megatron", "sglang", enable_memory_saver=False, version=version
        )


@pytest.mark.parametrize("backend", ["fsdp", "megatron"])
@pytest.mark.parametrize("colocated_role", ["actor", "rollout"])
def test_v2_lora_colocation_is_rejected_before_engine_initialization(
    backend, colocated_role
):
    trainer = object.__new__(PPOTrainer)
    trainer.actor_alloc = SimpleNamespace(backend=backend)
    trainer.rollout_alloc = SimpleNamespace(backend="sglang")
    trainer.config = SimpleNamespace(
        actor=SimpleNamespace(_version="v2", use_lora=True, scheduling_strategy=None),
        rollout=SimpleNamespace(scheduling_strategy=None),
    )
    getattr(trainer.config, colocated_role).scheduling_strategy = SimpleNamespace(
        type="colocation", target="rollout" if colocated_role == "actor" else "actor"
    )

    with pytest.raises(ValueError, match="V2 LoRA colocation is not supported"):
        trainer._validate_cfg()
