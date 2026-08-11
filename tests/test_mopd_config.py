# SPDX-License-Identifier: Apache-2.0

import math

import pytest
from omegaconf import OmegaConf

from areal.api.cli_args import (
    InferenceEngineConfig,
    MOPDConfig,
    MOPDLossConfig,
    MOPDTeacherEngineConfig,
    MOPDTeacherManagerConfig,
    MOPDTeacherSpec,
    PPOActorConfig,
    PPOConfig,
    SchedulerConfig,
    SchedulingStrategy,
    TeacherConfig,
    to_structured_cfg,
)

MEGATRON_BACKEND = "megatron:(attn:d1p1t2c2|ffn:d1p1e4)"


def _colocation(*, fork: bool) -> SchedulingStrategy:
    return SchedulingStrategy(type="colocation", target="actor", fork=fork)


def _mopd_config(**overrides) -> MOPDConfig:
    kwargs = {
        "teachers": {
            "agriculture": MOPDTeacherSpec(path="checkpoints/agriculture"),
            "swe_agent": MOPDTeacherSpec(path="checkpoints/swe-agent"),
        },
        "routes": {
            "single": {"agriculture": 1.0},
            "ensemble": {"agriculture": 0.3, "swe_agent": 0.7},
        },
        "teacher_engine": MOPDTeacherEngineConfig(
            backend=MEGATRON_BACKEND,
            optimizer=None,
            disable_dropout=True,
            scheduling_strategy=_colocation(fork=True),
        ),
    }
    kwargs.update(overrides)
    return MOPDConfig(**kwargs)


def _ppo_config(mopd: MOPDConfig | None = None, **overrides) -> PPOConfig:
    kwargs = {
        "experiment_name": "mopd-test",
        "trial_name": "trial",
        "actor": PPOActorConfig(
            backend=MEGATRON_BACKEND,
            weight_update_mode="awex",
        ),
        "rollout": InferenceEngineConfig(
            backend="sglang:d4",
            scheduling_strategy=_colocation(fork=False),
        ),
        "mopd": mopd,
    }
    kwargs.update(overrides)
    return PPOConfig(**kwargs)


def test_ppo_config_without_mopd_preserves_disabled_default():
    """MOPD stays opt-in for existing PPO and GRPO configurations."""
    config = _ppo_config()

    assert config.mopd is None


def test_mopd_config_valid_preserves_teacher_and_route_order_and_weights():
    """Teacher order follows insertion order and route weights stay unnormalized."""
    config = _ppo_config(_mopd_config())

    assert config.enable_offload is False
    assert list(config.mopd.teachers) == ["agriculture", "swe_agent"]
    assert list(config.mopd.routes) == ["single", "ensemble"]
    assert config.mopd.routes["ensemble"] == {
        "agriculture": 0.3,
        "swe_agent": 0.7,
    }
    assert config.mopd.loss.rl_coefficient == 0.0
    assert config.mopd.loss.distillation_coefficient == 0.005


def test_mopd_config_yaml_shape_converts_to_structured_config():
    """A YAML-shaped mapping survives the production OmegaConf conversion path."""
    raw_config = OmegaConf.create(
        {
            "experiment_name": "mopd-test",
            "trial_name": "trial",
            "train_dataset": {"path": "dataset", "type": "rl"},
            "saver": {
                "experiment_name": "mopd-test",
                "trial_name": "trial",
                "fileroot": "outputs",
            },
            "evaluator": {
                "experiment_name": "mopd-test",
                "trial_name": "trial",
                "fileroot": "outputs",
            },
            "stats_logger": {
                "experiment_name": "mopd-test",
                "trial_name": "trial",
                "fileroot": "outputs",
            },
            "recover": {
                "experiment_name": "mopd-test",
                "trial_name": "trial",
                "fileroot": "outputs",
            },
            "actor": {
                "experiment_name": "mopd-test",
                "trial_name": "trial",
                "backend": MEGATRON_BACKEND,
                "weight_update_mode": "awex",
            },
            "rollout": {
                "backend": "sglang:d4",
                "scheduling_strategy": {
                    "type": "colocation",
                    "target": "actor",
                    "fork": False,
                },
            },
            "mopd": {
                "teachers": {"agriculture": {"path": "checkpoints/agriculture"}},
                "routes": {"math": {"agriculture": 2.0}},
                "teacher_engine": {
                    "experiment_name": "mopd-test",
                    "trial_name": "trial",
                    "backend": MEGATRON_BACKEND,
                    "optimizer": None,
                    "disable_dropout": True,
                    "scheduling_strategy": {
                        "type": "colocation",
                        "target": "actor",
                        "fork": True,
                    },
                },
            },
        }
    )

    config = OmegaConf.to_object(to_structured_cfg(raw_config, PPOConfig))

    assert isinstance(config, PPOConfig)
    assert isinstance(config.mopd, MOPDConfig)
    assert isinstance(config.mopd.teachers["agriculture"], MOPDTeacherSpec)
    assert config.mopd.routes["math"]["agriculture"] == 2.0


@pytest.mark.parametrize("weight", [-1.0, math.inf, math.nan, True, "1.0"])
def test_mopd_config_invalid_route_weight_raises(weight):
    """Route weights reject negative, non-finite, boolean, and non-numeric values."""
    with pytest.raises(ValueError, match="finite non-negative|finite and non-negative"):
        _mopd_config(routes={"bad": {"agriculture": weight}})


def test_mopd_config_unknown_teacher_raises():
    """Every route entry must reference a configured teacher."""
    with pytest.raises(ValueError, match="unknown teacher"):
        _mopd_config(routes={"bad": {"missing": 1.0}})


def test_mopd_config_all_zero_route_raises():
    """A route must include at least one teacher with positive weight."""
    with pytest.raises(ValueError, match="at least one positive weight"):
        _mopd_config(routes={"bad": {"agriculture": 0.0}})


@pytest.mark.parametrize("coefficient", [-1.0, math.inf, math.nan, True])
def test_mopd_loss_config_invalid_coefficient_raises(coefficient):
    """Loss coefficients must be finite non-negative numbers."""
    with pytest.raises(ValueError, match="finite"):
        MOPDLossConfig(rl_coefficient=coefficient)


@pytest.mark.parametrize("fork", [False, True])
def test_mopd_config_rollout_colocation_accepts_both_fork_modes(fork):
    """MOPD supports reused and process-isolated rollout workers."""
    rollout = InferenceEngineConfig(
        backend="sglang:d4",
        scheduling_strategy=_colocation(fork=fork),
    )

    config = _ppo_config(_mopd_config(), rollout=rollout)

    assert config.rollout.scheduling_strategy.fork is fork


def test_mopd_config_rollout_reused_workers_warn_about_stats(monkeypatch):
    """Same-process rollout warns about the known stats export limitation."""
    warnings = []
    monkeypatch.setattr("areal.api.cli_args.logger.warning", warnings.append)

    _ppo_config(
        _mopd_config(),
        rollout=InferenceEngineConfig(
            backend="sglang:d4",
            scheduling_strategy=_colocation(fork=False),
        ),
    )

    assert len(warnings) == 1
    assert "shared stats tracker" in warnings[0]


def test_mopd_config_teacher_parallelism_mismatch_raises():
    """Actor and teacher must use exactly the same parallel dimensions."""
    teacher_engine = MOPDTeacherEngineConfig(
        backend="megatron:d4",
        optimizer=None,
        disable_dropout=True,
        scheduling_strategy=_colocation(fork=True),
    )

    with pytest.raises(ValueError, match="same parallel strategy"):
        _ppo_config(_mopd_config(teacher_engine=teacher_engine))


def test_mopd_config_pipeline_parallelism_accepts_matching_topology():
    """MOPD accepts pipeline parallelism shared by the actor and teachers."""
    backend = "megatron:d1p2t2"
    teacher_engine = MOPDTeacherEngineConfig(
        backend=backend,
        optimizer=None,
        disable_dropout=True,
        scheduling_strategy=_colocation(fork=True),
    )
    actor = PPOActorConfig(backend=backend, weight_update_mode="awex")

    config = _ppo_config(
        _mopd_config(teacher_engine=teacher_engine),
        actor=actor,
    )

    assert config.actor.backend == backend
    assert config.mopd.teacher_engine.backend == backend


def test_mopd_config_local_memory_multi_node_raises():
    """Local-memory checkpoint staging cannot span multiple compute nodes."""
    actor = PPOActorConfig(backend="megatron:d16", weight_update_mode="awex")
    teacher_engine = MOPDTeacherEngineConfig(
        backend="megatron:d16",
        optimizer=None,
        disable_dropout=True,
        scheduling_strategy=_colocation(fork=True),
    )
    manager = MOPDTeacherManagerConfig(type="local_memory")

    with pytest.raises(ValueError, match="single node"):
        _ppo_config(
            _mopd_config(teacher_engine=teacher_engine, manager=manager),
            actor=actor,
            scheduler=SchedulerConfig(type="local"),
        )


@pytest.mark.parametrize("scheduler_type", ["ray", "slurm", None])
def test_mopd_config_local_memory_requires_same_host_local_scheduler(scheduler_type):
    """Controller-local staging is rejected when workers may run remotely."""
    manager = MOPDTeacherManagerConfig(type="local_memory")

    with pytest.raises(ValueError, match="scheduler.type='local'"):
        _ppo_config(
            _mopd_config(manager=manager),
            scheduler=SchedulerConfig(type=scheduler_type),
        )


def test_mopd_config_local_memory_accepts_local_scheduler():
    """LocalScheduler guarantees controller and fork workers share one host."""
    config = _ppo_config(
        _mopd_config(manager=MOPDTeacherManagerConfig(type="local_memory")),
        scheduler=SchedulerConfig(type="local"),
    )

    assert config.mopd.manager.type == "local_memory"


def test_mopd_config_teacher_v2_raises():
    """MOPD fails fast before selecting a controller without v1 lifecycle APIs."""
    teacher_engine = MOPDTeacherEngineConfig(
        backend=MEGATRON_BACKEND,
        optimizer=None,
        disable_dropout=True,
        scheduling_strategy=_colocation(fork=True),
        _version="v2",
    )

    with pytest.raises(ValueError, match="requires _version='v1'"):
        _ppo_config(_mopd_config(teacher_engine=teacher_engine))


@pytest.mark.parametrize("teacher_id", ["../teacher", "/teacher", "org/teacher"])
def test_mopd_config_rejects_path_like_teacher_ids(teacher_id):
    """Teacher IDs cannot escape or create nested staging directories."""
    with pytest.raises(ValueError, match="filename-safe"):
        MOPDConfig(
            teachers={teacher_id: MOPDTeacherSpec(path="checkpoints/teacher")},
            routes={"route": {teacher_id: 1.0}},
        )


def test_mopd_and_legacy_teacher_are_mutually_exclusive():
    """Ambiguous legacy and MOPD teacher configuration fails fast."""
    legacy_teacher = TeacherConfig(
        engine_type="train",
        train=MOPDTeacherEngineConfig(backend=MEGATRON_BACKEND),
    )

    with pytest.raises(ValueError, match="cannot be configured at the same time"):
        _ppo_config(_mopd_config(), teacher=legacy_teacher)


def test_legacy_teacher_emits_deprecation_warning():
    """The single-teacher path remains available for one compatibility period."""
    legacy_teacher = TeacherConfig(
        engine_type="train",
        train=MOPDTeacherEngineConfig(backend=MEGATRON_BACKEND),
    )

    with pytest.warns(DeprecationWarning, match="single-teacher"):
        config = _ppo_config(teacher=legacy_teacher)

    assert config.teacher is legacy_teacher
