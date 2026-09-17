# SPDX-License-Identifier: Apache-2.0

import math

import pytest
from omegaconf import OmegaConf

from areal.api.cli_args import (
    DatasetSourceConfig,
    InferenceEngineConfig,
    MOPDConfig,
    MOPDLossConfig,
    MOPDTeacherEngineConfig,
    MOPDTeacherManagerConfig,
    MOPDTeacherSpec,
    PPOActorConfig,
    PPOConfig,
    SchedulerConfig,
    SchedulingSpec,
    SchedulingStrategy,
    TeacherConfig,
    TrainDatasetConfig,
    to_structured_cfg,
)
from areal.trainer.mopd.execution import MOPDExecutionPlan

MEGATRON_BACKEND = "megatron:(attn:d1p1t2c2|ffn:d1p1e4)"


def _colocation(*, fork: bool) -> SchedulingStrategy:
    return SchedulingStrategy(type="colocation", target="actor", fork=fork)


def _mopd_config(**overrides) -> MOPDConfig:
    kwargs = {
        "teachers": {
            "agriculture": MOPDTeacherSpec(path="checkpoints/agriculture"),
            "swe_agent": MOPDTeacherSpec(path="checkpoints/swe-agent"),
        },
        "teacher_groups": {
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
            scheduling_spec=(SchedulingSpec(port_count=3),),
        ),
        "rollout": InferenceEngineConfig(
            backend="sglang:d4",
            scheduling_strategy=_colocation(fork=True),
        ),
        "mopd": mopd,
    }
    if mopd is not None:
        kwargs["train_dataset"] = TrainDatasetConfig(
            sources=[
                DatasetSourceConfig(
                    path="datasets/train",
                    type="rl",
                    teacher_group=next(iter(mopd.teacher_groups)),
                )
            ]
        )
    kwargs.update(overrides)
    return PPOConfig(**kwargs)


def test_ppo_config_without_mopd_preserves_disabled_default():
    """MOPD stays opt-in for existing PPO and GRPO configurations."""
    config = _ppo_config()

    assert config.mopd is None


def test_mopd_loss_config_defaults_to_unscaled_distillation():
    """The standalone MOPD objective uses a neutral coefficient by default."""
    config = MOPDLossConfig()

    assert config.rl_coefficient == 0.0
    assert config.distillation_coefficient == 1.0


def test_mopd_teacher_config_exposes_only_scoring_engine_fields():
    config = MOPDTeacherEngineConfig(backend=MEGATRON_BACKEND)

    assert config.optimizer is None
    assert config.disable_dropout is True
    assert not hasattr(config, "ppo_n_minibatches")
    assert not hasattr(config, "eps_clip")


@pytest.mark.parametrize(
    ("rl_coefficient", "distillation_coefficient", "expected"),
    [
        (0.0, 1.0, (True, False)),
        (1.0, 0.0, (False, True)),
        (0.4, 0.6, (True, True)),
    ],
)
def test_mopd_execution_plan_short_circuits_disabled_objectives(
    rl_coefficient, distillation_coefficient, expected
):
    config = _ppo_config(
        _mopd_config(
            loss=MOPDLossConfig(
                rl_coefficient=rl_coefficient,
                distillation_coefficient=distillation_coefficient,
            )
        )
    )

    plan = MOPDExecutionPlan.from_config(config)

    assert plan is not None
    assert (plan.requires_teacher_scoring, plan.requires_rl) == expected


def test_mopd_forked_rollout_requires_worker_and_nccl_ports():
    actor = PPOActorConfig(
        backend=MEGATRON_BACKEND,
        weight_update_mode="awex",
        scheduling_spec=(SchedulingSpec(port_count=1),),
    )

    with pytest.raises(ValueError, match="port_count >= 2"):
        _ppo_config(_mopd_config(), actor=actor)


def test_mopd_config_valid_preserves_teacher_group_order_and_weights():
    """Teacher order follows insertion order and group weights stay unnormalized."""
    config = _ppo_config(_mopd_config())

    assert config.enable_offload is False
    assert list(config.mopd.teachers) == ["agriculture", "swe_agent"]
    assert list(config.mopd.teacher_groups) == ["single", "ensemble"]
    assert config.mopd.teacher_groups["ensemble"] == {
        "agriculture": 0.3,
        "swe_agent": 0.7,
    }
    assert config.mopd.loss.rl_coefficient == 0.0
    assert config.mopd.loss.distillation_coefficient == 1.0
    assert config.train_dataset.sources[0].teacher_group == "single"


def test_mopd_config_requires_routed_dataset_sources():
    """MOPD rejects the legacy single-path dataset shape."""
    with pytest.raises(ValueError, match="train_dataset.sources must not be empty"):
        _ppo_config(
            _mopd_config(),
            train_dataset=TrainDatasetConfig(path="dataset", type="rl"),
        )


def test_dataset_source_teacher_group_defaults_to_none_without_mopd():
    """Dataset mixtures outside MOPD do not require teacher selection."""
    source = DatasetSourceConfig(path="dataset", type="rl")

    assert source.teacher_group is None


@pytest.mark.parametrize("teacher_group", ["", " ", 7, True])
def test_dataset_source_rejects_invalid_optional_teacher_group(teacher_group):
    """An explicitly configured teacher group must be a non-empty string."""
    with pytest.raises(ValueError, match="non-empty string or null"):
        DatasetSourceConfig(
            path="dataset",
            type="rl",
            teacher_group=teacher_group,
        )


def test_mopd_dataset_source_requires_teacher_group():
    """Every MOPD source must select a teacher group explicitly."""
    train_dataset = TrainDatasetConfig(
        sources=[DatasetSourceConfig(path="dataset", type="rl")]
    )

    with pytest.raises(ValueError, match="teacher_group must be configured"):
        _ppo_config(_mopd_config(), train_dataset=train_dataset)


def test_mopd_dataset_source_rejects_unknown_teacher_group():
    """A source cannot select a group absent from mopd.teacher_groups."""
    train_dataset = TrainDatasetConfig(
        sources=[
            DatasetSourceConfig(path="dataset", type="rl", teacher_group="missing")
        ]
    )

    with pytest.raises(ValueError, match="unknown MOPD teacher group 'missing'"):
        _ppo_config(_mopd_config(), train_dataset=train_dataset)


def test_mopd_dataset_sources_reject_legacy_path_and_type():
    """Mixture sources and the legacy single-source fields are mutually exclusive."""
    with pytest.raises(ValueError, match="path/type cannot be combined"):
        TrainDatasetConfig(
            path="legacy",
            type="rl",
            sources=[
                DatasetSourceConfig(path="dataset", type="rl", teacher_group="single")
            ],
        )


def test_mopd_config_yaml_shape_converts_to_structured_config():
    """A YAML-shaped mapping survives the production OmegaConf conversion path."""
    raw_config = OmegaConf.create(
        {
            "experiment_name": "mopd-test",
            "trial_name": "trial",
            "train_dataset": {
                "sources": [{"path": "dataset", "type": "rl", "teacher_group": "math"}]
            },
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
                "scheduling_spec": [{"port_count": 3}],
            },
            "rollout": {
                "backend": "sglang:d4",
                "scheduling_strategy": {
                    "type": "colocation",
                    "target": "actor",
                    "fork": True,
                },
            },
            "mopd": {
                "teachers": {"agriculture": {"path": "checkpoints/agriculture"}},
                "teacher_groups": {"math": {"agriculture": 2.0}},
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
    assert config.mopd.teacher_groups["math"]["agriculture"] == 2.0
    assert isinstance(config.train_dataset.sources[0], DatasetSourceConfig)
    assert config.train_dataset.sources[0].teacher_group == "math"


@pytest.mark.parametrize("weight", [-1.0, math.inf, math.nan, True, "1.0"])
def test_mopd_config_invalid_teacher_group_weight_raises(weight):
    """Group weights reject negative, non-finite, boolean, and non-numeric values."""
    with pytest.raises(ValueError, match="finite non-negative|finite and non-negative"):
        _mopd_config(teacher_groups={"bad": {"agriculture": weight}})


def test_mopd_config_unknown_teacher_raises():
    """Every teacher group entry must reference a configured teacher."""
    with pytest.raises(ValueError, match="unknown teacher"):
        _mopd_config(teacher_groups={"bad": {"missing": 1.0}})


def test_mopd_config_all_zero_teacher_group_raises():
    """A teacher group must include at least one positive teacher weight."""
    with pytest.raises(ValueError, match="at least one positive weight"):
        _mopd_config(teacher_groups={"bad": {"agriculture": 0.0}})


@pytest.mark.parametrize("coefficient", [-1.0, math.inf, math.nan, True])
def test_mopd_loss_config_invalid_coefficient_raises(coefficient):
    """Loss coefficients must be finite non-negative numbers."""
    with pytest.raises(ValueError, match="finite"):
        MOPDLossConfig(rl_coefficient=coefficient)


def test_mopd_loss_config_rejects_disabled_objective():
    with pytest.raises(ValueError, match="cannot both be zero"):
        MOPDLossConfig(rl_coefficient=0.0, distillation_coefficient=0.0)


def test_mopd_config_rollout_colocation_accepts_forked_workers():
    """The current v1 runtime accepts process-isolated rollout workers."""
    rollout = InferenceEngineConfig(
        backend="sglang:d4",
        scheduling_strategy=_colocation(fork=True),
    )

    config = _ppo_config(_mopd_config(), rollout=rollout)

    assert config.rollout.scheduling_strategy.fork is True


def test_mopd_config_rollout_reused_workers_are_rejected():
    """Same-process rollout is outside the current v1 runtime capability."""
    with pytest.raises(ValueError, match="current MOPD v1 runtime"):
        _ppo_config(
            _mopd_config(),
            rollout=InferenceEngineConfig(
                backend="sglang:d4",
                scheduling_strategy=_colocation(fork=False),
            ),
        )


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
    actor = PPOActorConfig(
        backend=backend,
        weight_update_mode="awex",
        scheduling_spec=(SchedulingSpec(port_count=3),),
    )

    config = _ppo_config(
        _mopd_config(teacher_engine=teacher_engine),
        actor=actor,
    )

    assert config.actor.backend == backend
    assert config.mopd.teacher_engine.backend == backend


def test_mopd_config_local_memory_multi_node_raises():
    """Local-memory checkpoint staging cannot span multiple compute nodes."""
    actor = PPOActorConfig(
        backend="megatron:d16",
        weight_update_mode="awex",
        scheduling_spec=(SchedulingSpec(port_count=3),),
    )
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
            teacher_groups={"group": {teacher_id: 1.0}},
        )


def test_mopd_and_legacy_teacher_are_mutually_exclusive():
    """Ambiguous legacy and MOPD teacher configuration fails fast."""
    legacy_teacher = TeacherConfig(
        engine_type="train",
        train=MOPDTeacherEngineConfig(backend=MEGATRON_BACKEND),
    )

    with pytest.raises(ValueError, match="cannot be configured at the same time"):
        _ppo_config(_mopd_config(), teacher=legacy_teacher)
