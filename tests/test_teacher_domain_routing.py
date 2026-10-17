from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from areal.api.cli_args import (
    DistillationConfig,
    InferenceEngineConfig,
    SGLangConfig,
    TeacherConfig,
    vLLMConfig,
)
from areal.trainer.rl_trainer import PPOTrainer


class _FakeTeacher:
    def __init__(self, value: float):
        self.value = value
        self.seen_batches = []

    def compute_logp(self, batch):
        self.seen_batches.append(batch)
        return [torch.full_like(traj["loss_mask"], self.value) for traj in batch]


def _teacher_config(domain: str):
    return SimpleNamespace(domain=domain)


def _distillation_teacher():
    return SimpleNamespace(engine_type="rollout", weight=1.0, domain=None)


def test_distillation_config_owns_loss_settings():
    config = DistillationConfig(
        teachers=[_distillation_teacher()],
        distill_loss_type="mopd_pg",
        mopd_adv_clip=3.0,
    )

    assert config.distill_loss_type == "mopd_pg"
    assert config.mopd_adv_clip == 3.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"distill_loss_type": "invalid"}, "teacher.distill_loss_type"),
        ({"mopd_adv_clip": 0.0}, "teacher.mopd_adv_clip"),
    ],
)
def test_distillation_config_rejects_invalid_loss_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DistillationConfig(teachers=[_distillation_teacher()], **kwargs)


@pytest.mark.parametrize(
    ("backend", "override", "message"),
    [
        ("sglang:d1", {"vllm": vLLMConfig()}, "teacher.vllm"),
        ("vllm:d1", {"sglang": SGLangConfig()}, "teacher.sglang"),
    ],
)
def test_teacher_config_rejects_backend_override_mismatch(backend, override, message):
    with pytest.raises(ValueError, match=message):
        TeacherConfig(
            rollout=InferenceEngineConfig(backend=backend),
            **override,
        )


class _FakeRolloutEngine:
    def __init__(self, config):
        self.config = config
        self.train_data_parallel_size = None

    def initialize(self, train_data_parallel_size):
        self.train_data_parallel_size = train_data_parallel_size


@pytest.mark.parametrize("use_override", [False, True])
def test_init_teacher_rollout_uses_teacher_vllm_override(monkeypatch, use_override):
    shared_vllm = vLLMConfig(gpu_memory_utilization=0.9)
    teacher_vllm = vLLMConfig(gpu_memory_utilization=0.5) if use_override else None
    teacher_config = TeacherConfig(
        path="teacher-model",
        rollout=InferenceEngineConfig(backend="vllm:d1"),
        vllm=teacher_vllm,
    )
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(vllm=shared_vllm)
    trainer.teacher_allocs = [
        SimpleNamespace(
            backend="vllm",
            parallel=SimpleNamespace(tp_size=1, pp_size=1),
        )
    ]
    trainer.actor_alloc = SimpleNamespace(parallel=SimpleNamespace(dp_size=2))
    captured = {}

    def _build_args(vllm_config, tp_size, pp_size):
        captured["config"] = vllm_config
        captured["tp_size"] = tp_size
        captured["pp_size"] = pp_size
        return {}

    monkeypatch.setattr("areal.trainer.rl_trainer.is_single_controller", lambda: False)
    monkeypatch.setattr("areal.trainer.rl_trainer.RemotevLLMEngine", _FakeRolloutEngine)
    monkeypatch.setattr(vLLMConfig, "build_args", _build_args)

    engine = trainer._init_teacher_rollout(teacher_config, 0)

    expected_utilization = 0.5 if use_override else 0.9
    assert captured["config"].gpu_memory_utilization == expected_utilization
    assert captured["config"].model == "teacher-model"
    assert captured["tp_size"] == 1
    assert captured["pp_size"] == 1
    assert engine.config.tokenizer_path == "teacher-model"
    assert engine.train_data_parallel_size == 2


def test_assign_domain_teacher_logps_routes_each_trajectory_to_matching_teacher():
    """Domain routing scores each trajectory with exactly one teacher."""

    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        teacher=SimpleNamespace(
            routing_mode="domain",
            domain_key="domain",
            rl_loss_weight=1.0,
            distill_loss_weight=0.25,
            distill_loss_type="mopd_pg",
            mopd_adv_clip=3.0,
        ),
        actor=SimpleNamespace(
            packing_algorithm="ffd",
        ),
    )
    trainer.teachers = [_FakeTeacher(1.0), _FakeTeacher(2.0)]
    trainer.teacher_configs = [_teacher_config("math"), _teacher_config("coding")]
    trainer.teacher_domain_to_idx = {"math": 0, "coding": 1}

    rollout_batch = [
        {
            "domain": "math",
            "loss_mask": torch.ones(1, 3),
        },
        {
            "domain": "coding",
            "loss_mask": torch.ones(1, 2),
        },
        {
            "domain": "math",
            "loss_mask": torch.ones(1, 4),
        },
    ]

    trainer._assign_teacher_logps(rollout_batch)

    assert len(trainer.teachers[0].seen_batches) == 1
    assert len(trainer.teachers[1].seen_batches) == 1
    math_lengths = [
        traj["loss_mask"].shape[-1] for traj in trainer.teachers[0].seen_batches[0]
    ]
    coding_lengths = [
        traj["loss_mask"].shape[-1] for traj in trainer.teachers[1].seen_batches[0]
    ]
    assert math_lengths == [3, 4]
    assert coding_lengths == [2]
    assert "domain" not in trainer.teachers[0].seen_batches[0][0]
    assert "domain" not in trainer.teachers[1].seen_batches[0][0]
    torch.testing.assert_close(
        rollout_batch[0]["teacher_logp"],
        torch.ones(1, 3),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        rollout_batch[1]["teacher_logp"],
        torch.full((1, 2), 2.0),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        rollout_batch[2]["teacher_logp"],
        torch.ones(1, 4),
        rtol=0,
        atol=0,
    )
    assert rollout_batch[0]["distill_loss_weight"] == 0.25
    assert rollout_batch[0]["distill_loss_type"] == "mopd_pg"
    assert rollout_batch[0]["mopd_adv_clip"] == 3.0
    assert rollout_batch[1]["distill_loss_type"] == "mopd_pg"
