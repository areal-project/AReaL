"""Focused checks for the CPU-staged Megatron GSM8K example."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import examples.cpu_staged_offload.engine as example_engine
from examples.cpu_staged_offload.config import (
    CPU_STAGED_SNAPSHOT_ROOT_ENV,
    CPUStagedGRPOConfig,
    CPUStagedOffloadSettings,
    install_cpu_staged_worker_environment,
)
from examples.cpu_staged_offload.engine import (
    CPUStagedMegatronPPOActor,
    CPUStagedPPOTrainer,
)

import areal.api.cli_args as cli_args
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import load_expr_config
from areal.engine.megatron_engine import MegatronPPOActor
from areal.trainer import PPOTrainer
from areal.utils.stats_logger import StatsLogger

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_actor_configures_staged_adamw_before_parent_initialize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The worker configures staged AdamW before MCore creates the optimizer."""
    events: list[tuple[str, Any]] = []
    actor = object.__new__(CPUStagedMegatronPPOActor)
    monkeypatch.setenv("AREAL_CPU_STAGED_BUFFER_COUNT", "3")
    monkeypatch.setenv("AREAL_CPU_STAGED_BUCKET_SIZE_MB", "32")
    monkeypatch.setenv(CPU_STAGED_SNAPSHOT_ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(
        CPUStagedMegatronPPOActor,
        "configure_gpu_staged_adamw",
        lambda self, config: events.append(("configure", config)),
    )
    monkeypatch.setattr(
        MegatronPPOActor,
        "initialize",
        lambda self, addr, ft_spec, *args, **kwargs: events.append(
            ("parent", (addr, ft_spec))
        ),
    )

    actor.initialize(None, object())

    assert [event[0] for event in events] == ["configure", "parent"]
    staged_config = events[0][1]
    assert staged_config.buffer_count == 3
    assert staged_config.bucket_size_mb == 32
    assert staged_config.checkpoint_snapshot_root == str(tmp_path)


@pytest.mark.parametrize("single_controller", [True, False])
def test_trainer_uses_custom_engine_only_for_primary_actor(
    monkeypatch: pytest.MonkeyPatch, single_controller: bool
) -> None:
    """Controller and in-process modes both customize only the primary actor."""
    actor_config = object()
    scheduler = object()
    parallel = object()
    events: list[tuple[str, Any]] = []

    class FakeActor:
        def __init__(self, config: Any) -> None:
            events.append(("construct", config))

        @classmethod
        def as_controller(cls, config: Any, received_scheduler: Any) -> FakeActor:
            events.append(("controller", (config, received_scheduler)))
            return cls.__new__(cls)

        def create_process_group(self, *, parallel_strategy: Any) -> None:
            events.append(("process_group", parallel_strategy))

    trainer = object.__new__(CPUStagedPPOTrainer)
    trainer.config = SimpleNamespace(
        actor=actor_config,
        cpu_staged=SimpleNamespace(enabled=True),
    )
    trainer.scheduler = scheduler
    actor_alloc = SimpleNamespace(backend="megatron", name="actor", parallel=parallel)
    ref_alloc = SimpleNamespace(backend="megatron", name="ref", parallel=parallel)
    standard_ref = object()
    monkeypatch.setattr(example_engine, "CPUStagedMegatronPPOActor", FakeActor)
    monkeypatch.setattr(
        example_engine, "is_single_controller", lambda: single_controller
    )
    monkeypatch.setattr(
        PPOTrainer,
        "_create_train_engine",
        lambda self, config, alloc: standard_ref,
    )

    actor = trainer._create_train_engine(actor_config, actor_alloc)
    ref = trainer._create_train_engine(object(), ref_alloc)

    assert actor is not standard_ref
    assert ref is standard_ref
    assert ("process_group", parallel) in events
    expected = "controller" if single_controller else "construct"
    assert any(event[0] == expected for event in events)


def test_disabled_cpu_staged_uses_standard_actor_and_clean_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The baseline delegates actor construction and injects no staged settings."""
    trainer = object.__new__(CPUStagedPPOTrainer)
    trainer.config = SimpleNamespace(cpu_staged=SimpleNamespace(enabled=False))
    standard_actor = object()
    monkeypatch.setattr(
        PPOTrainer,
        "_create_train_engine",
        lambda self, config, alloc: standard_actor,
    )
    alloc = SimpleNamespace(name="actor", backend="megatron")

    assert trainer._create_train_engine(object(), alloc) is standard_actor

    task = SimpleNamespace(env_vars={})
    config = SimpleNamespace(
        cpu_staged=CPUStagedOffloadSettings(enabled=False),
        actor=SimpleNamespace(scheduling_spec=[task]),
    )
    environ: dict[str, str] = {}
    assert install_cpu_staged_worker_environment(config, environ) == {}
    assert environ == task.env_vars == {}


@pytest.mark.parametrize("snapshot_root", [None, "/job/scratch/rollback"])
def test_worker_environment_forwards_optional_snapshot_root(
    snapshot_root: str | None,
) -> None:
    """Example settings reach remote workers without a hard-coded machine path."""
    task = SimpleNamespace(env_vars={})
    config = SimpleNamespace(
        cpu_staged=CPUStagedOffloadSettings(
            buffer_count=3,
            bucket_size_mb=32,
            checkpoint_snapshot_root=snapshot_root,
        ),
        actor=SimpleNamespace(scheduling_spec=[task]),
    )
    environ: dict[str, str] = {}

    values = install_cpu_staged_worker_environment(config, environ)

    assert values == environ == task.env_vars
    assert values["AREAL_CPU_STAGED_BUFFER_COUNT"] == "3"
    assert values["AREAL_CPU_STAGED_BUCKET_SIZE_MB"] == "32"
    assert values.get(CPU_STAGED_SNAPSHOT_ROOT_ENV) == snapshot_root


def test_example_config_and_modules_load_from_repository_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The documented YAML and remote module imports form one valid config."""
    monkeypatch.chdir(REPOSITORY_ROOT)
    monkeypatch.setenv("QWEN3_30B_A3B_MODEL_PATH", "/models/Qwen3-30B-A3B")
    monkeypatch.setattr(cli_args.name_resolve, "reconfigure", lambda config: None)
    monkeypatch.setattr(cli_args, "save_config", lambda config, path: None)
    monkeypatch.setattr(
        StatsLogger, "get_log_path", staticmethod(lambda config: str(tmp_path))
    )

    config, config_path = load_expr_config(
        [
            "--config",
            "examples/cpu_staged_offload/gsm8k_grpo_cpu_staged.yaml",
        ],
        CPUStagedGRPOConfig,
    )
    allocation = ModelAllocation.from_str(config.actor.backend)

    assert config_path.endswith("gsm8k_grpo_cpu_staged.yaml")
    assert config.actor.path == config.tokenizer_path == "/models/Qwen3-30B-A3B"
    assert allocation.parallel.world_size == 8
    assert allocation.parallel.data_parallel_size == 8
    assert allocation.parallel.expert_parallel_size == 8
    assert config.rollout.backend == "sglang:d1t8p1"
    assert config.cpu_staged.enabled is True
    assert config.memory_profiler.profile_steps == [1]
