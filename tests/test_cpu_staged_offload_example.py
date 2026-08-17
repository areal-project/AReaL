"""Lightweight checks for the CPU-staged Megatron GSM8K example."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import examples.cpu_staged_offload.engine as example_engine
from examples.cpu_staged_offload.config import (
    CPU_STAGED_SNAPSHOT_ROOT_ENV,
    CPUStagedGRPOConfig,
    CPUStagedOffloadSettings,
    install_cpu_staged_worker_environment,
)
from examples.cpu_staged_offload.engine import (
    CPU_STAGED_ACTOR_IMPORT_PATH,
    CPUStagedMegatronPPOActor,
    CPUStagedPPOTrainer,
    cpu_staged_adamw_config_from_environment,
)

import areal.api.cli_args as cli_args
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import load_expr_config
from areal.engine.megatron_engine import MegatronPPOActor
from areal.trainer import PPOTrainer
from areal.utils.stats_logger import StatsLogger

EXAMPLE_ROOT = Path(__file__).parents[1] / "examples" / "cpu_staged_offload"
REPOSITORY_ROOT = Path(__file__).parents[1]


def test_actor_configures_staged_adamw_before_parent_initialize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The worker must install staged AdamW before MCore creates an optimizer."""
    events: list[tuple[str, Any]] = []
    actor = object.__new__(CPUStagedMegatronPPOActor)

    monkeypatch.setenv("AREAL_CPU_STAGED_BUFFER_COUNT", "3")
    monkeypatch.setenv("AREAL_CPU_STAGED_BUCKET_SIZE_MB", "32")
    monkeypatch.setenv("AREAL_CPU_STAGED_SNAPSHOT_ROOT", str(tmp_path))
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


def test_actor_has_stable_remotely_importable_module_path() -> None:
    """Remote controllers can import the example actor by its canonical path."""
    module_name, class_name = CPU_STAGED_ACTOR_IMPORT_PATH.rsplit(".", 1)

    imported = getattr(importlib.import_module(module_name), class_name)

    assert imported is CPUStagedMegatronPPOActor
    assert CPUStagedMegatronPPOActor.__module__ == module_name


def test_actor_controller_retains_custom_importable_engine_class() -> None:
    """The inherited controller must serialize the custom class, not its parent."""
    config = SimpleNamespace(_version="v1", backend="megatron:d1")

    controller = CPUStagedMegatronPPOActor.as_controller(config, object())

    assert controller.train_engine is CPUStagedMegatronPPOActor
    assert (
        f"{controller.train_engine.__module__}.{controller.train_engine.__name__}"
        == CPU_STAGED_ACTOR_IMPORT_PATH
    )


@pytest.mark.parametrize("single_controller", [True, False])
def test_trainer_uses_custom_engine_only_for_primary_actor(
    monkeypatch: pytest.MonkeyPatch, single_controller: bool
) -> None:
    """Controller and in-process branches both select the custom primary actor."""
    actor_config = object()
    ref_config = object()
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
        lambda self, config, received_alloc: (
            events.append(("standard", (config, received_alloc))) or standard_ref
        ),
    )

    actor = trainer._create_train_engine(actor_config, actor_alloc)
    ref = trainer._create_train_engine(ref_config, ref_alloc)

    assert actor is not standard_ref
    assert ref is standard_ref
    assert events[-1] == ("standard", (ref_config, ref_alloc))
    assert ("process_group", parallel) in events
    if single_controller:
        assert ("controller", (actor_config, scheduler)) in events
        assert not any(event[0] == "construct" for event in events[:-1])
    else:
        assert ("construct", actor_config) in events
        assert not any(event[0] == "controller" for event in events)


def test_trainer_uses_standard_actor_when_cpu_staged_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The disabled branch delegates actor construction to PPOTrainer."""
    trainer = object.__new__(CPUStagedPPOTrainer)
    trainer.config = SimpleNamespace(cpu_staged=SimpleNamespace(enabled=False))
    actor_config = object()
    alloc = SimpleNamespace(name="actor", backend="megatron")
    standard_actor = object()

    monkeypatch.setattr(
        PPOTrainer,
        "_create_train_engine",
        lambda self, config, received_alloc: standard_actor,
    )

    assert trainer._create_train_engine(actor_config, alloc) is standard_actor


def test_yaml_uses_gsm8k_and_megatron_actor_without_staging_ref() -> None:
    """The example data and actor backend are explicit while ref stays standard."""
    config = yaml.safe_load((EXAMPLE_ROOT / "gsm8k_grpo_cpu_staged.yaml").read_text())

    assert config["train_dataset"]["path"] == "openai/gsm8k"
    assert config["valid_dataset"]["path"] == "openai/gsm8k"
    assert config["actor"]["backend"] == ("megatron:(attn:d8p1t1|ffn:d1p1t1e8)")
    assert config["actor"]["path"] == (
        "${oc.env:QWEN3_30B_A3B_MODEL_PATH,Qwen/Qwen3-30B-A3B}"
    )
    assert config["actor"]["weight_update_mode"] == "awex"
    assert config["cpu_staged"]["enabled"] is True
    assert config["rollout"]["backend"] == "sglang:d1t8p1"
    assert config["rollout"]["setup_timeout"] == 900.0
    assert "scheduling_strategy" not in config["rollout"]
    assert config["sglang"]["enable_dp_attention"] is True
    assert config["sglang"]["dp_size"] == 8
    assert config["sglang"]["ep_size"] == 8
    assert config["ref"]["backend"] == "${actor.backend}"
    assert config["ref"]["optimizer"] is None
    assert config["cpu_staged"]["checkpoint_snapshot_root"] is None
    assert config["enable_offload"] is False
    assert config["memory_profiler"] == {
        "profile_steps": [1],
        "max_entries": 100000,
    }


def test_qwen3_moe_backend_is_attention_dp8_and_ffn_ep8() -> None:
    """The folded Megatron allocation keeps dense DP and expert EP distinct."""
    config = yaml.safe_load((EXAMPLE_ROOT / "gsm8k_grpo_cpu_staged.yaml").read_text())

    allocation = ModelAllocation.from_str(config["actor"]["backend"])
    parallel = allocation.parallel

    assert parallel.world_size == 8
    assert parallel.data_parallel_size == 8
    assert parallel.tensor_parallel_size == 1
    assert parallel.pipeline_parallel_size == 1
    assert parallel.context_parallel_size == 1
    assert parallel.expert_parallel_size == 8
    assert parallel.expert_data_parallel_size == 1


@pytest.mark.parametrize("snapshot_root", [None, "/job/scratch/rollback"])
def test_snapshot_root_environment_is_optional_and_never_hardcoded(
    monkeypatch: pytest.MonkeyPatch, snapshot_root: str | None
) -> None:
    """Workers construct the core config with or without a job-owned root."""
    monkeypatch.delenv(CPU_STAGED_SNAPSHOT_ROOT_ENV, raising=False)
    if snapshot_root is not None:
        monkeypatch.setenv(CPU_STAGED_SNAPSHOT_ROOT_ENV, snapshot_root)

    config = cpu_staged_adamw_config_from_environment()

    assert config.checkpoint_snapshot_root == snapshot_root


@pytest.mark.parametrize("snapshot_root", [None, "/job/scratch/rollback"])
def test_snapshot_root_is_forwarded_to_remote_actor_workers(
    snapshot_root: str | None,
) -> None:
    """Example-local settings reach scheduled workers without a machine path."""
    settings = CPUStagedOffloadSettings(checkpoint_snapshot_root=snapshot_root)
    task = SimpleNamespace(env_vars={})
    config = SimpleNamespace(
        cpu_staged=settings,
        actor=SimpleNamespace(scheduling_spec=[task]),
    )
    environ: dict[str, str] = {}

    values = install_cpu_staged_worker_environment(config, environ)

    assert values == environ == task.env_vars
    if snapshot_root is None:
        assert CPU_STAGED_SNAPSHOT_ROOT_ENV not in values
    else:
        assert values[CPU_STAGED_SNAPSHOT_ROOT_ENV] == snapshot_root


def test_disabled_cpu_staged_does_not_modify_worker_environment() -> None:
    """A baseline run must not receive staged-optimizer worker settings."""
    settings = CPUStagedOffloadSettings(enabled=False)
    task = SimpleNamespace(env_vars={})
    config = SimpleNamespace(
        cpu_staged=settings,
        actor=SimpleNamespace(scheduling_spec=[task]),
    )
    environ: dict[str, str] = {}

    values = install_cpu_staged_worker_environment(config, environ)

    assert values == {}
    assert environ == {}
    assert task.env_vars == {}


def test_example_modules_and_structured_config_load_from_repository_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The documented root-relative imports and YAML form a valid config."""
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
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import examples.cpu_staged_offload.engine; "
                "import examples.cpu_staged_offload.gsm8k_rl_cpu_staged"
            ),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert config_path.endswith("gsm8k_grpo_cpu_staged.yaml")
    assert config.actor.path == "/models/Qwen3-30B-A3B"
    assert config.tokenizer_path == "/models/Qwen3-30B-A3B"
    assert config.train_dataset.path == "openai/gsm8k"
    assert config.valid_dataset.path == "openai/gsm8k"
    assert config.export_style == "concat"
    assert config.agent_run_args == {"max_turns": 2}
    assert config.cpu_staged.enabled is True
    assert config.memory_profiler.profile_steps == [1]


def test_multi_turn_workflow_and_reward_wiring_match_math_example() -> None:
    """The staged actor does not change multi-turn rollout/reward semantics."""
    script = (EXAMPLE_ROOT / "gsm8k_rl_cpu_staged.py").read_text()
    config = yaml.safe_load((EXAMPLE_ROOT / "gsm8k_grpo_cpu_staged.yaml").read_text())

    assert config["export_style"] == "concat"
    assert config["agent_run_args"]["max_turns"] == 2
    assert "gsm8k_reward_fn" in script
    assert "MultiturnRLVRWorkflow" in script
    assert "client.apply_reward_discount(turn_discount=0.9)" in script
    assert "client.export_interactions(style=self.export_style)" in script


def test_readme_documents_launch_resources_and_snapshot_contract() -> None:
    """Operators receive the required distinction and filesystem constraints."""
    readme = (EXAMPLE_ROOT / "README.md").read_text()

    assert "uv run python examples/cpu_staged_offload/gsm8k_rl_cpu_staged.py" in readme
    assert "examples/cpu_staged_offload/gsm8k_grpo_cpu_staged.yaml" in readme
    assert "one node with eight GPUs" in readme
    assert "Attention DP8 and FFN EP8" in readme
    assert "Eight Megatron actor ranks" in readme
    assert "One eight-GPU SGLang server" in readme
    assert "QWEN3_30B_A3B_MODEL_PATH" in readme
    assert "AREAL_CPU_STAGED_SNAPSHOT_ROOT" in readme
    assert "real directory, not a symlink" in readme
    assert "fsync" in readme
    assert "non-zero" in readme
    assert "enable_offload" in readme
    assert "CPU-staged AdamW is an optimizer implementation" in readme


def test_example_directory_contains_only_the_documented_six_files() -> None:
    """Keep the standalone example surface small and remotely importable."""
    assert {path.name for path in EXAMPLE_ROOT.iterdir() if path.is_file()} == {
        "README.md",
        "__init__.py",
        "config.py",
        "engine.py",
        "gsm8k_grpo_cpu_staged.yaml",
        "gsm8k_rl_cpu_staged.py",
    }
