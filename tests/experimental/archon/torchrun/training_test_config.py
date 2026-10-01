"""Configuration types + YAML/CLI loader for ArchonEngine training tests.

The runner accepts regular AReaL YAML plus ``test_config``:

```yaml
experiment_name: archon_train_test
trial_name: trial0
cluster:
  fileroot: /storage/openpsi/experiments

actor:               # Standard AReaL TrainEngineConfig/PPOActorConfig fields.
  backend: archon:d2
  path: /path/to/model
  dtype: bfloat16
  mb_spec:
    max_tokens_per_mb: 5596
  optimizer:          # Ignored when test_config.disable_optimizer=true.
    type: adam
    lr: 1e-5
    ...
  tree_training_mode: dta
  packing_algorithm: ffd

test_config:         # Test-only knobs, see ``TestOnlyConfig``.
  step: 4
  data_dir: /path/to/data_dir
  disable_optimizer: false
  save_diff: true
```

OmegaConf-style dotlist overrides are supported on the CLI, eg::

    torchrun --nproc_per_node=2 run_archon_training_test.py \
        --config config.yaml \
        test_config.step=4 test_config.disable_optimizer=true
"""

from __future__ import annotations

import argparse
import dataclasses
import getpass
import os
import re
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from omegaconf import DictConfig, OmegaConf

from areal.api.alloc_mode import ParallelStrategy
from areal.api.alloc_mode import _AllocationMode as AllocationMode
from areal.api.cli_args import TrainEngineConfig
from areal.utils.logging import getLogger

_LOGGER = getLogger("TrainingTestConfig")


def _log_config_info(msg: str, *args: object) -> None:
    """Log once per node before process group init (torchrun sets LOCAL_RANK)."""
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return
    _LOGGER.info(msg, *args)


@dataclass
class TestOnlyConfig:
    """Test-only settings not inherited from AReaL configs."""

    step: int = -1
    data_dir: str = ""
    disable_optimizer: bool = False
    save_diff: bool = True
    save_params: bool = False
    save_initial_params: bool = False
    seed: int = 42
    entropy_coef: float = 0.0
    entropy_mode: str = "sum"
    save_full_diff_tensors_fp32: bool = False
    # After the final training step, write ``last_grads.pt`` (per-param grad stats).
    # With ``disable_optimizer=true``, grads are cleared inside the patched step, so
    # the runner captures them in that hook on the last step only.
    dump_last_grads: bool = False
    # When ``dump_last_grads``, also store full CPU fp32 tensors under
    # ``grad_tensors_fp32`` (like ``delta_tensors_fp32`` in diff.pt). Set false to
    # save disk/memory (stats-only).
    save_full_last_grad_tensors_fp32: bool = True
    is_critic: bool = False
    # After load, take at most this many sequences per .pt (0 = no cap).
    max_sequences_per_pt: int = 0
    # After ``ArchonLMEngine.initialize`` (HF load + buffers), export weights with
    # ``ArchonEngine.save`` / ``save_model_to_hf`` into this directory. ``None`` = skip.
    save_hf_checkpoint_dir: str | None = None
    # Optional forward parity check: compare current tree_training_mode output with
    # a baseline engine forced to ``tree_training_mode=disabled`` at one step.
    dump_forward_compare: bool = False

    def __post_init__(self) -> None:
        if self.step is None or int(self.step) < 0:
            raise ValueError(
                f"test_config.step must be a non-negative integer, got {self.step}."
            )
        if not self.data_dir:
            raise ValueError(
                "test_config.data_dir is required and must be a non-empty path."
            )
        if float(self.entropy_coef) < 0:
            raise ValueError(
                f"test_config.entropy_coef must be >= 0, got {self.entropy_coef}."
            )
        valid_entropy_modes = {"mean", "sum"}
        if self.entropy_mode not in valid_entropy_modes:
            raise ValueError(
                f"test_config.entropy_mode must be one of "
                f"{sorted(valid_entropy_modes)}, got '{self.entropy_mode}'."
            )
        if int(self.max_sequences_per_pt) < 0:
            raise ValueError(
                "test_config.max_sequences_per_pt must be >= 0 "
                f"(0 = no cap), got {self.max_sequences_per_pt}."
            )


@dataclass
class TestParallelConfig:
    """Subset of ParallelStrategy fields exposed to the test YAML."""

    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1

    def to_parallel_strategy(self) -> ParallelStrategy:
        return ParallelStrategy(
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            data_parallel_size=self.data_parallel_size,
            context_parallel_size=self.context_parallel_size,
            expert_parallel_size=self.expert_parallel_size,
            expert_tensor_parallel_size=self.expert_tensor_parallel_size,
        )

    def to_compact_tag(self) -> str:
        """Compact path-friendly tag, e.g. ``d8t2c2``."""
        parts = [f"d{int(self.data_parallel_size)}"]
        if int(self.pipeline_parallel_size) > 1:
            parts.append(f"p{int(self.pipeline_parallel_size)}")
        if int(self.tensor_parallel_size) > 1:
            parts.append(f"t{int(self.tensor_parallel_size)}")
        if int(self.context_parallel_size) > 1:
            parts.append(f"c{int(self.context_parallel_size)}")
        if int(self.expert_parallel_size) > 1:
            parts.append(f"e{int(self.expert_parallel_size)}")
        if int(self.expert_tensor_parallel_size) > 1:
            parts.append(f"et{int(self.expert_tensor_parallel_size)}")
        return "".join(parts)


@dataclass
class ArchonTrainingTestConfig:
    """Top-level container combining AReaL engine config + test knobs."""

    engine: TrainEngineConfig = field(default_factory=TrainEngineConfig)
    parallel: TestParallelConfig = field(default_factory=TestParallelConfig)
    test_config: TestOnlyConfig = field(default_factory=TestOnlyConfig)
    fileroot: str = ""

    @staticmethod
    def _safe_token(value: str, *, fallback: str) -> str:
        s = (value or "").strip()
        if not s:
            s = fallback
        s = s.replace(os.sep, "_")
        if os.altsep:
            s = s.replace(os.altsep, "_")
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
        s = s.strip("._-")
        return s or fallback

    @staticmethod
    def _expand_path(path: str) -> str:
        return os.path.expanduser(os.path.expandvars(path))

    def resolve_dump_dir(self) -> str:
        """Pick a compact dump_dir under regular training log roots."""
        exp = self._safe_token(
            str(self.engine.experiment_name or "archon_train_test"),
            fallback="archon_train_test",
        )
        trial = self._safe_token(
            str(self.engine.trial_name or "trial0"),
            fallback="trial0",
        )
        tree_mode = self._safe_token(
            str(getattr(self.engine, "tree_training_mode", "unknown") or "unknown"),
            fallback="unknown",
        )
        parallel_tag = self._safe_token(self.parallel.to_compact_tag(), fallback="d1")
        model_name = self._safe_token(
            Path(str(self.engine.path or "")).name, fallback="model"
        )
        leaf = f"{tree_mode}_{parallel_tag}_{model_name}"

        if self.fileroot:
            # Align with AReaL StatsLogger layout:
            # <fileroot>/logs/<user>/<experiment>/<trial>
            base = (
                Path(self._expand_path(self.fileroot))
                / "logs"
                / getpass.getuser()
                / exp
                / trial
            )
        else:
            base = Path.cwd() / exp / trial
        return str(base / leaf)


def _merge_yaml_and_overrides(
    yaml_path: str,
    overrides: list[str],
) -> DictConfig:
    yaml_cfg = OmegaConf.load(yaml_path)
    if not isinstance(yaml_cfg, DictConfig):
        raise ValueError(
            f"Top-level YAML at {yaml_path} must be a mapping, got {type(yaml_cfg)}."
        )
    override_cfg = OmegaConf.from_dotlist(list(overrides))
    return OmegaConf.merge(yaml_cfg, override_cfg)


def _as_dict(section: Any) -> dict[str, Any]:
    """Resolve an OmegaConf node into a plain ``dict``."""
    if section is None:
        return {}
    if isinstance(section, DictConfig):
        return OmegaConf.to_container(section, resolve=True)  # type: ignore[return-value]
    if isinstance(section, dict):
        return dict(section)
    raise TypeError(f"Expected mapping-like config section, got {type(section)}")


def _coerce_value(tp: Any, value: Any) -> Any:
    """Best-effort coercion of ``value`` to the dataclass field type ``tp``.

    Handles ``Optional[X]``, nested dataclasses, and ``list[DataClass]`` /
    ``tuple[DataClass, ...]``. Other annotations (``Literal``, ``int``, ``str``,
    ``dict``, ...) pass through unchanged so OmegaConf primitives continue to
    work.
    """
    if value is None:
        return None

    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin is Union or origin is types.UnionType:
        # Try the non-None variants in order; first one that accepts the value
        # wins. Primitives pass through since ``_coerce_value`` is a no-op for
        # non-dataclass leaf types.
        non_none = [a for a in args if a is not type(None)]
        for alt in non_none:
            if dataclasses.is_dataclass(alt) and isinstance(value, dict):
                return _build_dataclass(alt, value)
        return value

    if dataclasses.is_dataclass(tp):
        if isinstance(value, dict):
            return _build_dataclass(tp, value)
        return value

    if origin in (list, tuple) and args:
        inner = args[0]
        if dataclasses.is_dataclass(inner) and isinstance(value, (list, tuple)):
            coerced = [_coerce_value(inner, v) for v in value]
            return tuple(coerced) if origin is tuple else coerced

    return value


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate ``cls`` from ``data``, recursively coercing nested fields.

    Fields not present in ``data`` fall back to their dataclass defaults so
    partial YAML sections are allowed. Unknown keys raise ``TypeError`` to
    surface typos early.
    """
    assert dataclasses.is_dataclass(cls), f"{cls} is not a dataclass"
    hints = typing.get_type_hints(cls)
    init_kwargs: dict[str, Any] = {}
    known_names = {f.name for f in dataclasses.fields(cls) if f.init}
    for key, value in data.items():
        if key not in known_names:
            raise TypeError(
                f"Unknown field '{key}' for {cls.__name__}; "
                f"valid fields: {sorted(known_names)[:30]}..."
            )
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        if f.name not in data:
            continue
        tp = hints.get(f.name, f.type)
        init_kwargs[f.name] = _coerce_value(tp, data[f.name])
    return cls(**init_kwargs)


def _build_engine_config_from_actor(actor_section: Any) -> TrainEngineConfig:
    """Project actor config onto ``TrainEngineConfig`` fields only."""
    actor_data = _as_dict(actor_section)
    train_fields = {f.name for f in dataclasses.fields(TrainEngineConfig) if f.init}
    engine_data = {k: v for k, v in actor_data.items() if k in train_fields}
    return _build_dataclass(TrainEngineConfig, engine_data)


def _build_engine_config(
    actor_section: Any, merged_cfg: DictConfig
) -> TrainEngineConfig:
    """Build ``TrainEngineConfig`` from top-level ``actor`` section only."""
    top_exp = merged_cfg.get("experiment_name") if merged_cfg is not None else None
    top_trial = merged_cfg.get("trial_name") if merged_cfg is not None else None
    default_exp = str(top_exp or "archon_train_test")
    default_trial = str(top_trial or "trial0")

    if actor_section is None:
        raise ValueError("Missing required top-level 'actor' section in config YAML.")
    actor_data = _as_dict(actor_section)
    actor_data.setdefault("experiment_name", default_exp)
    actor_data.setdefault("trial_name", default_trial)
    cfg = _build_engine_config_from_actor(actor_data)

    _log_config_info(
        "Resolved TrainEngineConfig from top-level 'actor' section.",
    )
    return cfg


def _parallel_strategy_to_test_config(strategy: ParallelStrategy) -> TestParallelConfig:
    """Convert ``ParallelStrategy`` to ``TestParallelConfig``."""
    return TestParallelConfig(
        data_parallel_size=int(strategy.data_parallel_size),
        tensor_parallel_size=int(strategy.tensor_parallel_size),
        pipeline_parallel_size=int(strategy.pipeline_parallel_size),
        context_parallel_size=int(strategy.context_parallel_size),
        expert_parallel_size=int(strategy.expert_parallel_size),
        expert_tensor_parallel_size=int(strategy.expert_tensor_parallel_size),
    )


def _build_parallel_config(section: Any) -> TestParallelConfig:
    """Build ``TestParallelConfig`` from mapping or allocation-mode string.

    Supported forms:
    - Mapping (legacy):
        parallel:
          data_parallel_size: 2
          tensor_parallel_size: 1
          ...
    - String (reuses AllocationMode parser):
        parallel: archon:d8
        parallel: sglang:d16+archon:d8
    """
    if section is None:
        return TestParallelConfig()

    if isinstance(section, str):
        mode = AllocationMode.from_str(section)
        train_allocs = [
            a for a in mode.allocations if a.backend in ("fsdp", "megatron", "archon")
        ]
        if len(train_allocs) != 1:
            raise ValueError(
                "parallel string must resolve to exactly one training allocation "
                f"(got {len(train_allocs)}): {section}"
            )
        alloc = train_allocs[0]
        if alloc.backend != "archon":
            raise ValueError(
                "Only archon backend is supported by this test runner. "
                f"Got training backend '{alloc.backend}' from parallel='{section}'."
            )
        if alloc.parallel is None:
            raise ValueError(
                f"Resolved archon allocation has no parallel strategy: {section}"
            )
        return _parallel_strategy_to_test_config(alloc.parallel)

    return TestParallelConfig(**_as_dict(section))


def _resolve_output_fileroot(merged: DictConfig) -> str:
    """Resolve output fileroot with priority: stats_logger > cluster."""
    stats_logger = _as_dict(merged.get("stats_logger") if merged else None)
    stats_logger_fileroot = stats_logger.get("fileroot")
    if stats_logger_fileroot:
        return str(stats_logger_fileroot)

    cluster = _as_dict(merged.get("cluster") if merged else None)
    cluster_fileroot = cluster.get("fileroot")
    if cluster_fileroot:
        return str(cluster_fileroot)
    return ""


def load_training_test_config(
    argv: list[str] | None = None,
) -> tuple[ArchonTrainingTestConfig, str]:
    """Parse CLI and return a resolved ``ArchonTrainingTestConfig``."""
    parser = argparse.ArgumentParser(
        description="Run ArchonEngine training-side test under torchrun."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file.",
    )
    args, overrides = parser.parse_known_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    merged = _merge_yaml_and_overrides(str(config_path), overrides)

    if merged and merged.get("engine") is not None:
        raise ValueError(
            "This runner only supports regular AReaL YAML + test_config. "
            "Do not provide top-level 'engine'; use top-level 'actor' instead."
        )
    if merged and merged.get("parallel") is not None:
        raise ValueError(
            "This runner derives parallel strategy from actor.backend. "
            "Do not provide top-level 'parallel'."
        )

    engine_cfg = _build_engine_config(merged.get("actor") if merged else None, merged)

    actor_data = _as_dict(merged.get("actor") if merged else None)
    actor_backend = actor_data.get("backend")
    if not actor_backend:
        raise ValueError("actor.backend is required and must be a non-empty string.")
    parallel_section = actor_backend
    _log_config_info("Resolved 'parallel' from actor.backend=%s", actor_backend)
    parallel_cfg = _build_parallel_config(parallel_section)

    test_cfg = TestOnlyConfig(**_as_dict(merged.get("test_config") if merged else None))
    fileroot = _resolve_output_fileroot(merged)

    cfg = ArchonTrainingTestConfig(
        engine=engine_cfg,
        parallel=parallel_cfg,
        test_config=test_cfg,
        fileroot=fileroot,
    )
    return cfg, str(config_path)


def ensure_dump_dir(cfg: ArchonTrainingTestConfig, rank: int) -> str:
    """Create (on rank 0) and return the resolved dump_dir."""
    dump_dir = cfg.resolve_dump_dir()
    if rank == 0:
        os.makedirs(dump_dir, exist_ok=True)
    return dump_dir
