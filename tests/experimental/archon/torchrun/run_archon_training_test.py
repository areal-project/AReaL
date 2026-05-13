"""Training-side smoke/benchmark runner for ``ArchonLMEngine``.

This script runs ``N`` training steps of ``ArchonLMEngine.train_batch`` under
``torch.distributed`` and records global per-step loss, elapsed time and peak
GPU memory. At the end it optionally dumps ``diff.pt`` (parameter update
statistics) so two runs can be compared offline via
:mod:`compare_training_dumps`.

Launch with torchrun::

    torchrun --nproc_per_node=$WORLD_SIZE \\
        tests/experimental/archon/torchrun/run_archon_training_test.py \\
        --config tests/experimental/archon/torchrun/archon_training_test.yaml \\
        test_config.step=4 \\
        test_config.data_dir=/path/to/data

Primary outputs land under ``<dump_dir>/``:

- ``stats.jsonl``  -- one global JSON record per step (rank-aggregated)
- ``diff.pt``      -- per-parameter update stats (saved on rank 0 only)
- ``last_grads.pt`` -- optional per-parameter gradient stats and, by default, full
  fp32 gradient tensors (``grad_tensors_fp32``) after the final step
  (``test_config.dump_last_grads=true``; see :class:`TestOnlyConfig`).
  Each ``params`` entry includes ``requires_grad``; ``requires_grad_meta`` summarizes
  all ``named_parameters()`` without ``full_tensor``.

The runner is intentionally narrow: inputs are assumed to be ``list[Tensor]``
(1-D token ids) per ``.pt`` file, and the loss function is hard-wired to a
typical ``grpo_loss_fn`` setup (or ``ppo_critic_loss_fn`` when
``test_config.is_critic`` is true).

``test_config.is_critic`` is applied to ``TrainEngineConfig.is_critic`` before
constructing the engine so the model (e.g. ``score`` vs ``output`` head), HF
load path, and loss stay aligned. Checkpoints are always read from YAML
``actor.path`` (``cfg.engine.path``). For critic models, whether the value
head is filled from the HF ``lm_head`` tensor depends on the Archon
``state_dict_adapter`` for that architecture (see engine load warnings for
missing / unexpected keys).
"""

from __future__ import annotations

import dataclasses
import functools
import glob
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

# Make repo root importable when invoked via torchrun from any cwd.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.experimental.archon.torchrun.training_test_config import (  # noqa: E402
    ArchonTrainingTestConfig,
    ensure_dump_dir,
    load_training_test_config,
)

from areal.api.io_struct import FinetuneSpec, SaveLoadMeta  # noqa: E402
from areal.experimental.archon.utils import strip_wrapper_prefixes  # noqa: E402
from areal.experimental.engine.archon_engine import ArchonLMEngine  # noqa: E402
from areal.infra.platforms import current_platform  # noqa: E402
from areal.trainer.ppo.actor import grpo_loss_fn  # noqa: E402
from areal.trainer.ppo.critic import ppo_loss_fn as ppo_critic_loss_fn  # noqa: E402
from areal.utils.data import concat_batch  # noqa: E402
from areal.utils.logging import getLogger  # noqa: E402
from areal.utils.network import find_free_ports  # noqa: E402

# Fixed prompt ratio for synthetic loss mask construction.
_PROMPT_RATIO = 0.3

# -----------------------------------------------------------------------------
# Distributed setup
# -----------------------------------------------------------------------------


def _setup_distributed_environment() -> tuple[int, int]:
    """Initialize the global process group using torchrun env vars."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(find_free_ports(1)[0]))

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend="nccl",
        init_method=(f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"),
        world_size=world_size,
        rank=rank,
    )
    current_platform.set_device(int(os.environ["LOCAL_RANK"]))
    return rank, world_size


# -----------------------------------------------------------------------------
# Data loading / trajectory construction
# -----------------------------------------------------------------------------


def _list_step_files(data_dir: str) -> list[str]:
    """Sort .pt files in ``data_dir`` lexicographically (ascii dict order)."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir does not exist or is not a dir: {data_dir}")
    files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    if not files:
        raise FileNotFoundError(f"No .pt files under {data_dir}")
    return files


def _load_sequences(pt_path: str) -> list[torch.Tensor]:
    seqs = torch.load(pt_path, map_location="cpu", weights_only=True)
    if not isinstance(seqs, list) or not seqs:
        raise ValueError(
            f"Expected non-empty list[Tensor] in {pt_path}, got {type(seqs)}"
        )
    for i, s in enumerate(seqs):
        if not isinstance(s, torch.Tensor) or s.ndim != 1:
            raise ValueError(
                f"Entry {i} of {pt_path} is not a 1-D tensor: "
                f"type={type(s)}, ndim={getattr(s, 'ndim', None)}"
            )
    return seqs


def _synthetic_advantages(seq_len: int, global_idx: int) -> torch.Tensor:
    """Deterministic per-token advantages for tests (CPU float32).

    Depends on ``seq_len`` and ``global_idx`` so that changing truncation /
    ``max_tokens_per_mb`` or which sequence is loaded changes targets in a
    structured, reproducible way (unlike i.i.d. Gaussian noise).
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    t = torch.arange(seq_len, dtype=torch.float32)
    denom = max(float(seq_len), 1.0)
    phase = 2.0 * math.pi * (t + 0.5) / denom
    seq_phase = 2.0 * math.pi * float(global_idx % 997) / 997.0
    return (torch.sin(phase + seq_phase) * 0.5).unsqueeze(0)


def _synthetic_old_values(seq_len: int, global_idx: int) -> torch.Tensor:
    """Deterministic per-token old values (``input_data['values']``) for tests.

    Replaces i.i.d. ``randn`` so ``returns = values + advantages`` is a smooth
    function of position and ``global_idx`` only, making critic targets easier
    to reason about and reproduce across DTA / batching.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    t = torch.arange(seq_len, dtype=torch.float32)
    denom = max(float(seq_len), 1.0)
    # Different phase from :func:`_synthetic_advantages` so the sum is not redundant.
    phase = 2.0 * math.pi * (1.5 * t + 0.25) / denom
    seq_phase = 2.0 * math.pi * float((global_idx * 2 + 1) % 991) / 991.0
    return (torch.cos(phase + seq_phase) * 0.4).unsqueeze(0)


def _build_trajectory(
    input_ids: torch.Tensor,
    global_idx: int,
    base_seed: int,
    max_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    """Wrap one 1-D token sequence as a GRPO-ready trajectory dict.

    The per-sequence RNG seed is derived from ``global_idx`` for logprobs.
    ``advantages`` and ``values`` (hence ``returns``) are **non-random** (see
    :func:`_synthetic_advantages` and :func:`_synthetic_old_values`) so
    critic/actor tests behave reproducibly when sequence length or batch
    composition changes.
    """
    assert input_ids.ndim == 1
    seq_len = int(min(int(input_ids.numel()), int(max_tokens)))
    if seq_len <= 0:
        raise ValueError(f"Sequence at idx {global_idx} has non-positive length.")

    ids = input_ids[:seq_len].long().unsqueeze(0).contiguous()
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    loss_mask = torch.zeros(1, seq_len)
    prompt_len = max(1, int(seq_len * _PROMPT_RATIO))
    loss_mask[:, prompt_len:] = 1.0

    gen = torch.Generator(device="cpu").manual_seed(int(base_seed) + int(global_idx))
    logprobs = torch.randn(1, seq_len, generator=gen) * 0.5 - 2.0
    old_logprobs = logprobs.clone()
    advantages = _synthetic_advantages(seq_len, global_idx)
    rewards = torch.randint(0, 2, (1,), generator=gen).float()
    values = _synthetic_old_values(seq_len, global_idx)
    returns = values + advantages

    traj = {
        "input_ids": ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "logprobs": logprobs,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
        "rewards": rewards,
        "values": values,
        "returns": returns,
        "prox_logp": old_logprobs.clone(),
    }
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in traj.items()
    }


def _build_local_trajectories(
    seqs: list[torch.Tensor],
    dp_rank: int,
    dp_world_size: int,
    base_seed: int,
    max_tokens: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Each rank owns a disjoint stride of the sequence list.

    Per-rank sequence counts may differ (e.g. when ``len(seqs)`` is not a
    multiple of ``dp_world_size``). No all-gather / load-balancing redistribution.
    """
    if len(seqs) < dp_world_size:
        raise ValueError(
            f"Need at least dp_world_size={dp_world_size} sequences, got {len(seqs)}."
        )
    out: list[dict[str, Any]] = []
    for global_i in range(dp_rank, len(seqs), dp_world_size):
        out.append(
            _build_trajectory(
                input_ids=seqs[global_i],
                global_idx=global_i,
                base_seed=base_seed,
                max_tokens=max_tokens,
                device=device,
            )
        )
    return out


# -----------------------------------------------------------------------------
# Loss function / engine patching
# -----------------------------------------------------------------------------


# Reasonable defaults mirroring ``tests/experimental/archon/test_grpo.py``.
_GRPO_KW: dict[str, Any] = dict(
    eps_clip=0.2,
    eps_clip_higher=None,
    c_clip=None,
    importance_sampling_level="token",
    current_version=1,
    prox_logp_method="recompute",
    use_sapo_loss=False,
    use_decoupled_loss=False,
)


def _loss_weight_fn(input_data: dict[str, Any]) -> torch.Tensor:
    mask = input_data["loss_mask"]
    return mask.count_nonzero()


def _make_loss_fn(cfg: ArchonTrainingTestConfig):
    """Build test loss with optional entropy regularization."""
    if cfg.test_config.is_critic:
        return functools.partial(ppo_critic_loss_fn, eps_clip=3.0)

    base_loss_fn = functools.partial(grpo_loss_fn, **_GRPO_KW)
    entropy_coef = float(cfg.test_config.entropy_coef)
    entropy_mode = str(cfg.test_config.entropy_mode)
    if entropy_coef <= 0:
        return base_loss_fn

    def _loss_fn(logprobs, entropy, input_data, **kwargs):
        base_loss = base_loss_fn(logprobs, entropy, input_data, **kwargs)
        loss_mask = input_data["loss_mask"].bool()
        valid_entropy = entropy.float().masked_select(loss_mask)
        if valid_entropy.numel() == 0:
            entropy_term = base_loss * 0.0
        elif entropy_mode == "mean":
            entropy_term = -valid_entropy.mean()
        else:
            entropy_term = -valid_entropy.sum()
        return base_loss + entropy_coef * entropy_term

    return _loss_fn


def _patch_engine_for_test(
    engine: ArchonLMEngine,
    disable_optimizer: bool,
    *,
    dump_last_grads: bool = False,
    num_training_steps: int = 0,
    last_grad_holder: dict[str, Any] | None = None,
    save_full_last_grad_tensors_fp32: bool = True,
) -> None:
    """Inject optional optimizer no-ops onto the engine."""
    if not disable_optimizer:
        return

    noop_step_counter = {"n": 0}

    def _noop_zero_grad(self):
        for p in self._get_all_parameters():
            if p.grad is not None:
                p.grad = None

    def _noop_step(self):
        grad_norm = 0.0
        for p in self._get_all_parameters():
            if p.grad is not None:
                grad_norm += float(p.grad.detach().float().norm().item()) ** 2
        grad_norm = grad_norm**0.5
        noop_step_counter["n"] += 1
        if (
            dump_last_grads
            and last_grad_holder is not None
            and noop_step_counter["n"] == int(num_training_steps)
        ):
            payload = _build_grad_snapshot_payload(
                self,
                save_full_grad_tensors_fp32=save_full_last_grad_tensors_fp32,
            )
            if payload is not None:
                last_grad_holder.clear()
                last_grad_holder.update(payload)
        _noop_zero_grad(self)
        return {
            "update_successful": 1.0,
            "grad_norm": grad_norm,
            "lr": 0.0,
        }

    engine.optimizer_zero_grad = types.MethodType(_noop_zero_grad, engine)
    engine.optimizer_step = types.MethodType(_noop_step, engine)


# -----------------------------------------------------------------------------
# Engine lifecycle
# -----------------------------------------------------------------------------


def _resolve_test_hf_export_dir(cfg: ArchonTrainingTestConfig) -> str:
    """Return absolute export dir, or ``\"\"`` when ``save_hf_checkpoint_dir`` is None."""
    raw = cfg.test_config.save_hf_checkpoint_dir
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    return os.path.abspath(os.path.expanduser(s))


def _create_engine(cfg: ArchonTrainingTestConfig) -> ArchonLMEngine:
    """Construct + initialize an ``ArchonLMEngine`` from the test config."""
    parallel_strategy = cfg.parallel.to_parallel_strategy()

    engine_cfg = cfg.engine
    critic = bool(cfg.test_config.is_critic)
    if critic != bool(engine_cfg.is_critic) and int(os.environ.get("RANK", "0")) == 0:
        getLogger("[ArchonTrainingTest]").warning(
            "test_config.is_critic=%s overrides actor.is_critic=%s for engine "
            "(model head + checkpoint layout must match critic vs actor).",
            critic,
            engine_cfg.is_critic,
        )
    replace_kw: dict[str, Any] = {"is_critic": critic}
    if cfg.test_config.disable_optimizer:
        # Skip optimizer creation entirely so no Adam state is allocated.
        replace_kw["optimizer"] = None
    engine_cfg = dataclasses.replace(engine_cfg, **replace_kw)

    engine = ArchonLMEngine(engine_cfg)
    engine.create_process_group(parallel_strategy=parallel_strategy)

    ft_spec = FinetuneSpec(
        total_train_epochs=1,
        dataset_size=max(1, int(cfg.test_config.step)),
        train_batch_size=1,
    )
    engine.initialize(addr=None, ft_spec=ft_spec)

    hf_export = _resolve_test_hf_export_dir(cfg)
    if hf_export:
        meta = SaveLoadMeta(
            path=hf_export,
            weight_format="hf",
            with_optim=False,
            tokenizer=engine.tokenizer,
        )
        engine.save(meta)

    return engine


def _destroy_engine(engine: ArchonLMEngine | None) -> None:
    if engine is not None:
        engine.destroy()
    if dist.is_initialized():
        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# Parameter diff dump
# -----------------------------------------------------------------------------


def _materialize_full_param(param: torch.Tensor) -> torch.Tensor:
    """Return a full (unsharded) tensor for one parameter."""
    from torch.distributed.tensor import DTensor

    if isinstance(param, DTensor):
        return param.full_tensor()
    return param


def _to_dump_name_tensors(
    engine: ArchonLMEngine, raw_name: str, tensor: torch.Tensor
) -> list[tuple[str, torch.Tensor]]:
    """Convert one Archon parameter into dump-name/tensor pairs.

    Prefer HuggingFace keys when ``state_dict_adapter`` is available; otherwise
    use wrapper-stripped Archon keys.
    """
    adapter = engine.state_dict_adapter
    if adapter is not None:
        mapped = adapter.convert_single_to_hf(raw_name, tensor)
        if mapped:
            return [(strip_wrapper_prefixes(name), value) for name, value in mapped]
    return [(strip_wrapper_prefixes(raw_name), tensor)]


def _build_grad_snapshot_payload(
    engine: ArchonLMEngine,
    *,
    save_full_grad_tensors_fp32: bool = True,
) -> dict[str, Any] | None:
    """Collect per-parameter gradient statistics (all ranks must call).

    Optionally embeds full CPU fp32 gradient tensors under ``grad_tensors_fp32``
    (same role as ``delta_tensors_fp32`` in ``diff.pt``).

    Returns a payload dict on rank 0 only; other ranks return ``None`` after
    participating in any required ``full_tensor()`` collectives.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    if dist.is_initialized():
        dist.barrier(group=engine.cpu_group)

    if rank == 0:
        per_param: dict[str, dict[str, float]] = {}
        full_grad_tensors_fp32: dict[str, torch.Tensor] | None = (
            {} if save_full_grad_tensors_fp32 else None
        )
        global_numel = 0.0
        global_abs_sum = 0.0
        global_l2_sq = 0.0
        global_max_abs = 0.0
    else:
        per_param = {}
        full_grad_tensors_fp32 = None
        global_numel = 0.0
        global_abs_sum = 0.0
        global_l2_sq = 0.0
        global_max_abs = 0.0

    for raw_name, param in engine.model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        full_g = _materialize_full_param(grad)
        if rank == 0:
            for dump_name, dump_tensor in _to_dump_name_tensors(
                engine, raw_name, full_g
            ):
                tensor_f = dump_tensor.detach().to(device="cpu", dtype=torch.float32)
                numel = float(tensor_f.numel())
                if numel <= 0:
                    continue
                abs_t = tensor_f.abs()
                abs_sum = float(abs_t.sum().item())
                l2_sq = float(tensor_f.double().pow(2).sum().item())
                max_abs = float(abs_t.max().item())
                l2 = math.sqrt(max(l2_sq, 0.0))
                if dump_name in per_param:
                    raise ValueError(
                        f"Duplicate grad dump key '{dump_name}' from raw param '{raw_name}'."
                    )
                per_param[dump_name] = {
                    "numel": numel,
                    "mean_abs": abs_sum / numel,
                    "max_abs": max_abs,
                    "l2": l2,
                    "requires_grad": bool(param.requires_grad),
                }
                global_numel += numel
                global_abs_sum += abs_sum
                global_l2_sq += l2_sq
                global_max_abs = max(global_max_abs, max_abs)
                if full_grad_tensors_fp32 is not None:
                    full_grad_tensors_fp32[dump_name] = tensor_f.clone()
        del full_g

    if rank == 0:
        g_l2 = math.sqrt(max(global_l2_sq, 0.0))
        payload: dict[str, Any] = {
            "schema_version": 2,
            "aggregation": "full_tensor_grad_one_param_peak",
            "params": per_param,
            "global": {
                "num_params": len(per_param),
                "numel": global_numel,
                "mean_abs": global_abs_sum / max(global_numel, 1.0),
                "max_abs": global_max_abs,
                "l2": g_l2,
            },
        }
        if full_grad_tensors_fp32 is not None:
            payload["grad_tensors_fp32"] = full_grad_tensors_fp32

        rg_false: list[str] = []
        rg_true = 0
        for rn, p in engine.model.named_parameters():
            if p.requires_grad:
                rg_true += 1
            else:
                rg_false.append(rn)
        payload["requires_grad_meta"] = {
            "num_named_requires_grad_true": rg_true,
            "num_named_requires_grad_false": len(rg_false),
            "named_requires_grad_false": rg_false[:128],
        }
        return payload
    return None


def _snapshot_initial_full_params(
    engine: ArchonLMEngine,
) -> dict[str, torch.Tensor] | None:
    """Capture initial full params on CPU (rank 0 only).

    Each parameter is materialized one-by-one via ``full_tensor()`` and moved to
    CPU immediately. This keeps extra GPU memory bounded by one parameter tensor.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    out: dict[str, torch.Tensor] | None = {} if rank == 0 else None
    for raw_name, param in engine.model.named_parameters():
        full = _materialize_full_param(param)
        if rank == 0:
            assert out is not None
            for dump_name, dump_tensor in _to_dump_name_tensors(engine, raw_name, full):
                if dump_name in out:
                    raise ValueError(
                        f"Duplicate dump key '{dump_name}' from raw param '{raw_name}'."
                    )
                out[dump_name] = (
                    dump_tensor.detach().to(device="cpu", dtype=torch.float32).clone()
                )
        del full
    if dist.is_initialized():
        dist.barrier(group=engine.cpu_group)
    return out


def _save_diff_snapshot(
    engine: ArchonLMEngine,
    initial_params: dict[str, torch.Tensor] | None,
    dump_dir: str,
    filename: str,
    save_full_diff_tensors_fp32: bool = False,
) -> str | None:
    """Save ``diff.pt`` with per-parameter update metrics (rank 0 only)."""
    out_path: str | None = None
    rank = dist.get_rank() if dist.is_initialized() else 0
    if dist.is_initialized():
        dist.barrier(group=engine.cpu_group)

    if rank == 0 and initial_params is None:
        raise RuntimeError("Missing initial params on rank 0 for diff snapshot.")

    # Every rank participates in full_tensor() to ensure collective safety.
    if rank == 0:
        assert initial_params is not None
        per_param: dict[str, dict[str, float]] = {}
        full_delta_tensors_fp32: dict[str, torch.Tensor] | None = (
            {} if save_full_diff_tensors_fp32 else None
        )
        global_numel = 0.0
        global_abs_sum = 0.0
        global_l2_sq = 0.0
        global_ref_l2_sq = 0.0
        global_max_abs = 0.0
    else:
        per_param = {}
        global_numel = 0.0
        global_abs_sum = 0.0
        global_l2_sq = 0.0
        global_ref_l2_sq = 0.0
        global_max_abs = 0.0

    for raw_name, param in engine.model.named_parameters():
        full = _materialize_full_param(param)
        if rank == 0:
            assert initial_params is not None
            for dump_name, dump_tensor in _to_dump_name_tensors(engine, raw_name, full):
                if dump_name not in initial_params:
                    raise KeyError(
                        f"Missing initial parameter for dump key '{dump_name}' "
                        f"(raw='{raw_name}')."
                    )
                initial = initial_params[dump_name]
                current = dump_tensor.detach().to(device="cpu", dtype=torch.float32)
                if current.shape != initial.shape:
                    raise ValueError(
                        f"Shape mismatch for '{dump_name}': current={tuple(current.shape)} "
                        f"vs initial={tuple(initial.shape)}"
                    )
                delta = current - initial
                abs_delta = delta.abs()
                numel = float(delta.numel())
                abs_sum = float(abs_delta.sum().item())
                l2_sq = float(delta.double().pow(2).sum().item())
                ref_l2_sq = float(initial.double().pow(2).sum().item())
                max_abs = float(abs_delta.max().item()) if delta.numel() > 0 else 0.0
                l2 = math.sqrt(max(l2_sq, 0.0))
                ref_l2 = math.sqrt(max(ref_l2_sq, 0.0))
                if dump_name in per_param:
                    raise ValueError(
                        f"Duplicate final dump key '{dump_name}' from raw param '{raw_name}'."
                    )
                per_param[dump_name] = {
                    "numel": numel,
                    "mean_abs_update": abs_sum / max(numel, 1.0),
                    "max_abs_update": max_abs,
                    "l2_update": l2,
                    "rel_l2_update": l2 / max(ref_l2, 1e-12),
                }
                if full_delta_tensors_fp32 is not None:
                    full_delta_tensors_fp32[dump_name] = delta.clone()
                global_numel += numel
                global_abs_sum += abs_sum
                global_l2_sq += l2_sq
                global_ref_l2_sq += ref_l2_sq
                global_max_abs = max(global_max_abs, max_abs)
        del full

    if rank == 0:
        payload = {
            "schema_version": 1,
            "aggregation": "full_tensor_one_param_peak",
            "params": per_param,
            "global": {
                "num_params": len(per_param),
                "numel": global_numel,
                "mean_abs_update": global_abs_sum / max(global_numel, 1.0),
                "max_abs_update": global_max_abs,
                "l2_update": math.sqrt(max(global_l2_sq, 0.0)),
                "rel_l2_update": math.sqrt(max(global_l2_sq, 0.0))
                / max(math.sqrt(max(global_ref_l2_sq, 0.0)), 1e-12),
            },
        }
        if full_delta_tensors_fp32 is not None:
            payload["delta_tensors_fp32"] = full_delta_tensors_fp32

        os.makedirs(dump_dir, exist_ok=True)
        out_path = os.path.join(dump_dir, filename)
        torch.save(payload, out_path)

    if dist.is_initialized():
        dist.barrier(group=engine.cpu_group)
    return out_path


# -----------------------------------------------------------------------------
# Per-step training
# -----------------------------------------------------------------------------


def _build_step_batch(
    *,
    engine: ArchonLMEngine,
    cfg: ArchonTrainingTestConfig,
    step_idx: int,
    step_file: str,
    device: torch.device,
) -> dict[str, Any]:
    """Build one local stride-sharded batch for a data file."""
    dp_rank = engine.data_parallel_rank
    dp_world_size = engine.data_parallel_world_size
    seqs = _load_sequences(step_file)
    cap = int(cfg.test_config.max_sequences_per_pt)
    if cap > 0:
        seqs = seqs[:cap]
        if not seqs:
            raise ValueError(
                f"Step {step_idx}: after max_sequences_per_pt={cap}, "
                f"no sequences left in {step_file}."
            )
    max_tokens = int(engine.config.mb_spec.max_tokens_per_mb)
    trajectories = _build_local_trajectories(
        seqs=seqs,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        base_seed=cfg.test_config.seed + step_idx * 100003,
        max_tokens=max_tokens,
        device=device,
    )
    if not trajectories:
        raise RuntimeError(
            f"Step {step_idx}: local trajectory list is empty for rank {dp_rank}."
        )
    batch, _ = concat_batch(trajectories)
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def _dump_forward_outputs(
    *,
    engine: ArchonLMEngine,
    cfg: ArchonTrainingTestConfig,
    step_idx: int,
    step_file: str,
    device: torch.device,
    dump_dir: str,
) -> dict[str, Any]:
    """Run forward_batch and dump valid-token outputs for this step.

    Actor outputs are token logprobs; critic outputs are token values.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    batch = _build_step_batch(
        engine=engine,
        cfg=cfg,
        step_idx=step_idx,
        step_file=step_file,
        device=device,
    )
    out = engine.forward_batch(input_=batch)
    if not torch.is_tensor(out):
        raise TypeError(
            f"forward dump expects dict input and tensor output, got {type(out)}."
        )
    out_cpu = out.detach().to(device="cpu", dtype=torch.float32)
    valid_mask_cpu = batch["attention_mask"].detach().to(device="cpu").bool()
    local_input_tokens = int(valid_mask_cpu.sum().item())
    mask_matches_output = tuple(valid_mask_cpu.shape) == tuple(out_cpu.shape)
    if mask_matches_output:
        valid_positions = valid_mask_cpu.nonzero(as_tuple=False).to(torch.int32)
        valid_forward = out_cpu[valid_mask_cpu].contiguous()
    else:
        valid_positions = torch.empty((0, 2), dtype=torch.int32)
        valid_forward = torch.empty(0, dtype=torch.float32)
    mismatch_t = torch.tensor(
        [0.0 if mask_matches_output else 1.0],
        dtype=torch.float64,
        device=device,
    )
    token_reduce_t = torch.tensor(
        [
            float(local_input_tokens),
            float(valid_forward.numel()),
        ],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(mismatch_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_reduce_t, op=dist.ReduceOp.SUM)
    global_mask_mismatch_ranks = int(round(float(mismatch_t.item())))
    global_input_tokens = int(round(float(token_reduce_t[0].item())))
    global_valid_output_numel = int(round(float(token_reduce_t[1].item())))

    local_summary = {
        "rank": int(rank),
        "local_input_tokens": local_input_tokens,
        "local_valid_output_numel": int(valid_forward.numel()),
        "local_padded_output_numel": int(out_cpu.numel()),
        "shape": list(out_cpu.shape),
        "local_valid_mask_matches_output": bool(mask_matches_output),
        "global_input_tokens": global_input_tokens,
        "global_valid_output_numel": global_valid_output_numel,
        "global_mask_mismatch_ranks": global_mask_mismatch_ranks,
        "valid_token_positions": valid_positions,
        "valid_forward_fp32": valid_forward,
    }
    if dist.is_initialized():
        all_summaries: list[dict[str, Any] | None] = [
            None for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(all_summaries, local_summary)
    else:
        all_summaries = [local_summary]
    if rank == 0:
        summary_payload = {
            "schema_version": 1,
            "step": int(step_idx),
            "file": os.path.abspath(step_file),
            "tree_training_mode": str(engine.config.tree_training_mode),
            "is_critic": bool(engine.config.is_critic),
            "output_kind": "value" if engine.config.is_critic else "logprob",
            "global_input_tokens": global_input_tokens,
            "global_valid_output_numel": global_valid_output_numel,
            "global_mask_mismatch_ranks": global_mask_mismatch_ranks,
            "per_rank": all_summaries,
        }
        summary_path = os.path.join(dump_dir, f"forward.step{step_idx}.summary.pt")
        torch.save(summary_payload, summary_path)
    if dist.is_initialized():
        dist.barrier(group=engine.cpu_group)
    return {
        "step": int(step_idx),
        "file": os.path.abspath(step_file),
        "global_input_tokens": global_input_tokens,
        "global_valid_output_numel": global_valid_output_numel,
        "global_mask_mismatch_ranks": global_mask_mismatch_ranks,
    }


def _run_single_step(
    *,
    engine: ArchonLMEngine,
    cfg: ArchonTrainingTestConfig,
    step_idx: int,
    step_file: str,
    device: torch.device,
    loss_fn,
) -> dict[str, Any]:
    """Run one training step and return a global (all-rank) stats record."""
    dp_world_size = engine.data_parallel_world_size
    batch = _build_step_batch(
        engine=engine, cfg=cfg, step_idx=step_idx, step_file=step_file, device=device
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    result = engine.train_batch(
        input_=batch,
        loss_fn=loss_fn,
        loss_weight_fn=_loss_weight_fn,
        return_loss=True,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0
    peak_mem_mib = (
        float(torch.cuda.max_memory_allocated() / (1024**2))
        if torch.cuda.is_available()
        else 0.0
    )

    step_loss = float(result.get("loss", float("nan")))
    loss_source = "train_batch_return"

    num_local_seqs = int(batch["input_ids"].shape[0])
    num_local_tokens = int(batch["attention_mask"].sum().item())

    grad_norm_local = float(result.get("grad_norm", float("nan")))
    grad_norm_local_for_max = (
        grad_norm_local if math.isfinite(grad_norm_local) else float("-inf")
    )
    grad_norm_local_for_min = (
        grad_norm_local if math.isfinite(grad_norm_local) else float("inf")
    )
    lr_local = float(result.get("lr", 0.0))
    update_successful_local = float(result.get("update_successful", 0.0))

    # train_batch(return_loss=True) returns each rank's contribution to the
    # globally normalized objective. The correct global loss is the SUM across
    # DP ranks (not a second token-weighted average).
    loss_contrib_local = float(step_loss) if math.isfinite(step_loss) else 0.0
    loss_valid_local = 1.0 if math.isfinite(step_loss) else 0.0

    reduce_sum = torch.tensor(
        [
            loss_contrib_local,
            loss_valid_local,
            float(num_local_seqs),
            float(num_local_tokens),
        ],
        dtype=torch.float64,
        device=device,
    )
    reduce_max = torch.tensor(
        [
            float(elapsed_s),
            float(peak_mem_mib),
            float(grad_norm_local_for_max),
            float(lr_local),
        ],
        dtype=torch.float64,
        device=device,
    )
    reduce_min = torch.tensor(
        [float(update_successful_local), float(grad_norm_local_for_min)],
        dtype=torch.float64,
        device=device,
    )

    if dist.is_initialized():
        dist.all_reduce(reduce_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(reduce_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(reduce_min, op=dist.ReduceOp.MIN)

    global_loss_valid_count = int(round(float(reduce_sum[1].item())))
    global_loss = (
        float(reduce_sum[0].item()) if global_loss_valid_count > 0 else float("nan")
    )
    global_grad_norm = float(reduce_max[2].item())
    if not math.isfinite(global_grad_norm):
        global_grad_norm = float("nan")
    global_grad_norm_min = float(reduce_min[1].item())
    if not math.isfinite(global_grad_norm_min):
        global_grad_norm_min = float("nan")

    return {
        "step": int(step_idx),
        "file": os.path.abspath(step_file),
        "world_size": int(dist.get_world_size()) if dist.is_initialized() else 1,
        "dp_world_size": int(dp_world_size),
        "num_global_sequences": int(round(float(reduce_sum[2].item()))),
        "num_global_tokens": int(round(float(reduce_sum[3].item()))),
        "elapsed_s_max": float(reduce_max[0].item()),
        "peak_mem_mib_max": float(reduce_max[1].item()),
        "loss": float(global_loss),
        "loss_source": f"{loss_source}_global_dp_sum",
        "grad_norm_max": float(global_grad_norm),
        "grad_norm_min": float(global_grad_norm_min),
        "update_successful": float(reduce_min[0].item()),
        "lr_max": float(reduce_max[3].item()),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    cfg, config_path = load_training_test_config(argv)

    rank, world_size = _setup_distributed_environment()
    device = torch.device(current_platform.device_type)
    logger = getLogger(f"[ArchonTrainingTest Rank {rank}]")

    dump_dir = ensure_dump_dir(cfg, rank=rank)
    stats_path = os.path.join(dump_dir, "stats.jsonl")
    diff_path = os.path.join(dump_dir, "diff.pt")
    last_grads_path = os.path.join(dump_dir, "last_grads.pt")
    if rank == 0:
        # Truncate any prior stats file.
        open(stats_path, "w").close()
        # Remove stale diff snapshot so failed runs never expose old results.
        if cfg.test_config.save_diff and os.path.exists(diff_path):
            os.remove(diff_path)
        if cfg.test_config.dump_last_grads and os.path.exists(last_grads_path):
            os.remove(last_grads_path)

    if rank == 0:
        logger.info(
            "config=%s dump_dir=%s world_size=%s",
            config_path,
            dump_dir,
            world_size,
        )

    step_files = _list_step_files(cfg.test_config.data_dir)
    if rank == 0:
        logger.info(
            "Found %d .pt files in %s",
            len(step_files),
            cfg.test_config.data_dir,
        )

    engine: ArchonLMEngine | None = None

    try:
        engine = _create_engine(cfg)
        last_grad_holder: dict[str, Any] = {}
        _patch_engine_for_test(
            engine,
            disable_optimizer=cfg.test_config.disable_optimizer,
            dump_last_grads=cfg.test_config.dump_last_grads,
            num_training_steps=int(cfg.test_config.step),
            last_grad_holder=(
                last_grad_holder
                if (
                    cfg.test_config.disable_optimizer
                    and cfg.test_config.dump_last_grads
                )
                else None
            ),
            save_full_last_grad_tensors_fp32=(
                cfg.test_config.save_full_last_grad_tensors_fp32
            ),
        )
        if rank == 0 and cfg.test_config.save_params:
            logger.warning(
                "test_config.save_params is deprecated in low-memory mode and "
                "ignored. Use diff.pt.",
            )
        if rank == 0 and cfg.test_config.save_initial_params:
            logger.warning(
                "test_config.save_initial_params is ignored in low-memory mode.",
            )

        initial_params: dict[str, torch.Tensor] | None = None
        if cfg.test_config.save_diff:
            initial_params = _snapshot_initial_full_params(engine)

        loss_fn = _make_loss_fn(cfg)
        num_steps = int(cfg.test_config.step)
        for step_idx in range(num_steps):
            file_idx = step_idx % len(step_files)
            step_file = step_files[file_idx]
            if rank == 0:
                logger.info(
                    "Starting training step %d/%d (0-based index %d), data file=%s",
                    step_idx + 1,
                    num_steps,
                    step_idx,
                    os.path.abspath(step_file),
                )

            if cfg.test_config.dump_forward_compare:
                forward_record = _dump_forward_outputs(
                    engine=engine,
                    cfg=cfg,
                    step_idx=step_idx,
                    step_file=step_file,
                    device=device,
                    dump_dir=dump_dir,
                )
                if rank == 0:
                    logger.info(
                        "Forward output dumped: step=%d file=%s output_kind=%s "
                        "global_tokens=%d global_valid_outputs=%d mask_mismatch_ranks=%d",
                        forward_record["step"],
                        os.path.basename(forward_record["file"]),
                        "value" if engine.config.is_critic else "logprob",
                        forward_record["global_input_tokens"],
                        forward_record["global_valid_output_numel"],
                        forward_record["global_mask_mismatch_ranks"],
                    )

            record = _run_single_step(
                engine=engine,
                cfg=cfg,
                step_idx=step_idx,
                step_file=step_file,
                device=device,
                loss_fn=loss_fn,
            )

            if rank == 0:
                with open(stats_path, "a") as fp:
                    fp.write(json.dumps(record) + "\n")
                logger.info(
                    "Step %03d done: file=%s loss=%.6f grad_norm(min/max)=%.4f/%.4f "
                    "elapsed(max)=%.2fs peak_mem(max)=%.1fMiB",
                    step_idx,
                    os.path.basename(step_file),
                    record["loss"],
                    record["grad_norm_min"],
                    record["grad_norm_max"],
                    record["elapsed_s_max"],
                    record["peak_mem_mib_max"],
                )

        if cfg.test_config.dump_last_grads:
            if cfg.test_config.disable_optimizer:
                grad_payload: dict[str, Any] = dict(last_grad_holder)
            else:
                built = _build_grad_snapshot_payload(
                    engine,
                    save_full_grad_tensors_fp32=(
                        cfg.test_config.save_full_last_grad_tensors_fp32
                    ),
                )
                grad_payload = built if built is not None else {}
            if rank == 0:
                if not grad_payload:
                    logger.warning(
                        "dump_last_grads: empty payload (no .grad tensors?)."
                    )
                else:
                    os.makedirs(dump_dir, exist_ok=True)
                    torch.save(grad_payload, last_grads_path)
                    gmeta = grad_payload.get("global", {})
                    gtensors = grad_payload.get("grad_tensors_fp32")
                    n_full = len(gtensors) if isinstance(gtensors, dict) else 0
                    rg_meta = grad_payload.get("requires_grad_meta") or {}
                    n_rg_t = rg_meta.get("num_named_requires_grad_true")
                    n_rg_f = rg_meta.get("num_named_requires_grad_false")
                    in_dump_rg_f = sum(
                        1
                        for v in grad_payload.get("params", {}).values()
                        if isinstance(v, dict) and not v.get("requires_grad", True)
                    )
                    logger.info(
                        "Wrote %s: num_params=%s global_l2=%s full_grad_tensors=%s "
                        "named_requires_grad_true/false=%s/%s "
                        "(in this dump, params with requires_grad=False: %s)",
                        last_grads_path,
                        gmeta.get("num_params"),
                        gmeta.get("l2"),
                        n_full,
                        n_rg_t,
                        n_rg_f,
                        in_dump_rg_f,
                    )
                    if n_rg_f:
                        sample = (rg_meta.get("named_requires_grad_false") or [])[:8]
                        logger.warning(
                            "Some named_parameters have requires_grad=False (sample raw names): %s",
                            sample,
                        )
            if dist.is_initialized():
                dist.barrier(group=engine.cpu_group)

        if cfg.test_config.save_diff:
            _save_diff_snapshot(
                engine,
                initial_params,
                dump_dir,
                "diff.pt",
                save_full_diff_tensors_fp32=(
                    cfg.test_config.save_full_diff_tensors_fp32
                ),
            )
            if initial_params is not None:
                initial_params.clear()
    finally:
        _destroy_engine(engine)


if __name__ == "__main__":
    main()
