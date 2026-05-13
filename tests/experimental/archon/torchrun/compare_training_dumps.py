"""Offline comparison tool for two ArchonEngine training-test dumps.

Given two directories produced by :mod:`run_archon_training_test`, this script:

1. Loads global ``stats.jsonl`` and builds a per-step view, then performs a
   strict loss alignment check.
2. Preferentially loads ``diff.pt`` from each dump and compares parameter update
   signatures. If ``diff.pt`` is absent, falls back to legacy ``params.pt``
   (and optionally ``params_initial.pt``) tensor diffs.
3. If both dumps contain ``last_grads.pt`` (from ``test_config.dump_last_grads``),
   checks A/B alignment (signature + optional full ``grad_tensors_fp32``) and
   prints one compact table like the diff dump's "Top delta_gap" plus a one-line
   PASS/FAIL summary.  ``requires_grad=True`` only means a parameter *may* receive
   gradients; a zero ``.grad`` after backward is normal if the loss graph does not
   flow to that parameter for that step (e.g. critic value head only).
4. If both dumps contain ``forward.step*.summary.pt`` (from
   ``test_config.dump_forward_compare``), compares valid-token forward outputs.

The tool is launched as a plain Python script -- no distributed setup required.

Example::

    python tests/experimental/archon/torchrun/compare_training_dumps.py \\
        --dump-a /tmp/run_a --dump-b /tmp/run_b \\
        --loss-rtol 1e-6 --loss-atol 1e-6

Exit code is non-zero when any enabled alignment check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# -----------------------------------------------------------------------------
# Terminal colors
# -----------------------------------------------------------------------------


_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_RED = "\033[31m"
_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"


def _supports_color() -> bool:
    term = os.environ.get("TERM", "").lower()
    no_color = os.environ.get("NO_COLOR")
    return sys.stdout.isatty() and bool(term) and term != "dumb" and not no_color


def _colorize(text: str, color: str, *, bold: bool = False) -> str:
    if not _supports_color():
        return text
    prefix = f"{_ANSI_BOLD}{color}" if bold else color
    return f"{prefix}{text}{_ANSI_RESET}"


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


def _stats_file(dump_dir: str) -> str:
    path = os.path.join(dump_dir, "stats.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing stats file: {path}. Did the training run finish?"
        )
    return path


def _load_stats(dump_dir: str) -> dict[int, list[dict[str, Any]]]:
    """Return ``{step -> list[records]}`` for one dump."""
    by_step: dict[int, list[dict[str, Any]]] = {}
    path = _stats_file(dump_dir)
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            step = int(rec["step"])
            by_step.setdefault(step, []).append(rec)
    return by_step


def _rank_head_loss(records: list[dict[str, Any]]) -> float:
    """Pick one valid loss for a step.

    Current dumps store one global record per step; we read the first valid one.
    """
    for r in records:
        loss = r.get("loss")
        if loss is None:
            continue
        if isinstance(loss, float) and math.isnan(loss):
            continue
        return float(loss)
    return float("nan")


# -----------------------------------------------------------------------------
# Loss comparison
# -----------------------------------------------------------------------------


@dataclass
class LossDiff:
    step: int
    loss_a: float
    loss_b: float
    abs_gap: float
    rel_gap: float
    aligned: bool


def _compare_losses(
    stats_a: dict[int, list[dict[str, Any]]],
    stats_b: dict[int, list[dict[str, Any]]],
    *,
    atol: float,
    rtol: float,
) -> list[LossDiff]:
    steps_a = set(stats_a.keys())
    steps_b = set(stats_b.keys())
    shared = sorted(steps_a & steps_b)
    if steps_a != steps_b:
        print(
            f"[warn] step sets differ: only_a={sorted(steps_a - steps_b)[:10]} "
            f"only_b={sorted(steps_b - steps_a)[:10]}"
        )

    diffs: list[LossDiff] = []
    for step in shared:
        la = _rank_head_loss(stats_a[step])
        lb = _rank_head_loss(stats_b[step])
        gap = abs(la - lb)
        rel = gap / max(abs(lb), 1e-12)
        aligned = gap <= (atol + rtol * abs(lb))
        diffs.append(
            LossDiff(
                step=step,
                loss_a=la,
                loss_b=lb,
                abs_gap=gap,
                rel_gap=rel,
                aligned=aligned,
            )
        )
    return diffs


# -----------------------------------------------------------------------------
# Parameter comparison
# -----------------------------------------------------------------------------


@dataclass
class ParamDiff:
    name: str
    shape_match: bool
    max_diff: float
    mean_diff: float
    l2_diff: float
    l2_a: float
    l2_b: float
    norm_gap: float
    delta_gap: float


@dataclass
class ParamUpdateStat:
    name: str
    numel: float
    mean_abs_update: float
    max_abs_update: float
    l2_update: float
    rel_l2_update: float


@dataclass
class GradSnapshotStat:
    """Per-parameter stats from ``last_grads.pt`` (final-step .grad snapshot)."""

    name: str
    numel: float
    mean_abs: float
    max_abs: float
    l2: float


@dataclass
class DiffFileGap:
    name: str
    numel_match: bool
    mean_abs_gap: float
    mean_abs_rel_gap: float
    max_abs_gap: float
    max_abs_rel_gap: float
    l2_gap: float
    l2_rel_gap: float


@dataclass
class ForwardTensorGap:
    key: str
    shape_match: bool
    max_diff: float
    mean_diff: float
    l2_rel_gap: float


_FORWARD_SUMMARY_RE = re.compile(r"forward\.step(\d+)\.summary\.pt$")


def _load_diff_signatures(path: str) -> dict[str, ParamUpdateStat]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}, got {type(payload)}")
    params = payload.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"Expected 'params' dict in {path}, got {type(params)}")

    out: dict[str, ParamUpdateStat] = {}
    for name, item in params.items():
        if not isinstance(item, dict):
            raise ValueError(
                f"Expected metrics dict for parameter '{name}' in {path}, got {type(item)}"
            )
        out[name] = ParamUpdateStat(
            name=str(name),
            numel=float(item.get("numel", 0.0)),
            mean_abs_update=float(item.get("mean_abs_update", 0.0)),
            max_abs_update=float(item.get("max_abs_update", 0.0)),
            l2_update=float(item.get("l2_update", 0.0)),
            rel_l2_update=float(item.get("rel_l2_update", 0.0)),
        )
    return out


def _print_last_grads_requires_grad_one_line(path_a: str, path_b: str) -> None:
    """One summary line; list frozen param names only when non-empty."""

    payloads: list[dict[str, Any]] = []
    for path in (path_a, path_b):
        raw = torch.load(path, map_location="cpu")
        payloads.append(raw if isinstance(raw, dict) else {})

    line_parts: list[str] = []
    for label, payload in zip(("A", "B"), payloads, strict=True):
        meta = payload.get("requires_grad_meta")
        if not isinstance(meta, dict):
            line_parts.append(f"{label}: (no requires_grad_meta)")
        else:
            nt = meta.get("num_named_requires_grad_true")
            nf = meta.get("num_named_requires_grad_false")
            line_parts.append(f"{label}: requires_grad true={nt} false={nf}")
    print(f"  {'; '.join(line_parts)}")

    for label, payload in zip(("A", "B"), payloads, strict=True):
        meta = payload.get("requires_grad_meta")
        if not isinstance(meta, dict):
            continue
        false_names = meta.get("named_requires_grad_false") or []
        if false_names:
            print(f"  {label} requires_grad=False (sample): {false_names[:8]}")


def _load_grad_signatures(path: str) -> dict[str, GradSnapshotStat]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}, got {type(payload)}")
    params = payload.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"Expected 'params' dict in {path}, got {type(params)}")

    out: dict[str, GradSnapshotStat] = {}
    for name, item in params.items():
        if not isinstance(item, dict):
            raise ValueError(
                f"Expected metrics dict for parameter '{name}' in {path}, got {type(item)}"
            )
        out[name] = GradSnapshotStat(
            name=str(name),
            numel=float(item.get("numel", 0.0)),
            mean_abs=float(item.get("mean_abs", 0.0)),
            max_abs=float(item.get("max_abs", 0.0)),
            l2=float(item.get("l2", 0.0)),
        )
    return out


def _compare_grad_signatures(
    stats_a: dict[str, GradSnapshotStat],
    stats_b: dict[str, GradSnapshotStat],
) -> tuple[list[DiffFileGap], list[str], list[str]]:
    """Compare ``last_grads.pt`` entries; reuse :class:`DiffFileGap` for reporting."""
    names_a = set(stats_a.keys())
    names_b = set(stats_b.keys())
    shared = sorted(names_a & names_b)
    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)

    gaps: list[DiffFileGap] = []
    for name in shared:
        a = stats_a[name]
        b = stats_b[name]
        gaps.append(
            DiffFileGap(
                name=name,
                numel_match=int(round(a.numel)) == int(round(b.numel)),
                mean_abs_gap=abs(a.mean_abs - b.mean_abs),
                mean_abs_rel_gap=_relative_gap(a.mean_abs, b.mean_abs),
                max_abs_gap=abs(a.max_abs - b.max_abs),
                max_abs_rel_gap=_relative_gap(a.max_abs, b.max_abs),
                l2_gap=abs(a.l2 - b.l2),
                l2_rel_gap=_relative_gap(a.l2, b.l2),
            )
        )
    return gaps, only_a, only_b


def _global_grad_l2_relative_gap(
    stats_a: dict[str, GradSnapshotStat],
    stats_b: dict[str, GradSnapshotStat],
) -> float:
    shared = set(stats_a.keys()) & set(stats_b.keys())
    if not shared:
        return 0.0
    total_l2_a = math.sqrt(sum(float(stats_a[name].l2) ** 2 for name in shared))
    total_l2_b = math.sqrt(sum(float(stats_b[name].l2) ** 2 for name in shared))
    return _relative_gap(total_l2_a, total_l2_b)


def _load_full_delta_tensors(path: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return {}
    tensors = payload.get("delta_tensors_fp32")
    if not isinstance(tensors, dict):
        return {}
    out: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if isinstance(tensor, torch.Tensor):
            out[str(name)] = tensor.detach().float().cpu()
    return out


def _load_full_grad_tensors(path: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return {}
    tensors = payload.get("grad_tensors_fp32")
    if not isinstance(tensors, dict):
        return {}
    out: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if isinstance(tensor, torch.Tensor):
            out[str(name)] = tensor.detach().float().cpu()
    return out


def _relative_gap(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def _compare_diff_signatures(
    stats_a: dict[str, ParamUpdateStat],
    stats_b: dict[str, ParamUpdateStat],
) -> tuple[list[DiffFileGap], list[str], list[str]]:
    names_a = set(stats_a.keys())
    names_b = set(stats_b.keys())
    shared = sorted(names_a & names_b)
    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)

    gaps: list[DiffFileGap] = []
    for name in shared:
        a = stats_a[name]
        b = stats_b[name]
        gaps.append(
            DiffFileGap(
                name=name,
                numel_match=int(round(a.numel)) == int(round(b.numel)),
                mean_abs_gap=abs(a.mean_abs_update - b.mean_abs_update),
                mean_abs_rel_gap=_relative_gap(a.mean_abs_update, b.mean_abs_update),
                max_abs_gap=abs(a.max_abs_update - b.max_abs_update),
                max_abs_rel_gap=_relative_gap(a.max_abs_update, b.max_abs_update),
                l2_gap=abs(a.l2_update - b.l2_update),
                l2_rel_gap=_relative_gap(a.l2_update, b.l2_update),
            )
        )
    return gaps, only_a, only_b


def _compare_full_deltas(
    tensors_a: dict[str, torch.Tensor],
    tensors_b: dict[str, torch.Tensor],
) -> tuple[list[ParamDiff], list[ParamDiff], list[ParamDiff]]:
    """Compare full fp32 delta tensors.

    Returns (shared_diffs, only_a_vs_zero, only_b_vs_zero).
    """
    shared, only_a, only_b = _compare_state_dicts(tensors_a, tensors_b)

    def _vs_zero(name: str, tensor: torch.Tensor) -> ParamDiff:
        zeros = torch.zeros_like(tensor)
        delta = tensor - zeros
        l2_tensor = float(tensor.norm().item())
        l2_delta = float(delta.norm().item())
        return ParamDiff(
            name=name,
            shape_match=True,
            max_diff=float(delta.abs().max().item()) if delta.numel() > 0 else 0.0,
            mean_diff=float(delta.abs().mean().item()) if delta.numel() > 0 else 0.0,
            l2_diff=l2_delta,
            l2_a=l2_tensor,
            l2_b=0.0,
            norm_gap=_relative_gap(l2_tensor, 0.0),
            delta_gap=l2_delta / max(l2_tensor, 1e-12),
        )

    only_a_diffs = [_vs_zero(name, tensors_a[name]) for name in only_a]
    only_b_diffs = [_vs_zero(name, tensors_b[name]) for name in only_b]
    return shared, only_a_diffs, only_b_diffs


def _summarize_diff_gaps(gaps: list[DiffFileGap]) -> dict[str, float]:
    if not gaps:
        return {
            "num_params": 0.0,
            "numel_mismatch": 0.0,
            "max_abs_gap_max": 0.0,
            "max_abs_rel_gap_max": 0.0,
            "mean_abs_gap_mean": 0.0,
            "mean_abs_rel_gap_mean": 0.0,
            "l2_gap_mean": 0.0,
            "l2_rel_gap_mean": 0.0,
            "mean_abs_rel_gap_max": 0.0,
            "l2_rel_gap_max": 0.0,
        }
    return {
        "num_params": float(len(gaps)),
        "numel_mismatch": float(sum(0 if g.numel_match else 1 for g in gaps)),
        "max_abs_gap_max": float(max(g.max_abs_gap for g in gaps)),
        "max_abs_rel_gap_max": float(max(g.max_abs_rel_gap for g in gaps)),
        "mean_abs_gap_mean": float(sum(g.mean_abs_gap for g in gaps) / len(gaps)),
        "mean_abs_rel_gap_mean": float(
            sum(g.mean_abs_rel_gap for g in gaps) / len(gaps)
        ),
        "l2_gap_mean": float(sum(g.l2_gap for g in gaps) / len(gaps)),
        "l2_rel_gap_mean": float(sum(g.l2_rel_gap for g in gaps) / len(gaps)),
        "mean_abs_rel_gap_max": float(max(g.mean_abs_rel_gap for g in gaps)),
        "l2_rel_gap_max": float(max(g.l2_rel_gap for g in gaps)),
    }


def _global_l2_update_relative_gap(
    stats_a: dict[str, ParamUpdateStat],
    stats_b: dict[str, ParamUpdateStat],
) -> float:
    shared = set(stats_a.keys()) & set(stats_b.keys())
    if not shared:
        return 0.0
    total_l2_a = math.sqrt(sum(float(stats_a[name].l2_update) ** 2 for name in shared))
    total_l2_b = math.sqrt(sum(float(stats_b[name].l2_update) ** 2 for name in shared))
    return _relative_gap(total_l2_a, total_l2_b)


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Expected dict state_dict in {path}, got {type(state)}")
    return {k: v.detach().float() for k, v in state.items()}


def _compare_state_dicts(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
) -> tuple[list[ParamDiff], list[str], list[str]]:
    names_a = set(state_a.keys())
    names_b = set(state_b.keys())
    shared = sorted(names_a & names_b)
    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)

    diffs: list[ParamDiff] = []
    for name in shared:
        a = state_a[name]
        b = state_b[name]
        shape_ok = a.shape == b.shape
        if not shape_ok:
            diffs.append(
                ParamDiff(
                    name=name,
                    shape_match=False,
                    max_diff=float("inf"),
                    mean_diff=float("inf"),
                    l2_diff=float("inf"),
                    l2_a=float("inf"),
                    l2_b=float("inf"),
                    norm_gap=float("inf"),
                    delta_gap=float("inf"),
                )
            )
            continue
        delta = a - b
        l2_a = float(a.norm().item())
        l2_b = float(b.norm().item())
        l2_delta = float(delta.norm().item())
        diffs.append(
            ParamDiff(
                name=name,
                shape_match=True,
                max_diff=float(delta.abs().max().item()),
                mean_diff=float(delta.abs().mean().item()),
                l2_diff=l2_delta,
                l2_a=l2_a,
                l2_b=l2_b,
                norm_gap=_relative_gap(l2_a, l2_b),
                delta_gap=l2_delta / max(l2_a, l2_b, 1e-12),
            )
        )
    return diffs, only_a, only_b


def _discover_forward_summary_files(dump_dir: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for p in Path(dump_dir).glob("forward.step*.summary.pt"):
        m = _FORWARD_SUMMARY_RE.fullmatch(p.name)
        if m is None:
            continue
        step = int(m.group(1))
        out[step] = str(p)
    return out


def _load_forward_compare_payload(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}, got {type(payload)}")
    return payload


def _compare_forward_tensor(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    key: str,
) -> ForwardTensorGap:
    ta = a.detach().float().cpu()
    tb = b.detach().float().cpu()
    if ta.shape != tb.shape:
        return ForwardTensorGap(
            key=key,
            shape_match=False,
            max_diff=float("inf"),
            mean_diff=float("inf"),
            l2_rel_gap=float("inf"),
        )
    delta = (ta - tb).abs()
    max_diff = float(delta.max().item()) if delta.numel() > 0 else 0.0
    mean_diff = float(delta.mean().item()) if delta.numel() > 0 else 0.0
    l2a = float(ta.norm().item())
    l2b = float(tb.norm().item())
    l2_rel_gap = _relative_gap(l2a, l2b)
    return ForwardTensorGap(
        key=key,
        shape_match=True,
        max_diff=max_diff,
        mean_diff=mean_diff,
        l2_rel_gap=l2_rel_gap,
    )


def _summarize_param_diffs(diffs: list[ParamDiff]) -> dict[str, float]:
    if not diffs:
        return {
            "num_params": 0,
            "global_max_diff": 0.0,
            "global_mean_diff": 0.0,
            "global_l2_diff": 0.0,
            "global_norm_gap": 0.0,
            "max_norm_gap": 0.0,
            "global_delta_gap": 0.0,
            "max_delta_gap": 0.0,
        }
    matched = [d for d in diffs if d.shape_match]
    if not matched:
        return {
            "num_params": len(diffs),
            "global_max_diff": float("inf"),
            "global_mean_diff": float("inf"),
            "global_l2_diff": float("inf"),
            "global_norm_gap": float("inf"),
            "max_norm_gap": float("inf"),
            "global_delta_gap": float("inf"),
            "max_delta_gap": float("inf"),
        }
    max_diff = max(d.max_diff for d in matched)
    total_l2 = math.sqrt(sum(d.l2_diff**2 for d in matched))
    mean_diff = sum(d.mean_diff for d in matched) / len(matched)
    total_l2_a = math.sqrt(sum(d.l2_a**2 for d in matched))
    total_l2_b = math.sqrt(sum(d.l2_b**2 for d in matched))
    global_norm_gap = _relative_gap(total_l2_a, total_l2_b)
    max_norm_gap = max(d.norm_gap for d in matched)
    global_delta_gap = total_l2 / max(total_l2_a, total_l2_b, 1e-12)
    max_delta_gap = max(d.delta_gap for d in matched)
    return {
        "num_params": len(diffs),
        "global_max_diff": float(max_diff),
        "global_mean_diff": float(mean_diff),
        "global_l2_diff": float(total_l2),
        "global_norm_gap": float(global_norm_gap),
        "max_norm_gap": float(max_norm_gap),
        "global_delta_gap": float(global_delta_gap),
        "max_delta_gap": float(max_delta_gap),
    }


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def _print_loss_report(
    diffs: list[LossDiff],
    atol: float,
    rtol: float,
) -> bool:
    print("\n=== Per-step loss comparison (strict) ===")
    print(f"atol={atol:.3e} rtol={rtol:.3e}")
    print(
        f"{'step':>4} {'loss_a':>14} {'loss_b':>14} "
        f"{'abs_gap':>12} {'rel_gap':>12} {'status':>8}"
    )
    ok = True
    for d in diffs:
        status_raw = "OK" if d.aligned else "MISMATCH"
        status = (
            _colorize(status_raw, _ANSI_GREEN, bold=True)
            if d.aligned
            else _colorize(status_raw, _ANSI_RED, bold=True)
        )
        ok = ok and d.aligned
        print(
            f"{d.step:>4d} {d.loss_a:>14.6f} {d.loss_b:>14.6f} "
            f"{d.abs_gap:>12.3e} {d.rel_gap:>12.3e} {status:>8}"
        )
    overall = (
        _colorize("PASS", _ANSI_GREEN, bold=True)
        if ok
        else _colorize("FAIL", _ANSI_RED, bold=True)
    )
    print(f"Loss alignment overall: {overall}")
    return ok


def _print_param_report(
    label: str,
    diffs: list[ParamDiff],
    only_a: list[str],
    only_b: list[str],
    top_k: int = 10,
) -> None:
    summary = _summarize_param_diffs(diffs)
    shared_count = len(diffs)
    mismatch_count = len(only_a) + len(only_b)
    print(f"\n=== {label} parameter comparison ===")
    print(
        f"  key_coverage: shared={shared_count} only_a={len(only_a)} "
        f"only_b={len(only_b)} total_union={shared_count + mismatch_count}"
    )
    if only_a or only_b:
        warn = (
            f"[warn] parameter key mismatch: only_a={only_a[:5]}{'...' if len(only_a) > 5 else ''} "
            f"only_b={only_b[:5]}{'...' if len(only_b) > 5 else ''}"
        )
        print(_colorize(warn, _ANSI_YELLOW, bold=True))
    print(f"  global_norm_gap: {summary['global_norm_gap']}")
    print(f"  max_norm_gap: {summary['max_norm_gap']}")
    print(f"  global_delta_gap: {summary['global_delta_gap']}")
    print(f"  max_delta_gap: {summary['max_delta_gap']}")
    worst = sorted(diffs, key=lambda d: d.max_diff, reverse=True)[:top_k]
    print(f"  top-{len(worst)} tensors by max_diff:")
    for d in worst:
        print(
            f"    {d.name[:80]:<80} max={d.max_diff:.3e} "
            f"mean={d.mean_diff:.3e} norm_gap={d.norm_gap:.3e} "
            f"delta_gap={d.delta_gap:.3e} "
            f"shape_match={d.shape_match}"
        )


def _print_diff_file_report(
    title: str,
    gaps: list[DiffFileGap],
    only_a: list[str],
    only_b: list[str],
    top_k: int = 10,
) -> None:
    summary = _summarize_diff_gaps(gaps)
    shared_count = len(gaps)
    mismatch_count = len(only_a) + len(only_b)
    print(f"\n=== {title} ===")
    print(
        f"  key_coverage: shared={shared_count} only_a={len(only_a)} "
        f"only_b={len(only_b)} total_union={shared_count + mismatch_count}"
    )
    if only_a or only_b:
        warn = (
            f"[warn] parameter key mismatch: only_a={only_a[:5]}{'...' if len(only_a) > 5 else ''} "
            f"only_b={only_b[:5]}{'...' if len(only_b) > 5 else ''}"
        )
        print(_colorize(warn, _ANSI_YELLOW, bold=True))
    for k, v in summary.items():
        print(f"  {k}: {v}")
    worst = sorted(gaps, key=lambda g: g.max_abs_gap, reverse=True)[:top_k]
    print(f"  top-{len(worst)} tensors by max_abs_gap:")
    for g in worst:
        print(
            f"    {g.name[:80]:<80} max_abs_gap={g.max_abs_gap:.3e} "
            f"max_abs_rel_gap={g.max_abs_rel_gap:.3e} "
            f"mean_abs_gap={g.mean_abs_gap:.3e} "
            f"mean_abs_rel_gap={g.mean_abs_rel_gap:.3e} "
            f"l2_gap={g.l2_gap:.3e} "
            f"l2_rel_gap={g.l2_rel_gap:.3e} "
            f"numel_match={g.numel_match}"
        )


def _print_diff_alignment_report(
    summary: dict[str, float],
    *,
    rel_gap_tol: float,
    title: str = "diff.pt alignment",
) -> bool:
    l2_rel_max_value = float(summary.get("l2_rel_gap_max", float("inf")))
    global_l2_rel_value = float(summary.get("global_l2_rel_gap", float("inf")))
    l2_ok = l2_rel_max_value < rel_gap_tol
    global_l2_ok = global_l2_rel_value < rel_gap_tol
    aligned_ok = l2_ok and global_l2_ok

    print(f"\n=== {title} ===")
    print(
        f"max(l2_rel_gap)   < {rel_gap_tol:.3e}: "
        f"{l2_rel_max_value:.3e} "
        f"{_colorize('PASS', _ANSI_GREEN, bold=True) if l2_ok else _colorize('FAIL', _ANSI_RED, bold=True)}"
    )
    print(
        f"global_l2_rel_gap < {rel_gap_tol:.3e}: "
        f"{global_l2_rel_value:.3e} "
        f"{_colorize('PASS', _ANSI_GREEN, bold=True) if global_l2_ok else _colorize('FAIL', _ANSI_RED, bold=True)}"
    )
    print(
        f"{title} overall: "
        + (
            _colorize("PASS", _ANSI_GREEN, bold=True)
            if aligned_ok
            else _colorize("FAIL", _ANSI_RED, bold=True)
        )
    )
    return aligned_ok


def _print_unmatched_tensor_report(
    label: str,
    diffs: list[ParamDiff],
    side: str,
    top_k: int = 10,
) -> None:
    print(f"\n=== {label} unmatched full-delta tensors ({side} vs zeros) ===")
    if not diffs:
        print("  none")
        return
    summary = _summarize_param_diffs(diffs)
    print(f"  global_norm_gap: {summary['global_norm_gap']}")
    print(f"  max_norm_gap: {summary['max_norm_gap']}")
    print(f"  global_delta_gap: {summary['global_delta_gap']}")
    print(f"  max_delta_gap: {summary['max_delta_gap']}")
    worst = sorted(diffs, key=lambda d: d.max_diff, reverse=True)[:top_k]
    print(f"  top-{len(worst)} tensors by max_diff:")
    for d in worst:
        print(
            f"    {d.name[:80]:<80} max={d.max_diff:.3e} "
            f"mean={d.mean_diff:.3e} norm_gap={d.norm_gap:.3e} "
            f"delta_gap={d.delta_gap:.3e}"
        )


def _print_full_delta_alignment_report(
    param_diffs: list[ParamDiff],
    *,
    rel_gap_tol: float,
    heading: str = "full-delta fp32 alignment",
) -> bool:
    summary = _summarize_param_diffs(param_diffs)
    global_value = float(summary.get("global_delta_gap", float("inf")))
    max_value = float(summary.get("max_delta_gap", float("inf")))
    global_ok = global_value < rel_gap_tol
    max_ok = max_value < rel_gap_tol
    ok = global_ok and max_ok
    print(f"\n=== {heading} ===")
    print(
        f"global_delta_gap < {rel_gap_tol:.3e}: "
        f"{global_value:.3e} "
        f"{_colorize('PASS', _ANSI_GREEN, bold=True) if global_ok else _colorize('FAIL', _ANSI_RED, bold=True)}"
    )
    print(
        f"max_delta_gap    < {rel_gap_tol:.3e}: "
        f"{max_value:.3e} "
        f"{_colorize('PASS', _ANSI_GREEN, bold=True) if max_ok else _colorize('FAIL', _ANSI_RED, bold=True)}"
    )
    return ok


def _print_name_gap_report(
    *,
    norm_gaps: dict[str, float],
    delta_gaps: dict[str, float],
    l2_delta_a: dict[str, float] | None = None,
    l2_delta_b: dict[str, float] | None = None,
    top_k: int = 10,
    heading: str = "Top delta_gap parameters (shared names)",
    l2_note: str | None = None,
) -> None:
    """Print top-k shared parameter gaps sorted by delta_gap.

    When ``l2_delta_a`` / ``l2_delta_b`` are provided, each column is the L2
    norm of the saved fp32 update tensor ``Δθ = current - initial`` for that
    dump (so you can see whether a parameter actually moved vs. stayed at 0).
    """
    shared_names = sorted(set(norm_gaps.keys()) & set(delta_gaps.keys()))
    ranked = sorted(
        shared_names,
        key=lambda name: delta_gaps.get(name, float("-inf")),
        reverse=True,
    )[:top_k]
    print(f"\n=== {heading} ===")
    if l2_delta_a is not None and l2_delta_b is not None:
        print(
            l2_note
            or "(||A||, ||B|| are L2 norms of Δθ from each dump's delta_tensors_fp32.)"
        )
        print(
            f"{'name':<72} {'||A||':>12} {'||B||':>12} "
            f"{'norm_gap':>12} {'delta_gap':>12}"
        )
        for name in ranked:
            norm_value = norm_gaps.get(name, float("inf"))
            delta_value = delta_gaps.get(name, float("inf"))
            na = l2_delta_a.get(name, float("nan"))
            nb = l2_delta_b.get(name, float("nan"))
            print(
                f"{name[:72]:<72} {na:>12.3e} {nb:>12.3e} "
                f"{norm_value:>12.3e} {delta_value:>12.3e}"
            )
    else:
        print(f"{'name':<80} {'norm_gap':>12} {'delta_gap':>12}")
        for name in ranked:
            norm_value = norm_gaps.get(name, float("inf"))
            delta_value = delta_gaps.get(name, float("inf"))
            print(f"{name[:80]:<80} {norm_value:>12.3e} {delta_value:>12.3e}")


def _print_forward_compare_report(
    dump_a: str,
    dump_b: str,
    *,
    rel_gap_tol: float,
    max_abs_tol: float,
) -> bool:
    summary_a = _discover_forward_summary_files(dump_a)
    summary_b = _discover_forward_summary_files(dump_b)
    if not summary_a and not summary_b:
        print("\n[info] no forward.step*.summary.pt in either dump.")
        return True
    if not summary_a or not summary_b:
        print(
            "\n[warn] forward summaries present on only one side "
            f"(a={bool(summary_a)}, b={bool(summary_b)}); skipping forward comparison."
        )
        return False

    steps_a = set(summary_a.keys())
    steps_b = set(summary_b.keys())
    shared_steps = sorted(steps_a & steps_b)
    only_a_steps = sorted(steps_a - steps_b)
    only_b_steps = sorted(steps_b - steps_a)

    print("\n=== valid-token forward output comparison ===")
    print(
        f"  step_coverage: shared={len(shared_steps)} only_a={len(only_a_steps)} "
        f"only_b={len(only_b_steps)}"
    )
    if only_a_steps or only_b_steps:
        print(
            _colorize(
                f"[warn] forward summary mismatch: only_a={only_a_steps[:6]} "
                f"only_b={only_b_steps[:6]}",
                _ANSI_YELLOW,
                bold=True,
            )
        )

    ok = not only_a_steps and not only_b_steps
    tensor_gaps: list[ForwardTensorGap] = []
    global_max_gap = 0.0
    global_mean_gap = 0.0
    for step in shared_steps:
        pa = _load_forward_compare_payload(summary_a[step])
        pb = _load_forward_compare_payload(summary_b[step])
        meta_ok = (
            pa.get("output_kind") == pb.get("output_kind")
            and int(pa.get("global_input_tokens", -1))
            == int(pb.get("global_input_tokens", -2))
            and int(pa.get("global_valid_output_numel", -1))
            == int(pb.get("global_valid_output_numel", -2))
            and int(pa.get("global_mask_mismatch_ranks", -1))
            == int(pb.get("global_mask_mismatch_ranks", -2))
        )
        ok = ok and meta_ok
        ranks_a = {
            int(x["rank"]): x for x in pa.get("per_rank", []) if isinstance(x, dict)
        }
        ranks_b = {
            int(x["rank"]): x for x in pb.get("per_rank", []) if isinstance(x, dict)
        }
        rank_shared = sorted(set(ranks_a) & set(ranks_b))
        ok = ok and set(ranks_a) == set(ranks_b)
        for rank in rank_shared:
            ra = ranks_a[rank]
            rb = ranks_b[rank]
            pos_a = ra.get("valid_token_positions")
            pos_b = rb.get("valid_token_positions")
            positions_ok = (
                isinstance(pos_a, torch.Tensor)
                and isinstance(pos_b, torch.Tensor)
                and torch.equal(pos_a, pos_b)
            )
            ok = ok and positions_ok
            ta = ra.get("valid_forward_fp32")
            tb = rb.get("valid_forward_fp32")
            if isinstance(ta, torch.Tensor) and isinstance(tb, torch.Tensor):
                tg = _compare_forward_tensor(
                    ta, tb, key=f"step={step},rank={rank},valid_forward"
                )
                tensor_gaps.append(tg)
                global_max_gap = max(global_max_gap, tg.max_diff)
                global_mean_gap = max(global_mean_gap, tg.mean_diff)
                ok = (
                    ok
                    and tg.shape_match
                    and tg.max_diff <= max_abs_tol
                    and tg.l2_rel_gap < rel_gap_tol
                )
            else:
                ok = False
                print(
                    _colorize(
                        f"[warn] missing valid_forward_fp32 for step={step}, rank={rank}",
                        _ANSI_YELLOW,
                        bold=True,
                    )
                )

    print(
        f"  valid_forward_max_diff_max={global_max_gap:.3e}, "
        f"valid_forward_mean_diff_max={global_mean_gap:.3e} "
        f"(tol abs={max_abs_tol:.3e}, rel={rel_gap_tol:.3e})"
    )
    if tensor_gaps:
        worst = sorted(tensor_gaps, key=lambda g: g.max_diff, reverse=True)[:10]
        print(f"  top-{len(worst)} forward tensor gaps by max_diff:")
        for g in worst:
            print(
                f"    {g.key:<52} max={g.max_diff:.3e} "
                f"mean={g.mean_diff:.3e} l2_rel={g.l2_rel_gap:.3e} "
                f"shape_match={g.shape_match}"
            )

    print(
        "  forward_compare overall: "
        + (
            _colorize("PASS", _ANSI_GREEN, bold=True)
            if ok
            else _colorize("FAIL", _ANSI_RED, bold=True)
        )
    )
    return ok


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def run_comparison(
    dump_a: str,
    dump_b: str,
    *,
    loss_atol: float,
    loss_rtol: float,
    diff_rel_gap_tol: float,
    full_delta_rel_gap_tol: float,
    compare_initial: bool,
) -> bool:
    print(f"[compare] dump_a={dump_a}")
    print(f"[compare] dump_b={dump_b}")

    stats_a = _load_stats(dump_a)
    stats_b = _load_stats(dump_b)
    loss_diffs = _compare_losses(stats_a, stats_b, atol=loss_atol, rtol=loss_rtol)
    loss_ok = _print_loss_report(loss_diffs, loss_atol, loss_rtol)
    diff_ok = True
    full_delta_ok = True
    grad_ok = True
    grad_full_ok = True
    forward_ok = _print_forward_compare_report(
        dump_a,
        dump_b,
        rel_gap_tol=diff_rel_gap_tol,
        max_abs_tol=full_delta_rel_gap_tol,
    )

    diff_a = Path(dump_a) / "diff.pt"
    diff_b = Path(dump_b) / "diff.pt"
    if diff_a.exists() and diff_b.exists():
        sig_a = _load_diff_signatures(str(diff_a))
        sig_b = _load_diff_signatures(str(diff_b))
        gaps, only_a, only_b = _compare_diff_signatures(sig_a, sig_b)
        if only_a or only_b:
            print("\n=== diff.pt param keys only in one dump ===")
            print(f"  only_in_a: {only_a}")
            print(f"  only_in_b: {only_b}")
        diff_summary = _summarize_diff_gaps(gaps)
        diff_summary["global_l2_rel_gap"] = _global_l2_update_relative_gap(sig_a, sig_b)
        diff_ok = (
            float(diff_summary.get("l2_rel_gap_max", float("inf"))) < diff_rel_gap_tol
            and float(diff_summary.get("global_l2_rel_gap", float("inf")))
            < diff_rel_gap_tol
        )

        full_a = _load_full_delta_tensors(str(diff_a))
        full_b = _load_full_delta_tensors(str(diff_b))
        if full_a and full_b:
            shared_diffs, only_a_diffs, only_b_diffs = _compare_full_deltas(
                full_a, full_b
            )
            all_full_diffs = [*shared_diffs, *only_a_diffs, *only_b_diffs]
            full_summary = _summarize_param_diffs(all_full_diffs)
            full_delta_ok = (
                float(full_summary.get("global_delta_gap", float("inf")))
                < full_delta_rel_gap_tol
                and float(full_summary.get("max_delta_gap", float("inf")))
                < full_delta_rel_gap_tol
            )

            norm_gaps: dict[str, float] = {}
            for g in gaps:
                norm_gaps[g.name] = float(g.l2_rel_gap)

            delta_gaps: dict[str, float] = {}
            l2_delta_a: dict[str, float] = {}
            l2_delta_b: dict[str, float] = {}
            for d in shared_diffs:
                delta_gaps[d.name] = float(d.delta_gap)
                l2_delta_a[d.name] = float(d.l2_a)
                l2_delta_b[d.name] = float(d.l2_b)

            _print_name_gap_report(
                norm_gaps=norm_gaps,
                delta_gaps=delta_gaps,
                l2_delta_a=l2_delta_a,
                l2_delta_b=l2_delta_b,
                top_k=10,
            )
        else:
            print(
                "\n[info] full fp32 delta tensors not found in both diff.pt "
                "(enable test_config.save_full_diff_tensors_fp32=true in both runs)."
            )
    else:
        final_a = Path(dump_a) / "params.pt"
        final_b = Path(dump_b) / "params.pt"
        if final_a.exists() and final_b.exists():
            print(
                "\n[info] diff.pt not found in both dumps; "
                "falling back to legacy params.pt comparison."
            )
            state_a = _load_state_dict(str(final_a))
            state_b = _load_state_dict(str(final_b))
            diffs, only_a, only_b = _compare_state_dicts(state_a, state_b)
            _print_param_report("Final", diffs, only_a, only_b)
        else:
            print(
                f"\n[info] skipping final param comparison "
                f"(diff exists: a={diff_a.exists()}, b={diff_b.exists()}; "
                f"params exists: a={final_a.exists()}, b={final_b.exists()})"
            )

    if compare_initial:
        init_a = Path(dump_a) / "params_initial.pt"
        init_b = Path(dump_b) / "params_initial.pt"
        if init_a.exists() and init_b.exists():
            state_a = _load_state_dict(str(init_a))
            state_b = _load_state_dict(str(init_b))
            diffs, only_a, only_b = _compare_state_dicts(state_a, state_b)
            _print_param_report("Initial", diffs, only_a, only_b)

    lg_a = Path(dump_a) / "last_grads.pt"
    lg_b = Path(dump_b) / "last_grads.pt"
    if lg_a.is_file() and lg_b.is_file():
        print("\n=== last_grads.pt ===")
        _print_last_grads_requires_grad_one_line(str(lg_a), str(lg_b))

        gsig_a = _load_grad_signatures(str(lg_a))
        gsig_b = _load_grad_signatures(str(lg_b))
        ggaps, g_only_a, g_only_b = _compare_grad_signatures(gsig_a, gsig_b)
        if g_only_a or g_only_b:
            print("  param keys only in one dump:")
            print(f"    only_a={g_only_a[:8]}{'...' if len(g_only_a) > 8 else ''}")
            print(f"    only_b={g_only_b[:8]}{'...' if len(g_only_b) > 8 else ''}")
        grad_summary = _summarize_diff_gaps(ggaps)
        grad_summary["global_l2_rel_gap"] = _global_grad_l2_relative_gap(gsig_a, gsig_b)
        grad_ok = (
            float(grad_summary.get("l2_rel_gap_max", float("inf"))) < diff_rel_gap_tol
            and float(grad_summary.get("global_l2_rel_gap", float("inf")))
            < diff_rel_gap_tol
        )

        full_ga = _load_full_grad_tensors(str(lg_a))
        full_gb = _load_full_grad_tensors(str(lg_b))
        if full_ga and full_gb:
            shared_g, only_a_g, only_b_g = _compare_full_deltas(full_ga, full_gb)
            all_g = [*shared_g, *only_a_g, *only_b_g]
            full_g_summary = _summarize_param_diffs(all_g)
            grad_full_ok = (
                float(full_g_summary.get("global_delta_gap", float("inf")))
                < full_delta_rel_gap_tol
                and float(full_g_summary.get("max_delta_gap", float("inf")))
                < full_delta_rel_gap_tol
            )

            norm_gaps_g: dict[str, float] = {}
            for g in ggaps:
                norm_gaps_g[g.name] = float(g.l2_rel_gap)
            delta_gaps_g: dict[str, float] = {}
            l2_ga: dict[str, float] = {}
            l2_gb: dict[str, float] = {}
            for d in shared_g:
                delta_gaps_g[d.name] = float(d.delta_gap)
                l2_ga[d.name] = float(d.l2_a)
                l2_gb[d.name] = float(d.l2_b)
            _print_name_gap_report(
                norm_gaps=norm_gaps_g,
                delta_gaps=delta_gaps_g,
                l2_delta_a=l2_ga,
                l2_delta_b=l2_gb,
                top_k=10,
                heading="Top last_grads gaps (shared names)",
                l2_note=(
                    "(||A||, ||B|| are L2 norms of grad_tensors_fp32; "
                    "norm_gap from signature l2_rel; zero ||·||₂ is an all-zero grad.)"
                ),
            )
            print(
                f"  alignment: signature={'PASS' if grad_ok else 'FAIL'} "
                f"full_tensor={'PASS' if grad_full_ok else 'FAIL'} "
                f"(tol signature={diff_rel_gap_tol} elementwise={full_delta_rel_gap_tol})"
            )
        elif full_ga or full_gb:
            grad_full_ok = False
            print(
                "\n[warn] last_grads.pt grad_tensors_fp32 present on only one side "
                f"(a={bool(full_ga)}, b={bool(full_gb)}); skipping full grad tensor compare."
            )
            print(
                f"  alignment: signature={'PASS' if grad_ok else 'FAIL'} "
                "full_tensor=FAIL (asymmetric dumps)"
            )
        else:
            print(
                "\n  [info] no grad_tensors_fp32 in both dumps "
                "(enable test_config.save_full_last_grad_tensors_fp32=true); "
                f"signature_only={'PASS' if grad_ok else 'FAIL'}."
            )
    elif lg_a.is_file() or lg_b.is_file():
        print(
            "\n[warn] last_grads.pt present on only one side "
            f"(a={lg_a.is_file()}, b={lg_b.is_file()}); skipping grad comparison."
        )

    return (
        loss_ok
        and diff_ok
        and full_delta_ok
        and grad_ok
        and grad_full_ok
        and forward_ok
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two ArchonEngine training-test dump_dirs."
    )
    parser.add_argument("--dump-a", type=str, required=True, help="First dump dir.")
    parser.add_argument("--dump-b", type=str, required=True, help="Second dump dir.")
    parser.add_argument(
        "--loss-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for per-step loss alignment (default 1e-6).",
    )
    parser.add_argument(
        "--loss-rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for per-step loss alignment (default 1e-3).",
    )
    parser.add_argument(
        "--diff-rel-gap-tol",
        type=float,
        default=1e-2,
        help=(
            "Relative-gap threshold for diff.pt (max(l2_rel_gap), global_l2_rel_gap) "
            "and, when both dumps include last_grads.pt, the same checks on "
            "gradient signature gaps. Default 1e-2."
        ),
    )
    parser.add_argument(
        "--full-delta-rel-gap-tol",
        type=float,
        default=1e-2,
        help=(
            "Threshold for full tensor alignment: both global_delta_gap and "
            "max_delta_gap must be < tol (diff.pt delta_tensors_fp32 when present; "
            "last_grads.pt grad_tensors_fp32 when present). Default 1e-2."
        ),
    )
    parser.add_argument(
        "--compare-initial",
        action="store_true",
        help="Also compare legacy params_initial.pt if present in both dumps.",
    )
    args = parser.parse_args(argv)

    ok = run_comparison(
        dump_a=args.dump_a,
        dump_b=args.dump_b,
        loss_atol=args.loss_atol,
        loss_rtol=args.loss_rtol,
        diff_rel_gap_tol=args.diff_rel_gap_tol,
        full_delta_rel_gap_tol=args.full_delta_rel_gap_tol,
        compare_initial=args.compare_initial,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
