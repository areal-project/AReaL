# SPDX-License-Identifier: Apache-2.0
"""Pluggable change detectors for AWEX colocate delta weight transfer.

A *detector* answers one question each weight-update step: which bf16 elements
of the converted HF payload changed since the previous version? It returns
``{hf_name: bool mask}`` consumed by ``DeltaTracker.encode(masks=...)``.

Two implementations, selected by ``DTE_DELTA_DETECTOR``:

- ``snapshot`` (fallback): the change mask comes from ``DeltaTracker``'s own CPU
  bf16 baseline (a full-model snapshot). This detector is a no-op marker — it
  returns ``None`` so the writer keeps using ``encode(masks=None)`` (the
  snapshot path). Always available, no optimizer needed, but costs a resident
  full-model CPU snapshot per rank.

- ``inversion`` (design mainline, the actual win): reconstruct the *pre-step*
  weights from the optimizer's resident AdamW moments (``exp_avg`` /
  ``exp_avg_sq``) — **zero extra snapshot memory** — convert them to HF space,
  and bitwise-compare against the current HF payload to get the masks.

AdamW inversion (decoupled AdamW, the mcore default):

    theta_t = theta_{t-1}·(1 - lr·wd) - (lr/bc1)·m / (sqrt(v)/sqrt(bc2) + eps)

is element-wise invertible because ``m=exp_avg`` / ``v=exp_avg_sq`` stay resident
after ``optimizer.step()``:

    theta_{t-1} = (theta_t + (lr/bc1)·m / (sqrt(v)/sqrt(bc2) + eps)) / (1 - lr·wd)

with ``bc1 = 1 - beta1^step``, ``bc2 = 1 - beta2^step``.

Megatron specifics (confirmed against mcore distrib_optimizer):

- Moments live ONLY in mcore main-param space (HF space has QKV already split,
  gate/expert converted) so the mask MUST be computed AFTER convert.
- The distributed optimizer shards the fp32 main param + moments across DP, but
  the FULL bf16 model param (theta_t) is resident on every DP rank (post-step
  all-gather). So inversion is a per-rank, shard-local fp32 compute scattered
  into a full-param buffer, then ONE DP all-reduce(SUM) of the *correction*
  (zero outside the owned slice) assembles the full pre-step param.

Hard gates (any failure -> return None -> writer ships dense this step):

- ``use_precision_aware_optimizer`` (adam_bf16): moments are bf16/TE-fused, not
  plain fp32 ``exp_avg`` -> inversion infeasible.
- non-decoupled AdamW (L2-Adam folds wd into grad): the ``/(1-lr·wd)`` form is
  wrong.
- no reconstructable moment state for the whole step, or ``step < 1`` on every
  usable shard (first step, recover): division blow-up. Per-param/per-rank
  missing state contributes zeros to DP reduction so ranks do not deadlock.
- compact per-shard fingerprints are only a guard for ``step`` unchanged or
  missing-watermark recovery. A known one-step AdamW transition is always
  reconstructed; the fingerprint is not trusted as proof of no payload change.
- more than one distributed-optimizer instance: the DP group identity differs
  from the all-gather group.

Migration note (origin/gh AWEX): the detector now drives the new adapter's
convert path — ``adapter._get_inner_optimizers()`` for the moments and
``adapter._convert_hf_with_overrides(theta_by_id)`` to push reconstructed
pre-step weights through the same all_gather + convert_to_hf as the live
payload (the old awex ``_make_weight_converter`` / ``_convert_parameters_with``
are gone). The mcore reconstruction below is GPU-only and must be validated on
the cluster; ``dte.core.invert_adamw`` itself is CPU-unit-tested.
"""

from __future__ import annotations

import os

import torch

from areal.utils import logging

logger = logging.getLogger("AwexDeltaDetect")


def _env_value(name: str, legacy_name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is not None and value.strip() != "":
        return value
    value = os.environ.get(legacy_name)
    if value is not None and value.strip() != "":
        return value
    return default


def _env_bool(name: str, legacy_name: str, default: bool = False) -> bool:
    value = _env_value(name, legacy_name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def delta_detector_mode() -> str:
    """Return the configured detector: '' (off) | 'inversion' | 'snapshot'."""
    return (
        (_env_value("DTE_DELTA_DETECTOR", "AWEX_DELTA_DETECTOR", "") or "")
        .strip()
        .lower()
    )


def _inversion_debug_enabled() -> bool:
    return _env_bool("DTE_DELTA_INVERSION_DEBUG", "AWEX_DELTA_INVERSION_DEBUG")


def _inversion_bf16_margin_rel() -> float:
    return float(
        _env_value(
            "DTE_DELTA_INVERSION_BF16_MARGIN_REL",
            "AWEX_DELTA_INVERSION_BF16_MARGIN_REL",
            "1e-4",
        )
    )


def _dist_rank(group=None) -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return -1
    try:
        return torch.distributed.get_rank(group=group)
    except Exception:  # pragma: no cover - debug-only best effort
        return -1


def _dist_world_size(group=None) -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    try:
        return torch.distributed.get_world_size(group=group)
    except Exception:  # pragma: no cover - debug-only best effort
        return 1


def _adamw_hparams(param_group: dict) -> tuple[float, float, float, float, float]:
    """Extract (lr, weight_decay, beta1, beta2, eps) from a torch param_group."""
    betas = param_group.get("betas", (0.9, 0.999))
    beta1 = float(betas[0])
    beta2 = float(betas[1])
    if not param_group.get("bias_correction", True):
        # Apex FusedAdam stores step on the param group and can disable bias
        # correction there. invert_adamw uses beta**step only to compute those
        # correction factors, so beta=0 gives bc1=bc2=1.
        beta1 = 0.0
        beta2 = 0.0
    return (
        float(param_group["lr"]),
        float(param_group.get("weight_decay", 0.0)),
        beta1,
        beta2,
        float(param_group.get("eps", 1e-8)),
    )


def _resolve_offloaded_state_value(
    state: dict, offloaded_state: dict, key: str
) -> object | None:
    """Return optimizer state value, following AwexMegatronAdapter offload slots."""
    val = state.get(key)
    if isinstance(val, torch.Tensor) and val.numel() == 0 and key in offloaded_state:
        return offloaded_state[key]
    return val


def _state_step_value(state: dict, offloaded_state: dict) -> float | None:
    step_val = _resolve_offloaded_state_value(state, offloaded_state, "step")
    return _step_to_float(step_val)


def _param_group_step_value(param_group: dict) -> float | None:
    return _step_to_float(param_group.get("step"))


def _step_to_float(step_val: object | None) -> float | None:
    if step_val is None:
        return None
    if isinstance(step_val, torch.Tensor):
        if step_val.numel() == 0:
            return None
        return float(step_val.item())
    return float(step_val)


@torch.no_grad()
def _tensor_fingerprint(tensor: torch.Tensor) -> tuple:
    """Small value fingerprint for a tensor slice, not a stored snapshot.

    The detector uses this as a no-full-copy guard to decide whether a local
    model-visible shard changed since the last successful sync. The actual delta
    mask still comes from AdamW inversion; this fingerprint only prevents
    replaying moments for a shard whose synced bf16 payload did not move.
    """
    data = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    n = data.numel()
    if n == 0:
        return (str(tensor.dtype), 0, 0, 0, 0, 0)

    bins = min(4096, n)
    width = max(n // bins, 1)
    head_n = bins * width
    head = data[:head_n].reshape(bins, width)
    bin_sums = head.sum(dim=1, dtype=torch.int64)
    weights = torch.arange(1, bins + 1, device=data.device, dtype=torch.int64)
    total = bin_sums.sum()
    weighted = (bin_sums * weights).sum()
    if head_n < n:
        tail = data[head_n:].sum(dtype=torch.int64)
        total = total + tail
        weighted = weighted + tail * (bins + 1)

    sample_count = min(16, n)
    if sample_count == 1:
        sample_idx = torch.zeros(1, device=data.device, dtype=torch.long)
    else:
        sample_idx = (
            torch.arange(sample_count, device=data.device, dtype=torch.long)
            * (n - 1)
            // (sample_count - 1)
        )
    samples = data.index_select(0, sample_idx).to(torch.int64)
    sample_weights = torch.arange(
        1, sample_count + 1, device=data.device, dtype=torch.int64
    )
    sample_hash = (samples * sample_weights).sum()
    return (
        str(tensor.dtype),
        n,
        int(total.item()),
        int(weighted.item()),
        int(sample_hash.item()),
        int(data[0].item()),
        int(data[-1].item()),
    )


@torch.no_grad()
def _bf16_rounding_boundary_mask(
    old_fp32: torch.Tensor, cur_bf16: torch.Tensor
) -> torch.Tensor:
    """Conservatively include values near a BF16 rounding boundary."""
    cur_fp32 = cur_bf16.to(torch.float32)
    old_fp32 = old_fp32.to(torch.float32)

    # BF16 bins are not symmetric at exponent boundaries: for example the lower
    # neighbor of 1.0 is 0.99609375 while the upper neighbor is 1.0078125.  Using
    # one spacing for both sides misses values close to the narrower boundary.
    neg_inf = torch.full_like(cur_bf16, float("-inf"))
    pos_inf = torch.full_like(cur_bf16, float("inf"))
    lower = torch.nextafter(cur_bf16, neg_inf).to(torch.float32)
    upper = torch.nextafter(cur_bf16, pos_inf).to(torch.float32)
    lower_half_ulp = (cur_fp32 - lower).abs() * 0.5
    upper_half_ulp = (upper - cur_fp32).abs() * 0.5
    half_ulp = torch.where(old_fp32 < cur_fp32, lower_half_ulp, upper_half_ulp)
    threshold = half_ulp * (1.0 - _inversion_bf16_margin_rel())
    dist = (old_fp32 - cur_fp32).abs()
    return torch.isfinite(old_fp32) & torch.isfinite(cur_fp32) & (dist >= threshold)


class SnapshotDetector:
    """No-op detector: defers change detection to ``DeltaTracker``'s snapshot.

    ``compute_masks`` returns ``None`` so the writer calls ``encode(masks=None)``
    (the verified snapshot path). Exists so the writer has one uniform call site
    regardless of the configured detector.
    """

    name = "snapshot"

    def compute_masks(self, names, tensors, version):  # noqa: ARG002
        return None


class AdamWInversionDetector:
    """Compute change masks by inverting the resident AdamW moments.

    Bound to ``AwexMegatronAdapter`` (needs ``_get_inner_optimizers`` for the
    moments and ``_convert_hf_with_overrides`` to convert reconstructed pre-step
    params through the live path). All mcore/DP/convert work is GPU-only; any
    unmet precondition returns ``None`` so the writer falls back to dense for
    that step (and the snapshot tracker re-seeds), never silently corrupting.
    """

    name = "inversion"

    def __init__(self, adapter):
        self._adapter = adapter
        self._last_synced_steps: dict[int, float | None] = {}
        self._last_synced_fingerprints: dict[int, tuple] = {}

    # -- gating ------------------------------------------------------------
    def _inversion_feasible(self, inner_optimizers) -> bool:
        """All hard gates from the design; any failure -> dense fallback."""
        if not inner_optimizers:
            logger.warning("Inversion: no inner optimizers; falling back to dense.")
            return False
        for opt in inner_optimizers:
            cfg = getattr(opt, "config", None)
            if cfg is not None:
                # adam_bf16 / precision-aware: moments are bf16 / TE-fused, not
                # plain fp32 exp_avg -> inversion infeasible.
                pa = getattr(
                    cfg, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", None
                )
                if pa is None:
                    pa = getattr(cfg, "use_precision_aware_optimizer", False)
                if pa:
                    logger.warning("Inversion: use_precision_aware_optimizer -> dense.")
                    return False
                # decoupled AdamW: the (1-lr*wd) form requires decoupled wd.
                if not getattr(cfg, "decoupled_weight_decay", True):
                    logger.warning("Inversion: non-decoupled AdamW -> dense.")
                    return False
            # Single distributed-optimizer instance: otherwise the DP group
            # identity differs from the param all-gather group (design risk #3).
            n_inst = getattr(opt, "num_distributed_optimizer_instances", 1)
            if n_inst and n_inst > 1:
                logger.warning(
                    "Inversion: %d distributed-optimizer instances -> dense.",
                    n_inst,
                )
                return False
        return True

    def _collect_current_steps(self, inner_optimizers) -> dict[int, float | None]:
        """Return optimizer step watermarks keyed by ``id(model_param)``.

        A value of ``None`` means this param had no local optimizer state when the
        payload was synced. On the next step, ``None -> 1`` is still a valid
        one-step transition.
        """
        steps: dict[int, float | None] = {}
        offloaded_states = getattr(self._adapter, "_offloaded_optimizer_states", {})
        for opt in inner_optimizers:
            base_opt = getattr(opt, "optimizer", opt)
            state = getattr(base_opt, "state", None)
            group_index_map = getattr(opt, "model_param_group_index_map", None)
            fp32_groups = getattr(opt, "shard_fp32_from_float16_groups", None)
            model_groups = getattr(opt, "model_float16_groups", None)
            if state is None or fp32_groups is None or model_groups is None:
                continue
            for g, (fp32_group, model_group) in enumerate(
                zip(fp32_groups, model_groups)
            ):
                for main_shard, model_param in zip(fp32_group, model_group):
                    if main_shard is None:
                        continue
                    pid = id(model_param)
                    if pid in steps:
                        continue
                    gi = (
                        group_index_map[model_param][0]
                        if group_index_map is not None
                        and model_param in group_index_map
                        else g
                    )
                    st = state.get(main_shard)
                    step = None
                    if st:
                        offloaded_state = offloaded_states.get(main_shard, {})
                        step = _state_step_value(st, offloaded_state)
                        if step is None:
                            step = _param_group_step_value(base_opt.param_groups[gi])
                    steps[pid] = step
        return steps

    def _collect_current_fingerprints(self, inner_optimizers) -> dict[int, tuple]:
        """Record compact fingerprints of model-visible local shards.

        This is intentionally not a snapshot: it stores a few scalar values per
        optimizer-owned shard, not tensor contents. The fingerprint gates AdamW
        replay to slices whose synced payload actually changed.
        """
        fingerprints: dict[int, tuple] = {}
        for opt in inner_optimizers:
            fp32_groups = getattr(opt, "shard_fp32_from_float16_groups", None)
            model_groups = getattr(opt, "model_float16_groups", None)
            if fp32_groups is None or model_groups is None:
                continue
            for fp32_group, model_group in zip(fp32_groups, model_groups):
                for main_shard, model_param in zip(fp32_group, model_group):
                    if main_shard is None:
                        continue
                    pid = id(model_param)
                    if pid in fingerprints:
                        continue
                    try:
                        rng = opt._get_model_param_range_map(model_param)["param"]
                    except Exception:
                        continue
                    visible_slice = model_param.detach().reshape(-1)[
                        rng.start : rng.end
                    ]
                    fingerprints[pid] = _tensor_fingerprint(visible_slice)
        return fingerprints

    def mark_synced(self, version: int) -> None:
        """Record optimizer step watermarks after a successful payload sync.

        AdamW inversion only reconstructs the most recent optimizer step. Without
        this watermark, a later weight sync with no new optimizer step would replay
        the previous step's moments and report false positives.
        """
        inner = self._adapter._get_inner_optimizers()
        if not self._inversion_feasible(inner):
            self._last_synced_steps.clear()
            self._last_synced_fingerprints.clear()
            return
        self._last_synced_steps = self._collect_current_steps(inner)
        self._last_synced_fingerprints = self._collect_current_fingerprints(inner)
        known = len(self._last_synced_steps)
        stepped = sum(
            1 for step in self._last_synced_steps.values() if step is not None
        )
        max_step = max(
            (step for step in self._last_synced_steps.values() if step is not None),
            default=0.0,
        )
        logger.info(
            "Inversion: recorded optimizer step watermark for %d params "
            "at version %d (stepped=%d fingerprints=%d max_step=%.0f)",
            known,
            version,
            stepped,
            len(self._last_synced_fingerprints),
            max_step,
        )

    # -- reconstruction (GPU-only) ----------------------------------------
    @torch.no_grad()
    def _reconstruct_pre_step_mcore(
        self, inner_optimizers
    ) -> dict[int, torch.Tensor] | None:
        """Reconstruct theta_{t-1} for every model param the optimizer owns.

        Returns ``{id(model_param): full bf16 pre-step tensor}`` or ``None`` if
        no parameter has a usable global optimizer-state contribution.
        """
        from dte.core import invert_adamw

        theta_old_by_id: dict[int, torch.Tensor] = {}
        skipped_no_state = 0
        skipped_bad_state = 0
        skipped_global_no_state = 0
        skipped_step_unchanged = 0
        skipped_missing_watermark = 0
        skipped_step_jump = 0
        skipped_missing_fingerprint = 0
        skipped_tracked_unchanged = 0
        skipped_payload_changed_without_step = 0
        force_dense = False
        debug = _inversion_debug_enabled()
        global_rank = _dist_rank()
        opt_entries: dict[int, tuple] = {}
        fallback_order: list[torch.Tensor] = []
        seen_fallback: set[int] = set()
        default_dp_group = None
        for opt_idx, opt in enumerate(inner_optimizers):
            base_opt = getattr(opt, "optimizer", opt)
            state = getattr(base_opt, "state", None)
            if state is None:
                return None
            group_index_map = getattr(opt, "model_param_group_index_map", None)
            fp32_groups = getattr(opt, "shard_fp32_from_float16_groups", None)
            model_groups = getattr(opt, "model_float16_groups", None)
            if fp32_groups is None or model_groups is None:
                return None
            dp_group = getattr(opt, "data_parallel_group", None)
            if default_dp_group is None:
                default_dp_group = dp_group
            elif dp_group is not default_dp_group:
                logger.warning("Inversion: multiple DP groups -> dense.")
                return None
            if debug:
                logger.info(
                    "Inversion debug: rank=%d opt=%d dp_rank=%d/%d "
                    "fp32_groups=%d model_groups=%d state=%d",
                    global_rank,
                    opt_idx,
                    _dist_rank(dp_group),
                    _dist_world_size(dp_group),
                    len(fp32_groups),
                    len(model_groups),
                    len(state),
                )
            offloaded_states = getattr(self._adapter, "_offloaded_optimizer_states", {})
            for g, (fp32_group, model_group) in enumerate(
                zip(fp32_groups, model_groups)
            ):
                if debug:
                    logger.info(
                        "Inversion debug: rank=%d opt=%d group=%d "
                        "fp32_len=%d model_len=%d",
                        global_rank,
                        opt_idx,
                        g,
                        len(fp32_group),
                        len(model_group),
                    )
                for i, (main_shard, model_param) in enumerate(
                    zip(fp32_group, model_group)
                ):
                    if main_shard is None:
                        # precision-aware sentinel (should be gated out already)
                        return None
                    pid = id(model_param)
                    if pid not in seen_fallback:
                        fallback_order.append(model_param)
                        seen_fallback.add(pid)
                    if pid in opt_entries:
                        continue
                    gi = (
                        group_index_map[model_param][0]
                        if group_index_map is not None
                        and model_param in group_index_map
                        else g
                    )
                    opt_entries[pid] = (
                        opt_idx,
                        opt,
                        base_opt,
                        state,
                        main_shard,
                        gi,
                        dp_group,
                    )

        ordered_model_params: list[torch.Tensor] = []
        engine = getattr(self._adapter, "_engine", None)
        if engine is not None and getattr(engine, "model", None) is not None:
            from areal.engine.megatron_utils.megatron import get_named_parameters

            num_moe_experts = getattr(engine.tf_config, "num_moe_experts", None)
            seen_order: set[int] = set()
            for _mcore_name, model_param in get_named_parameters(
                engine.model, num_moe_experts
            ):
                pid = id(model_param)
                if pid in seen_order:
                    continue
                ordered_model_params.append(model_param)
                seen_order.add(pid)
        else:
            ordered_model_params = fallback_order

        if debug:
            logger.info(
                "Inversion debug: rank=%d ordered_params=%d opt_entries=%d",
                global_rank,
                len(ordered_model_params),
                len(opt_entries),
            )

        if not ordered_model_params:
            return None

        offloaded_states = getattr(self._adapter, "_offloaded_optimizer_states", {})
        for param_idx, model_param in enumerate(ordered_model_params):
            entry = opt_entries.get(id(model_param))
            dp_group = entry[6] if entry is not None else default_dp_group
            # theta_t for the full param is resident on every DP rank.
            theta_t_full = model_param.detach()
            flat_t = theta_t_full.reshape(-1)
            # Every DP rank must enter the same collectives for every model
            # param in the same order. Ranks without a usable local optimizer
            # shard contribute an all-zero correction and a zero contribution
            # count; ranks with moments fill only their owned slice.
            correction = torch.zeros_like(flat_t, dtype=torch.float32)
            has_local_update = False
            expected_one_step = False
            rng = None
            opt_idx = -1
            if entry is None:
                skipped_no_state += 1
            else:
                opt_idx, opt, base_opt, state, main_shard, gi, _dp_group = entry
                rng = opt._get_model_param_range_map(model_param)["param"]
                st = state.get(main_shard)
                offloaded_state = offloaded_states.get(main_shard, {}) if st else {}
                step = _state_step_value(st, offloaded_state) if st else None
                if step is None:
                    step = _param_group_step_value(base_opt.param_groups[gi])
                if step is None or step < 1:
                    skipped_bad_state += 1
                elif id(model_param) not in self._last_synced_steps:
                    skipped_missing_watermark += 1
                    force_dense = True
                else:
                    last_step = self._last_synced_steps[id(model_param)]
                    baseline_step = 0.0 if last_step is None else last_step
                    step_delta = step - baseline_step
                    if step_delta == 0:
                        last_fingerprint = self._last_synced_fingerprints.get(
                            id(model_param)
                        )
                        if last_fingerprint is not None and (
                            _tensor_fingerprint(flat_t[rng.start : rng.end])
                            != last_fingerprint
                        ):
                            skipped_payload_changed_without_step += 1
                            force_dense = True
                        else:
                            skipped_step_unchanged += 1
                    elif step_delta != 1:
                        skipped_step_jump += 1
                        force_dense = True
                    else:
                        expected_one_step = True
                        last_fingerprint = self._last_synced_fingerprints.get(
                            id(model_param)
                        )
                        if last_fingerprint is None:
                            skipped_missing_fingerprint += 1
                            force_dense = True
                        elif not st or "exp_avg" not in st or "exp_avg_sq" not in st:
                            # A rank may not own this shard in Megatron's
                            # distributed optimizer; do not force dense locally.
                            # If no DP rank contributes below, the global status
                            # check will fall back to dense.
                            skipped_bad_state += 1
                        else:
                            exp_avg = _resolve_offloaded_state_value(
                                st, offloaded_state, "exp_avg"
                            )
                            exp_avg_sq = _resolve_offloaded_state_value(
                                st, offloaded_state, "exp_avg_sq"
                            )
                            theta_t_slice = (
                                main_shard.detach().reshape(-1).to(torch.float32)
                            )
                            if (
                                not isinstance(exp_avg, torch.Tensor)
                                or not isinstance(exp_avg_sq, torch.Tensor)
                                or exp_avg.numel() == 0
                                or exp_avg_sq.numel() == 0
                                or exp_avg.numel() != theta_t_slice.numel()
                                or exp_avg_sq.numel() != theta_t_slice.numel()
                            ):
                                skipped_bad_state += 1
                            else:
                                lr, wd, b1, b2, eps = _adamw_hparams(
                                    base_opt.param_groups[gi]
                                )
                                # Moments may be offloaded to CPU; bring this
                                # shard's moments to the fp32 main-param device
                                # so inversion uses the exact post-step optimizer
                                # weight, not the bf16 model copy. Keep the
                                # reconstructed previous value in fp32 until mask
                                # creation so boundary-near BF16 roundoff is
                                # handled conservatively.
                                dev = theta_t_slice.device
                                theta_old_slice = invert_adamw(
                                    theta_t_slice,
                                    exp_avg.to(dev).reshape(-1),
                                    exp_avg_sq.to(dev).reshape(-1),
                                    step,
                                    lr,
                                    wd,
                                    b1,
                                    b2,
                                    eps,
                                )
                                visible_slice = flat_t[rng.start : rng.end]
                                correction[rng.start : rng.end] = (
                                    theta_old_slice - visible_slice.to(torch.float32)
                                )
                                has_local_update = True

            status = torch.tensor(
                [1 if has_local_update else 0, 1 if force_dense else 0],
                dtype=torch.int32,
                device=flat_t.device,
            )
            if debug:
                slice_start = -1 if rng is None else rng.start
                slice_end = -1 if rng is None else rng.end
                logger.info(
                    "Inversion debug: rank=%d param_idx=%d opt=%d "
                    "pre_reduce numel=%d slice=%d:%d local=%d",
                    global_rank,
                    param_idx,
                    opt_idx,
                    flat_t.numel(),
                    slice_start,
                    slice_end,
                    int(has_local_update),
                )
            if dp_group is not None:
                torch.distributed.all_reduce(
                    correction,
                    op=torch.distributed.ReduceOp.SUM,
                    group=dp_group,
                )
                torch.distributed.all_reduce(
                    status,
                    op=torch.distributed.ReduceOp.SUM,
                    group=dp_group,
                )
            if debug:
                logger.info(
                    "Inversion debug: rank=%d param_idx=%d post_reduce "
                    "contrib=%d bad=%d",
                    global_rank,
                    param_idx,
                    int(status[0].item()),
                    int(status[1].item()),
                )
            if int(status[1].item()) != 0:
                force_dense = True
                continue
            if int(status[0].item()) == 0:
                # No DP rank could reconstruct this param. For a known one-step
                # AdamW transition that is unsafe: current-vs-current would hide
                # a real update, so fall back to dense. If no optimizer step is
                # pending, leaving it without an override is still correct.
                skipped_global_no_state += 1
                if expected_one_step:
                    force_dense = True
                continue
            theta_old_full = (flat_t.to(torch.float32) + correction).reshape(
                theta_t_full.shape
            )
            theta_old_by_id[id(model_param)] = theta_old_full
        if force_dense:
            logger.warning(
                "Inversion: optimizer step replay is ambiguous "
                "(missing_watermark=%d step_jump=%d missing_fingerprint=%d "
                "payload_changed_without_step=%d) "
                "-> dense.",
                skipped_missing_watermark,
                skipped_step_jump,
                skipped_missing_fingerprint,
                skipped_payload_changed_without_step,
            )
            return None
        if not theta_old_by_id:
            logger.info(
                "Inversion: no BF16 payload shard changed since last sync "
                "(step_unchanged=%d tracked_unchanged=%d no_state=%d "
                "bad_state=%d global_no_state=%d)",
                skipped_step_unchanged,
                skipped_tracked_unchanged,
                skipped_no_state,
                skipped_bad_state,
                skipped_global_no_state,
            )
            return {}
        if (
            skipped_no_state
            or skipped_bad_state
            or skipped_global_no_state
            or skipped_step_unchanged
            or skipped_tracked_unchanged
            or skipped_missing_fingerprint
            or skipped_payload_changed_without_step
        ):
            logger.info(
                "Inversion: reconstructed %d params, skipped no_state=%d "
                "bad_state=%d global_no_state=%d step_unchanged=%d "
                "tracked_unchanged=%d missing_fingerprint=%d "
                "payload_changed_without_step=%d",
                len(theta_old_by_id),
                skipped_no_state,
                skipped_bad_state,
                skipped_global_no_state,
                skipped_step_unchanged,
                skipped_tracked_unchanged,
                skipped_missing_fingerprint,
                skipped_payload_changed_without_step,
            )
        return theta_old_by_id

    # -- entry point -------------------------------------------------------
    @torch.no_grad()
    def compute_masks(self, names, tensors, version):
        """Return ``{hf_name: bool mask}`` or ``None`` to force a dense step."""
        adapter = self._adapter
        inner = adapter._get_inner_optimizers()
        if not self._inversion_feasible(inner):
            return None

        theta_old_by_id = self._reconstruct_pre_step_mcore(inner)
        if theta_old_by_id is None:
            logger.warning("Inversion: reconstruction unavailable -> dense step.")
            return None

        # Convert the reconstructed pre-step params through the SAME all_gather +
        # convert_to_hf path as the live payload (overrides by id(model_param)).
        # Params without an override convert as-is (theta_old == theta_t ->
        # all-False mask).
        hf_old = adapter._convert_hf_with_overrides(theta_old_by_id)

        cur = dict(zip(names, tensors))
        masks: dict[str, torch.Tensor] = {}
        from dte.core import bitwise_changed_mask

        for name, cur_t in cur.items():
            old_t = hf_old.get(name)
            if old_t is None or old_t.shape != cur_t.shape:
                # No pre-step counterpart -> treat as fully changed (dense).
                masks[name] = torch.ones(
                    cur_t.numel(), dtype=torch.bool, device=cur_t.device
                )
                continue
            old_payload = old_t.to(cur_t.dtype)
            mask = bitwise_changed_mask(cur_t, old_payload).reshape(-1)
            if cur_t.dtype == torch.bfloat16 and old_t.dtype != cur_t.dtype:
                same_payload = ~mask
                boundary = _bf16_rounding_boundary_mask(old_t, cur_t).reshape(-1)
                mask = mask | (same_payload & boundary)
            masks[name] = mask
        logger.info(
            "Inversion: computed masks for %d HF params at version %d",
            len(masks),
            version,
        )
        return masks


def build_detector(mode: str, adapter):
    """Construct the detector for ``mode`` ('inversion' | 'snapshot')."""
    if mode == "inversion":
        return AdamWInversionDetector(adapter)
    return SnapshotDetector()
