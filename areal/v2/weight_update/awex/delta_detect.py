# SPDX-License-Identifier: Apache-2.0
"""AdamW inversion detector for AWEX separation delta weight transfer.

The detector answers one question each weight-update step: which bf16 elements
of the converted HF payload changed since the previous version? It returns
``{hf_name: bool mask | int32/int64 flat indices}`` consumed by
``DeltaTracker.encode(masks=...)``.

It reconstructs the *pre-step* weights from the optimizer's resident AdamW
moments (``exp_avg`` / ``exp_avg_sq``), converts them through the same HF path
as the live payload, and bitwise-compares both versions. Unsupported optimizer
states return ``None`` so every rank falls back to a full transfer.

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
  from the all-gather group. Multiple optimizer DP topologies in one model are
  allowed; MoE expert params can use a smaller expert-DP group while dense
  params use the regular DP group, so reconstruction reduces each param on its
  own optimizer DP group.

Migration note (origin/gh AWEX): the detector now drives the new adapter's
convert path — ``adapter._get_inner_optimizers()`` for the moments and
``adapter._convert_hf_with_overrides(theta_by_id)`` to push reconstructed
pre-step weights through the same all_gather + convert_to_hf as the live
payload (the old awex ``_make_weight_converter`` / ``_convert_parameters_with``
are gone). The mcore reconstruction below is GPU-only and must be validated on
the cluster; ``dte.core.invert_adamw`` itself is CPU-unit-tested.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import torch

from areal.utils import logging
from areal.utils.environ import get_bool_env_var, get_env_var

logger = logging.getLogger("AwexDeltaDetect")

_DTE_TRUTHY_VALUES = ("1", "true", "yes", "on")
_DTE_FALSY_VALUES = ("0", "false", "no", "off")


def _dte_bool_env(name: str, legacy_name: str, default: bool = False) -> bool:
    return get_bool_env_var(
        name,
        "true" if default else "false",
        fallback_names=(legacy_name,),
        truthy_values=_DTE_TRUTHY_VALUES,
        falsy_values=_DTE_FALSY_VALUES,
        strip_value=True,
    )


def _inversion_debug_enabled() -> bool:
    return _dte_bool_env("DTE_DELTA_INVERSION_DEBUG", "AWEX_DELTA_INVERSION_DEBUG")


def _streaming_reconstruct_enabled() -> bool:
    """Whether to reconstruct pre-step tensors lazily during HF conversion."""
    return _dte_bool_env(
        "DTE_STREAMING_RECONSTRUCT", "PYQ_STREAMING_RECONSTRUCT", False
    )


def _inversion_allreduce_window_bytes() -> int:
    """In-flight byte budget for pipelined inversion all-reduces.

    The reconstruct loop posts per-param correction/status all-reduces with
    ``async_op=True`` and only waits once this many correction bytes are in
    flight. ``0`` restores the fully synchronous per-param path (A/B knob).
    """
    mb = float(
        get_env_var(
            "DTE_INVERSION_ALLREDUCE_WINDOW_MB",
            "512",
            fallback_names=("AWEX_INVERSION_ALLREDUCE_WINDOW_MB",),
        )
    )
    return max(int(mb * 1024 * 1024), 0)


def _inversion_compute_device(flat_t: torch.Tensor) -> torch.device:
    """Device used for the AdamW-inversion elementwise math.

    Colocate offloads the fp32 main shards and AdamW moments to CPU before the
    weight sync runs, so deriving the compute device from ``main_shard`` makes
    the whole inversion (invert_adamw + mask probes) run on the CPU: ~18s/step
    for Qwen3-30B versus a few seconds of staged H2D copies. Default to the
    visible payload's device (GPU under colocate) and stage the offloaded
    operands there. Set DTE_INVERSION_COMPUTE_ON_CPU=1 to restore the legacy
    behaviour of computing wherever the fp32 main shard lives.
    """
    if _dte_bool_env("DTE_INVERSION_COMPUTE_ON_CPU", "AWEX_INVERSION_COMPUTE_ON_CPU"):
        return torch.device("cpu")
    return flat_t.device


def _inversion_bf16_margin_rel() -> float:
    return float(
        get_env_var(
            "DTE_DELTA_INVERSION_BF16_MARGIN_REL",
            "1e-4",
            fallback_names=("AWEX_DELTA_INVERSION_BF16_MARGIN_REL",),
        )
    )


def _inversion_dense_param_suffixes() -> tuple[str, ...]:
    """HF parameter suffixes to send dense under AdamW inversion.

    Qwen3 MoE router/gate weights have repeatedly shown one-element BF16 false
    negatives in snapshot verification. They are small relative to full model
    payloads, so sending them dense is the conservative default. Set the env to
    an empty string to disable, or to a comma-separated suffix list to override.
    """
    value = get_env_var(
        "DTE_DELTA_INVERSION_DENSE_PARAM_SUFFIXES",
        ".mlp.gate.weight",
        fallback_names=("AWEX_DELTA_INVERSION_DENSE_PARAM_SUFFIXES",),
        allow_empty=True,
    )
    assert value is not None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _cuda_mem_stats_mb(reset_peak: bool = True) -> tuple[float, float]:
    """Return current and peak CUDA allocation, or CPU sentinels."""
    if not torch.cuda.is_available():
        return -1.0, -1.0
    allocated = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
    peak = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    return allocated, peak


def _inversion_force_dense_param(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in _inversion_dense_param_suffixes())


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


def _adamw_hparams(
    param_group: dict,
) -> tuple[float, float, float, float, float] | None:
    """Extract (lr, weight_decay, beta1, beta2, eps) from a torch param_group."""
    last_step_lr = param_group.get("_areal_last_step_lr")
    if last_step_lr is None:
        return None
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
        float(last_step_lr),
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


@torch.no_grad()
def _bf16_rounding_boundary_mask_chunked(
    old_fp32: torch.Tensor, cur_bf16: torch.Tensor
) -> torch.Tensor:
    """Bound boundary-check temporaries by processing large tensors in chunks."""
    chunk_elems = int(
        get_env_var(
            "DTE_BOUNDARY_CHUNK_ELEMS",
            str(50_000_000),
            fallback_names=("PYQ_BOUNDARY_CHUNK_ELEMS",),
        )
    )
    if chunk_elems <= 0 or cur_bf16.numel() <= chunk_elems:
        return _bf16_rounding_boundary_mask(old_fp32, cur_bf16)

    flat_old = old_fp32.reshape(-1)
    flat_cur = cur_bf16.reshape(-1)
    result = torch.empty(flat_cur.numel(), dtype=torch.bool, device=flat_cur.device)
    for start in range(0, flat_cur.numel(), chunk_elems):
        end = min(start + chunk_elems, flat_cur.numel())
        result[start:end] = _bf16_rounding_boundary_mask(
            flat_old[start:end], flat_cur[start:end]
        ).reshape(-1)
    return result.reshape(cur_bf16.shape)


class _ReconstructionAborted(Exception):
    """Lazy reconstruction reached a rank-consistent dense fallback verdict."""


_NO_RECONSTRUCTION = object()


class _LazyReconstructDict:
    """Dict-like lazy view over reconstructed pre-step tensors.

    Current Megatron converters call ``get(id(param), param)`` and, after
    conversion, ``pop(id(param), None)``. ``get`` may advance the producer;
    the matching ``pop`` must only discard the value already returned and must
    never advance again. The producer emits a sentinel for params without an
    override so one missing value cannot force buffering the rest of the model.
    """

    def __init__(self, producer: Iterator[tuple[int, torch.Tensor | object]]) -> None:
        self._producer = producer
        self._buffer: dict[int, torch.Tensor | object] = {}
        self._consumed: set[int] = set()
        self._exhausted = False
        self._aborted = False
        self._producer_seconds = 0.0
        self._reconstructed_count = 0

    def _pull_one(self) -> tuple[int, torch.Tensor | object] | None:
        if self._exhausted:
            return None
        started = time.monotonic()
        try:
            item = next(self._producer)
        except _ReconstructionAborted:
            self._aborted = True
            self._exhausted = True
            return None
        except StopIteration:
            self._exhausted = True
            return None
        finally:
            self._producer_seconds += time.monotonic() - started
        if item[1] is not _NO_RECONSTRUCTION:
            self._reconstructed_count += 1
        return item

    def _fill(self, key: int) -> None:
        while key not in self._buffer and not self._exhausted:
            item = self._pull_one()
            if item is None:
                break
            item_key, value = item
            self._buffer[item_key] = value

    def get(self, key: int, default=None):
        if key in self._consumed:
            return default
        self._fill(key)
        value = self._buffer.get(key, _NO_RECONSTRUCTION)
        self._consumed.add(key)
        if value is _NO_RECONSTRUCTION:
            return default
        return value

    def pop(self, key: int, default=None):
        if key in self._consumed:
            value = self._buffer.pop(key, _NO_RECONSTRUCTION)
            return default if value is _NO_RECONSTRUCTION else value
        self._fill(key)
        self._consumed.add(key)
        value = self._buffer.pop(key, _NO_RECONSTRUCTION)
        return default if value is _NO_RECONSTRUCTION else value

    def finish(self) -> None:
        """Drain the producer so trailing collectives and aborts always run."""
        while not self._exhausted:
            item = self._pull_one()
            if item is None:
                break
            # Conversion has ended, so unrequested overrides can be released
            # immediately. Only their collectives and trailing verdict matter.
            del item

    def clear(self) -> None:
        self._buffer.clear()
        self._consumed.clear()

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def producer_seconds(self) -> float:
        return self._producer_seconds

    @property
    def reconstructed_count(self) -> int:
        return self._reconstructed_count


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
        # All watermark/fingerprint maps are keyed by the mcore param NAME, not
        # id(param): on EP topologies the optimizer can hold different param
        # OBJECTS than the live module traversal (observed d2p8e4: one DP
        # replica's ids never match), and ids are also not stable across
        # offload/reload cycles. Names are rank- and version-invariant.
        self._last_synced_steps: dict[str, float | None] = {}
        self._last_synced_fingerprints: dict[str, tuple] = {}
        self._last_synced_model_fingerprints: dict[str, tuple] = {}
        self._last_synced_payload_fingerprints: dict[str, tuple] = {}
        # Single-slot cache filled by precompute_masks() before the colocate
        # orchestration offloads the optimizer to CPU; consumed (or discarded)
        # by the next compute_masks()/mark_synced() call.
        self._precomputed_masks: (
            tuple[int, tuple[str, ...], dict[str, torch.Tensor] | None] | None
        ) = None

    def has_synced_watermark(self) -> bool:
        """Whether a successful full/delta payload sync recorded optimizer steps."""
        return bool(self._last_synced_steps)

    # -- gating ------------------------------------------------------------
    def _inversion_feasible(self, inner_optimizers) -> bool:
        """All hard gates from the design; any failure -> dense fallback."""
        if not inner_optimizers:
            logger.warning("Inversion: no inner optimizers; falling back to dense.")
            return False
        for opt in inner_optimizers:
            base_opt = getattr(opt, "optimizer", opt)
            adam_w_mode = getattr(base_opt, "adam_w_mode", None)
            if adam_w_mode is not None and int(adam_w_mode) == 0:
                logger.warning("Inversion: non-decoupled FusedAdam -> dense.")
                return False
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

    def _module_param_key_maps(
        self,
    ) -> tuple[dict[int, str], dict[tuple[int, tuple], str]]:
        """Map live module params to stable name keys, by id and (data_ptr, shape).

        The (data_ptr, shape) index bridges optimizer-held param objects to the
        module params when object identities diverge (observed on d2p8e4: one DP
        replica's optimizer groups hold different objects than the module
        traversal). Both views alias the same storage while weights are
        resident, so the pointer identifies the tensor. Pointers are only valid
        within one call (offload/reload reallocates storage) — never persist
        them, only the derived names.
        """
        id2key: dict[int, str] = {}
        ptr2key: dict[tuple[int, tuple], str] = {}
        engine = getattr(self._adapter, "_engine", None)
        if engine is None or getattr(engine, "model", None) is None:
            return id2key, ptr2key
        from areal.engine.megatron_utils.megatron import get_named_parameters

        num_moe_experts = getattr(engine.tf_config, "num_moe_experts", None)
        for name, param in get_named_parameters(engine.model, num_moe_experts):
            id2key.setdefault(id(param), name)
            try:
                if param.data.numel() > 0:
                    ptr2key.setdefault(
                        (param.data.data_ptr(), tuple(param.shape)), name
                    )
            except RuntimeError:
                pass  # released storage has no data_ptr
        return id2key, ptr2key

    @staticmethod
    def _param_key(param, id2key, ptr2key) -> str | int | None:
        key = id2key.get(id(param))
        if key is not None:
            return key
        try:
            if param.data.numel() > 0:
                key = ptr2key.get((param.data.data_ptr(), tuple(param.shape)))
                if key is not None:
                    return key
        except RuntimeError:
            pass
        if not id2key and not ptr2key:
            return id(param)
        return None

    def _collect_current_steps(self, inner_optimizers) -> dict[str, float | None]:
        """Return optimizer step watermarks keyed by mcore param name.

        A value of ``None`` means this param had no local optimizer state when the
        payload was synced. On the next step, ``None -> 1`` is still a valid
        one-step transition.
        """
        steps: dict[str, float | None] = {}
        id2key, ptr2key = self._module_param_key_maps()
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
                    key = self._param_key(model_param, id2key, ptr2key)
                    if key is None or key in steps:
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
                    steps[key] = step
        return steps

    def _collect_current_fingerprints(self, inner_optimizers) -> dict[str, tuple]:
        """Record compact fingerprints of model-visible local shards.

        This is intentionally not a snapshot: it stores a few scalar values per
        optimizer-owned shard, not tensor contents. The fingerprint gates AdamW
        replay to slices whose synced payload actually changed.
        """
        fingerprints: dict[str, tuple] = {}
        id2key, ptr2key = self._module_param_key_maps()
        for opt in inner_optimizers:
            fp32_groups = getattr(opt, "shard_fp32_from_float16_groups", None)
            model_groups = getattr(opt, "model_float16_groups", None)
            if fp32_groups is None or model_groups is None:
                continue
            for fp32_group, model_group in zip(fp32_groups, model_groups):
                for main_shard, model_param in zip(fp32_group, model_group):
                    if main_shard is None:
                        continue
                    key = self._param_key(model_param, id2key, ptr2key)
                    if key is None or key in fingerprints:
                        continue
                    try:
                        rng = opt._get_model_param_range_map(model_param)["param"]
                    except Exception:
                        continue
                    visible_slice = model_param.detach().reshape(-1)[
                        rng.start : rng.end
                    ]
                    fingerprints[key] = _tensor_fingerprint(visible_slice)
        return fingerprints

    def _ordered_model_params(self, fallback_order: list[torch.Tensor]):
        iter_fn = getattr(self._adapter, "_iter_model_params_for_delta", None)
        if iter_fn is not None:
            params = list(iter_fn())
            if params:
                return params

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
        return ordered_model_params

    def _collect_current_model_fingerprints(
        self, fallback_order: list[torch.Tensor] | None = None
    ) -> dict[str, tuple]:
        fingerprints: dict[str, tuple] = {}
        id2key, ptr2key = self._module_param_key_maps()
        for model_param in self._ordered_model_params(fallback_order or []):
            key = self._param_key(model_param, id2key, ptr2key)
            if key is None or key in fingerprints:
                continue
            fingerprints[key] = _tensor_fingerprint(model_param.detach().reshape(-1))
        return fingerprints

    def _collect_payload_fingerprints(
        self, payload_params: dict[str, torch.Tensor] | None = None
    ) -> dict[str, tuple]:
        if not payload_params:
            return {}
        return {
            name: _tensor_fingerprint(tensor) for name, tensor in payload_params.items()
        }

    @staticmethod
    def _mask_to_index_mask(mask: torch.Tensor) -> torch.Tensor:
        """Convert a temporary bool mask into compact external flat indices."""
        indices = mask.nonzero(as_tuple=False).squeeze(1)
        if mask.numel() <= torch.iinfo(torch.int32).max:
            return indices.to(torch.int32)
        return indices

    def capture_synced_state(
        self, payload_params: dict[str, torch.Tensor] | None = None
    ):
        """Capture post-step watermarks while model weights are still resident."""
        inner = self._adapter._get_inner_optimizers()
        if not self._inversion_feasible(inner):
            return None
        return (
            self._collect_current_steps(inner),
            self._collect_current_fingerprints(inner),
            self._collect_current_model_fingerprints(),
            self._collect_payload_fingerprints(payload_params),
        )

    def mark_synced(self, version: int, captured_state=None) -> None:
        """Record optimizer step watermarks after a successful payload sync.

        AdamW inversion only reconstructs the most recent optimizer step. Without
        this watermark, a later weight sync with no new optimizer step would replay
        the previous step's moments and report false positives.
        """
        # Drop any unconsumed precompute cache (e.g. the sync took the
        # full-sync branch and never called compute_masks).
        self._precomputed_masks = None
        if captured_state is None:
            captured_state = self.capture_synced_state()
        if captured_state is None:
            self._last_synced_steps.clear()
            self._last_synced_fingerprints.clear()
            self._last_synced_model_fingerprints.clear()
            self._last_synced_payload_fingerprints.clear()
            return
        if len(captured_state) == 3:
            (
                self._last_synced_steps,
                self._last_synced_fingerprints,
                self._last_synced_model_fingerprints,
            ) = captured_state
            self._last_synced_payload_fingerprints = {}
        else:
            (
                self._last_synced_steps,
                self._last_synced_fingerprints,
                self._last_synced_model_fingerprints,
                self._last_synced_payload_fingerprints,
            ) = captured_state
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
            "at version %d (stepped=%d shard_fingerprints=%d "
            "model_fingerprints=%d payload_fingerprints=%d max_step=%.0f)",
            known,
            version,
            stepped,
            len(self._last_synced_fingerprints),
            len(self._last_synced_model_fingerprints),
            len(self._last_synced_payload_fingerprints),
            max_step,
        )

    # -- reconstruction (GPU-only) ----------------------------------------
    @torch.no_grad()
    def _iter_reconstruct_pre_step_mcore(
        self, inner_optimizers
    ) -> Iterator[tuple[int, torch.Tensor | object]]:
        """Yield theta_{t-1} overrides in model-parameter order.

        Every model parameter produces exactly one item. Parameters without a
        usable override produce ``_NO_RECONSTRUCTION`` so an on-demand
        consumer never has to drain the rest of the model just to establish a
        missing key. Ambiguous replay is reported after all rank-aligned
        collectives finish by raising ``_ReconstructionAborted``.
        """
        from dte.core import invert_adamw

        reconstructed_count = 0
        skipped_no_state = 0
        skipped_bad_state = 0
        skipped_global_no_state = 0
        skipped_step_unchanged = 0
        skipped_missing_watermark = 0
        skipped_step_jump = 0
        skipped_missing_last_step_lr = 0
        skipped_missing_fingerprint = 0
        skipped_tracked_unchanged = 0
        skipped_payload_changed_without_step = 0
        skipped_untracked_non_optimizer = 0
        skipped_changed_non_optimizer = 0
        skipped_unchanged_non_optimizer = 0
        skipped_partial_state = 0
        unkeyed_opt_params = 0
        force_dense = False
        debug = _inversion_debug_enabled()
        global_rank = _dist_rank()
        id2key, ptr2key = self._module_param_key_maps()
        # opt_entries is keyed by the stable mcore name (see _param_key): on EP
        # topologies id(optimizer param) may never match id(module param), and
        # any rank-local miss must not change which collectives this rank
        # enters (P-deadlock 2026-07-04: half the ranks fell back to the
        # default group while their peers used the expert group -> crossed
        # communicators, all ranks spinning in the first expert-param
        # all_reduce).
        opt_entries: dict[object, tuple] = {}
        fallback_order: list[torch.Tensor] = []
        seen_fallback: set[int] = set()
        default_dp_group = None
        group_by_ranks: dict[tuple, object] = {}
        for opt_idx, opt in enumerate(inner_optimizers):
            base_opt = getattr(opt, "optimizer", opt)
            state = getattr(base_opt, "state", None)
            if state is None:
                raise _ReconstructionAborted
            group_index_map = getattr(opt, "model_param_group_index_map", None)
            fp32_groups = getattr(opt, "shard_fp32_from_float16_groups", None)
            model_groups = getattr(opt, "model_float16_groups", None)
            if fp32_groups is None or model_groups is None:
                raise _ReconstructionAborted
            dp_group = getattr(opt, "data_parallel_group", None)
            if default_dp_group is None:
                default_dp_group = dp_group
            if dp_group is not None:
                # Canonicalize group OBJECTS by their actual global-rank
                # membership: distinct communicators with identical members
                # must collapse to one object, or peers end up posting the
                # same logical reduce on different NCCL comms.
                try:
                    ranks_key = tuple(
                        torch.distributed.get_process_group_ranks(dp_group)
                    )
                except Exception:
                    ranks_key = (
                        "sig",
                        _dist_rank(dp_group),
                        _dist_world_size(dp_group),
                    )
                group_by_ranks.setdefault(ranks_key, dp_group)
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
                        raise _ReconstructionAborted
                    pid = id(model_param)
                    if pid not in seen_fallback:
                        fallback_order.append(model_param)
                        seen_fallback.add(pid)
                    key = self._param_key(model_param, id2key, ptr2key)
                    if key is None:
                        unkeyed_opt_params += 1
                        continue
                    if key in opt_entries:
                        continue
                    gi = (
                        group_index_map[model_param][0]
                        if group_index_map is not None
                        and model_param in group_index_map
                        else g
                    )
                    opt_entries[key] = (
                        opt_idx,
                        opt,
                        base_opt,
                        state,
                        main_shard,
                        gi,
                        model_param,
                    )

        # Resolve the reduce group(s) once, rank-invariantly. A param's group
        # must NEVER depend on whether THIS rank found an optimizer entry.
        expert_group = None
        if len(group_by_ranks) <= 1:
            canonical_group = (
                next(iter(group_by_ranks.values()))
                if group_by_ranks
                else default_dp_group
            )
        elif len(group_by_ranks) == 2:
            dense_ranks = None
            try:
                from megatron.core import parallel_state as mpu

                try:
                    dense_group = mpu.get_data_parallel_group(
                        with_context_parallel=True
                    )
                except TypeError:
                    dense_group = mpu.get_data_parallel_group()
                dense_ranks = tuple(
                    torch.distributed.get_process_group_ranks(dense_group)
                )
            except Exception:
                dense_ranks = None
            if dense_ranks is None or dense_ranks not in group_by_ranks:
                try:
                    dense_ranks = tuple(
                        torch.distributed.get_process_group_ranks(default_dp_group)
                    )
                except Exception:
                    dense_ranks = next(iter(group_by_ranks.keys()))
            canonical_group = group_by_ranks.get(dense_ranks, default_dp_group)
            expert_group = next(
                (grp for k, grp in group_by_ranks.items() if k != dense_ranks),
                None,
            )
        else:
            logger.warning(
                "Inversion: %d distinct DP replica sets; only dense+expert "
                "are supported -> dense.",
                len(group_by_ranks),
            )
            raise _ReconstructionAborted
        logger.info(
            "Inversion: canonical DP groups resolved (replica_sets=%d "
            "expert_split=%s entries=%d unkeyed_opt_params=%d)",
            len(group_by_ranks),
            expert_group is not None,
            len(opt_entries),
            unkeyed_opt_params,
        )

        ordered_model_params = self._ordered_model_params(fallback_order)

        if debug:
            logger.info(
                "Inversion debug: rank=%d ordered_params=%d opt_entries=%d",
                global_rank,
                len(ordered_model_params),
                len(opt_entries),
            )

        if not ordered_model_params:
            raise _ReconstructionAborted

        offloaded_states = getattr(self._adapter, "_offloaded_optimizer_states", {})

        # Post-reduce finalization, shared by the synchronous and pipelined
        # paths. Runs strictly in parameter order on every rank; only local
        # classification happens here so draining order can never diverge
        # across ranks.
        def _finalize_reduced(ctx) -> tuple[int, torch.Tensor | object]:
            nonlocal force_dense
            nonlocal reconstructed_count
            nonlocal skipped_no_state, skipped_global_no_state
            nonlocal skipped_partial_state, skipped_untracked_non_optimizer
            nonlocal skipped_changed_non_optimizer
            nonlocal skipped_unchanged_non_optimizer
            (
                param_idx,
                model_param,
                param_key,
                entry,
                rng,
                expected_one_step,
                has_local_update,
                flat_t,
                theta_t_full,
                correction,
                status,
                dp_group,
            ) = ctx
            # Single host sync for all three counters (was 3x .item()).
            status_vals = status.tolist()
            if debug:
                logger.info(
                    "Inversion debug: rank=%d param_idx=%d post_reduce "
                    "contrib=%d bad=%d",
                    global_rank,
                    param_idx,
                    int(status_vals[0]),
                    int(status_vals[1]),
                )
            if int(status_vals[1]) != 0:
                force_dense = True
                return id(model_param), _NO_RECONSTRUCTION
            if int(status_vals[0]) == 0:
                # No DP rank could reconstruct this param. For a known one-step
                # AdamW transition that is unsafe: current-vs-current would hide
                # a real update, so fall back to dense. If this rank had no
                # optimizer entry at all, only now (after the DP all-reduce)
                # classify it as a true non-AdamW payload tensor; a different DP
                # rank may have owned and contributed the optimizer shard.
                if entry is None:
                    last_model_fingerprint = (
                        self._last_synced_model_fingerprints.get(param_key)
                        if param_key is not None
                        else None
                    )
                    if last_model_fingerprint is None:
                        skipped_untracked_non_optimizer += 1
                    elif _tensor_fingerprint(flat_t) != last_model_fingerprint:
                        skipped_changed_non_optimizer += 1
                        force_dense = True
                    else:
                        skipped_unchanged_non_optimizer += 1
                else:
                    skipped_global_no_state += 1
                    if expected_one_step:
                        force_dense = True
                return id(model_param), _NO_RECONSTRUCTION
            covered = int(status_vals[2])
            if covered != flat_t.numel():
                # Some DP rank's owned state slice is missing (unkeyed entry or
                # unusable state on that rank). theta_old would silently equal
                # theta_t on the uncovered slice and hide real updates there —
                # never ship a sparse delta built from partial coverage.
                skipped_partial_state += 1
                if debug and skipped_partial_state <= 8:
                    slice_start = -1 if rng is None else rng.start
                    slice_end = -1 if rng is None else rng.end
                    logger.warning(
                        "Inversion debug: partial optimizer coverage "
                        "rank=%d param_idx=%d key=%s shape=%s covered=%d "
                        "numel=%d local_slice=%d:%d dp_rank=%d/%d "
                        "local_update=%d force_dense=%d",
                        global_rank,
                        param_idx,
                        param_key,
                        tuple(theta_t_full.shape),
                        covered,
                        flat_t.numel(),
                        slice_start,
                        slice_end,
                        _dist_rank(dp_group),
                        _dist_world_size(dp_group),
                        int(has_local_update),
                        int(force_dense),
                    )
                force_dense = True
                return id(model_param), _NO_RECONSTRUCTION
            theta_old_full = (flat_t.to(torch.float32) + correction).reshape(
                theta_t_full.shape
            )
            reconstructed_count += 1
            return id(model_param), theta_old_full

        # Pipelined all-reduce window: post async reduces and only drain once
        # this many correction bytes are in flight. Collective ORDER is
        # identical to the synchronous path on every rank; only completion is
        # overlapped ([dte-perf] showed ~18s of reconstruct dominated by
        # serialized per-param reduce + 3x .item() sync round-trips).
        window_bytes = _inversion_allreduce_window_bytes()
        pending: list[tuple] = []
        pending_bytes = 0

        def _drain_one_pending() -> tuple[int, torch.Tensor | object]:
            nonlocal pending_bytes
            ctx, work_c, work_s = pending.pop(0)
            correction = ctx[9]
            pending_bytes -= correction.numel() * correction.element_size()
            work_c.wait()
            work_s.wait()
            return _finalize_reduced(ctx)

        def _drain_pending_batch() -> list[tuple[int, torch.Tensor | object]]:
            # Never yield into the HF converter while a DP async collective is
            # still in flight. The converter may issue all-gathers on another
            # communicator, and interleaving those with this batch can deadlock
            # when ranks advance the lazy producer at slightly different times.
            ready = []
            while pending:
                ready.append(_drain_one_pending())
            return ready

        for param_idx, model_param in enumerate(ordered_model_params):
            param_key = self._param_key(model_param, id2key, ptr2key)
            entry = opt_entries.get(param_key) if param_key is not None else None
            # Group choice is a pure function of the (rank-invariant) param
            # name — never of local entry state.
            if (
                expert_group is not None
                and isinstance(param_key, str)
                and (".experts." in param_key)
            ):
                dp_group = expert_group
            else:
                dp_group = canonical_group
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
                opt_idx, opt, base_opt, state, main_shard, gi, opt_model_param = entry
                rng = opt._get_model_param_range_map(opt_model_param)["param"]
                st = state.get(main_shard)
                offloaded_state = offloaded_states.get(main_shard, {}) if st else {}
                step = _state_step_value(st, offloaded_state) if st else None
                if step is None:
                    step = _param_group_step_value(base_opt.param_groups[gi])
                if step is None or step < 1:
                    skipped_bad_state += 1
                elif param_key is None or param_key not in self._last_synced_steps:
                    skipped_missing_watermark += 1
                    force_dense = True
                else:
                    last_step = self._last_synced_steps[param_key]
                    baseline_step = 0.0 if last_step is None else last_step
                    step_delta = step - baseline_step
                    if step_delta == 0:
                        last_fingerprint = self._last_synced_fingerprints.get(param_key)
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
                        last_fingerprint = self._last_synced_fingerprints.get(param_key)
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
                            main_flat = main_shard.detach().reshape(-1)
                            if (
                                not isinstance(exp_avg, torch.Tensor)
                                or not isinstance(exp_avg_sq, torch.Tensor)
                                or exp_avg.numel() == 0
                                or exp_avg_sq.numel() == 0
                                or exp_avg.numel() != main_flat.numel()
                                or exp_avg_sq.numel() != main_flat.numel()
                            ):
                                skipped_bad_state += 1
                            else:
                                hparams = _adamw_hparams(base_opt.param_groups[gi])
                                if hparams is None:
                                    skipped_missing_last_step_lr += 1
                                    force_dense = True
                                else:
                                    lr, wd, b1, b2, eps = hparams
                                    # The fp32 main shard and moments may be
                                    # offloaded to CPU under colocate; stage them on
                                    # the inversion compute device (the visible
                                    # payload's device by default) so invert_adamw
                                    # runs on GPU instead of the CPU. Inversion
                                    # still uses the exact post-step optimizer
                                    # weight, not the bf16 model copy. Keep the
                                    # reconstructed previous value in fp32 until
                                    # mask creation so boundary-near BF16 roundoff
                                    # is handled conservatively.
                                    dev = _inversion_compute_device(flat_t)
                                    theta_t_slice = main_flat.to(
                                        device=dev, dtype=torch.float32
                                    )
                                    theta_old_slice = invert_adamw(
                                        theta_t_slice,
                                        exp_avg.to(
                                            device=dev, dtype=torch.float32
                                        ).reshape(-1),
                                        exp_avg_sq.to(
                                            device=dev, dtype=torch.float32
                                        ).reshape(-1),
                                        step,
                                        lr,
                                        wd,
                                        b1,
                                        b2,
                                        eps,
                                    )
                                    visible_slice = flat_t[rng.start : rng.end]
                                    visible_fp32 = visible_slice.to(
                                        device=theta_old_slice.device,
                                        dtype=torch.float32,
                                    )
                                    correction[rng.start : rng.end] = (
                                        theta_old_slice - visible_fp32
                                    ).to(
                                        device=correction.device,
                                        dtype=correction.dtype,
                                    )
                                    has_local_update = True

            owned_numel = (
                int(rng.end - rng.start) if has_local_update and rng is not None else 0
            )
            status = torch.tensor(
                [
                    1 if has_local_update else 0,
                    1 if force_dense else 0,
                    owned_numel,
                ],
                dtype=torch.int64,
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
            ctx = (
                param_idx,
                model_param,
                param_key,
                entry,
                rng,
                expected_one_step,
                has_local_update,
                flat_t,
                theta_t_full,
                correction,
                status,
                dp_group,
            )
            if dp_group is None:
                yield _finalize_reduced(ctx)
            elif window_bytes <= 0:
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
                yield _finalize_reduced(ctx)
            else:
                work_c = torch.distributed.all_reduce(
                    correction,
                    op=torch.distributed.ReduceOp.SUM,
                    group=dp_group,
                    async_op=True,
                )
                work_s = torch.distributed.all_reduce(
                    status,
                    op=torch.distributed.ReduceOp.SUM,
                    group=dp_group,
                    async_op=True,
                )
                pending.append((ctx, work_c, work_s))
                pending_bytes += correction.numel() * correction.element_size()
                if pending_bytes > window_bytes:
                    yield from _drain_pending_batch()
        yield from _drain_pending_batch()
        if force_dense:
            logger.warning(
                "Inversion: optimizer step replay is ambiguous "
                "(missing_watermark=%d step_jump=%d missing_last_step_lr=%d "
                "missing_fingerprint=%d "
                "payload_changed_without_step=%d "
                "untracked_non_optimizer=%d changed_non_optimizer=%d "
                "partial_state=%d unkeyed_opt_params=%d) "
                "-> dense.",
                skipped_missing_watermark,
                skipped_step_jump,
                skipped_missing_last_step_lr,
                skipped_missing_fingerprint,
                skipped_payload_changed_without_step,
                skipped_untracked_non_optimizer,
                skipped_changed_non_optimizer,
                skipped_partial_state,
                unkeyed_opt_params,
            )
            raise _ReconstructionAborted
        if reconstructed_count == 0:
            logger.info(
                "Inversion: no BF16 payload shard changed since last sync "
                "(step_unchanged=%d tracked_unchanged=%d no_state=%d "
                "bad_state=%d global_no_state=%d "
                "unchanged_non_optimizer=%d)",
                skipped_step_unchanged,
                skipped_tracked_unchanged,
                skipped_no_state,
                skipped_bad_state,
                skipped_global_no_state,
                skipped_unchanged_non_optimizer,
            )
            return
        if (
            skipped_no_state
            or skipped_bad_state
            or skipped_global_no_state
            or skipped_step_unchanged
            or skipped_tracked_unchanged
            or skipped_missing_fingerprint
            or skipped_payload_changed_without_step
            or skipped_unchanged_non_optimizer
        ):
            logger.info(
                "Inversion: reconstructed %d params, skipped no_state=%d "
                "bad_state=%d global_no_state=%d step_unchanged=%d "
                "tracked_unchanged=%d missing_fingerprint=%d "
                "payload_changed_without_step=%d "
                "unchanged_non_optimizer=%d",
                reconstructed_count,
                skipped_no_state,
                skipped_bad_state,
                skipped_global_no_state,
                skipped_step_unchanged,
                skipped_tracked_unchanged,
                skipped_missing_fingerprint,
                skipped_payload_changed_without_step,
                skipped_unchanged_non_optimizer,
            )
        return

    @torch.no_grad()
    def _reconstruct_pre_step_mcore(
        self, inner_optimizers
    ) -> dict[int, torch.Tensor] | None:
        """Eager compatibility wrapper around streaming reconstruction."""
        theta_old_by_id: dict[int, torch.Tensor] = {}
        try:
            for param_id, theta_old in self._iter_reconstruct_pre_step_mcore(
                inner_optimizers
            ):
                if theta_old is not _NO_RECONSTRUCTION:
                    theta_old_by_id[param_id] = theta_old
        except _ReconstructionAborted:
            return None
        return theta_old_by_id

    # -- entry point -------------------------------------------------------
    @torch.no_grad()
    def precompute_masks(self, names, tensors, version) -> bool:
        """Compute and cache masks while optimizer state is still GPU-resident.

        The colocate gateway offloads the optimizer (fp32 main shards + AdamW
        moments) to CPU right before the weight sync to make room for the
        inference-side apply. Calling this beforehand lets the inversion run
        with all operands on GPU and moves the mask computation off the sync
        critical path. The cached result (including ``None`` = infeasible) is
        consumed by the next ``compute_masks`` call for the same version and
        parameter list; any mismatch discards the cache and recomputes.

        All training ranks must call this together: the mask computation runs
        the same DP/TP collectives as ``compute_masks``.
        """
        self._precomputed_masks = None
        masks = self.compute_masks(names, tensors, version)
        self._precomputed_masks = (int(version), tuple(names), masks)
        return masks is not None

    def _pop_precomputed_masks(self, names, version):
        """Return ``(hit, masks)`` consuming the precompute cache if valid.

        The hit decision must be rank-invariant: a cache hit skips the DP/TP
        collectives inside the recompute path, so if any training rank misses
        (e.g. a restarted worker lost its cache, or a stale version), every
        rank must recompute or the collectives would mismatch and hang. A
        cheap global MIN-reduce promotes any local miss to a global miss.
        """
        cached = self._precomputed_masks
        self._precomputed_masks = None
        local_hit = False
        masks = None
        if cached is not None:
            c_version, c_names, c_masks = cached
            if c_version == int(version) and c_names == tuple(names):
                local_hit = True
                masks = c_masks
            else:
                logger.warning(
                    "Inversion: discarding stale precomputed masks "
                    "(cached v%d/%d params vs requested v%d/%d params)",
                    c_version,
                    len(c_names),
                    int(version),
                    len(names),
                )
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and _dist_world_size() > 1
        ):
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            hit_t = torch.tensor(
                [1 if local_hit else 0], dtype=torch.int32, device=device
            )
            torch.distributed.all_reduce(hit_t, op=torch.distributed.ReduceOp.MIN)
            if int(hit_t.item()) == 0:
                if local_hit:
                    logger.warning(
                        "Inversion: dropping precomputed masks for v%d; "
                        "another rank missed its cache, recomputing on all "
                        "ranks to keep collectives aligned",
                        int(version),
                    )
                return False, None
        return (local_hit, masks) if local_hit else (False, None)

    def compute_masks(self, names, tensors, version):
        """Return sparse index masks or ``None`` to force a dense step."""
        hit, cached_masks = self._pop_precomputed_masks(names, version)
        if hit:
            logger.info(
                "Inversion: using precomputed masks for v%d (%d params, %s)",
                int(version),
                len(names),
                "feasible" if cached_masks is not None else "dense_infeasible",
            )
            return cached_masks
        perf_start = time.monotonic()
        # Reset the peak counter so peak_mb below covers inversion only.
        _cuda_mem_stats_mb()
        adapter = self._adapter
        t_inner = time.monotonic()
        inner = adapter._get_inner_optimizers()
        inner_ms = (time.monotonic() - t_inner) * 1000
        t_feasible = time.monotonic()
        if not self._inversion_feasible(inner):
            logger.info(
                "[dte-perf][inversion] v%d result=dense_infeasible "
                "get_optimizers_ms=%.1f feasible_ms=%.1f total_ms=%.1f",
                version,
                inner_ms,
                (time.monotonic() - t_feasible) * 1000,
                (time.monotonic() - perf_start) * 1000,
            )
            return None
        feasible_ms = (time.monotonic() - t_feasible) * 1000

        cur = dict(zip(names, tensors))
        masks: dict[str, torch.Tensor] = {}
        from dte.core import bitwise_changed_mask

        # Convert reconstructed pre-step params through the SAME all_gather +
        # convert_to_hf path as the live payload. Prefer the streaming adapter
        # interface so AdamW inversion does not materialize a second full HF
        # payload (current HF payload + old HF payload) before mask creation.
        iter_hf = getattr(adapter, "_iter_hf_with_overrides", None)
        lazy: _LazyReconstructDict | None = None
        streaming_reconstruct = _streaming_reconstruct_enabled() and iter_hf is not None
        t_reconstruct = time.monotonic()
        if streaming_reconstruct:
            theta_old_by_id = _LazyReconstructDict(
                self._iter_reconstruct_pre_step_mcore(inner)
            )
            reconstruct_ms = 0.0
            theta_old_param_count = 0
        else:
            theta_old_by_id = self._reconstruct_pre_step_mcore(inner)
            reconstruct_ms = (time.monotonic() - t_reconstruct) * 1000
            if theta_old_by_id is None:
                logger.warning("Inversion: reconstruction unavailable -> dense step.")
                logger.info(
                    "[dte-perf][inversion] v%d "
                    "result=dense_reconstruct_unavailable "
                    "get_optimizers_ms=%.1f feasible_ms=%.1f "
                    "reconstruct_mcore_ms=%.1f total_ms=%.1f",
                    version,
                    inner_ms,
                    feasible_ms,
                    reconstruct_ms,
                    (time.monotonic() - perf_start) * 1000,
                )
                return None
            theta_old_param_count = len(theta_old_by_id)

        old_items = None
        stream_old_hf = iter_hf is not None
        convert_hf_ms = 0.0
        if stream_old_hf:
            if streaming_reconstruct:
                lazy = theta_old_by_id
            old_items = iter(iter_hf(theta_old_by_id))
        else:
            t_convert = time.monotonic()
            hf_old = adapter._convert_hf_with_overrides(theta_old_by_id)
            convert_hf_ms = (time.monotonic() - t_convert) * 1000
            old_items = iter(hf_old.items())

        mask_loop_ms = 0.0
        full_changed_masks = 0
        shape_mismatch_masks = 0
        forced_dense_params = 0
        forced_dense_elements = 0
        seen_names: set[str] = set()
        dense_verdict: str | None = None
        while True:
            if stream_old_hf:
                t_next = time.monotonic()
            try:
                name, old_t = next(old_items)
            except StopIteration:
                break
            if stream_old_hf:
                convert_hf_ms += (time.monotonic() - t_next) * 1000

            cur_t = cur.get(name)
            if cur_t is None:
                continue
            seen_names.add(name)
            t_mask = time.monotonic()
            if _inversion_force_dense_param(name):
                # Known-unsupported inversion cases (e.g. tiny Qwen3 MoE router
                # gates) stay dense so snapshot verification cannot miss
                # changed BF16 values (c3c80a935 false-negative fix).
                masks[name] = torch.ones(
                    cur_t.numel(), dtype=torch.bool, device=cur_t.device
                )
                forced_dense_params += 1
                forced_dense_elements += cur_t.numel()
                mask_loop_ms += (time.monotonic() - t_mask) * 1000
                continue
            if old_t is None or old_t.shape != cur_t.shape:
                # No pre-step counterpart -> treat as fully changed (dense).
                masks[name] = torch.ones(
                    cur_t.numel(), dtype=torch.bool, device=cur_t.device
                )
                full_changed_masks += 1
                if old_t is not None:
                    shape_mismatch_masks += 1
                mask_loop_ms += (time.monotonic() - t_mask) * 1000
                continue
            old_payload = old_t.to(cur_t.dtype)
            mask = bitwise_changed_mask(cur_t, old_payload).reshape(-1)
            if cur_t.dtype == torch.bfloat16 and old_t.dtype != cur_t.dtype:
                same_payload = ~mask
                boundary = _bf16_rounding_boundary_mask_chunked(old_t, cur_t).reshape(
                    -1
                )
                mask = mask | (same_payload & boundary)
            if not bool(mask.any().item()) and self._last_synced_payload_fingerprints:
                last_payload_fingerprint = self._last_synced_payload_fingerprints.get(
                    name
                )
                if last_payload_fingerprint is None:
                    if dense_verdict is None:
                        logger.warning(
                            "Inversion: payload %s has no synced fingerprint while "
                            "detector mask is empty -> dense step.",
                            name,
                        )
                        dense_verdict = "missing_payload_fingerprint"
                cur_payload_fingerprint = _tensor_fingerprint(cur_t)
                if (
                    last_payload_fingerprint is not None
                    and cur_payload_fingerprint != last_payload_fingerprint
                ):
                    if dense_verdict is None:
                        logger.warning(
                            "Inversion: payload %s changed since last sync but "
                            "detector mask is empty -> dense step.",
                            name,
                        )
                        dense_verdict = "payload_fingerprint_changed"
            masks[name] = self._mask_to_index_mask(mask)
            mask_loop_ms += (time.monotonic() - t_mask) * 1000

        if lazy is not None:
            # The converter may not request every model parameter. Draining is
            # still mandatory so trailing collectives and a late dense verdict
            # execute on every rank.
            producer_during_convert_ms = lazy.producer_seconds * 1000
            lazy.finish()
            reconstruct_ms = lazy.producer_seconds * 1000
            theta_old_param_count = lazy.reconstructed_count
            convert_hf_ms = max(0.0, convert_hf_ms - producer_during_convert_ms)
            if lazy.aborted:
                dense_verdict = dense_verdict or "reconstruction_aborted"
            lazy.clear()

        for name, cur_t in cur.items():
            if name in seen_names:
                continue
            t_mask = time.monotonic()
            masks[name] = torch.ones(
                cur_t.numel(), dtype=torch.bool, device=cur_t.device
            )
            full_changed_masks += 1
            mask_loop_ms += (time.monotonic() - t_mask) * 1000
        if dense_verdict is not None:
            logger.warning(
                "Inversion: completed HF conversion before dense fallback (reason=%s).",
                dense_verdict,
            )
            return None
        logger.info(
            "Inversion: computed masks for %d HF params at version %d "
            "(forced_dense_params=%d forced_dense_elements=%d)",
            len(masks),
            version,
            forced_dense_params,
            forced_dense_elements,
        )
        alloc_mb, peak_mb = _cuda_mem_stats_mb()
        logger.info(
            "[dte-perf][inversion] v%d result=sparse params=%d "
            "theta_old_params=%d full_changed_masks=%d "
            "shape_mismatch_masks=%d get_optimizers_ms=%.1f "
            "feasible_ms=%.1f reconstruct_mcore_ms=%.1f "
            "convert_hf_ms=%.1f mask_loop_ms=%.1f stream_old_hf=%d "
            "streaming_reconstruct=%d "
            "total_ms=%.1f alloc_mb=%.0f peak_mb=%.0f",
            version,
            len(masks),
            theta_old_param_count,
            full_changed_masks,
            shape_mismatch_masks,
            inner_ms,
            feasible_ms,
            reconstruct_ms,
            convert_hf_ms,
            mask_loop_ms,
            int(stream_old_hf),
            int(streaming_reconstruct),
            (time.monotonic() - perf_start) * 1000,
            alloc_mb,
            peak_mb,
        )
        return masks
