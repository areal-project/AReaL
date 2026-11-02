# SPDX-License-Identifier: Apache-2.0

"""FSDP2 CPU-offload work, separate from GPU collectives and AWEX publishing.

Traffic is logical eager-tensor traffic, not measured DDR bytes. Rates must
describe one concurrently running training rank, not the whole host's peak.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from ..core import Estimate, Features, nonnegative
from ._actor.config import ActorConfig, ModelShape
from ._actor.packing import PackingPlan
from .transition import TransferTask


@dataclass(frozen=True)
class FSDPHostRates:
    h2d_gbps: float | None = None
    d2h_gbps: float | None = None
    cpu_memory_gbps: float | None = None
    transfer_launch_seconds: float | None = None
    optimizer_fixed_seconds: float | None = None
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name != "source_ids" and value is not None:
                nonnegative(item.name, value, positive=item.name.endswith("gbps"))

    def estimate(self, work: Features) -> Estimate:
        values = dict(work)
        requirements = (
            ("h2d_bytes", "h2d_gbps", 1e9),
            ("d2h_bytes", "d2h_gbps", 1e9),
            ("cpu_logical_bytes", "cpu_memory_gbps", 1e9),
        )
        seconds = 0.0
        missing = []
        for counter, rate_name, units in requirements:
            if values.get(counter, 0):
                rate = getattr(self, rate_name)
                if rate is None:
                    missing.append(rate_name)
                else:
                    seconds += values[counter] / (rate * units)
        if values.get("h2d_bytes", 0) or values.get("d2h_bytes", 0):
            if self.transfer_launch_seconds is None:
                missing.append("transfer_launch_seconds")
            elif self.transfer_launch_seconds:
                if "transfer_calls" not in values:
                    missing.append("observed parameter tensor count for transfer_calls")
                else:
                    seconds += self.transfer_launch_seconds * values["transfer_calls"]
        for counter, rate_name in (("optimizer_steps", "optimizer_fixed_seconds"),):
            if values.get(counter, 0):
                rate = getattr(self, rate_name)
                if rate is None:
                    missing.append(rate_name)
                else:
                    seconds += values[counter] * rate
        if missing:
            return Estimate(
                None,
                "unsupported",
                notes=("Missing FSDP host rates: " + ", ".join(missing),),
            )
        return Estimate(
            seconds,
            "assumed",
            source_ids=self.source_ids,
            notes=(
                "Serial FSDP host-transfer/CPU prior; no overlap or whole-host bandwidth entitlement implied",
                "CPU logical eager traffic is not a measurement of physical DDR traffic; clipping uses an upper prior",
            ),
        )


def optimizer_bytes_per_element(config: ActorConfig) -> int:
    """Eager BF16 Adam/Kahan traffic from fsdp_utils/optimizer.py, lines 164-190."""
    if config.optimizer_type != "adam_bf16" or config.optimizer_dtype != "bfloat16":
        raise ValueError("FSDP traffic currently requires BF16 Adam/Kahan storage")
    return 64 + 4 * config.optimizer_weight_decay


def fsdp_work(
    config: ActorConfig, model: ModelShape, plan: PackingPlan
) -> tuple[tuple[str, Features], ...]:
    if config.backend != "fsdp":
        raise ValueError("FSDP work requires the fsdp backend")
    optimizer_bytes_per_element(config)
    shard = model.total_parameters / config.parallel.dp
    wrap_groups = model.layers + (1 if model.tied_embeddings else 2)
    logp_mbs = len(plan.logprob[0]) if plan.logprob else 0
    train_mbs = sum(len(call[0]) for call in plan.train)
    updates = len(plan.train)
    # The real engine explicitly sets reshard_after_forward=True, including root.
    if not config.reshard_after_forward or not config.fsdp_root_reshard_after_forward:
        raise ValueError(
            "FSDP profile requires explicit body/root reshard_after_forward=True"
        )
    # Resident BF16 shards first pack into flat_foreach_copy_input, then
    # copy into the collective input and out to parameters (Torch 2.9.1).
    gather_copies = (
        1
        + (1 + int(not config.offload_params and config.parallel.dp > 1))
        / config.parallel.dp
    )
    logp = {
        "ag_copy_logical_bytes": 4 * model.total_parameters * gather_copies * logp_mbs,
        "h2d_bytes": 2 * shard * logp_mbs if config.offload_params else 0.0,
        "d2h_bytes": 0.0,
        "cpu_logical_bytes": 0.0,
        "transfer_groups": wrap_groups * logp_mbs if config.offload_params else 0.0,
        "optimizer_steps": 0.0,
        "parameter_gathers": float(logp_mbs),
        "gradient_reductions": 0.0,
    }
    accum = 6 * shard * (train_mbs - updates) if config.offload_params else 0.0
    adam = optimizer_bytes_per_element(config) * shard * updates
    # Norm reads BF16 gradients; optional CPU clipping creates/scales/copies a
    # temporary. Its 8-byte upper prior is explicit, never a zero-filled gap.
    norm_and_clip = 10 * shard * updates
    train = {
        "ag_copy_logical_bytes": 8 * model.total_parameters * gather_copies * train_mbs,
        "rs_copy_logical_bytes": 2
        * config.grad_bytes
        * model.total_parameters
        * train_mbs,
        "h2d_bytes": 4 * shard * train_mbs if config.offload_params else 0.0,
        "d2h_bytes": 2 * shard * train_mbs if config.offload_params else 0.0,
        "cpu_accumulation_bytes": accum,
        "optimizer_logical_bytes": adam,
        "norm_and_clip_logical_bytes": norm_and_clip,
        "cpu_logical_bytes": accum + adam + norm_and_clip
        if config.offload_params
        else 0.0,
        "transfer_groups": 3 * wrap_groups * train_mbs
        if config.offload_params
        else 0.0,
        "optimizer_steps": float(updates) if config.offload_params else 0.0,
        "parameter_gathers": float(2 * train_mbs),
        "gradient_reductions": float(train_mbs),
    }
    if config.offload_params and config.fsdp_parameter_tensors is not None:
        logp["transfer_calls"] = float(config.fsdp_parameter_tensors * logp_mbs)
        train["transfer_calls"] = float(config.fsdp_parameter_tensors * 3 * train_mbs)
    return (
        ("recompute_logp", tuple(logp.items())),
        ("train_step", tuple(train.items())),
    )


def fsdp_publish_tasks(
    *,
    parameter_shard_bytes: float,
    staging_tensors: int,
    contiguous_tensor_bytes: float,
    rates: FSDPHostRates,
    gpu_copy_gbps: float | None,
    network: Estimate,
    cleanup: Estimate,
    contiguous_tensors: int | None = None,
    gpu_copy_launch_seconds: float | None = None,
) -> tuple[TransferTask, ...]:
    """Fresh CPU shards precede contiguous copies and the real AWEX peer plan.

    The caller supplies network volume/fanout from the validated transfer plan;
    model-bytes times replicas is not an adequate replacement. Cleanup covers
    pre-staging work; network must exclude staging/copy time. A calibrated whole
    publication label replaces this entire DAG rather than being its last task.
    """
    nonnegative("parameter_shard_bytes", parameter_shard_bytes)
    nonnegative("contiguous_tensor_bytes", contiguous_tensor_bytes)
    if type(staging_tensors) is not int or staging_tensors < 1:
        raise ValueError("AWEX staging needs an observed positive tensor count")
    if gpu_copy_gbps is not None:
        nonnegative("gpu_copy_gbps", gpu_copy_gbps, positive=True)
    if contiguous_tensors is not None and (
        type(contiguous_tensors) is not int
        or contiguous_tensors < 0
        or (contiguous_tensor_bytes > 0 and contiguous_tensors == 0)
    ):
        raise ValueError("Contiguous tensor count must agree with copy bytes")
    if gpu_copy_launch_seconds is not None:
        nonnegative("gpu_copy_launch_seconds", gpu_copy_launch_seconds)
    setup = gpu_copy_launch_seconds
    if setup:
        setup = setup * contiguous_tensors if contiguous_tensors is not None else None
    stage = rates.estimate(
        (
            ("h2d_bytes", parameter_shard_bytes),
            ("transfer_calls", float(staging_tensors)),
        )
    )
    return (
        TransferTask("release_cache", measured=cleanup),
        TransferTask("stage_fresh_shards", ("release_cache",), measured=stage),
        TransferTask(
            "contiguous_views",
            ("stage_fresh_shards",),
            measured=Estimate(
                0, "assumed", notes=("Explicit zero contiguous-copy payload",)
            ),
        )
        if not contiguous_tensor_bytes
        else TransferTask(
            "contiguous_views",
            ("stage_fresh_shards",),
            transferred_bytes=2 * contiguous_tensor_bytes,
            bandwidth_bytes_s=gpu_copy_gbps * 1e9
            if gpu_copy_gbps is not None
            else None,
            setup_seconds=setup,
        ),
        TransferTask("nccl_plan", ("contiguous_views",), measured=network),
    )
