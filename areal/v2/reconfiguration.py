# SPDX-License-Identifier: Apache-2.0

"""Coordinate Actor and rollout layouts at an idle training boundary."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any

from areal.api.io_struct import SaveLoadMeta
from areal.infra.utils.concurrent import run_async_task
from areal.utils.network import gethostip


@dataclass(frozen=True)
class Device:
    """A GPU identified by its worker's advertised node IP and local index."""

    node: str
    gpu: int

    def __post_init__(self) -> None:
        if not isinstance(self.node, str) or not self.node.strip():
            raise ValueError("Device.node must be a nonempty worker node IP")
        if type(self.gpu) is not int or self.gpu < 0:
            raise ValueError("Device.gpu must be a nonnegative integer")


def sglang_gpus_per_node(tp_size: int, pp_size: int, nnodes: int) -> int:
    """Number of local ranks in SGLang's node partition of a TP/PP mesh."""
    if min(tp_size, pp_size, nnodes) < 1:
        raise ValueError("SGLang TP, PP and node count must be positive")
    # SGLang first groups complete PP stages on a node. If there are more
    # nodes than stages, it splits the TP ranks of each stage across nodes.
    if nnodes <= pp_size:
        valid = pp_size % nnodes == 0
    else:
        valid = nnodes % pp_size == 0 and tp_size % (nnodes // pp_size) == 0
    if not valid:
        raise ValueError("SGLang TP/PP ranks cannot be evenly partitioned across nodes")
    return tp_size * pp_size // nnodes


class RayDevicePool:
    """Ordered physical devices. A Ray pool slot is an index into this sequence."""

    def __init__(self, devices: tuple[Device, ...]):
        self.devices = tuple(devices)
        self.slots = {device: slot for slot, device in enumerate(self.devices)}
        if (
            not self.devices
            or len(self.slots) != len(self.devices)
            or not all(isinstance(device, Device) for device in self.devices)
        ):
            raise ValueError("Ray reconfiguration requires unique physical devices")

    @classmethod
    def for_scheduler(cls, scheduler: Any) -> RayDevicePool | None:
        pool = getattr(scheduler, "reconfiguration_pool", None)
        return pool if isinstance(pool, cls) else None

    def local_gpu_ids(self, node: str, slots: list[int]) -> list[int]:
        devices = [self.devices[slot] for slot in slots]
        if any(device.node != node for device in devices):
            raise ValueError("GPU slots assigned to a Guard must be on its node")
        return [device.gpu for device in devices]

    def validate_replicas(self, slots: list[int], replica_size: int) -> None:
        for start in range(0, len(slots), replica_size):
            self.local_gpu_ids(
                self.devices[slots[start]].node, slots[start : start + replica_size]
            )

    def rollout_groups(
        self, slots: list[int], tp_size: int, pp_size: int
    ) -> list[list[list[int]]]:
        """Return DP groups of ordered node-local GPU slots."""
        width = tp_size * pp_size
        if (
            min(tp_size, pp_size) < 1
            or not slots
            or len(slots) % width
            or len(set(slots)) != len(slots)
            or any(
                type(slot) is not int or not 0 <= slot < len(self.devices)
                for slot in slots
            )
        ):
            raise ValueError(
                "SGLang layouts require complete groups of unique pool slots"
            )
        replicas = []
        for start in range(0, len(slots), width):
            chunk = slots[start : start + width]
            node_chunks = [
                list(group)
                for _, group in groupby(chunk, key=lambda slot: self.devices[slot].node)
            ]
            nodes = {self.devices[group[0]].node for group in node_chunks}
            ranks = sglang_gpus_per_node(tp_size, pp_size, len(nodes))
            if len(nodes) != len(node_chunks) or any(
                len(group) != ranks for group in node_chunks
            ):
                raise ValueError(
                    "SGLang requires one ordered GPU block of equal size per node"
                )
            replicas.append(node_chunks)
        return replicas


@dataclass(frozen=True)
class ExecutionLayout:
    """One role's parallel backend and ordered rank-to-device placement."""

    backend: str
    devices: tuple[Device, ...]
    # SGLang only: None inherits, 1 disables, >1 enables Attention DP.
    attention_dp_size: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "devices", tuple(self.devices))
        if not self.devices or any(not isinstance(d, Device) for d in self.devices):
            raise ValueError("ExecutionLayout requires an ordered sequence of Devices")
        if len(set(self.devices)) != len(self.devices):
            raise ValueError("ExecutionLayout devices must be unique")

    @classmethod
    def local(
        cls,
        backend: str,
        gpu_devices: list[int],
        *,
        attention_dp_size: int | None = None,
    ) -> ExecutionLayout:
        node = gethostip()
        return cls(
            backend,
            tuple(Device(node, gpu) for gpu in gpu_devices),
            attention_dp_size,
        )

    def local_gpu_devices(self) -> list[int]:
        if {device.node for device in self.devices} != {gethostip()}:
            raise ValueError("LocalScheduler requires target GPUs on the Trainer node")
        return [device.gpu for device in self.devices]

    def gpu_slots(self, pool: RayDevicePool | None) -> list[int]:
        if pool is None:
            return self.local_gpu_devices()
        if not all(device in pool.slots for device in self.devices):
            raise ValueError("Target layout contains a device outside the Ray pool")
        return [pool.slots[device] for device in self.devices]

    @classmethod
    def from_gpu_slots(
        cls, backend: str, slots: list[int], pool: RayDevicePool | None
    ) -> ExecutionLayout:
        if pool is None:
            return cls.local(backend, slots)
        if any(
            type(slot) is not int or not 0 <= slot < len(pool.devices) for slot in slots
        ):
            raise ValueError("GPU slots must belong to the Ray reconfiguration pool")
        return cls(backend, tuple(pool.devices[slot] for slot in slots))


async def _migrate(
    actor: Any,
    rollout: Any,
    actor_alloc: Any,
    actor_devices: list[int],
    rollout_alloc: Any,
    rollout_devices: list[int],
    actor_waits: bool,
    rollout_waits: bool,
    state: SaveLoadMeta | None,
    attention_dp_size: int | None = None,
) -> None:
    # Futures carry either a successful release or its error to the other role.
    # Gather joins both branches, including save/load threads, before returning.
    loop = asyncio.get_running_loop()
    released = {role: loop.create_future() for role in ("actor", "rollout")}

    async def actor_branch():
        if actor_alloc is None:
            released["actor"].set_result(None)
            return
        prefetch = None
        try:
            await asyncio.to_thread(actor.save, state)
            prefetch = asyncio.create_task(
                actor._async_prefetch_model_state(
                    state.path, actor.get_version(), actor_devices
                )
            )
            await actor._async_release_engines()
            released["actor"].set_result(None)
            if actor_waits:
                await asyncio.shield(released["rollout"])
            await actor._async_create_engines(actor_alloc, actor_devices)
        except BaseException as exc:
            if not released["actor"].done():
                released["actor"].set_exception(exc)
            raise
        finally:
            if prefetch is not None:
                await prefetch
        await asyncio.to_thread(actor.load, state)

    async def rollout_branch():
        try:
            if rollout_alloc is not None:
                await rollout._async_release_model_servers()
        except BaseException as exc:
            released["rollout"].set_exception(exc)
            raise
        released["rollout"].set_result(None)
        if rollout_alloc is not None:
            if rollout_waits:
                await asyncio.shield(released["actor"])
            options = (
                {"attention_dp_size": attention_dp_size}
                if attention_dp_size is not None
                else {}
            )
            await rollout._async_create_model_servers(
                rollout_alloc, rollout_devices, **options
            )

    results = await asyncio.gather(
        actor_branch(), rollout_branch(), return_exceptions=True
    )
    for future in released.values():
        if not future.cancelled():
            future.exception()
    for result in results:
        if isinstance(result, BaseException):
            raise result


def reconfigure(
    actor: Any,
    *,
    actor_layout: ExecutionLayout | None = None,
    rollout_layout: ExecutionLayout | None = None,
    state_path: str | None = None,
) -> None:
    """Apply supplied role layouts while retaining frontends and full Actor state.

    Omitted roles retain their layout. Enter synchronously after update/statistics
    and batch cleanup. Actor changes require a fresh shared-filesystem directory.
    Controllers use the same pool slots for ownership and device handoff.
    """
    with actor._reconfiguration_lock, actor._clear_batches_lock:
        actor._ensure_initialized()
        rollout = actor.rollout
        pool = RayDevicePool.for_scheduler(actor.scheduler)
        old_actor = {
            device
            for device, address in actor._worker_pool_devices.items()
            if address in actor._worker_addrs
        }
        old_rollout = (
            {
                device
                for devices in rollout._inference_gpu_devices.values()
                for device in devices
            }
            if rollout is not None
            else set()
        )
        actor_alloc = rollout_alloc = state = None
        actor_devices = list(old_actor)
        rollout_devices = list(old_rollout)
        if actor_layout is not None:
            if actor_layout.attention_dp_size is not None:
                raise ValueError("Attention DP applies only to SGLang rollout layouts")
            actor_devices = actor_layout.gpu_slots(pool)
            actor_alloc = actor._validate_engine_rebuild(
                actor_layout.backend, actor_devices
            )
            if state_path is None:
                raise ValueError("Actor reconfiguration requires a fresh state_path")
            directory = Path(state_path).resolve()
            if directory.exists() and any(directory.iterdir()):
                raise FileExistsError("Actor snapshot directory must be fresh")
            state = SaveLoadMeta(
                path=str(directory), weight_format="raw", with_optim=True
            )
        if rollout_layout is not None:
            if rollout is None:
                raise RuntimeError("Connect rollout before requesting its layout")
            rollout_devices = rollout_layout.gpu_slots(pool)
            rollout_alloc = rollout.validate_reconfiguration(
                rollout_layout.backend, rollout_devices
            )
            dp = rollout_layout.attention_dp_size
            if dp is None:
                args = rollout._model_server_args
                dp = (
                    args.get("dp_size", 1)
                    if args.get("enable_dp_attention", False)
                    else 1
                )
            if type(dp) is not int or dp < 1 or rollout_alloc.parallel.tp_size % dp:
                raise ValueError("Attention DP must be positive and divide rollout TP")
        if actor_alloc is None and rollout_alloc is None:
            return
        target_actor = set(actor_devices)
        target_rollout = set(rollout_devices)
        if target_actor & target_rollout:
            raise ValueError("Target Actor and rollout devices must be disjoint")
        if actor._pending_clear_shards:
            raise RuntimeError("Finish pending batch cleanup before reconfiguration")
        meta = actor._weight_update_meta
        if rollout is not None:
            if (
                not actor._weight_update_connected
                or meta is None
                or meta.type != "awex"
            ):
                raise RuntimeError("Reconfiguration requires a connected AWEX rollout")
            if rollout.get_version() != actor.get_version():
                raise RuntimeError(
                    "Actor and rollout versions differ at the switch boundary"
                )
            meta = meta.with_version(actor.get_version())
        try:
            actor._reconfiguration_phase = "pausing"
            actor.hold_training()
            if rollout is not None and not rollout._reconfiguration_prepared:
                rollout.pause_generation()
            actor._reconfiguration_phase = "disconnecting"
            actor.disconnect_engine()
            actor._reconfiguration_phase = "migrating"
            run_async_task(
                _migrate,
                actor,
                rollout,
                actor_alloc,
                actor_devices,
                rollout_alloc,
                rollout_devices,
                bool(target_actor & old_rollout),
                bool(target_rollout & old_actor),
                state,
                rollout_layout.attention_dp_size
                if rollout_layout is not None
                else None,
            )
            if rollout is not None:
                actor._reconfiguration_phase = "publishing"
                actor.connect_engine(rollout, meta)
                actor.update_weights(meta, resume_generation=False)
                rollout.continue_generation()
            actor.release_training()
            actor._reconfiguration_phase = "ready"
        except BaseException:
            # Keep both frontends held and the immutable source files available.
            actor._reconfiguration_phase = "failed"
            raise
