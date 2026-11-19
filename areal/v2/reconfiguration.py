# SPDX-License-Identifier: Apache-2.0

"""Coordinate Actor and rollout layouts at an idle training boundary."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
            raise ValueError(
                "Reconfiguration currently requires local single-node pools"
            )
        return [device.gpu for device in self.devices]


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
    Current execution uses local single-node pools. Device identities also carry
    a node IP for future multi-node placement.
    """
    with actor._reconfiguration_lock, actor._clear_batches_lock:
        actor._ensure_initialized()
        rollout = actor.rollout
        node = gethostip()
        old_actor = {
            Device(node, device)
            for device, address in actor._worker_pool_devices.items()
            if address in actor._worker_addrs
        }
        old_rollout = (
            {
                Device(node, device)
                for devices in rollout._inference_gpu_devices.values()
                for device in devices
            }
            if rollout is not None
            else set()
        )
        actor_alloc = rollout_alloc = state = None
        actor_devices = [device.gpu for device in old_actor]
        rollout_devices = [device.gpu for device in old_rollout]
        if actor_layout is not None:
            if actor_layout.attention_dp_size is not None:
                raise ValueError("Attention DP applies only to SGLang rollout layouts")
            actor_devices = actor_layout.local_gpu_devices()
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
            rollout_devices = rollout_layout.local_gpu_devices()
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
        target_actor = (
            set(actor_layout.devices) if actor_layout is not None else old_actor
        )
        target_rollout = (
            set(rollout_layout.devices) if rollout_layout is not None else old_rollout
        )
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
