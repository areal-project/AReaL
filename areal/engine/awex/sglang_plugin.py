# SPDX-License-Identifier: Apache-2.0

"""AWEX SGLang scheduler plugin for colocated weight transfer.

Patches SGLang's scheduler to inject CUDA IPC weight receiving capabilities.
When AWEX_META_SERVER_ADDR env var is set, starts a background thread that
fetches IPC handles from MetaServer (CPU I/O) and queues them for the
scheduler's main loop to process (CUDA copy on main thread).

Weight transfer flow (aligned with Asystem colocate mode):
  1. Training side: convert params → cuda_ipc_serialize → MetaServer put
  2. Background thread: MetaServer get → queue IPC data (CPU only)
  3. Scheduler main loop: release_memory → deserialize + copy → resume_memory
  4. Main loop: signal done → train side releases shared tensors

Usage:
    # Option 1: Register plugin then launch SGLang
    from areal.engine.awex.sglang_plugin import register_awex_plugin
    register_awex_plugin()

    # Option 2: Run as entry module (replaces sglang.launch_server)
    # python3 -m areal.engine.awex.sglang_plugin --model-path ...
"""

from __future__ import annotations

import importlib
import os
import queue
import threading
import time
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field
from typing import Any

from areal.engine.awex.memory_saver import patch_tms_hook_mode

# Must run before importing SGLang. Its scheduler may import Megatron while
# initializing the model, and Megatron otherwise switches torch-memory-saver
# away from the preload hook required by pauseable CUDA graphs.
patch_tms_hook_mode()


def assert_alloc_conf_supports_memory_saver(conf: str) -> None:
    """Reject allocator configs that silently disable SGLang's memory saver."""
    if "expandable_segments:true" in conf.lower().replace(" ", ""):
        raise RuntimeError(
            "SGLang's memory saver cannot unmap/remap expandable segments, so "
            f"it would disable itself (PYTORCH_CUDA_ALLOC_CONF={conf!r}). Give "
            "the rollout role its own scheduling_spec env_vars without "
            "expandable_segments instead of sharing the actor's."
        )


assert_alloc_conf_supports_memory_saver(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))

from areal.utils import pkg_version  # noqa: E402
from areal.utils.environ import (  # noqa: E402
    get_bool_env_var,
    get_float_env_var,
    get_int_env_var,
)
from areal.utils.logging import getLogger  # noqa: E402

logger = getLogger("AwexSGLangPlugin")
SUPPORTED_SGLANG_VERSIONS = ("0.5.9", "0.5.10.post1")


def assert_supported_sglang_version() -> None:
    """Refuse to patch a SGLang build whose internals were not verified."""
    installed = pkg_version.get_version("sglang")
    if installed not in SUPPORTED_SGLANG_VERSIONS:
        raise RuntimeError(
            "AWEX colocate patches SGLang internals and was verified against "
            f"{', '.join(SUPPORTED_SGLANG_VERSIONS)}, but found {installed}. "
            "Re-check Scheduler.__init__, the event loops, and "
            "execute_task_in_model_worker before allowing this version."
        )


def _load_sglang_plugins_if_available() -> bool:
    """Load SGLang runtime plugins when supported by the installed version.

    ``sglang.srt.plugins`` was added after the 0.5.10 runtime currently pinned
    by AReaL.  AWEX does not depend on that registry because it injects its
    scheduler entry point directly through ``launch_server``.  Treat the
    registry as optional so the same entry module works with both APIs.
    """
    try:
        plugins = importlib.import_module("sglang.srt.plugins")
    except ModuleNotFoundError as exc:
        if exc.name != "sglang.srt.plugins":
            raise
        logger.info(
            "[AWEX] SGLang plugin registry is unavailable; using the "
            "launch_server scheduler hook"
        )
        return False

    load_plugins = getattr(plugins, "load_plugins", None)
    if not callable(load_plugins):
        logger.info(
            "[AWEX] SGLang plugin registry has no load_plugins entry point; "
            "using the launch_server scheduler hook"
        )
        return False

    load_plugins()
    return True


def _resolve_transfer_rank(
    *,
    infer_world_size: int,
    gpu_id: int,
    node_id: int,
    nnodes: int,
    instance_world_size: int,
) -> int:
    """Resolve the inference rank in AWEX's global transfer world.

    A one-GPU SGLang server can use the colocated actor's inherited global
    rank when CUDA device isolation remaps its only GPU to device zero. For a
    multi-GPU server, every TP/PP scheduler inherits the same environment, so
    its scheduler-local physical GPU identity must be used instead.
    """
    explicit_rank = os.environ.get("AWEX_TRANSFER_RANK")
    if explicit_rank is not None:
        transfer_rank = int(explicit_rank)
    else:
        env_rank = os.environ.get("RANK")
        env_world_size = os.environ.get("WORLD_SIZE")
        if (
            instance_world_size == 1
            and env_rank is not None
            and env_world_size is not None
            and int(env_world_size) == infer_world_size
        ):
            transfer_rank = int(env_rank)
        else:
            n_gpus_per_node = max(1, infer_world_size // nnodes)
            transfer_rank = node_id * n_gpus_per_node + gpu_id

    if not 0 <= transfer_rank < infer_world_size:
        raise ValueError(
            "AWEX transfer rank must be in "
            f"[0, {infer_world_size}), got {transfer_rank}"
        )
    return transfer_rank


def _writer_version_key(ip_address: str, physical_gpu_id: int) -> str:
    return f"awex_writer_version_{ip_address}_{physical_gpu_id}"


def _try_get_writer_version(
    meta_server_client: Any,
    key: str,
    timeout_s: float,
) -> int | None:
    """Return the writer's current version if published, otherwise None."""

    try:
        wait_key = getattr(meta_server_client, "wait_key", None)
        if callable(wait_key):
            wait_key(key, timeout=timeout_s)
        return int(meta_server_client.get_object(key, timeout=timeout_s))
    except Exception:
        return None


class AwexSchedulerPlugin:
    """Binds awex weight-receive to a SGLang Scheduler instance.

    Architecture: background thread handles MetaServer I/O (CPU only),
    scheduler main loop handles CUDA weight copy (via process_awex_queue).
    """

    def __init__(self, scheduler: Any) -> None:
        self._scheduler = scheduler
        self._receiver = None
        self._bg_thread: threading.Thread | None = None
        self._weight_queue: queue.Queue = queue.Queue()
        self._version = 0
        self._paused_poll_interval_s = max(
            0.0, get_float_env_var("AWEX_PAUSED_POLL_INTERVAL_S", 0.01)
        )
        self._process_queue_when_idle = get_bool_env_var(
            "AREAL_AWEX_PROCESS_QUEUE_WHEN_IDLE", "true"
        )
        # Idle-poll throttle in *loop iterations*, not wall-clock time: TP
        # ranks run the scheduler loop in lockstep, so a loop-count gate is
        # deterministic across ranks (a time-based gate deadlocks, see
        # _maybe_process_awex_queue_when_idle).
        self._idle_poll_loops = max(1, get_int_env_var("AWEX_IDLE_POLL_LOOPS", 64))

    @staticmethod
    def _int_attr(scheduler: Any, name: str, default: int) -> int:
        for obj in (
            scheduler,
            getattr(scheduler, "ps", None),
            getattr(scheduler, "server_args", None),
        ):
            if obj is None or not hasattr(obj, name):
                continue
            value = getattr(obj, name)
            if value is not None:
                return int(value)
        return default

    @staticmethod
    def _callable(scheduler: Any, name: str) -> Callable:
        for obj in (scheduler, getattr(scheduler, "weight_updater", None)):
            if obj is None:
                continue
            method = getattr(obj, name, None)
            if callable(method):
                return method
        raise AttributeError(f"Scheduler has no callable {name!r}")

    def _logical_gpu_id(self) -> int:
        return self._int_attr(self._scheduler, "gpu_id", 0)

    def _physical_gpu_id(self) -> int:
        """Return the node-local physical GPU id used by AWEX keys."""
        logical_gpu_id = self._logical_gpu_id()
        visible_devices = [
            item.strip()
            for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if item.strip()
        ]
        if visible_devices:
            if logical_gpu_id >= len(visible_devices):
                raise ValueError(
                    f"SGLang gpu_id={logical_gpu_id} is outside "
                    f"CUDA_VISIBLE_DEVICES={visible_devices!r}"
                )
            physical_gpu_id = visible_devices[logical_gpu_id]
            if not physical_gpu_id.isdigit():
                raise ValueError(
                    "AWEX colocate requires numeric CUDA_VISIBLE_DEVICES entries "
                    "so MetaServer keys can use node-local physical GPU ids; "
                    f"got {physical_gpu_id!r}"
                )
            return int(physical_gpu_id)
        return logical_gpu_id

    def _instance_world_size(self) -> int:
        """Return scheduler GPU processes in one SGLang server."""
        return self._int_attr(self._scheduler, "tp_size", 1) * self._int_attr(
            self._scheduler, "pp_size", 1
        )

    def bind(self) -> None:
        methods = [
            "awex_init_receiver",
            "awex_receive_weights",
            "awex_release_memory",
            "awex_resume_memory",
            "awex_get_weight_metadata",
            "awex_get_parallelism",
            "process_awex_queue",
        ]
        for name in methods:
            setattr(self._scheduler, name, getattr(self, name))
        self._patch_memory_transitions()
        logger.info(
            f"[AWEX] AwexSchedulerPlugin bound {len(methods)} methods to scheduler",
        )

        meta_server_addr = os.environ.get("AWEX_META_SERVER_ADDR")
        if meta_server_addr:
            self._start_background_worker(meta_server_addr)
            self._patch_event_loop()

    def _require_receiver(self):
        if self._receiver is None:
            from areal.engine.awex.colocate_reader import AwexColocateReader

            self._receiver = AwexColocateReader(self._scheduler)
        return self._receiver

    def _patch_memory_transitions(self) -> None:
        """Make AWEX release/resume requests idempotent across retries."""
        scheduler = self._scheduler
        if getattr(scheduler, "_areal_awex_memory_transitions_patched", False):
            return
        original_release = getattr(scheduler, "release_memory_occupation", None)
        original_resume = getattr(scheduler, "resume_memory_occupation", None)
        if original_release is None or original_resume is None:
            return

        def _filtered_request(request: Any, *, release: bool) -> Any | None:
            tags = getattr(request, "tags", None)
            offload_tags = getattr(scheduler, "offload_tags", None)
            if tags is None or offload_tags is None:
                return request
            effective_tags = [
                tag
                for tag in tags
                if (tag not in offload_tags if release else tag in offload_tags)
            ]
            if not effective_tags:
                logger.info(
                    "[AWEX] skipping duplicate %s_memory_occupation(tags=%s)",
                    "release" if release else "resume",
                    tags,
                )
                return None
            if effective_tags == list(tags):
                return request
            filtered = copy(request)
            filtered.tags = effective_tags
            return filtered

        def _release(request: Any, *args: Any, **kwargs: Any) -> Any:
            filtered = _filtered_request(request, release=True)
            if filtered is None:
                return None
            return original_release(filtered, *args, **kwargs)

        def _resume(request: Any, *args: Any, **kwargs: Any) -> Any:
            filtered = _filtered_request(request, release=False)
            if filtered is None:
                return None
            return original_resume(filtered, *args, **kwargs)

        scheduler.release_memory_occupation = _release
        scheduler.resume_memory_occupation = _resume
        scheduler._areal_awex_memory_transitions_patched = True

    def awex_init_receiver(self, **kwargs: Any) -> None:
        self._require_receiver().initialize(**kwargs)

    def awex_receive_weights(self, version: int = 0) -> None:
        self._require_receiver().update_weights(version)

    def awex_release_memory(self, tags: list[str] | None = None) -> None:
        self._require_receiver().release_memory(tags)

    def awex_resume_memory(self, tags: list[str] | None = None) -> None:
        self._require_receiver().resume_memory(tags)

    def awex_get_weight_metadata(self) -> list:
        return self._require_receiver().get_weight_metadata()

    def awex_get_parallelism(self) -> dict:
        return self._require_receiver().get_parallelism()

    # ── Main loop hook: process queued weight updates ─────────────────

    def process_awex_queue(self, extra_ready: bool = True) -> None:
        """Called from scheduler main loop. Processes pending weight updates.

        This is a TP-collective operation: ALL TP ranks must call it together
        (since it's called between recv_requests() calls which use broadcast_pyobj).

        Uses all_reduce(MIN) to check if all TP ranks are ready to process a
        pending update. ``extra_ready`` lets callers fold per-rank conditions
        (e.g. idle state) into the collective decision instead of returning
        early, which would desynchronize the ranks. Only proceeds when ALL
        ranks are ready, preventing the deadlock where one rank blocks in CUDA
        ops while others wait in broadcast_pyobj.

        We act as the awex *driver* layer for the queued colocate update. The
        collect-IPC + StreamBatch transport + writer handshake is delegated to the
        awex-native worker reader (``AwexColocateReader.update_weights`` ->
        ``NCCLWorkerWeightsReader``). We only own the driver-equivalent steps
        around it:
          1. Wait for all_training_offloaded_weights (= driver _pre_update_weights)
          2. resume_memory_occupation(weights) — re-allocate infer weight buffers
          3. reader.update_weights(version) — awex worker reader does the rest:
             collect IPC + StreamBatch transport + put weights_update_finished
             + barrier + get_then_delete write_finished + flush_cache
          4. signal_finished_weights_update (= driver _resume_kvcache)
        """
        import torch
        import torch.distributed

        tp_cpu_group = self._scheduler.tp_cpu_group
        tp_size = self._int_attr(self._scheduler, "tp_size", 1)

        has_item = 1 if (extra_ready and not self._weight_queue.empty()) else 0

        if tp_size > 1:
            has_item_tensor = torch.tensor([has_item], dtype=torch.int32)
            torch.distributed.all_reduce(
                has_item_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=tp_cpu_group,
            )
            all_ready = has_item_tensor.item() == 1
        else:
            all_ready = has_item == 1

        if not all_ready:
            return

        item = self._weight_queue.get_nowait()
        version = item["version"]
        gpu_id = getattr(self._scheduler, "gpu_id", "?")
        logger.info(
            f"[AWEX] main loop: processing weight update v{version} (gpu_id={gpu_id})",
        )

        from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqInput

        receiver = self._require_receiver()

        # Step 1: Wait for writer to offload its model weights first (= awex driver
        # _pre_update_weights). Ensures no 2x model weights on GPU simultaneously.
        # The background thread already gated on this, so this returns immediately;
        # kept for driver-equivalent clarity.
        logger.info(
            f"[AWEX] main loop: waiting for all_training_offloaded_weights (gpu_id={gpu_id})",
        )
        receiver.wait_for_training_offloaded(version)
        logger.info(
            f"[AWEX] main loop: writer offloaded weights confirmed (gpu_id={gpu_id})",
        )

        # Step 2: Resume weight memory (memory_saver re-allocates buffers).
        resume_req = ResumeMemoryOccupationReqInput(tags=["weights"])
        resume_memory_occupation = self._callable(
            self._scheduler, "resume_memory_occupation"
        )
        resume_memory_occupation(resume_req)
        logger.info(
            f"[AWEX] main loop: resumed weight memory for v{version} (gpu_id={gpu_id})",
        )

        # Step 3: Delegate the whole collect-IPC + StreamBatch transport + writer
        # handshake (put weights_update_finished + barrier + get_then_delete
        # write_finished + flush_cache) to the awex-native worker reader.
        try:
            receiver.update_weights(version)
            logger.info(
                f"[AWEX] main loop: weight update done for v{version} (gpu_id={gpu_id})",
            )
        except Exception:
            logger.exception(
                "AWEX main loop failed to update weights v%s on gpu_id=%s",
                version,
                gpu_id,
            )
            raise

        # Step 4: Signal that this infer engine finished weight update, so the
        # writer can resume kv_cache (= awex driver _resume_kvcache).
        receiver.signal_finished_weights_update()
        self._version = version

    # ── Patch scheduler event loop to call process_awex_queue ─────────

    def _patch_event_loop(self) -> None:
        """Inject process_awex_queue into scheduler's event loops.

        SGLang uses event_loop_overlap by default. Patch both for safety.
        Weight updates process when engine is paused (no ongoing inference).
        """
        scheduler = self._scheduler
        plugin = self

        decode_stats_name = next(
            (
                name
                for name in ("log_decode_stats", "report_decode_stats")
                if callable(getattr(scheduler, name, None))
            ),
            None,
        )
        _orig_decode_stats = (
            getattr(scheduler, decode_stats_name) if decode_stats_name else None
        )
        every_iteration_name = "log_decode_stats_every_iteration"
        _orig_decode_stats_every_iteration = getattr(
            scheduler, every_iteration_name, None
        )
        has_decode_stats = callable(_orig_decode_stats)
        has_decode_iter_stats = callable(_orig_decode_stats_every_iteration)

        def _tracked_decode_stats(*args, **kwargs):
            scheduler._areal_awex_last_decode_stats_ct = getattr(
                scheduler, "forward_ct_decode", None
            )
            return _orig_decode_stats(*args, **kwargs)

        def _tracked_decode_stats_every_iteration(*args, **kwargs):
            scheduler._areal_awex_last_decode_stats_every_iter_ct = getattr(
                scheduler, "forward_ct_decode", None
            )
            return _orig_decode_stats_every_iteration(*args, **kwargs)

        if has_decode_stats:
            setattr(scheduler, decode_stats_name, _tracked_decode_stats)
        else:
            logger.info(
                "[AWEX] Scheduler has no decode stats method; "
                "skipping decode metrics restore hook",
            )
        if has_decode_iter_stats:
            setattr(
                scheduler,
                every_iteration_name,
                _tracked_decode_stats_every_iteration,
            )
        else:
            logger.info(
                "[AWEX] Scheduler has no log_decode_stats_every_iteration; "
                "skipping per-iteration decode metrics restore hook",
            )

        def _maybe_restore_decode_metrics(stage, batch, result):
            if os.environ.get("AREAL_AWEX_FORCE_SGLANG_METRICS", "1") != "1":
                return
            if stage != "after_process_batch_result" or batch is None:
                return
            mode = getattr(getattr(batch, "forward_mode", None), "name", None)
            if mode != "DECODE":
                return
            if not getattr(scheduler, "current_scheduler_metrics_enabled", False):
                return

            current_ct = getattr(scheduler, "forward_ct_decode", None)
            interval = (
                getattr(
                    getattr(scheduler, "server_args", None), "decode_log_interval", 1
                )
                or 1
            )
            should_log_decode = current_ct is not None and current_ct % interval == 0

            if (
                has_decode_stats
                and callable(getattr(scheduler, decode_stats_name, None))
                and should_log_decode
                and getattr(scheduler, "_areal_awex_last_decode_stats_ct", None)
                != current_ct
            ):
                can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
                logger.debug(
                    f"[AWEX-METRICS] restoring native {decode_stats_name} "
                    f"gpu_id={getattr(scheduler, 'gpu_id', '?')} "
                    f"forward_ct_decode={current_ct}",
                )
                decode_stats_kwargs = {"running_batch": batch}
                if decode_stats_name == "report_decode_stats":
                    decode_stats_kwargs["num_accepted_tokens"] = getattr(
                        result, "num_accepted_tokens", 0
                    )
                getattr(scheduler, decode_stats_name)(
                    can_run_cuda_graph, **decode_stats_kwargs
                )

            if (
                has_decode_iter_stats
                and callable(
                    getattr(scheduler, "log_decode_stats_every_iteration", None)
                )
                and getattr(
                    scheduler, "_areal_awex_last_decode_stats_every_iter_ct", None
                )
                != current_ct
            ):
                getattr(scheduler, every_iteration_name)(
                    batch,
                    num_accepted_tokens=getattr(result, "num_accepted_tokens", 0),
                )

        def _recv_requests():
            if hasattr(scheduler, "recv_requests"):
                return scheduler.recv_requests()
            return scheduler.request_receiver.recv_requests()

        def _on_idle():
            if hasattr(scheduler, "self_check_during_idle"):
                scheduler.self_check_during_idle()
            else:
                scheduler.on_idle()

        def _is_idle_for_awex_update() -> bool:
            is_fully_idle = getattr(scheduler, "is_fully_idle", None)
            if callable(is_fully_idle):
                try:
                    return bool(is_fully_idle())
                except TypeError:
                    return bool(is_fully_idle(for_health_check=False))

            for attr in ("cur_batch", "last_batch"):
                if getattr(scheduler, attr, None) is not None:
                    return False

            result_queue = getattr(scheduler, "result_queue", None)
            if result_queue is not None and len(result_queue) > 0:
                return False

            running_batch = getattr(scheduler, "running_batch", None)
            if running_batch is not None:
                is_empty = getattr(running_batch, "is_empty", None)
                if callable(is_empty) and not is_empty():
                    return False

            return True

        def _maybe_process_awex_queue_when_idle(loop_count: int) -> None:
            if not plugin._process_queue_when_idle:
                return
            # DEADLOCK WARNING: everything gating the all_reduce inside
            # process_awex_queue() MUST be deterministic and identical across
            # TP ranks. Loop iterations are lockstep (every iteration goes
            # through the recv_requests broadcast), so a loop-count throttle
            # is safe. A wall-clock throttle (time.monotonic) is NOT: ranks
            # hit the window at different times, some skip the all_reduce
            # while others enter it, and the next recv_requests broadcast
            # cross-deadlocks against the pending all_reduce (observed as
            # TP0 stuck in broadcast_pyobj vs TP1-7 stuck in all_reduce).
            if loop_count % plugin._idle_poll_loops != 0:
                return

            tp_size = self._int_attr(scheduler, "tp_size", 1)
            is_idle = _is_idle_for_awex_update()
            if tp_size == 1:
                if is_idle and not plugin._weight_queue.empty():
                    plugin.process_awex_queue()
                return

            # Rank-local idle state is folded into the collective vote instead
            # of gating it, so all ranks always enter the all_reduce together.
            plugin.process_awex_queue(extra_ready=is_idle)

        # Patch event_loop_overlap (the one actually used by SGLang)
        _orig_overlap = scheduler.event_loop_overlap

        def _patched_overlap():
            """Patched overlap loop that checks awex queue when paused."""
            from collections import deque

            scheduler.result_queue = deque()
            _loop_count = 0
            _paused_reported = False

            def pop_and_process():
                tmp_batch, tmp_result = scheduler.result_queue.popleft()
                scheduler.process_batch_result(tmp_batch, tmp_result)
                _maybe_restore_decode_metrics(
                    "after_process_batch_result", tmp_batch, tmp_result
                )

            logger.info(
                f"[AWEX] _patched_overlap STARTING (gpu_id={getattr(scheduler, 'gpu_id', '?')})",
            )

            while True:
                recv_reqs = _recv_requests()
                if recv_reqs:
                    req_types = [type(r).__name__ for r in recv_reqs]
                    has_control = any(
                        t
                        not in (
                            "TokenizedGenerateReqInput",
                            "TokenizedEmbeddingReqInput",
                        )
                        for t in req_types
                    )
                    if has_control:
                        logger.info(
                            f"[AWEX] loop gpu_id={getattr(scheduler, 'gpu_id', '?')}: "
                            f"recv {len(recv_reqs)} reqs, types={req_types[:5]}, "
                            f"_engine_paused={scheduler._engine_paused}, loop={_loop_count}",
                        )

                was_paused = scheduler._engine_paused
                scheduler.process_input_requests(recv_reqs)
                if scheduler._engine_paused != was_paused:
                    logger.info(
                        f"[AWEX] _engine_paused CHANGED: {was_paused} → {scheduler._engine_paused} "
                        f"(gpu_id={getattr(scheduler, 'gpu_id', '?')}, loop={_loop_count})",
                    )

                if scheduler._engine_paused:
                    if not _paused_reported:
                        logger.info(
                            f"[AWEX] _patched_overlap: _engine_paused=True detected! "
                            f"(gpu_id={getattr(scheduler, 'gpu_id', '?')}, loop_count={_loop_count})",
                        )
                        _paused_reported = True
                    plugin.process_awex_queue()
                    time.sleep(plugin._paused_poll_interval_s)
                    continue

                _loop_count += 1
                batch = scheduler.get_next_batch_to_run()
                scheduler.cur_batch = batch
                disable_overlap_for_batch = scheduler.is_disable_overlap_for_batch(
                    batch
                )

                if disable_overlap_for_batch:
                    pop_and_process()

                if batch:
                    batch_result = scheduler.run_batch(batch)
                    scheduler.result_queue.append((batch.copy(), batch_result))
                else:
                    batch_result = None

                if scheduler.last_batch:
                    if not disable_overlap_for_batch:
                        pop_and_process()
                elif batch is None:
                    _on_idle()

                _maybe_process_awex_queue_when_idle(_loop_count)

                if scheduler.is_generation:
                    scheduler.launch_batch_sample_if_needed(batch_result)

                scheduler.last_batch = batch

        scheduler.event_loop_overlap = _patched_overlap

        # Also patch event_loop_normal as fallback
        _orig_normal = scheduler.event_loop_normal

        def _patched_normal():
            logger.info(
                f"[AWEX] _patched_normal STARTING (gpu_id={getattr(scheduler, 'gpu_id', '?')})",
            )
            _loop_count = 0
            while True:
                recv_reqs = _recv_requests()
                scheduler.process_input_requests(recv_reqs)
                if scheduler._engine_paused:
                    plugin.process_awex_queue()
                    time.sleep(plugin._paused_poll_interval_s)
                    continue
                _loop_count += 1
                batch = scheduler.get_next_batch_to_run()
                scheduler.cur_batch = batch
                if batch:
                    result = scheduler.run_batch(batch)
                    scheduler.process_batch_result(batch, result)
                    _maybe_restore_decode_metrics(
                        "after_process_batch_result", batch, result
                    )
                else:
                    _on_idle()
                _maybe_process_awex_queue_when_idle(_loop_count)
                scheduler.last_batch = batch

        scheduler.event_loop_normal = _patched_normal
        logger.info(
            "[AWEX] Patched event_loop_overlap + event_loop_normal with awex queue",
        )

    # ── Background thread: MetaServer I/O only (no CUDA ops) ─────────

    def _start_background_worker(self, meta_server_addr: str) -> None:
        self._bg_thread = threading.Thread(
            target=self._background_worker,
            args=(meta_server_addr,),
            daemon=True,
        )
        self._bg_thread.start()
        gpu_id = self._logical_gpu_id()
        physical_gpu_id = self._physical_gpu_id()
        logger.info(
            f"[AWEX] Started background worker thread "
            f"(gpu_id={gpu_id}, physical_gpu_id={physical_gpu_id}, "
            f"meta_server={meta_server_addr})",
        )

    def _background_worker(self, meta_server_addr: str) -> None:
        """Initialize the reader, then gate weight-update triggers to the main loop.

        This thread does NOT perform any CUDA memory writes. It only:
        1. Connects to MetaServer and initializes the awex worker reader
        2. Blocks on the per-version writer-offload signal (a set-size wait)
        3. Enqueues a version marker so the TP-collective main-loop gate fires
           (the awex worker reader collects the IPC handles itself inside
           update_weights, so no large payload is prefetched here)
        """
        import torch

        gpu_id = self._logical_gpu_id()
        physical_gpu_id = self._physical_gpu_id()
        torch.cuda.set_device(gpu_id)
        logger.info(
            f"[AWEX] background worker: set CUDA device to {gpu_id} "
            f"(physical_gpu_id={physical_gpu_id})",
        )

        try:
            self._init_receiver_from_meta_server(meta_server_addr)
        except Exception:
            logger.exception("AWEX background worker initialization failed")
            return

        logger.info(
            "[AWEX] background worker: initialization complete, entering fetch loop",
        )
        # Recover can resume weight-transfer versions from the checkpoint step
        # instead of v1, so sync the first version from the writer.
        from awex.meta.meta_server import MetaServerClient as _MSC
        from awex.util.common import get_ip_address as _get_ip

        _host, _port = meta_server_addr.rsplit(":", 1)
        _ver_client = _MSC(_host, int(_port))
        _ver_key = _writer_version_key(_get_ip(), physical_gpu_id)
        writer_version_poll_timeout_s = max(
            0.1,
            get_float_env_var("AWEX_WRITER_VERSION_POLL_TIMEOUT_S", 5.0),
        )
        writer_version_log_interval_s = max(
            writer_version_poll_timeout_s,
            get_float_env_var("AWEX_WRITER_VERSION_LOG_INTERVAL_S", 60.0),
        )
        last_writer_wait_log_s = 0.0
        version = None
        while version is None:
            version = _try_get_writer_version(
                _ver_client,
                _ver_key,
                timeout_s=writer_version_poll_timeout_s,
            )
            if version is not None:
                break

            now = time.monotonic()
            if now - last_writer_wait_log_s >= writer_version_log_interval_s:
                logger.info(
                    "[AWEX] background worker: waiting for first writer version "
                    "key %s; initial RL rollout can run before actor.update_weights",
                    _ver_key,
                )
                last_writer_wait_log_s = now
            time.sleep(min(1.0, writer_version_poll_timeout_s))

        logger.info(
            f"[AWEX] background worker: writer stream starts at v{version}",
        )
        retries = 0
        # Slow online environments can spend tens of minutes between updates.
        # Keep the reader alive across transient wait timeouts.
        max_retries = int(os.environ.get("AWEX_READER_MAX_RETRIES", "1000"))

        while True:
            try:
                # Block on THIS version's writer-published IPC handles
                # (existence-only probe, no deserialization). This is the
                # per-version trigger: the writer only publishes v+1's key in the
                # next training cycle, so the background thread cannot fire early
                # off a stale unversioned set and dead-lock the main loop. See
                # AwexColocateReader.wait_for_weights_ready for the full rationale.
                logger.info(
                    f"[AWEX] background worker: waiting for writer weights v{version}",
                )
                receiver = self._require_receiver()
                receiver.wait_for_weights_ready(version)
                logger.info(
                    f"[AWEX] background worker: writer published v{version}, "
                    f"queuing for main loop",
                )

                # Queue a version marker for the main loop (no CUDA ops here).
                self._weight_queue.put({"version": version})

                # Wait for main loop to finish processing before gating the next.
                while self._version < version:
                    time.sleep(0.1)

                version += 1
                retries = 0
            except Exception as e:
                retries += 1
                logger.exception(
                    "AWEX background worker failed while waiting for writer "
                    "weights v%s (attempt %s/%s): %s",
                    version,
                    retries,
                    max_retries,
                    e,
                )
                if retries >= max_retries:
                    logger.info(
                        f"[AWEX] background worker: giving up after {max_retries} failures",
                    )
                    break
                time.sleep(min(2**retries, 30))

    def _init_receiver_from_meta_server(self, meta_server_addr: str):
        """Connect to MetaServer, get train info, initialize colocate receiver."""
        from awex.meta.meta_server import MetaServerClient

        host, port = meta_server_addr.rsplit(":", 1)

        client = None
        for attempt in range(60):
            try:
                client = MetaServerClient(host, int(port))
                break
            except Exception:
                if attempt % 10 == 0:
                    logger.info(
                        f"[AWEX] background worker: MetaServer not ready, retrying... "
                        f"(attempt {attempt + 1}, addr={meta_server_addr})",
                    )
                time.sleep(5)
        if client is None:
            raise RuntimeError(
                f"Failed to connect to MetaServer at {meta_server_addr} after 60 attempts"
            )

        logger.info(
            f"[AWEX] background worker: connected to MetaServer at {meta_server_addr}",
        )

        receiver = self._require_receiver()

        # `physical_gpu_id` is node-local. Multi-node colocate needs a globally
        # unique transfer rank that stays physically paired with the training
        # process. SGLang may run with logical gpu_id=0 under CUDA_VISIBLE_DEVICES
        # isolation, so do not use scheduler.gpu_id for AWEX keys.
        gpu_id = self._logical_gpu_id()
        physical_gpu_id = self._physical_gpu_id()
        node_id = int(os.environ.get("SLURM_NODEID", "0"))
        nnodes = int(os.environ.get("SLURM_NNODES", "1"))

        logger.info(
            f"[AWEX] background worker: waiting for awex_train_info "
            f"(gpu_id={gpu_id}, physical_gpu_id={physical_gpu_id}, "
            f"node_id={node_id}, nnodes={nnodes})",
        )
        # The driver publishes awex_train_info only after rollout init finishes,
        # so large models need the same timeout budget as the weight path.
        from areal.engine.awex.colocate_writer import awex_colocate_timeout_s

        train_info = client.get_object(
            "awex_train_info",
            timeout=awex_colocate_timeout_s(),
        )
        train_world_size = train_info["train_world_size"]
        # In colocate mode train and infer share the same N physical GPUs, so the
        # global infer NCCL world spans the same N ranks (numerically == train
        # world). This is a *physical* coincidence (same GPUs), NOT a requirement
        # that train/infer parallel topologies match: the infer side decomposes
        # into num_infer_engines DP replicas inside receiver.initialize().
        infer_world_size = train_world_size

        n_gpus_per_node = max(1, infer_world_size // nnodes)
        transfer_rank = _resolve_transfer_rank(
            infer_world_size=infer_world_size,
            gpu_id=gpu_id,
            node_id=node_id,
            nnodes=nnodes,
            instance_world_size=self._instance_world_size(),
        )

        logger.info(
            f"[AWEX] background worker: got train_world_size={train_world_size}, "
            f"infer_world_size={infer_world_size}, n_gpus_per_node={n_gpus_per_node}, "
            f"transfer_rank={transfer_rank}, physical_gpu_id={physical_gpu_id}",
        )

        receiver.initialize(
            meta_server_addr=meta_server_addr,
            transfer_rank=transfer_rank,
            infer_world_size=infer_world_size,
            train_world_size=train_world_size,
            local_gpu_id=physical_gpu_id,
        )
        logger.info(
            f"[AWEX] background worker: receiver initialized "
            f"(transfer_rank={transfer_rank}, infer_world_size={infer_world_size})",
        )


@dataclass
class ModelWorkerTask:
    """Task for execute_task_in_model_worker (PR #13595 backport for SGLang 0.5.9)."""

    task_func: Callable
    kwargs: dict = field(default_factory=dict)


def register_awex_plugin() -> None:
    """Patch Scheduler.__init__ to inject awex plugin after construction.

    Must be called INSIDE the scheduler child process (not the parent),
    because SGLang spawns scheduler processes via mp.Process with "spawn"
    start method, which doesn't inherit parent-process monkey-patches.
    """
    assert_supported_sglang_version()
    from sglang.srt.managers.scheduler import Scheduler

    _orig_init = Scheduler.__init__

    def _patched_init(self, *args, **kwargs):
        logger.info(
            "[AWEX] Scheduler.__init__ entering "
            "(pid=%s, CUDA_VISIBLE_DEVICES=%s, AWEX_META_SERVER_ADDR=%s)",
            os.getpid(),
            os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            os.environ.get("AWEX_META_SERVER_ADDR", ""),
        )
        try:
            _orig_init(self, *args, **kwargs)
        except BaseException:
            logger.exception("[AWEX] Scheduler.__init__ original init failed")
            raise
        plugin = AwexSchedulerPlugin(self)
        logger.info(
            "[AWEX] Scheduler.__init__ original init complete "
            "(gpu_id=%s, tp_rank=%s, tp_size=%s)",
            getattr(self, "gpu_id", "?"),
            getattr(self, "tp_rank", "?"),
            plugin._int_attr(self, "tp_size", 1),
        )
        try:
            plugin.bind()
            _patch_execute_task_in_model_worker(self, plugin)
        except BaseException:
            logger.exception("[AWEX] Scheduler.__init__ AWEX bind failed")
            raise
        logger.info("[AWEX] Scheduler.__init__ AWEX bind complete")

    Scheduler.__init__ = _patched_init
    logger.info("[AWEX] Patched Scheduler.__init__ with awex plugin")


def _patch_execute_task_in_model_worker(
    scheduler: Any, plugin: AwexSchedulerPlugin
) -> None:
    """Add execute_task_in_model_worker to Scheduler (backport from PR #13595)."""

    if callable(getattr(scheduler, "execute_task_in_model_worker", None)):
        logger.info(
            "[AWEX] Scheduler already has native execute_task_in_model_worker; "
            "skipping legacy backport",
        )
        return

    task_cls = _get_model_worker_task_cls()

    def execute_task_in_model_worker(task_spec):
        model_context = dict(
            tp_rank=plugin._int_attr(scheduler, "tp_rank", 0),
            tp_size=plugin._int_attr(scheduler, "tp_size", 1),
            server_args=scheduler.server_args,
            scheduler=scheduler,
        )
        kwargs = dict(task_spec.kwargs)
        kwargs["model_context"] = model_context
        kwargs["model"] = scheduler.tp_worker.model_runner.model
        kwargs["model_runner"] = scheduler.tp_worker.model_runner
        return task_spec.task_func(**kwargs)

    scheduler.execute_task_in_model_worker = execute_task_in_model_worker

    if hasattr(scheduler, "_request_dispatcher"):
        scheduler._request_dispatcher._mapping[task_cls] = execute_task_in_model_worker
        logger.info("[AWEX] Registered execute_task_in_model_worker in dispatcher")


def _get_model_worker_task_cls():
    try:
        from sglang.srt.managers.io_struct import ModelWorkerTask as SGLangTask

        return SGLangTask
    except (ImportError, AttributeError):
        return ModelWorkerTask


def awex_run_scheduler_process(*args, **kwargs):
    """Scheduler process entry point that registers awex plugin.

    Memory management (pause/resume weights, KV cache, CUDA graphs) is handled
    at runtime by AWEX's release_memory/resume_memory, matching HybridEngine.
    No init-time memory patching needed.
    """
    import os

    meta_addr = os.environ.get("AWEX_META_SERVER_ADDR")
    logger.info(
        "[AWEX] awex_run_scheduler_process starting "
        "(pid=%s, meta_server=%s, CUDA_VISIBLE_DEVICES=%s)",
        os.getpid(),
        meta_addr or "",
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )
    if meta_addr:
        register_awex_plugin()
    else:
        logger.info(
            "[AWEX] No AWEX_META_SERVER_ADDR, skipping plugin registration",
        )
    from sglang.srt.managers.scheduler import run_scheduler_process

    try:
        return run_scheduler_process(*args, **kwargs)
    except BaseException:
        logger.exception("[AWEX] run_scheduler_process failed")
        raise


if __name__ == "__main__":
    import os
    import sys

    logger.info(
        "[AWEX] awex_sglang_plugin __main__ starting "
        "(pid=%s, CUDA_VISIBLE_DEVICES=%s, AWEX_META_SERVER_ADDR=%s)",
        os.getpid(),
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        os.environ.get("AWEX_META_SERVER_ADDR", ""),
    )

    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    _load_sglang_plugins_if_available()
    server_args = prepare_server_args(sys.argv[1:])
    # In AWEX colocated mode the scheduler may not be able to serve requests
    # until the first weight sync (actor holds GPUs / writer version not yet
    # published). SGLang's warmup request would then hang for 600s
    # (_execute_server_warmup default timeout) and kill_process_tree() the
    # whole server. Skip the warmup: _wait_and_warmup marks the server Up
    # directly when skip_server_warmup is set.
    if not server_args.skip_server_warmup:
        logger.info(
            "[AWEX] forcing skip_server_warmup=True to avoid warmup-timeout "
            "suicide before the first weight sync"
        )
        server_args.skip_server_warmup = True
    try:
        launch_server(
            server_args,
            run_scheduler_process_func=awex_run_scheduler_process,
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
