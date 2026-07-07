# SPDX-License-Identifier: Apache-2.0

"""AWEX SGLang scheduler plugin for colocated weight transfer.

Patches SGLang's scheduler to inject CUDA IPC weight receiving capabilities.
When AWEX_META_SERVER_ADDR env var is set, starts a background thread that
fetches IPC handles from MetaServer (CPU I/O) and queues them for the
scheduler's main loop to process (CUDA copy on main thread).

Weight transfer flow (colocate mode):
  1. Training side: convert params → cuda_ipc_serialize → MetaServer put
  2. Background thread: MetaServer get → queue IPC data (CPU only)
  3. Scheduler main loop: release_memory → deserialize + copy → resume_memory
  4. Main loop: signal done → train side releases shared tensors

Usage:
    # Option 1: Register plugin then launch SGLang
    from areal.engine.awex_sglang_plugin import register_awex_plugin
    register_awex_plugin()

    # Option 2: Run as entry module (replaces sglang.launch_server)
    # python3 -m areal.engine.awex_sglang_plugin --model-path ...
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Any


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
        print(
            f"[AWEX] AwexSchedulerPlugin bound {len(methods)} methods to scheduler",
            flush=True,
        )

        meta_server_addr = os.environ.get("AWEX_META_SERVER_ADDR")
        if meta_server_addr:
            self._check_sglang_compat()
            self._trace_scheduler_methods()
            self._start_background_worker(meta_server_addr)
            self._patch_scheduler_hooks()

    def _trace_scheduler_methods(self) -> None:
        """Patch TypeBasedDispatcher to trace pause/release/continue methods.

        Must patch the dispatcher's _mapping directly because it captures
        bound method references at init time (before our monkey-patch).
        """
        scheduler = self._scheduler
        gpu_id = getattr(scheduler, "gpu_id", "?")

        from sglang.srt.managers.io_struct import (
            ContinueGenerationReqInput,
            PauseGenerationReqInput,
            ReleaseMemoryOccupationReqInput,
        )

        dispatcher = scheduler._request_dispatcher

        _orig_pause_gen = dispatcher._mapping[PauseGenerationReqInput]

        def _traced_pause_gen(recv_req):
            print(
                f"[AWEX-TRACE] pause_generation CALLED (gpu_id={gpu_id}, "
                f"_engine_paused_before={scheduler._engine_paused})",
                flush=True,
            )
            result = _orig_pause_gen(recv_req)
            print(
                f"[AWEX-TRACE] pause_generation DONE (gpu_id={gpu_id}, "
                f"_engine_paused_after={scheduler._engine_paused})",
                flush=True,
            )
            return result

        dispatcher._mapping[PauseGenerationReqInput] = _traced_pause_gen

        _orig_release_mem = dispatcher._mapping[ReleaseMemoryOccupationReqInput]

        def _traced_release_mem(recv_req):
            print(
                f"[AWEX-TRACE] release_memory_occupation CALLED (gpu_id={gpu_id}, "
                f"tags={getattr(recv_req, 'tags', None)}, "
                f"_engine_paused={scheduler._engine_paused})",
                flush=True,
            )
            result = _orig_release_mem(recv_req)
            print(
                f"[AWEX-TRACE] release_memory_occupation DONE (gpu_id={gpu_id})",
                flush=True,
            )
            return result

        dispatcher._mapping[ReleaseMemoryOccupationReqInput] = _traced_release_mem

        _orig_continue_gen = dispatcher._mapping[ContinueGenerationReqInput]

        def _traced_continue_gen(recv_req):
            print(
                f"[AWEX-TRACE] continue_generation CALLED (gpu_id={gpu_id}, "
                f"_engine_paused_before={scheduler._engine_paused})",
                flush=True,
            )
            result = _orig_continue_gen(recv_req)
            print(
                f"[AWEX-TRACE] continue_generation DONE (gpu_id={gpu_id}, "
                f"_engine_paused_after={scheduler._engine_paused})",
                flush=True,
            )
            return result

        dispatcher._mapping[ContinueGenerationReqInput] = _traced_continue_gen
        print(
            f"[AWEX] Traced dispatcher for pause/release/continue (gpu_id={gpu_id})",
            flush=True,
        )

    def _require_receiver(self):
        if self._receiver is None:
            from areal.engine.awex_colocate_reader import AwexColocateReader

            self._receiver = AwexColocateReader(self._scheduler)
        return self._receiver

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

    def process_awex_queue(self) -> None:
        """Called from scheduler main loop. Processes pending weight updates.

        This is a TP-collective operation: ALL TP ranks must call it together
        (it runs from the process_input_requests hook, between the event loop's
        broadcast_pyobj-driven recv_requests calls).

        Uses all_reduce(MIN) to check if all TP ranks have a pending update.
        Only proceeds when ALL ranks have queued an update, preventing the deadlock
        where one rank blocks in CUDA ops while others wait in broadcast_pyobj.

        We act as the awex *driver* layer (the community SGLang scheduler has no
        ``execute_task_in_model_worker`` driver). The collect-IPC + StreamBatch
        transport + writer handshake is delegated to the awex-native worker reader
        (``AwexColocateReader.update_weights`` -> ``NCCLWorkerWeightsReader``). We
        only own the driver-equivalent steps around it:
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
        tp_size = self._scheduler.tp_size

        has_item = 1 if not self._weight_queue.empty() else 0

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
        print(
            f"[AWEX] main loop: processing weight update v{version} (gpu_id={gpu_id})",
            flush=True,
        )

        from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqInput

        receiver = self._require_receiver()

        # Step 1: Wait for writer to offload its model weights first (= awex driver
        # _pre_update_weights). Ensures no 2x model weights on GPU simultaneously.
        # The background thread already gated on this, so this returns immediately;
        # kept for driver-equivalent clarity.
        print(
            f"[AWEX] main loop: waiting for all_training_offloaded_weights (gpu_id={gpu_id})",
            flush=True,
        )
        receiver.wait_for_training_offloaded()
        print(
            f"[AWEX] main loop: writer offloaded weights confirmed (gpu_id={gpu_id})",
            flush=True,
        )

        # Step 2: Resume weight memory (memory_saver re-allocates buffers).
        resume_req = ResumeMemoryOccupationReqInput(tags=["weights"])
        self._scheduler.resume_memory_occupation(resume_req)
        print(
            f"[AWEX] main loop: resumed weight memory for v{version} (gpu_id={gpu_id})",
            flush=True,
        )

        # Step 3: Delegate the whole collect-IPC + StreamBatch transport + writer
        # handshake (put weights_update_finished + barrier + get_then_delete
        # write_finished + flush_cache) to the awex-native worker reader.
        try:
            receiver.update_weights(version)
        except Exception:
            import traceback

            print(
                f"[AWEX] main loop: weight update FAILED v{version} (gpu_id={gpu_id})",
                flush=True,
            )
            traceback.print_exc()
            # Fail fast: a partially applied update must not serve traffic or
            # signal completion. Let the exception kill the scheduler so the
            # controller sees a dead worker instead of a silently stale engine;
            # the writer's wait then fails loudly by key name.
            raise
        print(
            f"[AWEX] main loop: weight update done for v{version} (gpu_id={gpu_id})",
            flush=True,
        )

        # Step 4: Signal that this infer engine finished weight update, so the
        # writer can resume kv_cache (= awex driver _resume_kvcache).
        receiver.signal_finished_weights_update()
        self._version = version

    # ── Scheduler hooks (no event-loop copies) ────────────────────

    def _check_sglang_compat(self) -> None:
        """Fail fast with a clear message when the scheduler surface moved.

        Developed against sglang v0.5.9; the depended symbols are checked up
        front so an incompatible sglang fails at startup instead of drifting
        silently.
        """
        import dataclasses

        import sglang
        from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput

        required = (
            "_engine_paused",
            "recv_requests",
            "process_input_requests",
            "process_batch_result",
            "release_memory_occupation",
            "resume_memory_occupation",
            "tp_cpu_group",
            "tp_size",
        )
        missing = [a for a in required if not hasattr(self._scheduler, a)]
        try:
            io_fields = {
                f.name for f in dataclasses.fields(ReleaseMemoryOccupationReqInput)
            }
            if "tags" not in io_fields:
                missing.append("ReleaseMemoryOccupationReqInput.tags")
        except TypeError:
            pass
        if missing:
            raise RuntimeError(
                f"awex plugin incompatible with sglang "
                f"{getattr(sglang, '__version__', '<unknown>')}: missing {missing} "
                f"(developed against v0.5.9)"
            )

    def _patch_scheduler_hooks(self) -> None:
        """Hook the scheduler through two narrow method wrappers.

        Both native event loops start every iteration with
        ``process_input_requests(recv_requests())`` followed by
        ``if self._engine_paused: continue``, so wrapping
        ``process_input_requests`` drains the awex queue at exactly the point
        a copied loop would — without pinning the loop bodies to one sglang
        version. ``process_batch_result`` is the shared result path of both
        loops, used to restore the native decode metrics.
        """
        scheduler = self._scheduler
        plugin = self

        _orig_process_input = scheduler.process_input_requests

        def _hooked_process_input_requests(*args, **kwargs):
            result = _orig_process_input(*args, **kwargs)
            if scheduler._engine_paused:
                plugin.process_awex_queue()
            return result

        scheduler.process_input_requests = _hooked_process_input_requests

        metrics_ok = hasattr(scheduler, "log_decode_stats") and hasattr(
            scheduler, "log_decode_stats_every_iteration"
        )
        if metrics_ok:
            _orig_log_decode_stats = scheduler.log_decode_stats
            _orig_log_decode_stats_every_iteration = (
                scheduler.log_decode_stats_every_iteration
            )

            def _tracked_log_decode_stats(*args, **kwargs):
                scheduler._areal_awex_last_decode_stats_ct = getattr(
                    scheduler, "forward_ct_decode", None
                )
                return _orig_log_decode_stats(*args, **kwargs)

            def _tracked_log_decode_stats_every_iteration(*args, **kwargs):
                scheduler._areal_awex_last_decode_stats_every_iter_ct = getattr(
                    scheduler, "forward_ct_decode", None
                )
                return _orig_log_decode_stats_every_iteration(*args, **kwargs)

            scheduler.log_decode_stats = _tracked_log_decode_stats
            scheduler.log_decode_stats_every_iteration = (
                _tracked_log_decode_stats_every_iteration
            )
        else:
            print(
                "[AWEX] decode-metrics restore disabled: scheduler lacks "
                "log_decode_stats symbols (sglang version drift)",
                flush=True,
            )

        def _maybe_restore_decode_metrics(batch, result):
            if not metrics_ok or batch is None:
                return
            if os.environ.get("AREAL_AWEX_FORCE_SGLANG_METRICS", "1") != "1":
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
                should_log_decode
                and getattr(scheduler, "_areal_awex_last_decode_stats_ct", None)
                != current_ct
            ):
                can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
                print(
                    f"[AWEX-METRICS] restoring native log_decode_stats "
                    f"gpu_id={getattr(scheduler, 'gpu_id', '?')} "
                    f"forward_ct_decode={current_ct}",
                    flush=True,
                )
                scheduler.log_decode_stats(can_run_cuda_graph, running_batch=batch)

            if (
                getattr(scheduler, "_areal_awex_last_decode_stats_every_iter_ct", None)
                != current_ct
            ):
                scheduler.log_decode_stats_every_iteration(
                    batch,
                    num_accepted_tokens=getattr(result, "num_accepted_tokens", 0),
                )

        _orig_process_batch_result = scheduler.process_batch_result

        def _hooked_process_batch_result(batch, result, *args, **kwargs):
            out = _orig_process_batch_result(batch, result, *args, **kwargs)
            _maybe_restore_decode_metrics(batch, result)
            return out

        scheduler.process_batch_result = _hooked_process_batch_result

        print(
            "[AWEX] Installed scheduler hooks: process_input_requests "
            "(paused queue drain) + process_batch_result (decode metrics)",
            flush=True,
        )

    # ── Background thread: MetaServer I/O only (no CUDA ops) ─────────

    def _start_background_worker(self, meta_server_addr: str) -> None:
        self._bg_thread = threading.Thread(
            target=self._background_worker,
            args=(meta_server_addr,),
            daemon=True,
        )
        self._bg_thread.start()
        gpu_id = int(getattr(self._scheduler, "gpu_id", -1))
        print(
            f"[AWEX] Started background worker thread "
            f"(gpu_id={gpu_id}, meta_server={meta_server_addr})",
            flush=True,
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

        gpu_id = int(getattr(self._scheduler, "gpu_id", 0))
        torch.cuda.set_device(gpu_id)
        print(f"[AWEX] background worker: set CUDA device to {gpu_id}", flush=True)

        try:
            self._init_receiver_from_meta_server(meta_server_addr)
        except Exception as e:
            import traceback

            print(f"[AWEX] background worker: initialization FAILED: {e}", flush=True)
            traceback.print_exc()
            return

        print(
            "[AWEX] background worker: initialization complete, entering fetch loop",
            flush=True,
        )
        # The writer numbers transfers by global_step+1, so a recover run
        # resumes the stream mid-way rather than at v=1; sync the starting
        # version from the writer-published key or the fetch loop deadlocks.
        from awex.meta.meta_server import MetaServerClient as _MSC
        from awex.util.common import get_ip_address as _get_ip

        _host, _port = meta_server_addr.rsplit(":", 1)
        _ver_client = _MSC(_host, int(_port))
        _ver_key = f"awex_writer_version_{_get_ip()}_{gpu_id}"
        version = int(
            _ver_client.get_object(
                _ver_key,
                timeout=int(os.environ.get("AWEX_COLOCATE_TIMEOUT_S", "1800")),
            )
        )
        print(
            f"[AWEX] background worker: writer stream starts at v{version}",
            flush=True,
        )
        retries = 0
        # The retry budget must survive slow-rollout workloads: one actor step
        # can take tens of minutes before the next version is published, while
        # each wait_for_weights_ready cycle times out in ~600 s. A small cap
        # makes this background reader give up permanently and deadlocks the
        # writer; keep it large and rely on the writer-side
        # AWEX_COLOCATE_TIMEOUT_S as the real failure bound.
        max_retries = int(os.environ.get("AWEX_READER_MAX_RETRIES", "1000"))

        while True:
            try:
                # Block on THIS version's writer-published IPC handles
                # (existence-only probe, no deserialization). This is the
                # per-version trigger: the writer only publishes v+1's key in the
                # next training cycle, so the background thread cannot fire early
                # off a stale unversioned set and dead-lock the main loop. See
                # AwexColocateReader.wait_for_weights_ready for the full rationale.
                print(
                    f"[AWEX] background worker: waiting for writer weights v{version}",
                    flush=True,
                )
                receiver = self._require_receiver()
                receiver.wait_for_weights_ready(version)
                print(
                    f"[AWEX] background worker: writer published v{version}, "
                    f"queuing for main loop",
                    flush=True,
                )

                # Queue a version marker for the main loop (no CUDA ops here).
                self._weight_queue.put({"version": version})

                # Wait for main loop to finish processing before gating the next.
                while self._version < version:
                    time.sleep(0.1)

                version += 1
                retries = 0
            except Exception as e:
                import traceback

                retries += 1
                print(
                    f"[AWEX] background worker: fetch failed at v{version} "
                    f"(attempt {retries}/{max_retries}): {e}",
                    flush=True,
                )
                traceback.print_exc()
                if retries >= max_retries:
                    print(
                        f"[AWEX] background worker: giving up after {max_retries} failures",
                        flush=True,
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
                    print(
                        f"[AWEX] background worker: MetaServer not ready, retrying... "
                        f"(attempt {attempt + 1}, addr={meta_server_addr})",
                        flush=True,
                    )
                time.sleep(5)
        if client is None:
            raise RuntimeError(
                f"Failed to connect to MetaServer at {meta_server_addr} after 60 attempts"
            )

        print(
            f"[AWEX] background worker: connected to MetaServer at {meta_server_addr}",
            flush=True,
        )

        receiver = self._require_receiver()

        # `gpu_id` is node-local (0..n_gpus_per_node-1). Multi-node colocate needs a
        # *global* transfer_rank that is unique across all infer processes and
        # physically paired with the train side: train uses
        # transfer_rank = global dist rank = SLURM_NODEID * n_gpus_per_node + local_gpu.
        # Using node-local gpu_id alone collapses every node onto ranks 0..7, so the
        # MetaServer only ever holds 8 unique infer_local_meta_* keys while
        # infer_world_size expects all N -> the rank-0 merge loop hangs forever on
        # infer_local_meta_8, and train update_weights times out waiting for
        # infer_conf.
        gpu_id = int(getattr(self._scheduler, "gpu_id", 0))
        node_id = int(os.environ.get("SLURM_NODEID", "0"))
        nnodes = int(os.environ.get("SLURM_NNODES", "1"))

        print(
            f"[AWEX] background worker: waiting for awex_train_info "
            f"(gpu_id={gpu_id}, node_id={node_id}, nnodes={nnodes})",
            flush=True,
        )
        # 1800s, not 600: the driver only publishes awex_train_info after every
        # rollout init RPC returns, and cold-cache 128k startups can exceed
        # 600s. Align with AWEX_COLOCATE_TIMEOUT_S.
        train_info = client.get_object(
            "awex_train_info",
            timeout=int(os.environ.get("AWEX_COLOCATE_TIMEOUT_S", "1800")),
        )
        train_world_size = train_info["train_world_size"]
        # In colocate mode train and infer share the same N physical GPUs, so the
        # global infer NCCL world spans the same N ranks (numerically == train
        # world). This is a *physical* coincidence (same GPUs), NOT a requirement
        # that train/infer parallel topologies match: the infer side decomposes
        # into num_infer_engines DP replicas inside receiver.initialize().
        infer_world_size = train_world_size

        n_gpus_per_node = max(1, infer_world_size // nnodes)
        transfer_rank = node_id * n_gpus_per_node + gpu_id

        print(
            f"[AWEX] background worker: got train_world_size={train_world_size}, "
            f"infer_world_size={infer_world_size}, n_gpus_per_node={n_gpus_per_node}, "
            f"transfer_rank={transfer_rank}",
            flush=True,
        )

        receiver.initialize(
            meta_server_addr=meta_server_addr,
            transfer_rank=transfer_rank,
            infer_world_size=infer_world_size,
            train_world_size=train_world_size,
            local_gpu_id=gpu_id,
        )
        print(
            f"[AWEX] background worker: receiver initialized "
            f"(transfer_rank={transfer_rank}, infer_world_size={infer_world_size})",
            flush=True,
        )


def register_awex_plugin() -> None:
    """Patch Scheduler.__init__ to inject awex plugin after construction.

    Must be called INSIDE the scheduler child process (not the parent),
    because SGLang spawns scheduler processes via mp.Process with "spawn"
    start method, which doesn't inherit parent-process monkey-patches.
    """
    from sglang.srt.managers.scheduler import Scheduler

    _orig_init = Scheduler.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        AwexSchedulerPlugin(self).bind()

    Scheduler.__init__ = _patched_init
    print("[AWEX] Patched Scheduler.__init__ with awex plugin", flush=True)


def awex_run_scheduler_process(*args, **kwargs):
    """Scheduler process entry point that registers awex plugin.

    Memory management (pause/resume weights, KV cache, CUDA graphs) is handled
    at runtime by AWEX's release_memory/resume_memory.
    No init-time memory patching needed.
    """
    import os

    meta_addr = os.environ.get("AWEX_META_SERVER_ADDR")
    if meta_addr:
        register_awex_plugin()
    else:
        print(
            "[AWEX] No AWEX_META_SERVER_ADDR, skipping plugin registration",
            flush=True,
        )
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)


if __name__ == "__main__":
    import os
    import sys

    # The actor-side env may ship expandable_segments:True (fragmentation
    # fix), and the colocated rollout worker inherits
    # the same scheduling_spec env. SGLang's memory saver (and CUDA graph
    # pools) cannot run on expandable segments, so flip it to False for
    # this process tree BEFORE any CUDA initialization.
    # Only touch the env when expandable_segments is explicitly set:
    # leaving "" untouched keeps the legacy (validated) behavior bit-exact.
    _conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" in _conf.lower():
        _tokens = [
            t.strip()
            for t in _conf.split(",")
            if t.strip() and not t.strip().lower().startswith("expandable_segments")
        ]
        _tokens.append("expandable_segments:False")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(_tokens)

    print("[AWEX] awex_sglang_plugin __main__ starting", flush=True)

    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(sys.argv[1:])
    try:
        launch_server(
            server_args,
            run_scheduler_process_func=awex_run_scheduler_process,
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
