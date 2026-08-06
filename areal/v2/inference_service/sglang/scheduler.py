# SPDX-License-Identifier: Apache-2.0
"""AwexSchedulerBridge + PPSchedulerBridge: compose weight-update methods onto SGLang Scheduler."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import torch.distributed as dist
import zmq
from sglang.srt.server_args import PortArgs, ServerArgs

from areal.infra.rpc.serialization import serialize_value

logger = logging.getLogger("AwexSchedulerBridge")

RESULT_IPC_ENV = "AREAL_AWEX_RESULT_IPC"


class AwexSchedulerBridge:
    """Compose awex weight-update capabilities onto a plain Scheduler instance.

    Lifecycle:
      1. Created after ``Scheduler.__init__()`` in :func:`areal_run_scheduler_process`
      2. :meth:`bind` attaches ``awex_*`` methods to the scheduler via ``setattr``
      3. ``handle_rpc_request`` dispatches via ``getattr(self, method)`` and finds them
      4. Methods delegate to :class:`AwexSGLangAdapter` for actual work
      5. Data-returning methods push results via ZMQ PUSH (tp_rank 0, dp_rank 0 only)

    No inheritance.  No monkey-patch.  The scheduler instance remains a plain
    ``sglang.srt.managers.scheduler.Scheduler``.
    """

    def __init__(self, scheduler: Any) -> None:
        self._scheduler = scheduler
        self._adapter: Any | None = None
        self._result_push: zmq.Socket | None = None

        result_ipc = os.environ.get(RESULT_IPC_ENV)
        # Only tp_rank==0 AND dp_rank==0 should push results to avoid
        # duplicate/corrupted messages on the single PULL socket.
        if (
            result_ipc
            and scheduler.tp_rank == 0
            and (getattr(scheduler, "dp_rank", None) is None or scheduler.dp_rank == 0)
        ):
            ctx = zmq.Context(1)
            self._result_push = ctx.socket(zmq.PUSH)
            self._result_push.connect(result_ipc)

    def bind(self) -> None:
        """Attach ``awex_*`` methods to the scheduler instance.

        After this call, ``handle_rpc_request`` can dispatch to them via
        ``getattr(scheduler, 'awex_report_weight_meta')`` etc.
        """
        methods = [
            "awex_report_weight_meta",
            "awex_report_parallelism",
            "awex_init_weights_update_group",
            "awex_execute_weight_update",
            "awex_batch_isend_irecv",
            "awex_get_parameters",
            "awex_randomize_parameters",
            "awex_init_colocate_weight_update",
            "awex_execute_colocate_weight_update",
            "awex_seed_delta_base",
            "awex_release_memory",
            "awex_resume_memory",
        ]
        for name in methods:
            setattr(self._scheduler, name, getattr(self, name))
        self._install_decode_stats_tracker()

    def _require_adapter(self) -> Any:
        if self._adapter is None:
            from areal.v2.weight_update.awex.sglang_adapter import (
                AwexSGLangAdapter,
            )

            self._adapter = AwexSGLangAdapter(self._scheduler)
        return self._adapter

    def _push_result(self, result: Any) -> None:
        if self._result_push is not None:
            self._result_push.send_pyobj(result)

    def awex_report_weight_meta(self) -> None:
        adapter = self._require_adapter()
        local_meta = adapter.get_weight_metadata()
        s = self._scheduler

        # All-gather across TP ranks so rank 0 returns aggregated metadata
        if s.tp_size > 1:
            gathered: list[list] = [[] for _ in range(s.tp_size)]
            dist.all_gather_object(gathered, local_meta, group=s.tp_cpu_group)
            all_meta: list = []
            for rank_meta in gathered:
                all_meta.extend(rank_meta)
            self._push_result(serialize_value(all_meta))
        else:
            self._push_result(serialize_value(local_meta))

    def awex_report_parallelism(self) -> None:
        from areal.utils.network import gethostip
        from areal.v2.weight_update.awex import resolve_physical_gpu_id

        result = dict(self._require_adapter().parallelism_strategy)
        result["ip"] = gethostip()
        result["device_id"] = resolve_physical_gpu_id()
        self._push_result(result)

    def awex_init_weights_update_group(self, **kwargs: Any) -> None:
        self._require_adapter().init_weight_update_group(**kwargs)

    def awex_execute_weight_update(self, version: int = 0) -> None:
        self._require_adapter().execute_weight_update(version)

    def awex_batch_isend_irecv(self, **kwargs: Any) -> None:
        self._require_adapter().batch_isend_irecv(**kwargs)

    def awex_get_parameters(
        self, save_path: str, names: list[str] | None = None
    ) -> None:
        adapter = self._require_adapter()
        if self._scheduler.tp_rank == 0:
            adapter.save_parameters(save_path, names)

    def awex_randomize_parameters(self) -> None:
        self._require_adapter().randomize_parameters()

    def awex_init_colocate_weight_update(self, **kwargs: Any) -> None:
        self._require_adapter().init_colocate_weight_update(**kwargs)

    def awex_execute_colocate_weight_update(self, version: int = 0) -> None:
        self._require_adapter().execute_colocate_weight_update(version)

    def awex_seed_delta_base(self, version: int = 0) -> None:
        self._require_adapter().seed_delta_base(version)

    def awex_release_memory(self, tags: list[str] | None = None) -> None:
        self._require_adapter().release_memory(tags)

    def awex_resume_memory(self, tags: list[str] | None = None) -> None:
        self._require_adapter().resume_memory(tags)

    def _install_decode_stats_tracker(self) -> None:
        """Keep SGLang decode-stats logging alive across pause/resume cycles.

        Colocate weight updates pause generation and release/resume memory;
        the native decode-stats logging then goes silent because its
        ``forward_ct_decode % decode_log_interval`` firing points are missed.
        Wrapping ``log_decode_stats*`` records the counter at each native log,
        and the ``process_batch_result`` wrapper re-invokes logging for decode
        batches whose firing point was skipped. Instance-attribute wraps only
        (same setattr territory as :meth:`bind`); disable with
        ``AREAL_AWEX_FORCE_SGLANG_METRICS=0``.
        """
        if os.environ.get("AREAL_AWEX_FORCE_SGLANG_METRICS", "1") != "1":
            return
        scheduler = self._scheduler
        orig_log = getattr(scheduler, "log_decode_stats", None)
        orig_log_iter = getattr(scheduler, "log_decode_stats_every_iteration", None)
        if orig_log is None or orig_log_iter is None:
            # sglang >= 0.5.10 reworked decode-stats logging and removed these
            # hooks; the tracker only restores metrics silenced by pause/resume
            # cycles, so skip it rather than block server startup.
            logger.warning(
                "sglang Scheduler.log_decode_stats(_every_iteration) not found; "
                "skipping AWEX decode-stats tracking for this sglang version"
            )
            return

        def _tracked_log_decode_stats(*args, **kwargs):
            scheduler._areal_awex_last_decode_stats_ct = getattr(
                scheduler, "forward_ct_decode", None
            )
            return orig_log(*args, **kwargs)

        def _tracked_log_decode_stats_every_iteration(*args, **kwargs):
            scheduler._areal_awex_last_decode_stats_every_iter_ct = getattr(
                scheduler, "forward_ct_decode", None
            )
            return orig_log_iter(*args, **kwargs)

        scheduler.log_decode_stats = _tracked_log_decode_stats
        scheduler.log_decode_stats_every_iteration = (
            _tracked_log_decode_stats_every_iteration
        )

        orig_process = scheduler.process_batch_result

        def _tracked_process_batch_result(batch, result, *args, **kwargs):
            out = orig_process(batch, result, *args, **kwargs)
            self._maybe_restore_decode_metrics(batch, result)
            return out

        scheduler.process_batch_result = _tracked_process_batch_result

    def _maybe_restore_decode_metrics(self, batch: Any, result: Any) -> None:
        if os.environ.get("AREAL_AWEX_FORCE_SGLANG_METRICS", "1") != "1":
            return
        if batch is None:
            return
        scheduler = self._scheduler
        mode = getattr(getattr(batch, "forward_mode", None), "name", None)
        if mode != "DECODE":
            return
        if not getattr(scheduler, "current_scheduler_metrics_enabled", False):
            return

        current_ct = getattr(scheduler, "forward_ct_decode", None)
        interval = (
            getattr(getattr(scheduler, "server_args", None), "decode_log_interval", 1)
            or 1
        )
        should_log_decode = current_ct is not None and current_ct % interval == 0

        if (
            should_log_decode
            and getattr(scheduler, "_areal_awex_last_decode_stats_ct", None)
            != current_ct
        ):
            can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
            scheduler.log_decode_stats(can_run_cuda_graph, running_batch=batch)

        if (
            getattr(scheduler, "_areal_awex_last_decode_stats_every_iter_ct", None)
            != current_ct
        ):
            scheduler.log_decode_stats_every_iteration(
                batch,
                num_accepted_tokens=getattr(result, "num_accepted_tokens", 0),
            )


# ---------------------------------------------------------------------------
# Duplicated from sglang.srt.managers.scheduler.run_scheduler_process
# (SGLang commit pinned in this repo).
#
# AReaL additions are between # ---- BEGIN AREAL ---- / # ---- END AREAL ----
# markers.  Deltas vs upstream:
#   1. AwexSchedulerBridge(scheduler).bind()   -- awex weight update service
#   2. PPSchedulerBridge(scheduler, server_args).bind()  -- per-PP-rank NCCL groups
# ---------------------------------------------------------------------------


def areal_run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: int | None,
    pipe_writer,
) -> None:
    """Drop-in for ``sglang.srt.managers.scheduler.run_scheduler_process``.

    Duplicated from SGLang source.  AReaL additions are between
    ``# ---- BEGIN AREAL ----`` / ``# ---- END AREAL ----`` markers.

    Deltas vs upstream:
      1. After ``Scheduler()`` creation -> ``AwexSchedulerBridge(scheduler).bind()``
      2. After ``Scheduler()`` creation -> ``PPSchedulerBridge(scheduler, server_args).bind()``
    """
    import signal

    import psutil
    from sglang.srt.environ import envs
    from sglang.srt.managers.scheduler import Scheduler, configure_scheduler
    from sglang.srt.observability.trace import (
        process_tracing_init,
        trace_set_thread_info,
    )
    from sglang.srt.utils import (
        get_bool_env_var,
        kill_itself_when_parent_died,
        set_gpu_proc_affinity,
    )
    from sglang.srt.utils.numa_utils import (
        get_numa_node_if_available,
        numa_bind_to_node,
    )
    from sglang.utils import get_exception_traceback

    from areal.v2.inference_service.sglang.pp_bridge import (
        PPSchedulerBridge,
    )

    logger = logging.getLogger(__name__)
    dp_rank = configure_scheduler(
        server_args, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank, pp_rank, dp_rank
    )

    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, gpu_id
        )
    numa_node = get_numa_node_if_available(server_args, gpu_id)
    if numa_node is not None and not envs.SGLANG_NUMA_BIND_V2.get():
        numa_bind_to_node(numa_node)

    # Set up tracing
    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        thread_label = "Scheduler"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill Scheduler"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode Scheduler"
        trace_set_thread_info(thread_label, tp_rank, dp_rank)

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            attn_cp_rank,
            moe_dp_rank,
            dp_rank,
        )

        # ---- BEGIN AREAL ----
        AwexSchedulerBridge(scheduler).bind()
        PPSchedulerBridge(scheduler, server_args).bind()
        # ---- END AREAL ----

        pipe_writer.send(scheduler.get_init_info())
        scheduler.run_event_loop()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)


def create_result_ipc() -> str:
    path = f"ipc://{tempfile.mktemp(prefix='areal_result_')}"
    os.environ[RESULT_IPC_ENV] = path
    return path
