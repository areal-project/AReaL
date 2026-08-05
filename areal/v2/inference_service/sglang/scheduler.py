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

RESULT_IPC_ENV = "AREAL_AWEX_RESULT_IPC"
logger = logging.getLogger(__name__)


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
        self._push_result(self._require_adapter().parallelism_strategy)

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
        try:
            self._require_adapter().init_colocate_weight_update(**kwargs)
        except Exception:
            logger.exception("awex_init_colocate_weight_update failed")
            raise

    def awex_execute_colocate_weight_update(self, version: int = 0) -> None:
        self._require_adapter().execute_colocate_weight_update(version)

    def awex_seed_delta_base(self, version: int = 0) -> None:
        self._require_adapter().seed_delta_base(version)

    def awex_release_memory(self, tags: list[str] | None = None) -> None:
        self._require_adapter().release_memory(tags)

    def awex_resume_memory(self, tags: list[str] | None = None) -> None:
        self._require_adapter().resume_memory(tags)


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
    import faulthandler
    import signal

    import psutil
    import setproctitle
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.environ import envs
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.tracing.trace import (
        process_tracing_init,
        trace_set_thread_info,
    )
    from sglang.srt.utils import (
        configure_logger,
        get_bool_env_var,
        kill_itself_when_parent_died,
        numa_bind_to_node,
        set_gpu_proc_affinity,
        suppress_other_loggers,
    )
    from sglang.utils import get_exception_traceback

    from areal.v2.inference_service.sglang.pp_bridge import (
        PPSchedulerBridge,
    )

    logger = logging.getLogger(__name__)

    prefix = ""
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])
    if dp_rank is not None:
        prefix += f" DP{dp_rank}"
    if server_args.pp_size > 1:
        prefix += f" PP{pp_rank}"
    if server_args.attn_cp_size > 1:
        prefix += f" ATTN_CP{attn_cp_rank}"
    if server_args.moe_dp_size > 1:
        prefix += f" MOE_DP{moe_dp_rank}"
    if server_args.tp_size > 1:
        prefix += f" TP{tp_rank}"
    if server_args.ep_size > 1:
        prefix += f" EP{moe_ep_rank}"

    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, gpu_id
        )
    if (
        numa_node := server_args.numa_node
    ) is not None and not envs.SGLANG_NUMA_BIND_V2.get():
        numa_bind_to_node(numa_node[gpu_id])

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

        result_dict = {
            "status": "ready",
            "max_total_num_tokens": scheduler.max_total_num_tokens,
            "max_req_input_len": scheduler.max_req_input_len,
        }
        if server_args.remote_instance_weight_loader_use_transfer_engine():
            (
                remote_instance_transfer_engine_session_id,
                remote_instance_transfer_engine_weights_info_dict,
            ) = scheduler.get_remote_instance_transfer_engine_info()
            result_dict.update(
                {
                    "tp_rank": tp_rank,
                    "remote_instance_transfer_engine_session_id": remote_instance_transfer_engine_session_id,
                    "remote_instance_transfer_engine_weights_info_dict": remote_instance_transfer_engine_weights_info_dict,
                }
            )

        pipe_writer.send(result_dict)

        disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
        if disaggregation_mode == DisaggregationMode.NULL:
            if scheduler.enable_pdmux:
                scheduler.event_loop_pdmux()
            elif server_args.pp_size > 1:
                scheduler.event_loop_pp()
            elif scheduler.enable_overlap:
                scheduler.event_loop_overlap()
            else:
                scheduler.event_loop_normal()
        elif disaggregation_mode == DisaggregationMode.PREFILL:
            if server_args.pp_size > 1:
                scheduler.event_loop_pp_disagg_prefill()
            elif scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_prefill()
            else:
                scheduler.event_loop_normal_disagg_prefill()
        elif disaggregation_mode == DisaggregationMode.DECODE:
            if server_args.pp_size > 1:
                scheduler.event_loop_pp_disagg_decode()
            elif scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_decode()
            else:
                scheduler.event_loop_normal_disagg_decode()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)


def create_result_ipc() -> str:
    path = f"ipc://{tempfile.mktemp(prefix='areal_result_')}"
    os.environ[RESULT_IPC_ENV] = path
    return path
