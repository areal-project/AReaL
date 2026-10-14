# SPDX-License-Identifier: Apache-2.0
"""AwexSchedulerBridge + PPSchedulerBridge: compose weight-update methods onto SGLang Scheduler."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any

import torch.distributed as dist
import zmq
from sglang.srt.server_args import PortArgs, ServerArgs

from areal.infra.rpc.serialization import serialize_value

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
            and scheduler.ps.tp_rank == 0
            and (scheduler.ps.dp_rank is None or scheduler.ps.dp_rank == 0)
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
            "awex_release_memory",
            "awex_resume_memory",
        ]
        for name in methods:
            bridge_method = getattr(self, name)

            def call_awex(*, payload: str, _method=bridge_method) -> None:
                parameters = json.loads(payload)
                if not isinstance(parameters, dict):
                    raise ValueError("AWEX RPC payload must be a JSON object")
                _method(**parameters)

            setattr(self._scheduler, name, call_awex)

    def _require_adapter(self) -> Any:
        if self._adapter is None:
            from areal.engine.awex.adapters.sglang_adapter import (
                AwexSGLangAdapter,
            )

            self._adapter = AwexSGLangAdapter(self._scheduler)
        return self._adapter

    def _push_result(self, result: Any) -> None:
        if self._result_push is not None:
            self._result_push.send_pyobj(result)

    def awex_report_weight_meta(self, parameter_layout: str = "hf") -> None:
        adapter = self._require_adapter()
        local_meta = adapter.get_weight_metadata(parameter_layout=parameter_layout)
        s = self._scheduler

        # All-gather across TP ranks so rank 0 returns aggregated metadata
        if s.ps.tp_size > 1:
            gathered: list[list] = [[] for _ in range(s.ps.tp_size)]
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
        if self._scheduler.ps.tp_rank == 0:
            adapter.save_parameters(save_path, names)

    def awex_randomize_parameters(self) -> None:
        self._require_adapter().randomize_parameters()

    def awex_init_colocate_weight_update(self, **kwargs: Any) -> None:
        self._require_adapter().init_colocate_weight_update(**kwargs)

    def awex_execute_colocate_weight_update(self, version: int = 0) -> None:
        self._require_adapter().execute_colocate_weight_update(version)

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
#   2. MTPDistributedWeightUpdateBridge(...).bind() -- NEXTN draft weight routing
#   3. PPSchedulerBridge(scheduler, server_args).bind()  -- per-PP-rank NCCL groups
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
    display_tp_rank: int | None = None,
    display_dp_rank: int | None = None,
    display_moe_ep_rank: int | None = None,
) -> None:
    """Drop-in for ``sglang.srt.managers.scheduler.run_scheduler_process``.

    Duplicated from SGLang source.  AReaL additions are between
    ``# ---- BEGIN AREAL ----`` / ``# ---- END AREAL ----`` markers.

    Deltas vs upstream:
      1. After ``Scheduler()`` creation -> ``AwexSchedulerBridge(scheduler).bind()``
      2. After ``Scheduler()`` creation ->
         ``MTPDistributedWeightUpdateBridge(scheduler, server_args).bind()``
      3. After ``Scheduler()`` creation ->
         ``PPSchedulerBridge(scheduler, server_args).bind()``
    """
    import signal

    import psutil
    from sglang.srt.distributed.parallel_state import abort_distributed_environment
    from sglang.srt.environ import envs
    from sglang.srt.managers.scheduler import Scheduler, configure_scheduler_process
    from sglang.srt.observability.trace import (
        process_tracing_init,
        trace_set_thread_info,
    )
    from sglang.srt.plugins import load_plugins
    from sglang.srt.runtime_context import (
        get_disagg,
        get_observability,
        publish,
    )
    from sglang.utils import get_exception_traceback

    from areal.v2.inference_service.sglang.mtp_weight_update_bridge import (
        MTPDistributedWeightUpdateBridge,
    )
    from areal.v2.inference_service.sglang.pp_bridge import (
        PPSchedulerBridge,
    )
    from areal.v2.inference_service.sglang.routed_experts_compat import (
        apply_sglang_routed_experts_token_cache_patch,
    )

    logger = logging.getLogger(__name__)
    load_plugins()
    publish(server_args, role="scheduler")
    dp_rank = configure_scheduler_process(
        server_args,
        gpu_id,
        tp_rank,
        attn_cp_rank,
        moe_dp_rank,
        moe_ep_rank,
        pp_rank,
        dp_rank,
        display_tp_rank=display_tp_rank,
        display_dp_rank=display_dp_rank,
        display_moe_ep_rank=display_moe_ep_rank,
    )

    parent_process = psutil.Process().parent()

    # Set up tracing
    if get_observability().enable_trace:
        process_tracing_init(
            get_observability().otlp_traces_endpoint,
            "sglang",
            trace_modules=get_observability().trace_modules,
        )
        thread_label = "Scheduler"
        if get_disagg().disaggregation_mode == "prefill":
            thread_label = "Prefill Scheduler"
        elif get_disagg().disaggregation_mode == "decode":
            thread_label = "Decode Scheduler"
        trace_set_thread_info(thread_label, tp_rank, dp_rank, pp_rank)

    # Create a scheduler and run the event loop
    scheduler = None
    try:
        # ---- BEGIN AREAL ----
        if server_args.enable_return_routed_experts:
            apply_sglang_routed_experts_token_cache_patch()
        # ---- END AREAL ----

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
        MTPDistributedWeightUpdateBridge(scheduler, server_args).bind()
        PPSchedulerBridge(scheduler, server_args).bind()
        # ---- END AREAL ----

        pipe_writer.send(scheduler.get_init_info())
        scheduler.run_event_loop()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
        if envs.SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION.get():
            try:
                os.killpg(os.getpgrp(), signal.SIGKILL)
            except Exception:
                pass
    finally:
        if scheduler is not None:
            scheduler.metrics_reporter._shutdown_fpm()
            if scheduler.gracefully_exit:
                scheduler.release_host_resources()
                abort_distributed_environment()


def create_result_ipc() -> str:
    path = f"ipc://{tempfile.mktemp(prefix='areal_result_')}"
    os.environ[RESULT_IPC_ENV] = path
    return path
