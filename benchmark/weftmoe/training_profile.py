"""Bounded PP1 training diagnostics; profiling runs are not performance samples.

The replay runner installs this instrumentation explicitly. No production engine
hook or extra GPU synchronization is required in the timed method wrappers.
"""

import json
import os
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

from areal.utils import logging

logger = logging.getLogger("PerfTracer")


def instrument_engine():
    """Instrument PP1 replay without modifying AReaL's production schedule."""
    import areal.engine.megatron_engine as module

    original = module.MegatronEngine.forward_backward_batch

    def measured(engine, *args, forward_only=False, **kwargs):
        if forward_only:
            return original(engine, *args, forward_only=True, **kwargs)
        diagnostic = TrainingProfile(engine)
        factory = module.get_forward_backward_func

        def profiled_factory():
            schedule = factory()

            def profiled_schedule(*args, **kwargs):
                forward = kwargs["forward_step_func"]

                def step(*args, **kwargs):
                    diagnostic.microbatch()
                    return forward(*args, **kwargs)

                kwargs["forward_step_func"] = step
                return schedule(*args, **kwargs)

            return profiled_schedule

        module.get_forward_backward_func = profiled_factory
        try:
            return original(engine, *args, forward_only=False, **kwargs)
        finally:
            module.get_forward_backward_func = factory
            diagnostic.finish()

    module.MegatronEngine.forward_backward_batch = measured


class TrainingProfile:
    def __init__(self, engine):
        self.directory = Path(os.environ["AREAL_MEGATRON_PROFILE_DIR"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank = dist.get_rank()
        self.step = getattr(engine, "_diagnostic_training_step", 0)
        engine._diagnostic_training_step = self.step + 1
        self.mb = -1
        self.profiler = None
        self.elapsed = defaultdict(list)
        self.runtime_setup = None
        self.originals = []
        self.routes = []
        self.gates = []
        self.profile_step = int(os.environ.get("AREAL_MEGATRON_PROFILE_STEP", "0"))
        self.profile_mb = int(os.environ.get("AREAL_MEGATRON_PROFILE_MB", "2"))
        self.profile_ranks = {
            int(r)
            for r in os.environ.get("AREAL_MEGATRON_PROFILE_RANKS", "0").split(",")
        }
        self.started = time.perf_counter()
        if self.step == 0:
            from megatron.core import parallel_state as mpu

            if mpu.get_tensor_model_parallel_rank() == 0:
                # The diagnostic hook is called from forward_backward_batch,
                # directly under train_batch. Save CPU training inputs only,
                # allowing identical-batch replay without another rollout.
                batch = sys._getframe(2).f_locals["input_"]
                torch.save(batch, self.directory / f"batch.rank{self.rank}.pt")
        from megatron.core.transformer.moe.moe_layer import MoELayer

        for chunk in engine.model:
            for layer in chunk.modules():
                if isinstance(layer, MoELayer):
                    self._annotate(layer, "forward", "moe/full")
                    self._annotate(layer, "route", "moe/router", capture=True)
                    self._annotate(layer, "shared_experts_compute", "moe/shared")
                    for method in (
                        "preprocess",
                        "dispatch",
                        "routed_experts_compute",
                        "combine",
                        "postprocess",
                    ):
                        self._annotate(layer, method, "moe/routed/" + method)
        cfg = engine.model[0].config
        self.model_config = {
            key: getattr(cfg, key, None)
            for key in (
                "hidden_size",
                "moe_ffn_hidden_size",
                "num_moe_experts",
                "moe_router_topk",
                "moe_deepep_num_sms",
                "moe_permute_fusion",
                "bias_activation_fusion",
                "moe_shared_expert_overlap",
                "recompute_modules",
                "sequence_parallel",
                "moe_router_load_balancing_type",
                "moe_aux_loss_coeff",
                "moe_router_enable_expert_bias",
                "gradient_accumulation_fusion",
            )
        }
        adapter = getattr(engine, "_weftmoe_adapter", None)
        if adapter is not None:
            self._wrap(adapter, "get_runtime", "runtime_prepare")
            self._wrap(adapter, "pack", "weight_stack")
            from weftmoe.runtime import TensorRuntime

            self._wrap(TensorRuntime, "forward", "native_forward")
            self._wrap(TensorRuntime, "backward", "native_backward")
            if hasattr(TensorRuntime, "forward_async"):
                self._wrap(TensorRuntime, "forward_async", "native_forward")
                self._wrap(TensorRuntime, "backward_async", "native_backward")

    def _wrap(self, target, method, label):
        original = getattr(target, method)
        self.originals.append((target, method, original))

        def measured(*args, **kwargs):
            scope = (
                torch.profiler.record_function("weft/" + label)
                if self.profiler
                else nullcontext()
            )
            with scope:
                begin = time.perf_counter()
                result = original(*args, **kwargs)
                self.elapsed[label].append((self.mb, time.perf_counter() - begin))
                if label == "runtime_prepare" and self.runtime_setup is None:
                    if hasattr(result.native, "setup_seconds"):
                        self.runtime_setup = result.native.setup_seconds()
                        if self.rank == 0:
                            logger.info(
                                "WEFT_SETUP_JSON %s",
                                json.dumps(
                                    {
                                        "step": self.step,
                                        "seconds": self.runtime_setup,
                                        "usage": result.native.usage(),
                                    }
                                ),
                            )
                return result

        setattr(target, method, measured)

    def _annotate(self, target, method, label, capture=False):
        original = getattr(target, method)
        self.originals.append((target, method, original))

        def measured(*args, **kwargs):
            scope = (
                torch.profiler.record_function(label)
                if self.profiler
                else nullcontext()
            )
            with scope:
                result = original(*args, **kwargs)
            if (
                capture
                and self.step == self.profile_step
                and self.mb == self.profile_mb
            ):
                # Outside the annotated router scope. Only route metadata is retained.
                gates, routes = (
                    result[0].detach().topk(target.config.moe_router_topk, dim=-1)
                )
                self.routes.append(routes.int())
                self.gates.append(gates)
            return result

        setattr(target, method, measured)

    def microbatch(self):
        self.mb += 1
        if self.step == self.profile_step:
            if self.mb == self.profile_mb:
                if self.rank in self.profile_ranks:
                    self.profile_label = "warm"
                    self.profiler = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        record_shapes=False,
                        with_stack=False,
                    )
                    self.profiler.start()
                # CUPTI's first initialization is rank-skewed. Align after it
                # so startup time is not charged to the first router collective.
                scope = (
                    torch.profiler.record_function("diagnostic/start_barrier")
                    if self.profiler
                    else nullcontext()
                )
                with scope:
                    dist.barrier()
                    torch.cuda.synchronize()
            elif self.mb == self.profile_mb + 1:
                self._stop()

    def _stop(self):
        if self.profiler is None:
            return
        self.profiler.stop()
        prefix = (
            self.directory / f"step{self.step}.rank{self.rank}.{self.profile_label}"
        )
        self.profiler.export_chrome_trace(str(prefix) + ".trace.json")
        self.profiler = None

    def finish(self):
        self._stop()
        for target, method, original in reversed(self.originals):
            setattr(target, method, original)
        prefix = self.directory / f"step{self.step}.rank{self.rank}"
        summary = {
            "rank": self.rank,
            "step": self.step,
            "microbatches": self.mb + 1,
            "wall_seconds": time.perf_counter() - self.started,
            "host_call_seconds": dict(self.elapsed),
            "runtime_setup_seconds": self.runtime_setup,
            "model_config": self.model_config,
            "profile_microbatch": self.profile_mb,
        }
        prefix.with_suffix(prefix.suffix + ".timings.json").write_text(
            json.dumps(summary)
        )
        if self.routes:
            torch.save([r.cpu() for r in self.routes], str(prefix) + ".routes.pt")
            torch.save([g.cpu() for g in self.gates], str(prefix) + ".gates.pt")
