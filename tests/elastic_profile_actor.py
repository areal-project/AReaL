# SPDX-License-Identifier: Apache-2.0
"""GPU-only profiling Actor: forward/backward/optimizer updates are prohibited."""

import hashlib
import os
from contextlib import ExitStack
from unittest.mock import patch

import torch
from transformer_engine.pytorch.optimizers import FusedAdam

from tests.elastic_probe_actor import ProbeActor
from tests.elastic_profile import OSProxy, Recorder, hook

from areal.engine import MegatronPPOActor


def forbidden_update(*args, **kwargs):
    raise RuntimeError("Training and parameter updates are prohibited in this profile")


class ProfileActor(MegatronPPOActor):
    def __init__(self, *args, **kwargs):
        self.profile = Recorder(
            os.environ["AREAL_ELASTIC_PROFILE_DIR"],
            "rank" + os.environ["RANK"],
            int(os.environ["AREAL_ELASTIC_PROFILE_GENERATION"]),
        )
        self.profile_hooks = ExitStack()
        with self.profile.span("engine.construct"):
            super().__init__(*args, **kwargs)
        self.hf_load_calls = 0
        self.hf_load_seconds = self.dcp_load_seconds = 0.0
        self._install_hooks()

    def _install_hooks(self):
        from megatron.core.dist_checkpointing.strategies import filesystem_async
        from megatron.core.dist_checkpointing.strategies.filesystem_async import (
            FileSystemWriterAsync,
        )
        from megatron.core.dist_checkpointing.strategies.torch import (
            TorchDistSaveShardedStrategy,
        )
        from torch.distributed.checkpoint.filesystem import FileSystemReader

        from areal.engine.megatron_utils import checkpointer
        from areal.v2.training_service import checkpoint

        stack, recorder = self.profile_hooks, self.profile
        for name in ("_build_hf_mcore_bridge", "_create_optimizer"):
            hook(stack, self, name, recorder, "engine." + name.lstrip("_"))
        for name in ("save_dist_checkpointing", "load_dist_checkpointing"):
            hook(stack, checkpointer, name, recorder, "dcp." + name)
        for name in ("_capture", "_restore_rng", "_validate_payload"):
            hook(stack, checkpoint, name, recorder, "runtime." + name.lstrip("_"))
        hook(stack, checkpoint, "atomic_json", recorder, "runtime.sidecar_write")
        hook(
            stack,
            TorchDistSaveShardedStrategy,
            "async_save",
            recorder,
            "save.plan_and_prepare",
        )
        for name, label in (
            ("prepare_write_data", "save.prepare_buckets"),
            ("preload_tensors", "save.device_to_host"),
            ("write_preloaded_data_multithread", "save.write_and_sync"),
            ("write_preloaded_data", "save.bucket_write_and_sync"),
            ("finish", "save.metadata_finish"),
        ):
            hook(stack, FileSystemWriterAsync, name, recorder, label)
        stack.enter_context(
            patch.object(
                filesystem_async, "os", OSProxy(filesystem_async.os, recorder, "writer")
            )
        )
        hook(stack, FileSystemReader, "read_data", recorder, "load.read_data")
        hook(stack, torch.distributed, "barrier", recorder, "distributed.barrier")

    def initialize(self, *args, **kwargs):
        with self.profile.span("engine.initialize"):
            super().initialize(*args, **kwargs)
        for name in (
            "generate_state_dict",
            "save_checkpoint",
            "load_checkpoint",
            "wait_async_saves",
        ):
            hook(
                self.profile_hooks,
                self.checkpointer,
                name,
                self.profile,
                "checkpointer." + name,
            )
        # Fence both public Actor APIs and every actual optimizer implementation.
        self.optimizer.step = forbidden_update
        for optimizer in self._optimizers():
            optimizer.step = forbidden_update
            optimizer.optimizer.step = forbidden_update

    def _load_model_from_hf(self, *args, **kwargs):
        import time

        start = time.monotonic()
        with self.profile.span("engine.hf_weights"):
            result = super()._load_model_from_hf(*args, **kwargs)
        self.hf_load_calls += 1
        self.hf_load_seconds += time.monotonic() - start
        return result

    def save(self, meta):
        import gc

        with ExitStack() as stack, self.profile.span("engine.save"):
            hook(stack, gc, "collect", self.profile, "save.gc_collect")
            hook(
                stack, torch.cuda, "empty_cache", self.profile, "save.cuda_empty_cache"
            )
            if self._awex_adapter is not None:
                hook(
                    stack,
                    self._awex_adapter,
                    "_release_grad_memory",
                    self.profile,
                    "save.release_grad_buffers",
                )
            return super().save(meta)

    def load(self, meta):
        import time

        start = time.monotonic()
        with self.profile.span("engine.load"):
            result = super().load(meta)
        self.dcp_load_seconds = time.monotonic() - start
        return result

    def _optimizers(self):
        return getattr(self.optimizer, "chained_optimizers", [self.optimizer])

    def _model_digest(self):
        digest = hashlib.sha256()
        for model in self.model:
            for parameter in model.parameters():
                digest.update(
                    parameter.detach()
                    .cpu()
                    .contiguous()
                    .reshape(-1)
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
        return digest.hexdigest()

    def get_device_stats(self, *, initialize_optimizer=False):
        """Audit is outside measured phases; no optimizer step is ever needed."""
        before = self._model_digest() if initialize_optimizer else None
        scheduler_before = self.lr_scheduler.state_dict()
        slots, steps = {}, []
        for wrapper in self._optimizers():
            optimizer = wrapper.optimizer
            if (
                not isinstance(optimizer, FusedAdam)
                or not wrapper.config.store_param_remainders
            ):
                raise ValueError(
                    "Profile requires the demonstrated TE Adam BF16 remainder layout"
                )
            for group in optimizer.param_groups:
                steps.append(int(group.get("step", 0)))
                for parameter in group["params"]:
                    if initialize_optimizer and not optimizer.state.get(parameter):
                        # Native TE initializer: zero moments and exact zero
                        # residuals for the freshly loaded BF16 parameters.
                        optimizer.initialize_state(
                            parameter, store_param_remainders=True
                        )
                    for name in ("master_param", "exp_avg", "exp_avg_sq"):
                        value = optimizer.state[parameter][name]
                        expected = (
                            torch.int16 if name == "master_param" else torch.float32
                        )
                        assert value.dtype == expected
                        assert torch.count_nonzero(value).item() == 0
                        key = name + ":" + str(value.dtype)
                        slots[key] = (
                            slots.get(key, 0) + value.numel() * value.element_size()
                        )
        assert all(step == 0 for step in steps)
        assert steps == [
            int(group.get("step", 0))
            for wrapper in self._optimizers()
            for group in wrapper.optimizer.param_groups
        ]
        assert self.lr_scheduler.state_dict() == scheduler_before
        assert scheduler_before["num_steps"] == 0
        if before is not None:
            assert self._model_digest() == before, (
                "Slot materialization changed model weights"
            )
        # Reuse only the read-only audit method, never ProbeActor's step method.
        result = ProbeActor.get_device_stats(self)
        return {
            **result,
            "optimizer_steps": steps,
            "scheduler_steps": scheduler_before["num_steps"],
            "slot_bytes": slots,
            "model_unchanged_by_slot_materialization": before is not None,
        }

    train_batch = forbidden_update
    forward_batch = forbidden_update
    backward_batch = forbidden_update
    forward_backward_batch = forbidden_update
    eval_batch = forbidden_update
    optimizer_step = forbidden_update
