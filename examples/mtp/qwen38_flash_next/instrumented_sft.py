# SPDX-License-Identifier: Apache-2.0
"""Run SFT while recording run-acceptance memory and throughput metrics."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import time
from pathlib import Path

import torch
import torch.distributed as dist


def _token_count(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel()
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple, torch.Tensor)):
            return sum(_token_count(item) for item in value)
        return len(value)
    raise TypeError(f"cannot count input tokens in {type(value).__name__}")


def _parameter_digest(models, *, trainable: bool) -> tuple[str, int]:
    """Hash complete rank-local parameter payloads outside measured train steps."""
    digest = hashlib.sha256()
    count = 0
    from examples.mtp.qwen38_flash_next.parameter_checks import classify_parameters

    for name, parameter, is_mtp in classify_parameters(models).values():
        if is_mtp != trainable:
            continue
        digest.update(f"{name}:{parameter.dtype}:{tuple(parameter.shape)}".encode())
        value = parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(value.numpy().tobytes())
        count += 1
    return digest.hexdigest(), count


def install_step_instrumentation() -> None:
    """Add per-step max-across-ranks CUDA and token-throughput statistics."""
    from megatron.core import parallel_state

    from examples.mtp.qwen38_flash_next.recovery_checks import (
        snapshot_training_state,
        state_digest,
    )

    from areal.engine.megatron_engine import MegatronEngine
    from areal.trainer.sft_trainer import SFTTrainer

    original_save_checkpoint = SFTTrainer._save_recover_checkpoint

    def save_checkpoint(trainer, epoch, epoch_step, global_step):
        result = original_save_checkpoint(trainer, epoch, epoch_step, global_step)
        if os.environ.get("RECOVERY_STATE_DIR") and global_step == int(
            os.environ.get("SAVED_GLOBAL_STEP", "3")
        ):
            snapshot_training_state(trainer.actor, "saved-boundary")
        return result

    SFTTrainer._save_recover_checkpoint = save_checkpoint
    original_train_batch = MegatronEngine.train_batch

    def train_batch(self, input_, *args, **kwargs):
        verify_steps = int(os.environ.get("VERIFY_PARAMETER_STEPS", "0"))
        if verify_steps and not hasattr(self, "_initial_parameter_digests"):
            from examples.mtp.qwen38_flash_next.parameter_checks import (
                verify_mtp_optimizer_membership,
            )

            self._parameter_membership = verify_mtp_optimizer_membership(
                self.model, self.optimizer
            )
            self._initial_parameter_digests = {
                trainable: _parameter_digest(self.model, trainable=trainable)
                for trainable in (False, True)
            }
            snapshot_training_state(
                self,
                os.environ.get("MTP_SNAPSHOT_LABEL", "run") + "-initial",
                frozen_only=True,
            )
        normalized, _ = self._normalize_batch_input(input_)
        input_digest = state_digest(
            {
                key: normalized[key]
                for key in ("input_ids", "attention_mask", "loss_mask")
            }
        )
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        stats = original_train_batch(self, input_, *args, **kwargs)
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started

        input_batched, _ = self._normalize_batch_input(input_)
        valid = input_batched["attention_mask"].bool()
        supervised = input_batched.get(
            "input_token_loss_mask", input_batched["loss_mask"]
        ).bool()
        if torch.any(supervised & ~valid):
            raise AssertionError("Padding cannot be supervised")
        valid_lengths = valid.sum(dim=-1)
        supervised_lengths = supervised.sum(dim=-1)
        local_tokens = int(valid_lengths.sum())
        dp_size = parallel_state.get_data_parallel_world_size(
            with_context_parallel=True
        )
        metrics = torch.tensor(
            [
                elapsed,
                float(torch.cuda.max_memory_allocated(self.device)),
                float(torch.cuda.max_memory_reserved(self.device)),
            ],
            dtype=torch.float64,
            device=self.device,
        )
        # Every TP/PP/EP rank participates. Reporting the slowest rank and the
        # largest allocator peak gives a conservative whole-job measurement.
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        step_seconds, peak_allocated, peak_reserved = metrics.tolist()
        stats.update(
            step_seconds=step_seconds,
            tokens_per_second=local_tokens * dp_size / step_seconds,
            effective_tokens=local_tokens * dp_size,
            supervised_tokens=int(supervised_lengths.sum()) * dp_size,
            min_effective_sample_tokens=int(valid_lengths.min()),
            max_effective_sample_tokens=int(valid_lengths.max()),
            min_supervised_sample_tokens=int(supervised_lengths.min()),
            max_supervised_sample_tokens=int(supervised_lengths.max()),
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
        )

        step_index = getattr(self, "_acceptance_step_index", 0) + 1
        self._acceptance_step_index = step_index
        if verify_steps and step_index == verify_steps:
            from examples.mtp.qwen38_flash_next.parameter_checks import (
                verify_mtp_optimizer_membership,
            )

            verify_mtp_optimizer_membership(self.model, self.optimizer)
            before = self._initial_parameter_digests
            after = {
                trainable: _parameter_digest(self.model, trainable=trainable)
                for trainable in (False, True)
            }
            if before[False] != after[False]:
                raise AssertionError("Frozen non-MTP parameters changed")
            if after[True][1] and before[True][0] == after[True][0]:
                raise AssertionError("MTP parameters did not update")
            destination = Path(os.environ["PARAMETER_VERIFICATION_DIR"])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / f"rank-{dist.get_rank():02d}.json").write_text(
                json.dumps(
                    {
                        "rank": dist.get_rank(),
                        "membership": self._parameter_membership,
                        "steps": step_index,
                        "frozen_before": before[False],
                        "frozen_after": after[False],
                        "mtp_before": before[True],
                        "mtp_after": after[True],
                    },
                    indent=2,
                )
                + "\n"
            )
        snapshot_step = int(os.environ.get("MTP_SNAPSHOT_STEP", "0"))
        snapshot_dir = os.environ.get("MTP_SNAPSHOT_DIR")
        if snapshot_dir and snapshot_step == step_index:
            label = os.environ.get("MTP_SNAPSHOT_LABEL", f"step-{step_index}")
            snapshot_training_state(self, label + "-state")
            destination = Path(snapshot_dir)
            destination.mkdir(parents=True, exist_ok=True)
            parameters = {
                f"chunk_{chunk_index}.{name}": parameter.detach().cpu().clone()
                for chunk_index, model in enumerate(self.model)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            torch.save(
                {
                    "rank": dist.get_rank(),
                    "world_size": dist.get_world_size(),
                    "input_digest": input_digest,
                    "step_index_in_process": step_index,
                    "parameters": parameters,
                },
                destination / f"{label}.rank-{dist.get_rank():02d}.pt",
            )
            dist.barrier()
        return stats

    MegatronEngine.train_batch = train_batch


def install_structured_metrics() -> None:
    """Append reduced rank-zero trainer statistics as machine-readable JSONL."""
    from areal.utils.stats_logger import StatsLogger

    original_commit = StatsLogger.commit

    def commit(self, epoch, step, global_step, data):
        path = os.environ.get("ACCEPTANCE_METRICS_FILE")
        if path and (not dist.is_initialized() or dist.get_rank() == 0):
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "epoch": epoch,
                "epoch_step": step,
                "global_step": global_step,
                "stats": data,
            }
            with destination.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        return original_commit(self, epoch, step, global_step, data)

    StatsLogger.commit = commit


def main() -> None:
    install_step_instrumentation()
    install_structured_metrics()
    runpy.run_module(
        "examples.mtp.qwen38_flash_next.fixed_data_sft", run_name="__main__"
    )


if __name__ == "__main__":
    main()
