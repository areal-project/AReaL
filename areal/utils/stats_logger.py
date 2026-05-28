# SPDX-License-Identifier: Apache-2.0

import getpass
import os
import time
from dataclasses import asdict
from typing import Any

import swanlab
import torch.distributed as dist
import trackio
import wandb
from tensorboardX import SummaryWriter

from areal.api import FinetuneSpec
from areal.api.cli_args import BaseExperimentConfig, StatsLoggerConfig
from areal.utils import logging
from areal.utils.printing import tabulate_stats
from areal.version import version_info

logger = logging.getLogger("StatsLogger", "system")


class StatsLogger:
    def __init__(self, config: BaseExperimentConfig, ft_spec: FinetuneSpec):
        if isinstance(config, StatsLoggerConfig):
            raise ValueError(
                "Passing config.stats_logger as the config is deprecated. "
                "Please pass the full config instead."
            )
        self.exp_config = config
        self.config = config.stats_logger
        self.ft_spec = ft_spec
        self.init()

        self._last_commit_step = -1

    def init(self):
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        if self.config.wandb.wandb_base_url:
            os.environ["WANDB_BASE_URL"] = self.config.wandb.wandb_base_url
        if self.config.wandb.wandb_api_key:
            os.environ["WANDB_API_KEY"] = self.config.wandb.wandb_api_key

        self.start_time = time.perf_counter()
        # wandb init, connect to remote wandb host
        if self.config.wandb.mode != "disabled":
            wandb.login()

        suffix = self.config.wandb.id_suffix
        if suffix == "timestamp":
            suffix = time.strftime("%Y_%m_%d_%H_%M_%S")

        exp_config_dict = asdict(self.exp_config)
        exp_config_dict["version_info"] = {
            "commit_id": version_info.commit,
            "branch": version_info.branch,
            "is_dirty": version_info.is_dirty,
            "version": version_info.full_version_with_dirty_description,
        }

        wandb.init(
            mode=self.config.wandb.mode,
            entity=self.config.wandb.entity,
            project=self.config.wandb.project or self.config.experiment_name,
            name=self.config.wandb.name or self.config.trial_name,
            job_type=self.config.wandb.job_type,
            group=self.config.wandb.group
            or f"{self.config.experiment_name}_{self.config.trial_name}",
            notes=self.config.wandb.notes,
            tags=self.config.wandb.tags,
            config=exp_config_dict,  # save all experiment config to wandb
            dir=self.get_log_path(self.config),
            force=True,
            id=f"{self.config.experiment_name}_{self.config.trial_name}_{suffix}",
            resume="allow",
        )

        swanlab_config = self.config.swanlab
        if swanlab_config.mode != "disabled":
            if swanlab_config.api_key:
                swanlab.login(swanlab_config.api_key)
            else:
                swanlab.login()

        swanlab_config = self.config.swanlab
        swanlab.init(
            project=swanlab_config.project or self.config.experiment_name,
            experiment_name=swanlab_config.name or self.config.trial_name + "_train",
            # NOTE: change from swanlab_config.config to log all experiment config, to be tested
            config=exp_config_dict,
            logdir=self.get_log_path(self.config),
            mode=swanlab_config.mode,
        )

        # trackio init
        self._trackio_enabled = False
        trackio_config = self.config.trackio
        if trackio_config.mode != "disabled":
            trackio.init(
                project=trackio_config.project or self.config.experiment_name,
                name=trackio_config.name or self.config.trial_name,
                config=exp_config_dict,
                space_id=trackio_config.space_id,
            )
            self._trackio_enabled = True

        # tensorboard logging
        self.summary_writer = None
        if self.config.tensorboard.path is not None:
            self.summary_writer = SummaryWriter(log_dir=self.config.tensorboard.path)

    def state_dict(self):
        return {
            "last_commit_step": self._last_commit_step,
        }

    def load_state_dict(self, state_dict):
        self._last_commit_step = state_dict["last_commit_step"]

    def close(self):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        logger.info(
            f"Training completes! Total time elapsed {time.monotonic() - self.start_time:.2f}."
        )
        wandb.finish()
        swanlab.finish()
        if getattr(self, "_trackio_enabled", False):
            trackio.finish()
        if self.summary_writer is not None:
            self.summary_writer.close()

    def commit(self, epoch: int, step: int, global_step: int, data: dict | list[dict]):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        logger.info(
            f"Epoch {epoch + 1}/{self.ft_spec.total_train_epochs} "
            f"Step {step + 1}/{self.ft_spec.steps_per_epoch} "
            f"Train step {global_step + 1}/{self.ft_spec.total_train_steps} done."
        )
        if isinstance(data, dict):
            data = [data]
        log_step = max(global_step, self._last_commit_step + 1)
        for i, item in enumerate(data):
            # Filter out counter keys for scalar variables
            item = {k: v for k, v in item.items() if not k.endswith("__count")}

            logger.info(f"Stats ({i + 1}/{len(data)}):")
            self.print_stats(item)
            wandb.log(item, step=log_step + i)
            swanlab.log(item, step=log_step + i)
            if getattr(self, "_trackio_enabled", False):
                trackio.log(item, step=log_step + i)
            if self.summary_writer is not None:
                for key, val in item.items():
                    self.summary_writer.add_scalar(f"{key}", val, log_step + i)
        self._last_commit_step = log_step + len(data) - 1

    def log_rollout_traces(
        self,
        trajectories: list[dict[str, Any] | None],
        *,
        split: str,
        global_step: int,
        tokenizer,
    ) -> None:
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        if not getattr(self, "_trackio_enabled", False):
            return

        max_traces = self.config.trackio.max_rollout_traces_per_step
        if max_traces <= 0:
            return

        traces = []
        for trajectory_index, trajectory in enumerate(trajectories):
            if len(traces) >= max_traces:
                break
            traces.extend(
                self._trajectory_to_trackio_traces(
                    trajectory,
                    split=split,
                    global_step=global_step,
                    trajectory_index=trajectory_index,
                    tokenizer=tokenizer,
                    remaining=max_traces - len(traces),
                )
            )

        if traces:
            trackio.log({f"{split}/trajectories": traces}, step=global_step)

    def _trajectory_to_trackio_traces(
        self,
        trajectory: dict[str, Any] | None,
        *,
        split: str,
        global_step: int,
        trajectory_index: int,
        tokenizer,
        remaining: int,
    ) -> list[Any]:
        if trajectory is None:
            return []

        input_ids = trajectory.get("input_ids")
        loss_mask = trajectory.get("loss_mask")
        attention_mask = trajectory.get("attention_mask")
        rewards = trajectory.get("rewards")
        versions = trajectory.get("versions")

        if input_ids is None or loss_mask is None or attention_mask is None:
            return []

        traces = []
        batch_size = input_ids.shape[0]
        input_ids_cpu = input_ids.detach().cpu()
        loss_mask_cpu = loss_mask.detach().cpu()
        attention_mask_cpu = attention_mask.detach().cpu()
        rewards_cpu = rewards.detach().cpu() if rewards is not None else None
        versions_cpu = versions.detach().cpu() if versions is not None else None
        for sample_index in range(batch_size):
            if len(traces) >= remaining:
                break
            seqlen = int(attention_mask_cpu[sample_index].sum().item())
            if seqlen <= 0:
                continue

            ids = input_ids_cpu[sample_index, :seqlen].tolist()
            mask = loss_mask_cpu[sample_index, :seqlen].tolist()
            if not mask or mask[-1] != 1:
                continue

            messages = self._trajectory_messages(
                trajectory,
                sample_index=sample_index,
                ids=ids,
                mask=mask,
                tokenizer=tokenizer,
            )
            if not messages:
                continue

            metadata = {
                "split": split,
                "global_step": global_step,
                "trajectory_index": trajectory_index,
                "sample_index": sample_index,
                "seqlen": seqlen,
                "prompt_len": mask.index(1),
            }
            if rewards_cpu is not None:
                metadata["reward"] = self._metadata_value(rewards_cpu[sample_index])
            if versions_cpu is not None:
                sample_versions = versions_cpu[sample_index, :seqlen].tolist()
                metadata["head_version"] = min(sample_versions)
                metadata["tail_version"] = max(sample_versions)

            traces.append(
                trackio.Trace(
                    messages=messages,
                    metadata=metadata,
                )
            )
        return traces

    @staticmethod
    def _metadata_value(value):
        if hasattr(value, "numel"):
            if value.numel() == 1:
                return value.item()
            return value.tolist()
        return value

    def _trajectory_messages(
        self,
        trajectory: dict[str, Any],
        *,
        sample_index: int,
        ids: list[int],
        mask: list[int],
        tokenizer,
    ) -> list[dict[str, str]]:
        messages = self._structured_messages(trajectory, sample_index)
        if messages is not None:
            return messages

        trace_messages = []
        span_start = 0
        for span_end in range(1, len(ids) + 1):
            if span_end < len(ids) and bool(mask[span_end]) == bool(mask[span_start]):
                continue

            content = tokenizer.decode(
                ids[span_start:span_end], skip_special_tokens=False
            )
            if content:
                if bool(mask[span_start]):
                    role = "assistant"
                else:
                    role = "user" if not trace_messages else "tool"
                trace_messages.append({"role": role, "content": content})
            span_start = span_end

        return trace_messages

    @staticmethod
    def _structured_messages(
        trajectory: dict[str, Any], sample_index: int
    ) -> list[dict[str, str]] | None:
        for key in ("messages", "conversation", "conversation_text"):
            value = trajectory.get(key)
            if value is None:
                continue
            if (
                isinstance(value, list)
                and len(value) > sample_index
                and isinstance(value[sample_index], list)
            ):
                value = value[sample_index]
            if isinstance(value, list) and all(
                isinstance(message, dict) for message in value
            ):
                return [
                    {
                        "role": str(message.get("role", "user")),
                        "content": str(message.get("content", "")),
                    }
                    for message in value
                ]
        return None

    def print_stats(self, stats: dict[str, float]):
        logger.info("\n" + tabulate_stats(stats))

    @staticmethod
    def get_log_path(
        config: StatsLoggerConfig | None = None,
        experiment_name: str | None = None,
        trial_name: str | None = None,
        fileroot: str | None = None,
    ) -> str:
        if config is not None:
            experiment_name = config.experiment_name
            trial_name = config.trial_name
            fileroot = config.fileroot
        if not fileroot or not experiment_name or not trial_name:
            raise ValueError(
                "fileroot, experiment_name, and trial_name must be provided."
            )
        path = f"{fileroot}/logs/{getpass.getuser()}/{experiment_name}/{trial_name}"
        os.makedirs(path, exist_ok=True)
        return path
