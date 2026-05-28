# SPDX-License-Identifier: Apache-2.0

import getpass
import os
import time
from dataclasses import asdict

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
