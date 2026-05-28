# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import getpass
import os
from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizerFast

if TYPE_CHECKING:
    from transformers import AutoProcessor

from areal.api import FinetuneSpec, SaveLoadMeta, TrainEngine
from areal.api.cli_args import SaverConfig
from areal.infra import TrainController
from areal.utils import timeutil
from areal.utils.async_checkpoint import AsyncCheckpointManager, AsyncMode
from areal.utils.logging import getLogger

logger = getLogger("Saver")


class Saver:
    def __init__(self, config: SaverConfig, ft_spec: FinetuneSpec):
        self.config = config
        self.ft_spec = ft_spec
        self.freq_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.freq_epochs,
            freq_step=config.freq_steps,
            freq_sec=config.freq_secs,
        )
        self._async_mode = AsyncMode(config.mode)
        self._managers: dict[str, AsyncCheckpointManager] = {}

    @staticmethod
    def get_save_root(
        experiment_name: str,
        trial_name: str,
        fileroot: str,
    ):
        path = os.path.join(
            f"{fileroot}/checkpoints/{getpass.getuser()}/{experiment_name}/{trial_name}",
        )
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_model_save_root(
        experiment_name: str,
        trial_name: str,
        fileroot: str,
        name: str = "default",
    ):
        path = os.path.join(
            Saver.get_save_root(experiment_name, trial_name, fileroot),
            name,
        )
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_model_save_path(
        experiment_name: str,
        trial_name: str,
        fileroot: str,
        epoch: int,
        step: int,
        globalstep: int,
        name: str = "default",
    ):
        path = os.path.join(
            Saver.get_model_save_root(experiment_name, trial_name, fileroot, name),
            f"epoch{epoch}epochstep{step}globalstep{globalstep}",
        )
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_recover_checkpoint_path(
        experiment_name: str,
        trial_name: str,
        fileroot: str,
        name: str = "default",
    ):
        path = os.path.join(
            Saver.get_model_save_root(experiment_name, trial_name, fileroot, name),
            "recover_checkpoint",
        )
        os.makedirs(path, exist_ok=True)
        return path

    def state_dict(self):
        return self.freq_ctl.state_dict()

    def load_state_dict(self, state_dict):
        self.freq_ctl.load_state_dict(state_dict)

    @property
    def is_async(self) -> bool:
        if self._async_mode in (AsyncMode.ASYNC, AsyncMode.AUTO):
            # True if any manager is configured for async saves.
            return any(mgr.is_async for mgr in self._managers.values())
        return False

    def _should_use_async(self, engine: TrainEngine | TrainController) -> bool:
        """Decide whether to use async save for this engine."""
        from areal.experimental.engine.archon_engine import ArchonEngine

        if self._async_mode == AsyncMode.ASYNC:
            if not isinstance(engine, ArchonEngine):
                logger.warning(
                    "Async checkpoint only supports ArchonEngine, "
                    "got %s; falling back to sync",
                    type(engine).__name__,
                )
                return False
            return True
        if self._async_mode == AsyncMode.AUTO:
            return isinstance(engine, ArchonEngine)
        return False

    def save(
        self,
        engine: TrainEngine | TrainController,
        epoch: int,
        step: int,
        global_step: int,
        name: str = "default",
        tokenizer: PreTrainedTokenizerFast | None = None,
        processor: AutoProcessor | None = None,
        base_model_path: str | None = None,
    ):
        if not self.freq_ctl.check(
            epochs=int(step == self.ft_spec.steps_per_epoch - 1), steps=1
        ):
            return
        path = Saver.get_model_save_path(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
            epoch,
            step,
            global_step,
            name,
        )

        if self._should_use_async(engine):
            self._async_save(engine, path, name, tokenizer, processor)
        else:
            meta = SaveLoadMeta(
                path=path,
                weight_format="hf",
                with_optim=False,
                tokenizer=tokenizer,
                processor=processor,
                base_model_path=base_model_path,
            )
            engine.save(meta)

    def _async_save(
        self,
        engine: TrainEngine | TrainController,
        path: str,
        name: str,
        tokenizer: PreTrainedTokenizerFast | None,
        processor: AutoProcessor | None,
    ):
        """Archon async save."""
        from areal.experimental.engine.archon_engine import ArchonEngine

        assert isinstance(engine, ArchonEngine)

        mgr = self._managers.get(name)
        if mgr is None:
            mgr = AsyncCheckpointManager(AsyncMode.ASYNC)
            self._managers[name] = mgr

        from areal.experimental.engine.archon_checkpoint import save_model_to_hf

        save_model_to_hf(engine, path, tokenizer, processor, async_mgr=mgr)

    def maybe_wait_for_staging(self):
        """Wait for all engines' staging to complete. Call before ppo_update."""
        for mgr in self._managers.values():
            mgr.maybe_wait_for_staging()

    def finalize(self):
        """Training end: wait for last upload + cleanup."""
        for mgr in self._managers.values():
            mgr.finalize()
        self._managers.clear()
