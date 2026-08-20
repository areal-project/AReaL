# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import inspect
import json
import os
import pickle
from typing import TYPE_CHECKING, Any

import torch.distributed as dist
from transformers import PreTrainedTokenizerFast

if TYPE_CHECKING:
    from transformers import AutoProcessor

from areal.api import (
    FinetuneSpec,
    InferenceEngine,
    SaveLoadMeta,
    StepInfo,
    TrainEngine,
    WeightUpdateMeta,
)
from areal.api.cli_args import RecoverConfig
from areal.infra import TrainController
from areal.utils import checkpoint_pointer as cp
from areal.utils import logging, timeutil
from areal.utils.environ import is_single_controller
from areal.utils.evaluator import Evaluator
from areal.utils.saver import Saver

if TYPE_CHECKING:
    from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger("Recover")


class InValidRecoverInfo(Exception):
    pass


@dataclasses.dataclass
class RecoverInfo:
    # Last step info is the counter of the saved checkpoint.
    # Recover will start from the next iteration, obtained by `last_step_info.next()`.
    last_step_info: StepInfo

    saver_info: dict
    evaluator_info: dict
    stats_logger_info: dict
    dataloader_info: dict | list[dict]
    checkpoint_info: dict

    def dump(self, dump_dir: str):
        # Dumps the recover info to multiple files in `dump_dir`:
        # 1. step_info.json: contains the recover info
        # 2. *_info.json or *_info.pkl: contains other informantion required for recover.

        if dist.is_initialized():
            # Since dataloader state is different across distributed ranks,
            # we need to all gather the dataloader state from all ranks.
            # In this situation, saved dataloader_info is a list of states from all ranks.
            dataloader_info = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(dataloader_info, self.dataloader_info)

            # To avoid contention, do not dump on multiple ranks
            if dist.get_rank() != 0:
                return
        else:
            dataloader_info = self.dataloader_info

        os.makedirs(dump_dir, exist_ok=True)
        step_info_path = os.path.join(dump_dir, "step_info.json")
        with open(step_info_path, "w") as f:
            json.dump(dataclasses.asdict(self.last_step_info), f, indent=4)

        saver_info_path = os.path.join(dump_dir, "saver_info.json")
        with open(saver_info_path, "w") as f:
            json.dump(self.saver_info, f, indent=4)

        evaluator_info_path = os.path.join(dump_dir, "evaluator_info.json")
        with open(evaluator_info_path, "w") as f:
            json.dump(self.evaluator_info, f, indent=4)

        stats_logger_info_path = os.path.join(dump_dir, "stats_logger_info.json")
        with open(stats_logger_info_path, "w") as f:
            json.dump(self.stats_logger_info, f, indent=4)

        checkpoint_info_path = os.path.join(dump_dir, "checkpoint_info.json")
        with open(checkpoint_info_path, "w") as f:
            json.dump(self.checkpoint_info, f, indent=4)

        dataloader_info_path = os.path.join(dump_dir, "dataloader_info.pkl")
        with open(dataloader_info_path, "wb") as f:
            pickle.dump(dataloader_info, f)

    @classmethod
    def load(cls, load_dir: str):
        # Loads the recover info from multiple files in `load_dir`:
        if not os.path.exists(load_dir):
            raise FileNotFoundError(
                f"Recover info directory {load_dir} does not exist."
            )

        try:
            step_info_path = os.path.join(load_dir, "step_info.json")
            with open(step_info_path) as f:
                step_info_dict = json.load(f)
                last_step_info = StepInfo(**step_info_dict)

            evaluator_info_path = os.path.join(load_dir, "evaluator_info.json")
            with open(evaluator_info_path) as f:
                evaluator_info = json.load(f)

            saver_info_path = os.path.join(load_dir, "saver_info.json")
            with open(saver_info_path) as f:
                saver_info = json.load(f)

            stats_logger_info_path = os.path.join(load_dir, "stats_logger_info.json")
            with open(stats_logger_info_path) as f:
                stats_logger_info = json.load(f)

            checkpoint_info_path = os.path.join(load_dir, "checkpoint_info.json")
            with open(checkpoint_info_path) as f:
                checkpoint_info = json.load(f)

            dataloader_info_path = os.path.join(load_dir, "dataloader_info.pkl")
            with open(dataloader_info_path, "rb") as f:
                dataloader_info = pickle.load(f)
                if isinstance(dataloader_info, list):
                    # If dataloader_info a list, it means it is saved from a distributed run.
                    if dist.is_initialized():
                        # Loading dataloader states in a distributed context.
                        assert dist.get_world_size() == len(dataloader_info), (
                            f"Dataloader info list length {len(dataloader_info)} does not match "
                            f"the world size {dist.get_world_size()}."
                        )
                        dataloader_info = dataloader_info[dist.get_rank()]

            return cls(
                last_step_info=last_step_info,
                saver_info=saver_info,
                evaluator_info=evaluator_info,
                stats_logger_info=stats_logger_info,
                dataloader_info=dataloader_info,
                checkpoint_info=checkpoint_info,
            )
        except Exception as e:
            logger.error(f"Failed to load recover info from {load_dir}: {e}")
            raise InValidRecoverInfo(f"Invalid recover info in {load_dir}") from e


class RecoverHandler:
    def __init__(self, config: RecoverConfig, ft_spec: FinetuneSpec):
        self.config = config
        self.ft_spec = ft_spec
        self.last_step_info = StepInfo(
            epoch=-1,
            epoch_step=-1,
            global_step=-1,
            steps_per_epoch=ft_spec.steps_per_epoch,
        )
        self.freq_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.freq_epochs,
            freq_step=config.freq_steps,
            freq_sec=config.freq_secs,
        )

    @staticmethod
    def recover_info_path(
        experiment_name: str,
        trial_name: str,
        fileroot: str,
    ):
        return os.path.join(
            Saver.get_save_root(experiment_name, trial_name, fileroot),
            "recover_info",
        )

    @staticmethod
    def _is_gateway_train_controller(
        engine: TrainEngine
        | TrainController
        | dict[str, TrainEngine | TrainController],
    ) -> bool:
        from areal.v2.training_service.controller.controller import (
            GatewayTrainController,
        )

        if isinstance(engine, GatewayTrainController):
            return True
        if isinstance(engine, dict):
            return any(
                isinstance(controller, GatewayTrainController)
                for controller in engine.values()
            )
        return False

    def _ensure_recover_supported(
        self,
        engine: TrainEngine
        | TrainController
        | dict[str, TrainEngine | TrainController],
    ) -> None:
        if self._is_gateway_train_controller(engine):
            raise NotImplementedError(
                "Recovery is not supported with GatewayTrainController "
                '(`_version="v2"`) yet. Disable `recover.mode` or use '
                '`_version="v1"`.'
            )

    @staticmethod
    def _normalize_recover_engines(
        engine: TrainEngine
        | TrainController
        | dict[str, TrainEngine | TrainController],
    ) -> dict[str, TrainEngine | TrainController]:
        if isinstance(engine, dict):
            return engine
        return {"default": engine}

    @staticmethod
    def _supports_checkpoint_pointer() -> bool:
        multi_rank = dist.is_initialized() and dist.get_world_size() > 1
        return is_single_controller() and not multi_rank

    @staticmethod
    def _should_run_awex_colocate_transfer(
        inference_engine: InferenceEngine | None,
        weight_update_meta: WeightUpdateMeta | None,
        colocated_rollout: bool,
    ) -> bool:
        """Whether recovery must drive the AWEX colocate pre-transfer sequence.

        The transport type alone is not enough: v2 selects AWEX for every
        non-LoRA run regardless of placement, so the caller has to state whether
        actor and rollout physically share devices.
        """
        return (
            inference_engine is not None
            and getattr(weight_update_meta, "type", None) == "awex"
            and colocated_rollout
        )

    @staticmethod
    def _require_colocate_rollout_protocol(
        inference_engine: InferenceEngine,
    ) -> None:
        missing = []
        if not callable(getattr(inference_engine, "pause_generation_sync", None)):
            missing.append("pause_generation_sync()")

        offload = getattr(inference_engine, "offload", None)
        if not callable(offload):
            missing.append("offload(tags=...)")
        else:
            try:
                accepts_tags = "tags" in inspect.signature(offload).parameters
            except (TypeError, ValueError):
                accepts_tags = True
            if not accepts_tags:
                missing.append("offload(tags=...)")

        if missing:
            raise NotImplementedError(
                "Colocated AWEX recovery needs a rollout engine implementing "
                f"{', '.join(missing)}, which {type(inference_engine).__name__} "
                "does not provide. Disable `recover.mode` or run this "
                "configuration without actor-rollout colocation."
            )

    @staticmethod
    def _uses_async_checkpoint(
        engine: TrainEngine | TrainController,
    ) -> bool:
        config = getattr(engine, "config", None)
        backend = getattr(config, "backend", "")
        megatron_config = getattr(config, "megatron", None)
        return (
            isinstance(backend, str)
            and backend.split(":", 1)[0] == "megatron"
            and bool(getattr(megatron_config, "async_save", False))
        )

    def dump(
        self,
        engine: TrainEngine
        | TrainController
        | dict[str, TrainEngine | TrainController],
        step_info: StepInfo,
        saver: Saver,
        evaluator: Evaluator,
        stats_logger: StatsLogger,
        dataloader: Any,
        tokenizer: PreTrainedTokenizerFast | None = None,
        processor: AutoProcessor | None = None,
        base_model_path: str | None = None,
    ):
        if self.config.mode in ("disabled", "off"):
            return
        self._ensure_recover_supported(engine)
        # currently only support recover on one engine
        if not self.freq_ctl.check(
            epochs=int(step_info.epoch_step == self.ft_spec.steps_per_epoch - 1),
            steps=1,
        ):
            return
        normalized_engine: dict[str, TrainEngine | TrainController] = (
            self._normalize_recover_engines(engine)
        )
        self.last_step_info = step_info
        recover_info = RecoverInfo(
            last_step_info=self.last_step_info,
            saver_info=saver.state_dict(),
            evaluator_info=evaluator.state_dict(),
            stats_logger_info=stats_logger.state_dict(),
            dataloader_info=dataloader.state_dict(),
            checkpoint_info=self.freq_ctl.state_dict(),
        )
        save_root = Saver.get_save_root(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
        )

        if not self._supports_checkpoint_pointer():
            if cp.read_latest(save_root) is not None:
                raise cp.CheckpointConsistencyError(
                    "Cannot write a legacy recovery checkpoint while LATEST "
                    "selects a transactional checkpoint generation"
                )
            for name, engine_ in normalized_engine.items():
                self._save_checkpoint(
                    engine_,
                    path=Saver.get_recover_checkpoint_path(
                        self.config.experiment_name,
                        self.config.trial_name,
                        self.config.fileroot,
                        name=name,
                    ),
                    name=name,
                    tokenizer=tokenizer,
                    processor=processor,
                    base_model_path=base_model_path,
                )
            recover_info.dump(
                self.recover_info_path(
                    self.config.experiment_name,
                    self.config.trial_name,
                    self.config.fileroot,
                )
            )
            return

        engine_names = list(normalized_engine)
        async_engines = [
            name
            for name, engine_ in normalized_engine.items()
            if self._uses_async_checkpoint(engine_)
        ]
        # TODO(agent): coordinate per-engine finalize callbacks before allowing
        # an asynchronous generation containing both actor and critic payloads.
        if async_engines and len(normalized_engine) != 1:
            raise NotImplementedError(
                "Crash-safe asynchronous recovery currently supports one train "
                "engine. Disable Megatron async_save when saving multiple engines."
            )

        generation, pointer_record = cp.prepare_generation(
            save_root, step_info.global_step, engine_names
        )

        recover_info.dump(cp.manifest_dir(generation))

        pointer_value = pointer_record.to_json()
        for name, engine_ in normalized_engine.items():
            self._save_checkpoint(
                engine_,
                path=cp.payload_dir(generation, name),
                name=name,
                tokenizer=tokenizer,
                processor=processor,
                base_model_path=base_model_path,
                checkpoint_pointer_path=(
                    cp.latest_path(save_root) if name in async_engines else None
                ),
                checkpoint_pointer_value=(
                    pointer_value if name in async_engines else None
                ),
            )

        if async_engines:
            logger.info(
                "Checkpoint generation %s will be published after Megatron "
                "async finalize completes",
                generation,
            )
        else:
            cp.publish_latest(save_root, pointer_value)
            logger.info(
                "Published recovery checkpoint generation %s at step %s",
                generation,
                step_info.global_step,
            )

    def load(
        self,
        engine: TrainEngine | dict[str, TrainEngine] | TrainController,
        saver: Saver,
        evaluator: Evaluator,
        stats_logger: StatsLogger,
        dataloader: Any,
        inference_engine: InferenceEngine | None = None,
        weight_update_meta: WeightUpdateMeta | None = None,
        inference_engine_update_from: str = "default",
        colocated_rollout: bool = False,
    ) -> RecoverInfo | None:
        if self.config.mode in ("disabled", "off"):
            return
        self._ensure_recover_supported(engine)
        if inference_engine is not None and weight_update_meta is None:
            raise ValueError("Weight update meta is required for recovery.")

        # TODO(agent): GatewayTrainController is currently duck-typed and does
        # not satisfy this TrainController type check. Extend recovery to accept
        # controller-v2 instances (or make v2 inherit TrainController) before
        # relying on resumed runs with `_version="v2"`.
        normalized_engine: dict[str, TrainEngine | TrainController] = (
            self._normalize_recover_engines(engine)
        )

        save_root = Saver.get_save_root(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
        )
        source = cp.resolve_checkpoint(save_root, list(normalized_engine))
        if source is None:
            logger.warning(
                f"Resume info not found under {save_root}. "
                f"This should not be a resumed experiment!"
            )
            return None
        logger.info(f"Loading recover info from {source.manifest}")
        try:
            recover_info: RecoverInfo = RecoverInfo.load(source.manifest)
            logger.info(
                f"Recovering from {recover_info.last_step_info.next()} using "
                f"{source.label}."
            )
            saver.load_state_dict(recover_info.saver_info)
            self.freq_ctl.load_state_dict(recover_info.checkpoint_info)
            evaluator.load_state_dict(recover_info.evaluator_info)
            stats_logger.load_state_dict(recover_info.stats_logger_info)
            dataloader.load_state_dict(recover_info.dataloader_info)

            global_step = recover_info.last_step_info.global_step
            recovery_version = global_step + 1

            is_awex_colocate = self._should_run_awex_colocate_transfer(
                inference_engine=inference_engine,
                weight_update_meta=weight_update_meta,
                colocated_rollout=colocated_rollout,
            )
            if is_awex_colocate:
                self._require_colocate_rollout_protocol(inference_engine)

            if not is_awex_colocate:
                for name, engine_ in normalized_engine.items():
                    self._load_checkpoint(
                        engine_, path=source.payloads[name], name=name
                    )

            if inference_engine is not None:
                assert weight_update_meta is not None
                update_engine = normalized_engine[inference_engine_update_from]
                versioned_meta = weight_update_meta.with_version(recovery_version)
                update_engine.connect_engine(inference_engine, versioned_meta)
                inference_engine.pause()
                try:
                    # AWEX colocate transfer requires the full engine-level
                    # pause/offload protocol, not just the controller pause. The
                    # sglang plugin's patched event loop only drains the weight-
                    # update queue while scheduler._engine_paused is True (set by
                    # pause_generation), and the reader-side protocol expects the
                    # engine's kv/weights released before the writer publishes.
                    # Without this the recover-path transfer deadlocks: reader
                    # never consumes the queued version marker, writer blocks on
                    # weights_update_finished forever.
                    # Mirror of the trainer's pre-update sequence; the reverse
                    # side (kv_cache onload) happens inside update_weights.
                    if is_awex_colocate:
                        inference_engine.pause_generation_sync()
                        inference_engine.offload(tags=["kv_cache"])
                        inference_engine.offload(tags=["weights"])
                        # Load the actor checkpoint only after the colocated
                        # rollout engine has released its GPU memory; loading
                        # first would stack DCP weights/optimizer on top of the
                        # still-resident sglang allocation and risk OOM.
                        for name, engine_ in normalized_engine.items():
                            self._load_checkpoint(
                                engine_, path=source.payloads[name], name=name
                            )
                    update_engine.update_weights(versioned_meta)
                finally:
                    # Always resume: leaving rollout paused after a failed
                    # checkpoint load or transfer would hang every later step.
                    inference_engine.resume()
                update_engine.set_version(recovery_version)
                inference_engine.set_version(recovery_version)
            return recover_info
        except (FileNotFoundError, InValidRecoverInfo) as e:
            if source.transactional:
                raise cp.CheckpointConsistencyError(
                    f"Published checkpoint {source.label} is not loadable: {e}"
                ) from e
            logger.warning(
                f"Resume info not found at {source.manifest}. "
                f"This should not be a resumed experiment!"
            )

    def _save_checkpoint(
        self,
        engine: TrainEngine,
        path: str,
        name: str = "default",
        tokenizer: PreTrainedTokenizerFast | None = None,
        processor: AutoProcessor | None = None,
        base_model_path: str | None = None,
        checkpoint_pointer_path: str | None = None,
        checkpoint_pointer_value: str | None = None,
    ):
        weight_format = "dcp"
        with_optim = not self.config.no_save_optim
        meta = SaveLoadMeta(
            path=path,
            weight_format=weight_format,
            with_optim=with_optim,
            tokenizer=tokenizer,
            processor=processor,
            base_model_path=base_model_path,
            checkpoint_pointer_path=checkpoint_pointer_path,
            checkpoint_pointer_value=checkpoint_pointer_value,
        )
        engine.save(meta)
        logger.info(f"Saved recover checkpoint to {path} (with_optim={with_optim})")

    def _load_checkpoint(
        self,
        engine: TrainEngine | TrainController,
        path: str,
        name: str = "default",
        tokenizer: PreTrainedTokenizerFast | None = None,
        base_model_path: str | None = None,
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint path {path} does not exist.")
        weight_format = "dcp"
        with_optim = not self.config.no_load_optim
        meta = SaveLoadMeta(
            path=path,
            weight_format=weight_format,
            with_optim=with_optim,
            tokenizer=None,
            processor=None,
            base_model_path=None,
        )
        engine.load(meta)


def check_if_auto_recover(config: RecoverConfig) -> bool:
    # This method is called by check_if_recover to check if the experiment should
    # recover from a previous run when recovery is enabled ("on" or "auto" mode).
    save_root = Saver.get_save_root(
        config.experiment_name, config.trial_name, config.fileroot
    )
    logger.info(f"Searching for recovery checkpoint under {save_root}.")
    source = cp.resolve_checkpoint(save_root, None)
    if source is None:
        logger.warning(f"Recover info not found under: {save_root}")
        return False
    try:
        info = RecoverInfo.load(source.manifest)
    except Exception as e:
        if source.transactional:
            raise cp.CheckpointConsistencyError(
                f"Published checkpoint {source.label} is not loadable: {e}"
            ) from e
        logger.warning(f"Failed to load recover info from {source.manifest}: {e}")
        return False
    if info.last_step_info.epoch < 0:
        logger.warning(
            "Recover checkpoint is not valid. Expected last_step_info.epoch "
            f">= 0, but found {info.last_step_info.epoch}"
        )
        return False
    return True


def check_if_recover(config: RecoverConfig, _run_id: int) -> bool:
    """Check if the experiment should be a recover run.

    When recovery is enabled ('on' or 'auto'), this checks if valid recover
    info and checkpoints are available for automatic recovery.

    Args:
        config: Recovery configuration.
        _run_id: Unused. Kept for API compatibility.

    Returns:
        True if the experiment should recover from a previous run.
    """
    if config.mode in ("disabled", "off"):
        return False
    # Both "on" and "auto" use auto-recovery behavior
    return check_if_auto_recover(config)
