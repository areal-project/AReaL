# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

from areal.api import FinetuneSpec
from areal.api.cli_args import EvaluatorConfig
from areal.utils import timeutil


class Evaluator:
    def __init__(self, config: EvaluatorConfig, ft_spec: FinetuneSpec):
        self.config = config
        self.ft_sepc = ft_spec
        self._eval_before_train_pending = config.eval_before_train
        self.freq_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.freq_epochs,
            freq_step=config.freq_steps,
            freq_sec=config.freq_secs,
        )

    def state_dict(self):
        return self.freq_ctl.state_dict()

    def load_state_dict(self, state_dict):
        # Checkpoints written by the original eval_before_train implementation may
        # still carry a deferred initial trigger. Loading a checkpoint is a resume,
        # so that trigger must not be interpreted as a version-zero evaluation.
        state_dict = {
            **state_dict,
            "epoch": {**state_dict["epoch"], "initial_value": False},
        }
        self.freq_ctl.load_state_dict(state_dict)
        self._eval_before_train_pending = False

    def evaluate_before_train(self, evaluate_fn: Callable[[], None] | None) -> bool:
        """Run the configured initial evaluation without advancing its cadence.

        Passing ``None`` consumes the one-shot opportunity when evaluation inputs
        are unavailable, preventing a later trained model from being mislabeled as
        the initial baseline. A callback failure leaves the opportunity pending.
        """
        if not self._eval_before_train_pending:
            return False
        if evaluate_fn is None:
            self._eval_before_train_pending = False
            return False
        evaluate_fn()
        self._eval_before_train_pending = False
        return True

    def evaluate(
        self,
        evaluate_fn: Callable,
        epoch: int,
        step: int,
        global_step: int,
    ):
        if not self.freq_ctl.check(
            epochs=int(step == self.ft_sepc.steps_per_epoch - 1), steps=1
        ):
            return
        evaluate_fn()
