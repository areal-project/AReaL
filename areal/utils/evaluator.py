# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

from areal.api import FinetuneSpec
from areal.api.cli_args import EvaluatorConfig
from areal.utils import timeutil


class Evaluator:
    def __init__(self, config: EvaluatorConfig, ft_spec: FinetuneSpec):
        self.config = config
        self.ft_sepc = ft_spec
        self.freq_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.freq_epochs,
            freq_step=config.freq_steps,
            freq_sec=config.freq_secs,
            initial_epoch_value=config.eval_before_train,
        )

    def state_dict(self):
        return self.freq_ctl.state_dict()

    def load_state_dict(self, state_dict):
        self.freq_ctl.load_state_dict(state_dict)

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
