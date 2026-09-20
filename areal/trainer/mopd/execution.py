# SPDX-License-Identifier: Apache-2.0

"""Static execution planning for optional MOPD objectives."""

from __future__ import annotations

from dataclasses import dataclass

from areal.api.cli_args import PPOConfig


@dataclass(frozen=True)
class MOPDExecutionPlan:
    """Infrastructure and objective phases required by configured coefficients."""

    requires_teacher_scoring: bool
    requires_rl: bool
    requires_critic: bool
    requires_ref: bool
    requires_prox_logp: bool

    @classmethod
    def from_config(cls, config: PPOConfig) -> MOPDExecutionPlan | None:
        """Derive one immutable plan before any engines are initialized."""
        if config.mopd is None:
            return None
        requires_rl = config.mopd.loss.rl_coefficient > 0
        requires_distillation_filter = (
            config.mopd.loss.distillation_coefficient > 0
            and (
                config.actor.m2_threshold is not None
                or config.actor.rejection_sampling is not None
            )
        )
        return cls(
            requires_teacher_scoring=(config.mopd.loss.distillation_coefficient > 0),
            requires_rl=requires_rl,
            requires_critic=requires_rl and config.critic is not None,
            requires_ref=(
                requires_rl and config.actor.kl_ctl > 0 and config.ref is not None
            ),
            requires_prox_logp=(
                (requires_rl or requires_distillation_filter)
                and config.actor.should_compute_prox_logp()
            ),
        )
