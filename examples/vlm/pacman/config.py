# SPDX-License-Identifier: Apache-2.0

"""Recipe configuration stays outside AReaL's generic engine configuration."""

from dataclasses import dataclass, field, fields

from areal_pacman.configs import PacmanAgentConfig
from areal_pacman.level1.recipe import recipe_contract_metadata
from omegaconf import OmegaConf

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import GenerationHyperparameters, PPOConfig


@dataclass
class PacmanConfig(PacmanAgentConfig):
    curriculum: int = field(default=1, metadata={"help": "Release curriculum: 1 or 2."})

    def __post_init__(self):
        super().__post_init__()
        if self.curriculum not in (1, 2):
            raise ValueError("curriculum must be 1 or 2")
        if self.critic is not None or self.teacher is not None or self.mopd is not None:
            raise ValueError("The Pacman release uses critic-free PPO")
        actor = ModelAllocation.from_str(self.actor.backend)
        parallel = actor.parallel
        if actor.backend != "megatron" or any(
            size != 1
            for size in (
                parallel.data_parallel_size,
                parallel.pipeline_parallel_size,
                parallel.context_parallel_size,
                parallel.expert_parallel_size,
            )
        ):
            raise ValueError(
                "This recipe requires Megatron with DP=PP=CP=EP=1; TP is configurable"
            )
        if ModelAllocation.from_str(self.rollout.backend).backend != "sglang":
            raise ValueError("This recipe uses SGLang rollout")
        if self.ref is None or self.ref.backend != self.actor.backend:
            raise ValueError("Reference must use the same Megatron topology")
        if (
            self.actor.temperature != self.gconfig.temperature
            or self.ref.temperature != self.gconfig.temperature
        ):
            raise ValueError(
                "Behavior, actor and reference policy temperatures must agree"
            )
        if (
            not self.sglang.enable_custom_logit_processor
            or not self.sglang.enable_multimodal
        ):
            raise ValueError(
                "SGLang requires multimodal and custom-logit-processor support"
            )
        if self.sglang.speculative_algorithm is not None:
            raise ValueError("The one-token recipe does not use speculative decoding")
        for generation in (self.gconfig, self.eval_gconfig):
            if (
                generation.n_samples != 12
                or generation.max_new_tokens != 1
                or generation.min_new_tokens != 1
                or generation.greedy
                or generation.temperature != 0.7
                or generation.top_p != 1.0
                or generation.reward_normalization
            ):
                raise ValueError(
                    "Preserve sampled12, one-token, temperature=0.7, top_p=1 decoding"
                )
            if (
                generation.request_plugin is None
                or generation.request_plugin.target
                != "examples.vlm.pacman.policy.TokenSetRequest"
            ):
                raise ValueError(
                    "Both generation configs require the token-set request plugin"
                )
        for engine in (self.actor, self.ref):
            plugin = engine.megatron.policy_distribution
            if (
                plugin is None
                or plugin.target != "examples.vlm.pacman.policy.TokenSetDistribution"
            ):
                raise ValueError(
                    "Actor and reference must use the recorded rollout token support"
                )
            if engine.megatron.bridge_type != "megatron-bridge":
                raise ValueError("Qwen3.5 requires the Megatron Bridge model adapter")
        objective = self.actor.objective_plugin
        if (
            objective is None
            or objective.target != "examples.vlm.pacman.objectives.PacmanObjective"
            or objective.kwargs != {"curriculum": self.curriculum}
        ):
            raise ValueError("The PPO objective must match the selected curriculum")
        direct = self.curriculum == 1
        if (
            self.environment.ghost_mode != ("disabled" if direct else "normal")
            or self.edward_options != (not direct)
            or self.open_action_mask != direct
            or self.reward_objective_contract
            != ("step_local_raw_v1" if direct else "episode_return_group_v1")
        ):
            raise ValueError(
                "Ghost, action and reward contracts must match the curriculum"
            )
        if self.enable_thinking is not False or self.actor.adv_norm is not None:
            raise ValueError(
                "The release disables thinking and advantage normalization"
            )

    def workflow_kwargs(
        self, generation: GenerationHyperparameters, *, training: bool
    ) -> dict:
        # Reuse release field definitions and its audited reward/environment loop.
        # Only the inference/training adapters and resource placement change.
        raw = OmegaConf.to_container(OmegaConf.structured(self), resolve=True)
        excluded = {item.name for item in fields(PPOConfig)}
        options = {key: value for key, value in raw.items() if key not in excluded}
        options.update(
            recipe_contract=recipe_contract_metadata(raw),
            ghost_mode=self.environment.ghost_mode,
            environment_max_steps=self.environment.max_steps,
            tokenizer_path=self.tokenizer_path,
            temperature=generation.temperature,
            top_p=generation.top_p,
            max_tokens=generation.max_tokens,
            max_completion_tokens=generation.max_new_tokens,
            reward_objective_contract=self.reward_objective_contract
            if training
            else "evaluation_only_v1",
        )
        return {
            "gconfig": generation,
            "tokenizer": self.tokenizer_path,
            "options": options,
        }
