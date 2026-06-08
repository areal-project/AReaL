"""
ScaffoldingWorkflow - RolloutWorkflow with generation and reward via Scaffolding.

Architecture
------------
- Generation: via scaffolding Worker (SGLangWorker calls SGLang OpenAI API)
- Reward: via scaffolding RLVRRewardController
- Logprobs: placeholder (0.0) since recompute_logprob=true in training config
  causes the actor to recompute exact logprobs during PPO update.
- Worker & ScaffoldingLlm: lazily created from engine server addresses,
  exposed for subclasses (e.g., multi-turn workflows).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging
from areal.utils.dynamic_import import import_from_string
from areal.utils.hf_utils import apply_chat_template

from ._compat import (
    NativeGenerationController,
    ScaffoldingLlm,
)
from .controllers import (
    PipelineTrajectoryMaker,
    RLVRRewardController,
)
from .worker import SGLangWorker

logger = logging.getLogger("ScaffoldingWorkflow")


class ScaffoldingWorkflow(RolloutWorkflow):
    """RolloutWorkflow with generation and reward via scaffolding components.

    Both generation and reward computation go through scaffolding:
    - Generation: SGLangWorker calls SGLang's OpenAI-compatible completions API
    - Reward: RLVRRewardController computes verifiable rewards

    Since the OpenAI API does not return per-token logprobs in AReaL's format,
    placeholder logprobs are used. Set ``recompute_logprob: true`` in the actor
    config so the training engine recomputes exact logprobs during PPO update.

    Parameters
    ----------
    reward_fn : Callable | str
        The reward function, or an importable string path.
    gconfig : GenerationHyperparameters
        Generation hyperparameters.
    tokenizer : PreTrainedTokenizerFast | str
        Tokenizer or path to load it.
    enable_thinking : bool
        Whether to enable thinking tokens.
    """

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
    ):
        if isinstance(reward_fn, str):
            reward_fn = import_from_string(reward_fn)
        self.reward_fn = reward_fn

        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            self.tokenizer = load_hf_tokenizer(self.tokenizer)
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking

        # Lazily created from engine server addresses via build_scaffolding_llm
        self.worker: SGLangWorker | None = None
        self.gen_controller: NativeGenerationController | None = None
        self.reward_controller: RLVRRewardController | None = None
        self.trajectory_maker: PipelineTrajectoryMaker | None = None
        self.scaffolding_llm: ScaffoldingLlm | None = None

    def _lazy_init_scaffolding(self, engine: InferenceEngine) -> None:
        """Create Worker, PipelineTrajectoryMaker, and ScaffoldingLlm."""
        import openai

        addr = engine.addresses[0]
        base_url = f"http://{addr}"
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        async_client = openai.AsyncOpenAI(base_url=base_url, api_key="EMPTY")
        self.worker = SGLangWorker(
            async_client=async_client, model="default", engine=engine
        )

        self.scaffolding_llm = self.build_scaffolding_llm(engine)
        logger.info("Initialized scaffolding components with server at %s", addr)

    def build_scaffolding_llm(self, engine: InferenceEngine) -> ScaffoldingLlm:
        """Build the ScaffoldingLlm instance.

        Override this method in subclasses to use different scaffolding
        controllers or worker configurations. Subclasses should set
        ``self.gen_controller`` and ``self.reward_controller`` here.

        When this method is called, ``self.worker`` is already initialized.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine (available for address lookup if needed).

        Returns
        -------
        ScaffoldingLlm
            The constructed ScaffoldingLlm instance.
        """
        # Convert gconfig to sampling params for NativeGenerationController
        stop_strings = []
        if self.gconfig.stop_token_ids:
            for tid in self.gconfig.stop_token_ids:
                decoded = self.tokenizer.decode([tid])
                if decoded:
                    stop_strings.append(decoded)

        sampling_params = {
            "max_tokens": self.gconfig.max_new_tokens,
            "temperature": self.gconfig.temperature or 1.0,
        }
        if stop_strings:
            sampling_params["stop"] = stop_strings

        self.gen_controller = NativeGenerationController(
            sampling_params=sampling_params
        )
        self.reward_controller = RLVRRewardController(self.reward_fn)
        self.trajectory_maker = PipelineTrajectoryMaker(
            self.gen_controller, self.reward_controller
        )
        return ScaffoldingLlm(
            self.trajectory_maker,
            {NativeGenerationController.WorkerTag.GENERATION: self.worker},
        )

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Run a single episode via scaffolding pipeline.

        Delegates the full episode (generation + reward) to
        ``self.scaffolding_llm``, which wraps a ``PipelineTrajectoryMaker``.
        The result is an ``InteractionWithTokenLogpReward`` whose
        ``to_tensor_dict()`` produces the training tensors.

        Note: logprobs are placeholders (0.0). Set ``recompute_logprob: true``
        in actor config so the training engine computes exact logprobs.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine (used for server addresses on first call).
        data : dict[str, Any]
            Input data containing messages and ground truth.

        Returns
        -------
        dict[str, torch.Tensor]
            Trajectory tensors for PPO training.
        """
        if self.worker is None:
            self._lazy_init_scaffolding(engine)

        # Tokenize prompt
        input_ids = apply_chat_template(
            self.tokenizer,
            data["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        input_ids = list(input_ids)
        prompt_str = self.tokenizer.decode(input_ids)

        # Configure per-episode data on trajectory maker
        # (clone() in scaffolding_llm will deep-copy these)
        self.trajectory_maker.task_data = data
        self.trajectory_maker.prompt_str = prompt_str
        self.trajectory_maker.input_tokens = input_ids

        # Run full pipeline via scaffolding_llm
        result = self.scaffolding_llm.generate_async(prompt_str)
        await result

        # Extract interaction and convert to tensor dict
        scaffolding_output = result.outputs[0]
        interactions = scaffolding_output.data
        interaction = next(iter(interactions.values()))

        # If output_tokens is missing (e.g., SGLang didn't return token_ids),
        # tokenize the output_str as a fallback so loss_mask is non-zero.
        resp = interaction.model_response
        if resp is not None and not resp.output_tokens:
            output_text = scaffolding_output.text or ""
            output_tokens = self.tokenizer.encode(output_text, add_special_tokens=False)
            resp.output_tokens = output_tokens
            resp.output_logprobs = [0.0] * len(output_tokens)
            resp.output_versions = [-1] * len(output_tokens)

        return interaction.to_tensor_dict()
