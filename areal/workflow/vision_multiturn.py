"""Multi-turn agentic multi-modal RL workflow for vision-language models.

This workflow combines:
1. Multi-modal image processing (via VisionRLVRWorkflow)
2. Multi-turn agentic interactions (via MultiTurnWorkflow)

Designed for scenarios like visual question answering with tool use, visual reasoning
with multi-step agentic interactions, and multi-turn visual problem solving.
"""

import uuid
from collections.abc import Callable
from typing import Any, cast

import torch
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal import workflow_context
from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.image import image2base64
from areal.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_session,
)

logger = logging.getLogger("VisionMultiTurnWorkflow")


class VisionMultiTurnWorkflow(RolloutWorkflow):
    """Multi-turn agentic workflow for vision-language models.

    This workflow enables:
    - Multi-modal image inputs processed via a vision processor
    - Multi-turn conversations with retry on failure
    - Configurable turn limits and turn discounting

    The workflow follows these steps for each episode:
    1. Process images using the provided processor
    2. Run up to max_turns iterations:
       - Generate response from the model
       - Compute reward based on response
       - If reward < 1.0 and turns remain, append failure message and retry
    3. Return training data with turn-discounted rewards
    """

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        processor: AutoProcessor | str,
        max_turns: int = 2,
        turn_discount: float = 0.95,
    ):
        """Initialize the vision multi-turn agentic workflow.

        Parameters
        ----------
        reward_fn : Callable or str
            Reward function that takes (result, answer) or similar. Can be a callable
            or a string import path (e.g., "module.submodule.function_name").
        gconfig : GenerationHyperparameters
            Generation configuration (temperature, max_tokens, etc.).
        tokenizer : PreTrainedTokenizerFast or str
            Tokenizer for encoding/decoding. If string, will load from HF.
        processor : AutoProcessor or str
            Vision processor for handling images. If string, will load from HF.
        max_turns : int, optional
            Maximum number of turns for multi-turn interaction. Default: 2.
        turn_discount : float, optional
            Discount factor applied to reward at each turn. Default: 0.95.
        """

        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if not (0.0 < turn_discount <= 1.0):
            raise ValueError("turn_discount must be in (0, 1].")

        # Load tokenizer
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer

        # Load processor
        if isinstance(processor, str):
            processor = AutoProcessor.from_pretrained(processor)
        self.processor = processor

        # Load reward function
        self.reward_fn = reward_fn
        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)

        # Setup generation config
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer).new(
            n_samples=1
        )
        self.max_turns = max_turns
        self.turn_discount = turn_discount

        # Create tokens that should be amended if the answer is incorrect.
        # This method eliminates the encode-decode inconsistency issue and cancels system prompts.
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "some random message."}],
            }
        ]
        s1 = list(self.tokenizer.apply_chat_template(messages, tokenize=True))
        self.feedback_str = "Your answer is either wrong or not parsable to the reward function. Try to answer it again. The final answer MUST BE put in \\boxed{}."
        messages += [
            {
                "role": "user",
                "content": [{"type": "text", "text": self.feedback_str}],
            }
        ]
        s2 = list(
            self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        )
        self.feedback_str_ids = s2[len(s1) :]

    @trace_session("reward")
    async def _compute_rewards(
        self,
        resp: ModelResponse,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> float:
        """Decode completion and compute reward.

        Parameters
        ----------
        resp : ModelResponse
            Model response with output tokens.
        prompt_str : str
            Decoded prompt string.
        task_data : dict
            Task-specific data (e.g., ground truth answer).

        Returns
        -------
        float
            Reward value.
        """
        completions_str = self.tokenizer.decode(resp.output_tokens)
        reward = await self.async_reward_fn(
            prompt=prompt_str,
            completions=completions_str,
            prompt_ids=resp.input_tokens,
            completion_ids=resp.output_tokens,
            **task_data,
        )
        return reward

    @session_context()
    async def _collect_samples(
        self,
        engine: InferenceEngine,
        req: ModelRequest,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[ModelResponse, float]:
        """Generate one sample and compute its reward.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine for generation.
        req : ModelRequest
            Model request with input IDs and generation config.
        prompt_str : str
            Decoded prompt string for logging.
        task_data : dict
            Task-specific data.

        Returns
        -------
        tuple[ModelResponse, float]
            Model response and reward value.
        """
        async with atrace_session_phase("generate"):
            resp = await engine.agenerate(req)

        reward = await self._compute_rewards(resp, prompt_str, task_data)

        return resp, reward

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Run a single episode with multi-turn agentic interaction.

        This method:
        1. Processes images using the vision processor
        2. Initializes conversation from data
        3. Runs up to max_turns iterations of generate-reward-feedback loop
        4. Returns training data with turn-discounted rewards

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine for generation.
        data : dict
            Input data containing:
            - "messages": Initial conversation messages
            - "images": Images to process (list or single image)
            - "answer": Ground truth answer for reward computation
            - Other fields needed by reward function

        Returns
        -------
        dict[str, torch.Tensor]
            Training data tensors including:
            - input_ids: Full sequence (prompt + responses)
            - loss_mask: Mask for which tokens to train on
            - logprobs: Log probabilities
            - multi_modal_input: Image tensors for vision model
            - attention_mask: Attention mask
            - versions: Token version tracking
            - rewards: Final discounted reward
        """
        # NOTE: load reward function dynamically if given as string
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)
        # Process images via vision processor
        processor_callable = cast(Callable[..., dict[str, Any]], self.processor)

        # Ensure images is in the right format
        images = data.get("images")
        if images is None or (isinstance(images, list) and len(images) == 0):
            raise ValueError(
                "data must contain non-empty 'images' key for multi-modal workflow"
            )

        messages = data.get("messages", [])
        messages_chat = data.get("messages_chat", None)

        processed_input = processor_callable(
            images=images,
            text=messages,
            padding=False,
            return_tensors="pt",
        )

        input_ids: list[int] = processed_input["input_ids"].tolist()[0]
        byte_images = image2base64(images)

        # Initialize tracking for multi-turn interaction
        seq = input_ids.copy()
        logprobs = [0.0] * len(input_ids)
        loss_mask = [0] * len(input_ids)
        versions = [-1] * len(input_ids)
        reward = 0.0
        discount = 1.0
        # Multi-turn interaction loop
        for turn in range(self.max_turns):
            # Prepare model request
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=seq,
                image_data=byte_images,
                vision_msg_vllm=[messages_chat] if messages_chat else None,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
                processor=self.processor,
            )

            prompt_str = self.tokenizer.decode(seq)

            # Generate and collect samples
            resp, turn_reward = await self._collect_samples(
                engine, req, prompt_str, data
            )

            # Amend results
            input_len = len(resp.input_tokens) - len(seq)
            assert len(seq) == 0 or resp.input_tokens[:-input_len] == seq, (
                seq,
                resp.input_tokens[:-input_len],
                len(seq),
                len(resp.input_tokens[:-input_len]),
            )
            seq += resp.input_tokens[-input_len:] + resp.output_tokens
            logprobs += resp.output_logprobs
            loss_mask += [1] * len(resp.output_tokens)
            versions += resp.output_versions

            # Track rewards with discounting
            if turn == 0:
                reward = turn_reward * discount
            else:
                reward = max(reward, turn_reward * discount)

            # If reward is positive or this is the last turn, stop
            if turn_reward >= 1.0 or turn == self.max_turns - 1:
                break

            # Handling missing EOS token before appending feedback
            if (
                resp.output_tokens
                and resp.output_tokens[-1] != self.tokenizer.eos_token_id
            ):
                resp.output_tokens.append(self.tokenizer.eos_token_id)
                seq.append(self.tokenizer.eos_token_id)
                logprobs += [0.0]
                loss_mask += [0]
                versions += [-1]

            if messages_chat:
                output_text = self.tokenizer.decode(resp.output_tokens)
                model_output = {
                    "role": "assistant",
                    "content": [{"type": "text", "text": output_text}],
                }
                messages_chat = messages_chat + [model_output]

            # Append failure feedback for next turn
            seq.extend(self.feedback_str_ids)

            if messages_chat:
                messages_chat = messages_chat + [self.feedback_str]

            logprobs += [0.0] * len(self.feedback_str_ids)
            loss_mask += [0] * len(self.feedback_str_ids)
            versions += [-1] * len(self.feedback_str_ids)

            # Apply discount for next turn
            discount *= self.turn_discount

        # Log stats
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            turn=turn,
            reward=reward,
        )

        # Build multi-modal input for training
        multi_modal_input = [
            {
                "pixel_values": processed_input["pixel_values"],
            }
        ]
        if "image_grid_thw" in processed_input:
            multi_modal_input[0]["image_grid_thw"] = processed_input["image_grid_thw"]

        # Return training data
        return {
            "input_ids": torch.tensor(seq, dtype=torch.int32).unsqueeze(0),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
            "multi_modal_input": multi_modal_input,
            "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            "rewards": torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
        }
