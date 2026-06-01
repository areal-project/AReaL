from functools import partial
from math import isnan
import os
import re
from typing import Any, List, Dict, Union, Optional

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from deepscaler.globals import (
    ALTERNATIVE_THOUGHT_DELIMITER_END,
    ALTERNATIVE_THOUGHT_DELIMITER_START,
    THOUGHT_DELIMITER_END,
    THOUGHT_DELIMITER_START,
)

from deepscaler.rewards import (
    RewardConfig,
    RewardFn,
    RewardInput,
    RewardOutput,
    RewardType,
)
from deepscaler.rewards.math_utils.utils import (
    extract_answer,
    grade_answer_mathd,
    grade_answer_sympy,
)

special_tokens = {
    "<Parallel>",
    "</Parallel>",
    "<Outlines>",
    "</Outlines>",
    "<Outline>",
    "</Outline>",
    "<Thread>",
    "</Thread>",
    "<Conclusion>",
    "</Conclusion>",
}

optional_stage_tokens = {
    "<Subtask>",
    "<Trial>",
}

def get_token_id(tokenizer: AutoTokenizer, token: str, required: bool = True) -> Optional[int]:
    """Helper to get a single token ID for a special token string."""
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) == 1:
        return token_ids[0]

    print(f"WARNING: Token '{token}' has multiple token IDs: {token_ids}, trying to strip </> wrappers")

    # This is helpful when we run sequential models without additional special tokens.
    token = token.strip("</>")
    token_ids = tokenizer.encode(token, add_special_tokens=False)

    if len(token_ids) == 1:
        return token_ids[0]

    if required:
        raise ValueError(f"Expected single token ID for '{token}', got {token_ids} even after stripping </>")

    return None

def get_special_token_ids(tokenizer: PreTrainedTokenizerBase):
    special_token_ids = {}
    for token in special_tokens:
        token_id = get_token_id(tokenizer, token, required=True)
        special_token_ids[token] = token_id
    for token in optional_stage_tokens:
        token_id = get_token_id(tokenizer, token, required=False)
        if token_id is not None:
            special_token_ids[token] = token_id
    return special_token_ids


def get_parallel_stats(model_response_token_ids: List[int], special_token_ids: Dict[str, int]) -> Dict[str, Any]:
    """
    One-pass parser for parallel stats.

    Rules:
      - </Parallel> only closes when not inside a <Thread>.
      - <Thread> cannot nest: nested <Thread> is literal thread content; </Thread> ends the thread.
      - <Thread> and </Thread> wrappers contribute to the block's sequential part.
      - <Parallel> and </Parallel> are counted as non-block sequential tokens.
    """
    pid_start   = special_token_ids["<Parallel>"]
    pid_end     = special_token_ids["</Parallel>"]
    thread_start  = special_token_ids["<Thread>"]
    thread_end    = special_token_ids["</Thread>"]
    outlines_start = special_token_ids.get("<Outlines>")
    outlines_end = special_token_ids.get("</Outlines>")
    subtask_start = special_token_ids.get("<Subtask>")
    trial_start = special_token_ids.get("<Trial>")

    # Totals for core reward logic
    non_block_seq = 0                     # counted outside any <Parallel> and for boundary tokens
    blocks_seq_total = 0                  # sum over blocks of (sequential_part + sum(thread_lengths))
    blocks_par_total = 0                  # sum over blocks of (sequential_part + max(thread_lengths))

    # State for current block
    in_parallel = False
    in_thread = False
    blk_seq = 0                           # sequential part inside the current block (outside any thread) + wrappers
    blk_threads: List[int] = []           # completed thread lengths
    cur_thread_len = 0                    # length of the current thread (if in_thread)

    # Metric accumulators
    with_parallel = False
    parallel_count = 0
    num_blocks = 0
    thread_counts_per_block: List[int] = [] # Collect thread counts for each block
    subtask_tokens = 0
    trial_tokens = 0

    in_outlines = False
    blk_has_subtask_outline = False
    blk_has_trial_outline = False

    for tok in model_response_token_ids:
        if not in_parallel:
            # Everything is non-block sequential until we see <Parallel>
            non_block_seq += 1
            if tok == pid_start:
                # Track metrics
                with_parallel = True
                parallel_count += 1
                # Start a new block (the <Parallel> token itself is already counted above)
                in_parallel = True
                in_thread = False
                in_outlines = False
                blk_seq = 0
                blk_threads = []
                cur_thread_len = 0
                blk_has_subtask_outline = False
                blk_has_trial_outline = False
            # Continue to next token
            continue

        # We are inside a <Parallel>…</Parallel> block
        if tok == pid_end and not in_thread:
            # This is a complete block. Finalize its stats.
            block_seq_len = blk_seq + sum(blk_threads)
            block_par_len = blk_seq + max(blk_threads, default=0)
            blocks_seq_total += block_seq_len
            blocks_par_total += block_par_len

            # Update metrics for parallel blocks
            num_blocks += 1
            thread_counts_per_block.append(len(blk_threads))
            thread_tokens_in_block = sum(blk_threads)
            if blk_has_subtask_outline and not blk_has_trial_outline:
                subtask_tokens += thread_tokens_in_block
            elif blk_has_trial_outline and not blk_has_subtask_outline:
                trial_tokens += thread_tokens_in_block

            # Count the </Parallel> token as non-block sequential
            non_block_seq += 1

            # Reset block state
            in_parallel = False
            in_thread = False
            in_outlines = False
            # blk_seq, blk_threads, cur_thread_len are reset on next block start
            continue

        if not in_thread and outlines_start is not None and tok == outlines_start:
            in_outlines = True
        elif not in_thread and outlines_end is not None and tok == outlines_end:
            in_outlines = False
        elif in_outlines and subtask_start is not None and tok == subtask_start:
            blk_has_subtask_outline = True
        elif in_outlines and trial_start is not None and tok == trial_start:
            blk_has_trial_outline = True

        # Still inside a block; handle threads and literals
        if tok == thread_start:
            if not in_thread:
                in_thread = True
                cur_thread_len = 0
                blk_seq += 1          # count the <Thread> wrapper as sequential
            else:
                # nested <Thread> is literal content of the current thread
                cur_thread_len += 1
        elif tok == thread_end:
            if in_thread:
                in_thread = False
                blk_threads.append(cur_thread_len)
                cur_thread_len = 0
                blk_seq += 1          # count the </Thread> wrapper as sequential
            else:
                # stray </Thread> outside a thread: treat as sequential content
                blk_seq += 1
        elif tok == pid_end and in_thread:
            # </Parallel> inside a thread is literal thread content
            cur_thread_len += 1
        elif tok == pid_start:
            # Nested <Parallel> inside a block is treated literally
            if in_thread:
                cur_thread_len += 1
            else:
                blk_seq += 1
        else:
            # Regular token inside the block
            if in_thread:
                cur_thread_len += 1
            else:
                blk_seq += 1

    # EOF: flush unterminated block, if any
    if in_parallel:
        if in_thread:
            blk_threads.append(cur_thread_len)
            cur_thread_len = 0
            in_thread = False
        block_seq_len = blk_seq + sum(blk_threads)
        block_par_len = blk_seq + max(blk_threads, default=0)
        blocks_seq_total += block_seq_len
        blocks_par_total += block_par_len

        # Update metrics for parallel blocks
        num_blocks += 1
        thread_counts_per_block.append(len(blk_threads))
        thread_tokens_in_block = sum(blk_threads)
        if blk_has_subtask_outline and not blk_has_trial_outline:
            subtask_tokens += thread_tokens_in_block
        elif blk_has_trial_outline and not blk_has_subtask_outline:
            trial_tokens += thread_tokens_in_block

        # No </Parallel> token to count at EOF

    total_num_tokens = non_block_seq + blocks_seq_total
    ref_total = len(model_response_token_ids)
    assert total_num_tokens == ref_total, (
        f"sequential_lengths {total_num_tokens} != ref_total_num_tokens {ref_total} "
        f"for model_response_token_ids {model_response_token_ids}"
    )

    # Calculate new metrics
    parallel_ratio = (blocks_seq_total / total_num_tokens) if total_num_tokens > 0 else 0.0
    avg_tokens_per_parallel_block = (blocks_seq_total / num_blocks) if num_blocks > 0 else 0.0

    num_tokens_in_the_longest_thread = non_block_seq + blocks_par_total
    acceleration_ratio = (1 - num_tokens_in_the_longest_thread / total_num_tokens) if total_num_tokens > 0 else 0.0
    subtask_ratio = (subtask_tokens / total_num_tokens) if total_num_tokens > 0 else 0.0
    trial_ratio = (trial_tokens / total_num_tokens) if total_num_tokens > 0 else 0.0

    return {
        # Core metrics for reward calculation
        "total_num_tokens": total_num_tokens,
        "num_tokens_in_the_longest_thread": num_tokens_in_the_longest_thread,
        "num_subtask_tokens": subtask_tokens,
        "num_trial_tokens": trial_tokens,
        "subtask_ratio": subtask_ratio,
        "trial_ratio": trial_ratio,
        # Detailed metrics for analysis
        "with_parallel": with_parallel,
        "parallel_count": parallel_count,
        "parallel_ratio": parallel_ratio,
        "acceleration_ratio": acceleration_ratio,
        "avg_tokens_per_parallel_block": avg_tokens_per_parallel_block,
        "thread_counts_per_block": sum(thread_counts_per_block) / len(thread_counts_per_block) if thread_counts_per_block else 0,
        # Reward v2 does not have parallel format errors
        "parallel_format_correct": True,
        "parallel_format_correct_v2": True,
    }

def compute_ratio_reward(
    model_response: str,
    ratio: Optional[float],
    ratio_reward: float,
    ratio_reward_factor: float,
    ratio_clip_max: float,
) -> float:
    """
    Reward proportional to fraction of tokens inside properly paired parallel blocks.
    Returns negative reward if blocks are missing/mismatched/empty.
    """

    # Technically, if `parallel_format_correct_v2` is Enabled, there should not be format errors.
    if ratio is None:
        print(f"WARNING: ratio is None for model_response: {model_response}, treat as 0.")
        ratio = 0.0

    ratio = min(ratio, ratio_clip_max)

    # `ratio_reward_factor` is deprecated. Please set to 1 in config.
    assert ratio_reward * ratio_reward_factor > 0, f"ratio_reward * ratio_reward_factor must be > 0, got {ratio_reward * ratio_reward_factor}"
    return ratio_reward * ratio_reward_factor * ratio

def calculate_reward(config: RewardConfig, model_response: str, correct_lenient: bool, parallel_stats: dict) -> tuple:
    # model_response is for printing debugging info only

    reward = 0
    extra_info = {}

    if correct_lenient:
        correctness_reward = config.correct_reward
    else:
        correctness_reward = config.incorrect_reward

    reward += correctness_reward

    extra_info.update({"correct": correct_lenient, "correct_lenient": correct_lenient, "correctness_reward": correctness_reward})

    acceleration_reward = 0.
    parallel_reward = 0.
    if correct_lenient:
        # Only apply parallel-related rewards/penalties if the answer is correct
        if config.acceleration_ratio_reward > 0.:
            acceleration_reward = compute_ratio_reward(
                model_response=model_response,
                ratio=parallel_stats["acceleration_ratio"],
                ratio_reward=config.acceleration_ratio_reward,
                ratio_reward_factor=config.acceleration_ratio_reward_factor,
                ratio_clip_max=config.acceleration_ratio_clip_max,
            )

        if config.parallel_rewardv2 > 0. and parallel_stats["with_parallel"]:
            parallel_reward = config.parallel_rewardv2

    reward += acceleration_reward
    reward += parallel_reward

    extra_info.update({
        "acceleration_reward": acceleration_reward,
        "parallel_reward": parallel_reward,
    })

    return reward, extra_info

class RewardMathFnv2(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    special_token_ids = None

    def __init__(self, config: RewardConfig):
        super().__init__(config)

        if self.config.require_think_end:
            print("WARNING: require_think_end is deprecated and ignored.")

        assert self.config.second_reward_type == "none", f"second_reward_type must be 'none', got {self.config.second_reward_type}"
        assert self.config.parallel_reward == 0, "parallel_reward is deprecated, please set to 0 in config."

        assert self.config.parallel_ratio_reward == 0., "parallel_ratio_reward is deprecated"
        assert not self.config.parallel_format_error_reward_enabled, "parallel_format_error_reward_enabled is deprecated"
        assert not self.config.parallel_format_error_v2_reward_enabled, "parallel_format_error_v2_reward_enabled is deprecated"

        # allow_immediate_stop is deprecated and ignored (allowed to be set to True or False)

    def __call__(self, input: RewardInput, tokenizer: Optional[PreTrainedTokenizerBase], skip_reward_fn: bool = False, verbose: bool = False) -> RewardOutput:
        """
        Evaluate the input and return the corresponding reward output.

        Args:
            input (RewardInput): The input to be evaluated.
            skip_reward_fn (bool): Whether to skip the reward function. Defaults to False.
        """

        assert (
            input.problem_type == RewardType.MATH
        ), "Invalid problem type: expected 'MATH', but got '{}'".format(
            input.problem_type
        )

        problem = input.problem
        model_response = input.model_response
        model_response_token_ids = input.model_response_token_ids

        model_response = model_response.replace(
            ALTERNATIVE_THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_START
        )
        model_response = model_response.replace(
            ALTERNATIVE_THOUGHT_DELIMITER_END, THOUGHT_DELIMITER_END
        )

        if self.config.require_think_end:
            print("WARNING: require_think_end is deprecated and ignored.")

        ## Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        assert ground_truths is not None, f"Ground truths must be provided. Got: {input.ground_truth}"

        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]

        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)

        ## Calculate correct_lenient
        model_answer_lenient = extract_answer(model_response)
        if self.config.strip_comma_from_answer:
            # If the config is set to strip commas, we do so for lenient checking
            if model_answer_lenient is not None:
                model_answer_lenient = model_answer_lenient.replace(",", "")

        correct_lenient = False

        if model_answer_lenient is not None:
            for ground_truth in processed_ground_truths:
                is_correct = grade_answer_mathd(
                    model_answer_lenient, ground_truth
                ) or grade_answer_sympy(model_answer_lenient, ground_truth)
                if is_correct:
                    correct_lenient = True
                    break

        if self.special_token_ids is None:
            self.special_token_ids = get_special_token_ids(tokenizer)

        ## Parse the model response to extract parallel statistics
        if model_response_token_ids is None:
            model_response_token_ids = tokenizer.encode(model_response, add_special_tokens=False)

        parallel_stats = get_parallel_stats(model_response_token_ids, self.special_token_ids)
        reward, extra_info = calculate_reward(self.config, model_response, correct_lenient, parallel_stats)

        return RewardOutput(
            is_correct=correct_lenient,
            reward=reward,
            second_reward=0.,
            extra_info={
                **parallel_stats,
                **extra_info,
            }
        )


def deepscaler_reward_fn(
    solution_str: str,
    ground_truth: Union[str, List[str]],
    config: Any,
    correctness_as_reward: bool,
    skip_reward_fn: bool = False,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    verbose: Optional[int] = None
):
    """
    Dispatcher function for reward computation.

    This function routes to the V2 reward implementation. V1 is deprecated.

    Args:
        solution_str: The model's solution string
        ground_truth: The correct answer(s)
        config: Reward configuration
        correctness_as_reward: If True, return only correctness (0 or 1) as reward
        skip_reward_fn: If True, skip reward computation (used for timeouts)
        tokenizer: Optional tokenizer for token counting
        verbose: Verbosity level

    Returns:
        Tuple of (reward_dict, extra_info_dict)
    """
    reward_config = config if isinstance(config, RewardConfig) else RewardConfig(**config)
    if verbose is None:
        verbose = reward_config.verbose if reward_config.verbose is not None else True
    if "REWARD_VERSION" in os.environ:
        reward_config.version = os.environ["REWARD_VERSION"]
    if reward_config.version is None:
        reward_config.version = "v2"
        print("Warning: reward_config.version is not set. Defaulting to 'v2'.")

    if reward_config.version == "v1":
        raise NotImplementedError("Reward function v1 is deprecated. Please use v2.")
    elif reward_config.version != "v2":
        raise ValueError(f"Unknown reward config version: {reward_config.version}")

    reward_fn = RewardMathFnv2(reward_config)
    reward_response = reward_fn(
        RewardInput(
            problem=solution_str,
            problem_type=RewardType.MATH,
            model_response=solution_str,
            ground_truth={"answer": ground_truth},
        ),
        tokenizer=tokenizer,
        verbose=verbose,
        skip_reward_fn=skip_reward_fn,
    )
    extra_info = {
        "correct": reward_response.is_correct,
        "ground_truth": ground_truth,
        **reward_response.extra_info
    }
    if correctness_as_reward:
        return {"reward": reward_response.is_correct, "second_reward": 0.}, extra_info
    else:
        return {"reward": reward_response.reward, "second_reward": reward_response.second_reward}, extra_info
