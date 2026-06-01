import argparse
import json
import os
import re
import sys
import time
import warnings
from typing import List, Optional

from openai import OpenAI
from rewards import get_parallel_stats, get_special_token_ids
from termcolor import colored
from transformers import AutoTokenizer
from utils import popen_launch_server, terminate_process


INLINE_PROBLEM_NAME = "inline_problem"
INLINE_PROBLEM = """Replace this string with the problem you want to debug.

Example:
Solve for x: 2x + 3 = 11.
"""


parser = argparse.ArgumentParser(
    description="Single-problem pause debug runner with inline prompt input."
)
parser.add_argument(
    "--model_name", type=str, required=True, help="Path to the model to evaluate"
)
parser.add_argument(
    "--launch_server", action="store_true", help="Whether to launch the model server"
)
parser.add_argument(
    "--verbose",
    type=int,
    default=1,
    help="Verbosity level. Higher value means more output",
)
parser.add_argument(
    "--timeout", type=int, default=600, help="Timeout for the server in seconds"
)
parser.add_argument(
    "--port",
    type=int,
    default=None,
    help="Port for the OpenAI API server. If not specified, a random port will be used.",
)
parser.add_argument(
    "--template-type",
    choices=["model"],
    required=True,
    help="Template type for the prompts",
)
parser.add_argument(
    "--suffix",
    type=str,
    default="",
    help="Suffix to append to the model name for saving results",
)
parser.add_argument(
    "--no-stop-at-eos",
    action="store_true",
    help="If set, the model will not stop at EOS token.",
)
parser.add_argument(
    "--skip-model-check",
    action="store_true",
    help="If set, skip the model availability check.",
)
parser.add_argument(
    "--tp",
    type=int,
    default=None,
    help="Tensor parallelism size. If not specified, will use auto configuration based on model size.",
)
parser.add_argument(
    "--dp",
    type=int,
    default=None,
    help="Data parallelism size. If not specified, will use auto configuration based on available GPUs.",
)
parser.add_argument(
    "--bfloat16",
    action="store_true",
    help="Use bfloat16 precision for model inference.",
)
parser.add_argument(
    "--no-terminate-on-exit",
    action="store_true",
    help="If set, the model server will not be terminated on exit.",
)
parser.add_argument(
    "--skip-actual-launch",
    action="store_true",
    help="If set, the model server will not be actually launched.",
)
parser.add_argument(
    "--use-os-system",
    action="store_true",
    help="If set, use os.system to launch the server instead of subprocess.Popen.",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.6,
    help="Temperature for sampling. Default is 0.6.",
)
parser.add_argument(
    "--top-p",
    type=float,
    default=0.95,
    help="Top-p sampling parameter. Default is 0.95.",
)
parser.add_argument(
    "--wait-before-health-check",
    type=int,
    default=0,
    help="Wait time in seconds before performing the health check after launching the server.",
)
parser.add_argument(
    "--branching-generate",
    action="store_true",
    help="If set, use branching generation instead of standard generation.",
)
parser.add_argument(
    "--max-context-length",
    type=int,
    default=32768,
    help="Maximum context length for the model. Please set to 40k for Qwen3 if needed.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="If set, overwrite the existing results file.",
)
parser.add_argument(
    "--pause-at-longest-thread-tokens",
    type=int,
    default=None,
    help="If set, branching generation stops as soon as num_tokens_in_the_longest_thread reaches this value.",
)
args = parser.parse_args()

if (
    args.pause_at_longest_thread_tokens is not None
    and args.pause_at_longest_thread_tokens < 0
):
    raise ValueError(
        "--pause-at-longest-thread-tokens must be a non-negative integer when set."
    )

openai_api_key = "EMPTY"
if args.port is None:
    import socket

    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    openai_api_port = find_free_port()
else:
    openai_api_port = args.port

openai_api_base = f"http://localhost:{openai_api_port}/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    timeout=3600,
)

model_name = args.model_name
model_path = model_name
max_context_length = args.max_context_length


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _resolve_results_root_for_model(path_value: str) -> str:
    expanded = os.path.abspath(os.path.expanduser(path_value))
    if os.path.isdir(expanded):
        return expanded
    if os.path.exists(expanded):
        return os.path.dirname(expanded)
    return os.path.abspath(".")


def _extract_generated_suffix(prompt_text: str, full_text: str) -> str:
    if full_text.startswith(prompt_text):
        return full_text[len(prompt_text) :]
    return full_text


def _print_debug_generated(
    label: str,
    prompt_text: str,
    full_text: str,
    finish_reason: str,
    matched_stop: Optional[str] = None,
    note: Optional[str] = None,
):
    generated_text = _extract_generated_suffix(prompt_text, full_text)
    print(colored("\n" + "=" * 80, "cyan"))
    print(colored(f"[DEBUG] {label}", "cyan", attrs=["bold"]))
    print(colored(f"finish_reason={finish_reason}", "cyan"))
    if matched_stop is not None:
        print(colored(f"matched_stop={matched_stop!r}", "cyan"))
    if note:
        print(colored(note, "cyan"))
    print(colored("-" * 80, "cyan"))
    print(generated_text)
    print(colored("=" * 80 + "\n", "cyan"))


if args.launch_server:
    num_gpus = (
        len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        if "CUDA_VISIBLE_DEVICES" in os.environ
        else 8
    )

    assert (args.tp is not None) == (
        args.dp is not None
    ), "Please specify both --tp and --dp or neither."

    if args.tp is not None and args.dp is not None:
        tp_size = args.tp
        dp_size = args.dp
    elif "32b" in model_name.lower():
        tp_size = 2
        dp_size = num_gpus // 2
    elif "7b" in model_name.lower():
        tp_size = 1
        dp_size = num_gpus
    else:
        tp_size = 1
        dp_size = num_gpus

    other_args = [
        "--tp",
        str(tp_size),
        "--dp",
        str(dp_size),
        "--disable-overlap-schedule",
        "--mem-fraction-static",
        "0.8",
        "--decode-log-interval",
        "60",
    ]

    if args.bfloat16:
        other_args.extend(["--dtype", "bfloat16"])

    process = popen_launch_server(
        model=model_path,
        base_url=openai_api_base.removesuffix("/v1"),
        timeout=args.timeout,
        api_key=openai_api_key,
        model_name=model_name,
        other_args=other_args,
        return_stdout_stderr=(sys.stdout, sys.stderr) if args.verbose > 1 else None,
        skip_actual_launch=args.skip_actual_launch,
        use_os_system=args.use_os_system,
        wait_before_check=args.wait_before_health_check,
    )

    if not args.no_terminate_on_exit:
        import atexit

        def exit_handler():
            print("Exiting... Terminating the model server process.")
            terminate_process(process)

        atexit.register(exit_handler)

tokenizer = AutoTokenizer.from_pretrained(model_path)

thread_end = "</Thread>"
planning_end_tags = ["</Outlines>"]
planning_start_tags = ["<Outlines>"]
outline_number_pattern = re.compile(
    r"<(?:Outline|Trial|Subtask)>\s*([0-9]+(?:\.[0-9]+)*)\s*:"
)
thread_end_id = tokenizer.convert_tokens_to_ids(thread_end)
thread_end_is_single_token = (
    thread_end_id is not None and thread_end_id != tokenizer.unk_token_id
)
if not thread_end_is_single_token:
    warnings.warn(
        f"{thread_end} is not configured as a single tokenizer token. "
        "Falling back to sequential generation without thread-end stop token.",
        RuntimeWarning,
    )

planning_end_ids = [
    token_id
    for token_id in (tokenizer.convert_tokens_to_ids(tag) for tag in planning_end_tags)
    if token_id is not None and token_id != tokenizer.unk_token_id
]
if not planning_end_ids:
    warnings.warn(
        f"{planning_end_tags} are not configured as single tokenizer tokens. "
        "Falling back to sequential generation without planning-end stop tokens.",
        RuntimeWarning,
    )

eos_id = tokenizer.eos_token_id

special_token_ids = None
parallel_stats_available = False
try:
    special_token_ids = get_special_token_ids(tokenizer)
    parallel_stats_available = True
except ValueError as exc:
    warnings.warn(
        f"Some required parallel special tokens are missing or not single-token: {exc}. "
        "Continuing with sequential-compatible stats (no parallel parsing).",
        RuntimeWarning,
    )

branching_supported = (
    bool(planning_end_ids) and thread_end_is_single_token and parallel_stats_available
)
if args.branching_generate and not branching_supported:
    warnings.warn(
        "Branching generation requested but required special tokens are missing. "
        "Falling back to sequential generation for this run.",
        RuntimeWarning,
    )
    args.branching_generate = False


def check_model_availability(model):
    models = client.models.list()
    available_models = [model_info.id for model_info in models.data]
    print(f"Available models: {available_models}")

    if model in available_models:
        print(f"Model '{model}' is available.")
        return True

    print(
        f"WARNING: Model '{model}' was not found in available models. "
        f"Available models are: {available_models}"
    )
    return False


if not args.skip_model_check:
    if not check_model_availability(model_name):
        raise RuntimeError(
            f"Model '{model_name}' is not available. Please verify the model name."
        )

text_completion = True
THINK_END_PATTERN = re.compile(r"</think>", re.IGNORECASE)


def apply_chat_template(messages):
    assert len(messages) == 1, f"Expected a single message, got {len(messages)}"
    assert (
        messages[0]["role"] == "user"
    ), f"Expected a user message, got {messages[0]['role']}"

    if args.template_type == "model":
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    raise ValueError(f"Unknown template type: {args.template_type}")


def compute_parallel_stats_safe(response_token_ids):
    if special_token_ids is None:
        total_num_tokens = len(response_token_ids)
        return {
            "total_num_tokens": total_num_tokens,
            "num_tokens_in_the_longest_thread": total_num_tokens,
            "with_parallel": False,
            "parallel_count": 0,
            "parallel_ratio": 0.0,
            "acceleration_ratio": 0.0,
            "avg_tokens_per_parallel_block": 0.0,
            "thread_counts_per_block": 0.0,
            "parallel_format_correct": True,
            "parallel_format_correct_v2": True,
        }
    return get_parallel_stats(response_token_ids, special_token_ids)


def _longest_thread_tokens_for_text(text: str) -> int:
    if not text:
        return 0
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    stats = compute_parallel_stats_safe(token_ids)
    return stats["num_tokens_in_the_longest_thread"]


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _remaining_longest_thread_budget(
    current_text: str, reserve_tokens: int = 0
) -> Optional[int]:
    if args.pause_at_longest_thread_tokens is None:
        return None
    used = _longest_thread_tokens_for_text(current_text)
    return args.pause_at_longest_thread_tokens - used - max(0, reserve_tokens)


def _reserve_tokens_for_branch_merge(num: str) -> int:
    structural = f"\n<Thread>\n{num}:</Thread>\n"
    return _count_tokens(structural)


def _truncate_to_longest_thread_budget(text: str, budget: int) -> str:
    if budget <= 0 or not text:
        return ""

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return ""

    if (
        compute_parallel_stats_safe(token_ids)["num_tokens_in_the_longest_thread"]
        <= budget
    ):
        return text

    lo, hi = 0, len(token_ids)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        prefix_ids = token_ids[:mid]
        longest = compute_parallel_stats_safe(prefix_ids)[
            "num_tokens_in_the_longest_thread"
        ]
        if longest <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return tokenizer.decode(
        token_ids[:best],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _continue_generation_from_prompt(
    prompt_text: str, stop_tokens_ids: List[int], label: str
) -> str:
    current_text = prompt_text
    while True:
        prompt_token_ids = tokenizer.encode(current_text, add_special_tokens=False)
        remaining_context_tokens = max_context_length - len(prompt_token_ids) - 1
        if remaining_context_tokens <= 0:
            _print_debug_generated(
                label=label,
                prompt_text=prompt_text,
                full_text=current_text,
                finish_reason="context_exhausted",
                note="No remaining context budget.",
            )
            return current_text

        try:
            completion = client.completions.create(
                model=model_name,
                prompt=prompt_token_ids,
                max_tokens=remaining_context_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                extra_body={
                    "add_special_tokens": False,
                    "skip_special_tokens": False,
                    "stop_tokens_ids": stop_tokens_ids,
                },
            )
        except Exception as exc:
            _print_debug_generated(
                label=label,
                prompt_text=prompt_text,
                full_text=current_text,
                finish_reason="exception",
                note=f"Continuation request raised {type(exc).__name__}: {exc}",
            )
            raise

        current_text += completion.choices[0].text
        finish_reason = completion.choices[0].finish_reason
        _print_debug_generated(
            label=label,
            prompt_text=prompt_text,
            full_text=current_text,
            finish_reason=str(finish_reason),
            note="Continuation request completed.",
        )

        if finish_reason in {"stop", "length"}:
            return current_text

        _print_debug_generated(
            label=label,
            prompt_text=prompt_text,
            full_text=current_text,
            finish_reason="unexpected_finish_reason",
            note=f"Unexpected finish reason before raising: {finish_reason}",
        )
        raise ValueError(f"Unexpected finish reason during continuation: {finish_reason}")


def _apply_hard_prethink_limit(
    prompt_text: str,
    generated_text: str,
    continue_stop_tokens_ids: List[int],
) -> str:
    if args.pause_at_longest_thread_tokens is None:
        return generated_text

    limit = args.pause_at_longest_thread_tokens
    match = THINK_END_PATTERN.search(generated_text)
    if match:
        pre_think = generated_text[: match.start()]
        suffix_from_close = generated_text[match.start() :]
    else:
        pre_think = generated_text
        suffix_from_close = ""

    capped_pre_think = _truncate_to_longest_thread_budget(pre_think, limit)
    pre_think_tokens = _longest_thread_tokens_for_text(capped_pre_think)
    truncated = capped_pre_think != pre_think

    if truncated and suffix_from_close:
        rebuilt = capped_pre_think + suffix_from_close
        _print_debug_generated(
            label="hard_prethink_limit",
            prompt_text=prompt_text,
            full_text=prompt_text + rebuilt,
            finish_reason="pause_at_longest_thread_tokens",
            note="Trimmed generation to fit the longest-thread budget and kept the existing </think> suffix.",
        )
        return rebuilt

    if truncated and not suffix_from_close:
        rebuilt = capped_pre_think
        if not rebuilt.endswith("\n"):
            rebuilt += "\n"
        rebuilt += "</think>\n"
        _print_debug_generated(
            label="hard_prethink_limit",
            prompt_text=prompt_text,
            full_text=prompt_text + rebuilt,
            finish_reason="pause_at_longest_thread_tokens",
            note="Trimmed generation to fit the longest-thread budget, inserted </think>, and will continue normally.",
        )
        full_text = _continue_generation_from_prompt(
            prompt_text + rebuilt,
            continue_stop_tokens_ids,
            label="post_pause_continuation",
        )
        return (
            full_text[len(prompt_text) :]
            if full_text.startswith(prompt_text)
            else full_text
        )

    if not suffix_from_close and pre_think_tokens >= limit:
        rebuilt = generated_text
        if not rebuilt.endswith("\n"):
            rebuilt += "\n"
        rebuilt += "</think>\n"
        _print_debug_generated(
            label="hard_prethink_limit",
            prompt_text=prompt_text,
            full_text=prompt_text + rebuilt,
            finish_reason="pause_at_longest_thread_tokens",
            note="Pause threshold hit exactly, inserted </think>, and will continue normally.",
        )
        full_text = _continue_generation_from_prompt(
            prompt_text + rebuilt,
            continue_stop_tokens_ids,
            label="post_pause_continuation",
        )
        return (
            full_text[len(prompt_text) :]
            if full_text.startswith(prompt_text)
            else full_text
        )

    return generated_text


def _get_first_turn_max_tokens(prompt_token_ids: List[int]) -> int:
    remaining_context_tokens = max_context_length - len(prompt_token_ids) - 1
    if args.pause_at_longest_thread_tokens is None:
        return remaining_context_tokens
    return min(remaining_context_tokens, args.pause_at_longest_thread_tokens)


def generate_single_sample(prompt_token_ids, prompt, messages, stop_tokens_ids):
    first_turn_max_tokens = _get_first_turn_max_tokens(prompt_token_ids)
    if text_completion:
        try:
            completion = client.completions.create(
                model=model_name,
                prompt=prompt_token_ids,
                max_tokens=first_turn_max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                extra_body={
                    "add_special_tokens": False,
                    "skip_special_tokens": False,
                    "stop_tokens_ids": stop_tokens_ids,
                },
            )
        except Exception as exc:
            _print_debug_generated(
                label="single_sample",
                prompt_text=prompt,
                full_text=prompt,
                finish_reason="exception",
                note=f"Initial completion raised {type(exc).__name__}: {exc}",
            )
            raise

        result_text = completion.choices[0].text
        finish_reason = completion.choices[0].finish_reason
        _print_debug_generated(
            label="single_sample",
            prompt_text=prompt,
            full_text=prompt + result_text,
            finish_reason=str(finish_reason),
            note="Initial completion request completed.",
        )
        return _apply_hard_prethink_limit(
            prompt_text=prompt,
            generated_text=result_text,
            continue_stop_tokens_ids=stop_tokens_ids,
        )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=first_turn_max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            extra_body={
                "add_special_tokens": False,
                "skip_special_tokens": False,
                "stop_tokens_ids": stop_tokens_ids,
            },
        )
    except Exception as exc:
        _print_debug_generated(
            label="single_sample_chat",
            prompt_text=prompt,
            full_text=prompt,
            finish_reason="exception",
            note=f"Initial chat completion raised {type(exc).__name__}: {exc}",
        )
        raise

    result_text = completion.choices[0].message.content
    finish_reason = completion.choices[0].finish_reason
    _print_debug_generated(
        label="single_sample_chat",
        prompt_text=prompt,
        full_text=prompt + result_text,
        finish_reason=str(finish_reason),
        note="Initial chat completion request completed.",
    )
    return _apply_hard_prethink_limit(
        prompt_text=prompt,
        generated_text=result_text,
        continue_stop_tokens_ids=stop_tokens_ids,
    )


def generate_until_any(
    model_name,
    tokenizer,
    prompt,
    stop,
    max_new_tokens,
    temperature,
    top_p,
    label,
):
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    max_tokens = min(max_new_tokens, max_context_length - len(prompt_token_ids) - 1)

    if max_tokens < 0:
        _print_debug_generated(
            label=label,
            prompt_text=prompt,
            full_text=prompt,
            finish_reason="invalid_max_tokens",
            note=(
                f"max_new_tokens={max_new_tokens} is too small for prompt length "
                f"{len(prompt_token_ids)} and max_context_length={max_context_length}."
            ),
        )
        raise ValueError(
            f"max_new_tokens ({max_new_tokens}) is too small for the prompt length "
            f"({len(prompt_token_ids)}) and max context length ({max_context_length})."
        )

    try:
        completion = client.completions.create(
            model=model_name,
            prompt=prompt_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            extra_body={"add_special_tokens": False, "skip_special_tokens": False},
        )
    except Exception as exc:
        _print_debug_generated(
            label=label,
            prompt_text=prompt,
            full_text=prompt,
            finish_reason="exception",
            note=f"generate_until_any raised {type(exc).__name__}: {exc}",
        )
        raise

    gen_text = completion.choices[0].text
    finish_reason = completion.choices[0].finish_reason
    matched_stop = None
    if finish_reason == "length":
        matched_stop = None
    elif finish_reason == "stop":
        matched_stop = completion.choices[0].matched_stop
        if matched_stop != eos_id:
            gen_text += matched_stop
    else:
        full_text = prompt + gen_text
        _print_debug_generated(
            label=label,
            prompt_text=prompt,
            full_text=full_text,
            finish_reason="unexpected_finish_reason",
            note=f"Unexpected finish reason before raising: {finish_reason}",
        )
        raise ValueError(f"Unexpected finish reason: {finish_reason}")

    full_text = prompt + gen_text
    _print_debug_generated(
        label=label,
        prompt_text=prompt,
        full_text=full_text,
        finish_reason=str(finish_reason),
        matched_stop=matched_stop,
        note="generate_until_any request completed.",
    )
    return gen_text, full_text, matched_stop, finish_reason


def branching_generate(
    model_name,
    tokenizer,
    base_prompt: str,
    sampling_params: dict,
    newlines_between_path: bool = False,
    verbose: bool = False,
):
    def _get_longest_thread_tokens(text: str) -> int:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        stats = compute_parallel_stats_safe(token_ids)
        return stats["num_tokens_in_the_longest_thread"]

    def _append_think_close_if_missing(text: str) -> str:
        lowered = text.lower()
        if "</think>" in lowered:
            return text
        if not text.endswith("\n"):
            text += "\n"
        return text + "</think>\n"

    def _continue_until_normal_stop(prompt_text: str) -> str:
        stop_tokens_ids = [] if args.no_stop_at_eos else [eos_id]
        return _continue_generation_from_prompt(
            prompt_text,
            stop_tokens_ids,
            label="branching_continuation",
        )

    def _generate_branch(
        outlines_full: str, num: str, longest_thread_tokens_before_round: int
    ):
        branch_prompt = outlines_full + f"\n<Thread>\n{num}:"
        branch_prompt_token_ids = tokenizer.encode(branch_prompt, add_special_tokens=False)
        remaining_context_tokens = max_context_length - len(branch_prompt_token_ids) - 1
        if remaining_context_tokens <= 0:
            fallback_text = thread_end
            _print_debug_generated(
                label=f"branch_{num}",
                prompt_text=branch_prompt,
                full_text=branch_prompt + fallback_text,
                finish_reason="context_exhausted",
                note="Skipping branch because there is no remaining context budget.",
            )
            return fallback_text, False

        per_branch_max_new_tokens = min(
            sampling_params["max_new_tokens"], remaining_context_tokens
        )
        capped_by_longest_thread_limit = False
        if args.pause_at_longest_thread_tokens is not None:
            branch_structure_reserve = _reserve_tokens_for_branch_merge(num)
            remaining_for_longest = (
                args.pause_at_longest_thread_tokens
                - longest_thread_tokens_before_round
                - branch_structure_reserve
            )
            if remaining_for_longest <= 0:
                fallback_text = thread_end
                _print_debug_generated(
                    label=f"branch_{num}",
                    prompt_text=branch_prompt,
                    full_text=branch_prompt + fallback_text,
                    finish_reason="pause_at_longest_thread_tokens",
                    note=(
                        "Skipping branch because the longest-thread pause threshold "
                        "was already reached before branch generation."
                    ),
                )
                return fallback_text, True
            per_branch_max_new_tokens = min(
                per_branch_max_new_tokens, remaining_for_longest
            )
            capped_by_longest_thread_limit = (
                per_branch_max_new_tokens == remaining_for_longest
            )

        per_branch_max_new_tokens = max(1, per_branch_max_new_tokens)
        if verbose:
            print(colored(f"Generating branch for outline: {num}", "blue"))

        branch_gen, _, _, finish_reason = generate_until_any(
            model_name,
            tokenizer,
            prompt=branch_prompt,
            stop=[thread_end],
            temperature=sampling_params["temperature"],
            top_p=sampling_params["top_p"],
            max_new_tokens=per_branch_max_new_tokens,
            label=f"branch_{num}",
        )
        hit_longest_thread_limit = (
            capped_by_longest_thread_limit and finish_reason == "length"
        )
        return branch_gen, hit_longest_thread_limit

    while True:
        if verbose:
            print(colored("--- Step 1: Generating up to planning end tag ---", "blue"))

        try:
            planning_max_new_tokens = sampling_params["max_new_tokens"]
            if args.pause_at_longest_thread_tokens is not None:
                planning_stop_reserve = max(
                    (_count_tokens(tag) for tag in planning_end_tags), default=0
                )
                remaining_for_planning = _remaining_longest_thread_budget(
                    base_prompt, reserve_tokens=planning_stop_reserve
                )
                if remaining_for_planning is not None and remaining_for_planning <= 0:
                    paused_text = _append_think_close_if_missing(base_prompt)
                    _print_debug_generated(
                        label="planning",
                        prompt_text=base_prompt,
                        full_text=paused_text,
                        finish_reason="pause_at_longest_thread_tokens",
                        note="Pause threshold reached before planning generation. Appending </think> and continuing generation.",
                    )
                    return _continue_until_normal_stop(paused_text)
                planning_max_new_tokens = min(
                    planning_max_new_tokens, remaining_for_planning
                )
                planning_max_new_tokens = max(1, planning_max_new_tokens)

            outlines_text, outlines_full, hit, _ = generate_until_any(
                model_name,
                tokenizer,
                prompt=base_prompt,
                stop=planning_end_tags,
                temperature=sampling_params["temperature"],
                top_p=sampling_params["top_p"],
                max_new_tokens=planning_max_new_tokens,
                label="planning",
            )
        except ValueError as exc:
            _print_debug_generated(
                label="planning",
                prompt_text=base_prompt,
                full_text=base_prompt,
                finish_reason="exception",
                note=f"Planning step raised ValueError: {exc}",
            )
            return base_prompt

        if hit is None:
            _print_debug_generated(
                label="planning",
                prompt_text=base_prompt,
                full_text=outlines_full,
                finish_reason="no_planning_end_tag",
                note="No planning end tag found, returning the full output as-is.",
            )
            return outlines_full

        outlines_start_index = max(
            outlines_text.rfind(tag) for tag in planning_start_tags
        )
        search_region = (
            outlines_text
            if outlines_start_index == -1
            else outlines_text[outlines_start_index:]
        )
        outline_nums = outline_number_pattern.findall(search_region)

        if verbose:
            print(colored(f"Found outline numbers: {outline_nums}", "green"))

        if not outline_nums:
            _print_debug_generated(
                label="planning",
                prompt_text=base_prompt,
                full_text=outlines_full,
                finish_reason="no_outline_numbers",
                note="No outline numbers were found. Returning the planning output.",
            )
            return outlines_full

        branches_gen = {}
        branches_hit_longest_limit = {}
        longest_thread_tokens_before_round = _get_longest_thread_tokens(outlines_full)

        for num in outline_nums:
            try:
                branch_gen, hit_longest_limit = _generate_branch(
                    outlines_full,
                    num,
                    longest_thread_tokens_before_round,
                )
            except Exception as exc:
                _print_debug_generated(
                    label=f"branch_{num}",
                    prompt_text=outlines_full,
                    full_text=outlines_full,
                    finish_reason="exception",
                    note=f"Branch generation raised {type(exc).__name__}: {exc}",
                )
                return outlines_full

            branches_gen[num] = branch_gen
            branches_hit_longest_limit[num] = hit_longest_limit

        merged = outlines_full
        end_seq = False
        paused_by_longest_thread_limit = False

        for idx, num in enumerate(outline_nums):
            thread_content = branches_gen[num]
            if not thread_content.endswith(thread_end):
                if branches_hit_longest_limit.get(num, False):
                    paused_by_longest_thread_limit = True
                else:
                    end_seq = True
                thread_content += thread_end

            if thread_content.endswith("\n"):
                raise AssertionError(
                    f"Thread content should not end with a newline. {thread_content=}"
                )

            if newlines_between_path:
                merged += f"\n<Thread>\n{num}:{thread_content}"
            else:
                if idx == 0:
                    merged += "\n"
                merged += f"<Thread>\n{num}:{thread_content}"

        if end_seq:
            _print_debug_generated(
                label="merge",
                prompt_text=base_prompt,
                full_text=merged,
                finish_reason="thread_missing_end_tag",
                note="Some thread did not end properly, returning the merged text without continuing.",
            )
            return merged

        merged += "\n"
        _print_debug_generated(
            label="merge",
            prompt_text=base_prompt,
            full_text=merged,
            finish_reason="merged_round_complete",
            note="Merged all generated thread content for this round.",
        )

        if args.pause_at_longest_thread_tokens is not None:
            longest_thread_tokens_after_round = _get_longest_thread_tokens(merged)
            if (
                longest_thread_tokens_after_round
                >= args.pause_at_longest_thread_tokens
            ):
                paused_by_longest_thread_limit = True
                merged = _append_think_close_if_missing(merged)
                _print_debug_generated(
                    label="merge",
                    prompt_text=base_prompt,
                    full_text=merged,
                    finish_reason="pause_at_longest_thread_tokens",
                    note="Pause threshold reached after merge. Appending </think> and continuing generation.",
                )
                return _continue_until_normal_stop(merged)

        if paused_by_longest_thread_limit:
            merged = _append_think_close_if_missing(merged)
            _print_debug_generated(
                label="merge",
                prompt_text=base_prompt,
                full_text=merged,
                finish_reason="pause_at_longest_thread_tokens",
                note="A branch hit the longest-thread pause threshold. Appending </think> and continuing generation.",
            )
            return _continue_until_normal_stop(merged)

        if verbose:
            print(
                colored(
                    "--- Loop will continue with the merged text as the new base_prompt ---",
                    "blue",
                )
            )
        base_prompt = merged


def generate_single_sample_branching(prompt_token_ids, base_prompt, stop_tokens_ids):
    assert text_completion, "Branching generation is only supported for text completion."
    assert not stop_tokens_ids, "Stop tokens are not supported for branching generation."

    full_text = branching_generate(
        model_name,
        tokenizer,
        base_prompt=base_prompt,
        sampling_params={
            "max_new_tokens": max_context_length - len(prompt_token_ids) - 1,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        verbose=args.verbose > 2,
    )

    generated_text = (
        full_text[len(base_prompt) :] if full_text.startswith(base_prompt) else full_text
    )
    continuation_stop_tokens = [] if args.no_stop_at_eos else [eos_id]
    return _apply_hard_prethink_limit(
        prompt_text=base_prompt,
        generated_text=generated_text,
        continue_stop_tokens_ids=continuation_stop_tokens,
    )


messages = [{"role": "user", "content": INLINE_PROBLEM}]
if not INLINE_PROBLEM.strip():
    raise ValueError("INLINE_PROBLEM is empty. Please put the problem directly in code.")

print(colored(f"Using inline problem: {INLINE_PROBLEM_NAME}", "green", attrs=["bold"]))
print(colored("Single-thread debug mode is enabled.", "green"))
print(colored("Problem content:", "green"))
print(INLINE_PROBLEM)

prompt = apply_chat_template(messages)
print(colored("Rendered prompt:", "green"))
print(prompt)

prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
if args.branching_generate:
    stop_tokens_ids = []
else:
    stop_tokens_ids = list(planning_end_ids)
    if thread_end_is_single_token:
        stop_tokens_ids.append(thread_end_id)
    if not args.no_stop_at_eos:
        stop_tokens_ids.append(eos_id)

if args.branching_generate:
    result = generate_single_sample_branching(
        prompt_token_ids,
        base_prompt=prompt,
        stop_tokens_ids=stop_tokens_ids,
    )
else:
    result = generate_single_sample(
        prompt_token_ids,
        prompt,
        messages,
        stop_tokens_ids,
    )

final_full_text = prompt + result
_print_debug_generated(
    label="final_result",
    prompt_text=prompt,
    full_text=final_full_text,
    finish_reason="final",
    note="Final generated content returned by the script.",
)

normalized_model_path = os.path.normpath(model_path.rstrip("/\\"))
model_leaf_name = os.path.basename(normalized_model_path) or "model"
model_parent_leaf = os.path.basename(os.path.dirname(normalized_model_path))
run_name = model_leaf_name
if model_leaf_name.startswith("global_step_") and model_parent_leaf:
    run_name = f"{model_parent_leaf}_{model_leaf_name}"
run_name = _sanitize_name(run_name)
if args.suffix:
    run_name += f"_{_sanitize_name(args.suffix)}"

pause_suffix = (
    f"_pause{args.pause_at_longest_thread_tokens}"
    if args.pause_at_longest_thread_tokens is not None
    else ""
)
file_stem = f"{_sanitize_name(INLINE_PROBLEM_NAME)}{pause_suffix}"
checkpoint_results_root = _resolve_results_root_for_model(model_path)
results_dir = os.path.join(
    checkpoint_results_root,
    "eval_pause_inline_debug_outputs",
    run_name,
)
os.makedirs(results_dir, exist_ok=True)

output_json_file = os.path.join(results_dir, f"{file_stem}.json")
output_txt_file = os.path.join(results_dir, f"{file_stem}.txt")

if not args.overwrite:
    for candidate in [output_json_file, output_txt_file]:
        if os.path.exists(candidate):
            raise FileExistsError(
                f"{candidate} already exists. Re-run with --overwrite or change --suffix."
            )

payload = {
    "model_name": model_name,
    "model_path": model_path,
    "problem_name": INLINE_PROBLEM_NAME,
    "problem": INLINE_PROBLEM,
    "rendered_prompt": prompt,
    "result": result,
    "branching_generate": args.branching_generate,
    "pause_at_longest_thread_tokens": args.pause_at_longest_thread_tokens,
    "temperature": args.temperature,
    "top_p": args.top_p,
    "max_context_length": max_context_length,
    "output_json_file": output_json_file,
    "output_txt_file": output_txt_file,
    "saved_at_unix_time": int(time.time()),
}

with open(output_json_file, "w", encoding="utf-8") as file_obj:
    json.dump(payload, file_obj, indent=2, ensure_ascii=False)

with open(output_txt_file, "w", encoding="utf-8") as file_obj:
    file_obj.write(result)

print(colored(f"Saved JSON output to: {output_json_file}", "green"))
print(colored(f"Saved text output to: {output_txt_file}", "green"))
