import argparse
import json
import os
import re
import sys
import time
from typing import List, Union

import numpy as np
import pandas as pd
from openai import OpenAI
from rewards import rllm_reward_fn_math
from rewards import grade_answer_verl
from rewards import get_special_token_ids, get_parallel_stats
from termcolor import colored
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import popen_launch_server, terminate_process

# Setup argument parser
parser = argparse.ArgumentParser(description="Evaluate a model on AIME 2024")
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
    help="If set, the model will not stop at EOS token. Useful for debugging or generating longer outputs.",
)
parser.add_argument(
    "--skip-model-check",
    action="store_true",
    help="If set, skip the model availability check. Useful if you are sure the model is available.",
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
    "--data-type",
    type=str,
    default="./data/mult-10k-par_pq/train.parquet",
    help="Type of dataset to evaluate. Default is './data/mult-10k-par_pq/train.parquet'.",
)
parser.add_argument(
    "-n",
    "--n_samples",
    type=int,
    default=16,
    help="Number of samples to generate for each prompt. Default is 16.",
)
parser.add_argument(
    "--autoregressive",
    action="store_true",
    help="Evaluate non-multiverse model (autoregressive)",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Save results to a temporary file, useful for debugging. Only query the first question.",
)
parser.add_argument(
    "--no-terminate-on-exit",
    action="store_true",
    help="If set, the model server will not be terminated on exit. Useful for debugging or if you want to keep the server running.",
)
parser.add_argument(
    "--skip-actual-launch",
    action="store_true",
    help="If set, the model server will not be actually launched. Useful for debugging or if you want to use an existing server. Health check will still be performed.",
)
parser.add_argument(
    "--use-os-system",
    action="store_true",
    help="If set, use os.system to launch the server instead of subprocess.Popen. This is useful for debugging.",
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
    help="Wait time in seconds before performing the health check after launching the server. Default is 0 seconds.",
)
parser.add_argument(
    "--branching-generate",
    action="store_true",
    help="If set, use branching generation instead of standard generation. This is useful for models that support structured generation.",
)
parser.add_argument(
    "--data-parallel-workers",
    type=int,
    default=32,
    help="Worker threads for parallelizing prompts. Default is 32.",
)
parser.add_argument(
    "--reasoning-parallel-workers",
    type=int,
    default=4,
    help="Worker threads for parallelizing branches in branching generation. Default is 4.",
)
parser.add_argument(
    "--total-splits",
    type=int,
    default=1,
    help="Total number of splits to divide the dataset into. Default is 1 (no splitting).",
)
parser.add_argument(
    "--current-split",
    type=int,
    default=0,
    help="Current split index to process (0-indexed). Must be less than total-splits. Default is 0.",
)
parser.add_argument(
    "--max-context-length",
    type=int,
    default=32768,
    help="Maximum context length for the model. Default is 32768. Please set to 40k for Qwen3.",
)
parser.add_argument(
    "--strip-comma-from-answer",
    action="store_true",
    help="If set, commas will be stripped from the model's answer before checking correctness. This is sometimes needed because `_strip_properly_formatted_commas` does not always remove commas in the answer.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="If set, overwrite the existing results file. Default is False (do not overwrite).",
)
parser.add_argument(
    "--skip-system-prompt-check",
    action="store_true",
    help="If set, skip the check for the system prompt in the chat template. This is useful if you are sure the system prompt is correct or if you are using a model that does not have a system prompt.",
)
args = parser.parse_args()

# Validate split arguments
if args.current_split >= args.total_splits:
    raise ValueError(f"current-split ({args.current_split}) must be less than total-splits ({args.total_splits})")
if args.current_split < 0:
    raise ValueError(f"current-split ({args.current_split}) must be non-negative")
if args.total_splits < 1:
    raise ValueError(f"total-splits ({args.total_splits}) must be at least 1")

openai_api_key = "EMPTY"
if args.port is None:
    import socket
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
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

if "/" in args.data_type:
    if args.data_type.split("/")[-1].split(".")[0] in ["train", "test", "val"]:
        # If the data_type is a split of a dataset, we need to extract the base name
        # and use it to construct the path.
        data_path = args.data_type
        args.data_type = data_type = args.data_type.split("/")[-2]
    else:
        data_path = args.data_type.rstrip("/") + "/val.parquet"
        args.data_type = data_type = args.data_type.rstrip("/").split("/")[-1]
else:
    data_type = args.data_type
    data_path = os.path.expanduser(f"data/{data_type}.parquet")

print(colored(f"Using {data_type} dataset from {data_path}", "green", attrs=["bold"]))

pd.set_option("display.max_columns", None)  # Show all columns in the output
pd.set_option("display.max_rows", None)  # Show all rows in the output
pd.set_option("display.max_colwidth", None)  # Show full content of each column

# Load parquet file
df = pd.read_parquet(data_path)

model_name = args.model_name
model_path = model_name
max_context_length = args.max_context_length

if args.launch_server:
    num_gpus = (
        len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        if "CUDA_VISIBLE_DEVICES" in os.environ
        else 8
    )

    # need to specify both or nont
    assert (args.tp is not None) == (
        args.dp is not None
    ), "Please specify both --tp and --dp or neither. They should be either both set or both None."

    # Use user-specified tp/dp values or auto-configure based on model size
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
        # This is used to print the server logs to the console, enable verbose level > 1 to print
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

if args.debug:
    df = df.head(1)  # For debugging, only take the first row
    print("Debug mode: Only processing the first message.")

# Apply dataset splitting if total_splits > 1
if args.total_splits > 1:
    total_rows = len(df)
    split_size = (total_rows + args.total_splits - 1) // args.total_splits  # Ceiling division
    start_idx = args.current_split * split_size
    end_idx = min(start_idx + split_size, total_rows)

    df = df.iloc[start_idx:end_idx].reset_index(drop=True)
    print(f"Dataset splitting: Processing split {args.current_split}/{args.total_splits-1}")
    print(f"Original dataset size: {total_rows}, current split size: {len(df)} (rows {start_idx}-{end_idx-1})")

messages_all = df["prompt"].to_list()
messages_all = [[message.item()] for message in messages_all]

tokenizer = AutoTokenizer.from_pretrained(model_path)

thread_end = "</Thread>"
planning_end_tags = ["</Outlines>"]
planning_start_tags = ["<Outlines>"]
outline_number_pattern = re.compile(r"<(?:Outline|Trial|Subtask)>\s*([0-9]+(?:\.[0-9]+)*)\s*:")
thread_end_id = tokenizer.convert_tokens_to_ids(thread_end)
if thread_end_id is None or thread_end_id == tokenizer.unk_token_id:
    raise ValueError(
        f"{thread_end} is not configured as a single tokenizer token. "
        "Please add thread tags as special tokens before evaluation."
    )
planning_end_ids = [
    token_id
    for token_id in (tokenizer.convert_tokens_to_ids(tag) for tag in planning_end_tags)
    if token_id is not None and token_id != tokenizer.unk_token_id
]
if not planning_end_ids:
    raise ValueError(
        f"Neither {planning_end_tags} is configured as a single tokenizer token. "
        "Please add planning tags as special tokens before evaluation."
    )
eos_id = tokenizer.eos_token_id

# Get special token IDs for parallel stats computation
special_token_ids = get_special_token_ids(tokenizer)

# Display the first few rows to check the data
print(f"Loaded {data_type} dataset with {len(df)} rows")


def check_model_availability(model):
    models = client.models.list()
    available_models = [model.id for model in models.data]
    print(f"Available models: {available_models}")

    if model in available_models:
        print(f"Model '{model}' is available.")
        return True
    else:
        print(
            f"WARNING: Model '{model}' was not found in available models! Available models are: {available_models}"
        )
        print(f"Please verify the model name or ensure the model is loaded.")
        return False


# Check if the specified model is available before proceeding
if not args.skip_model_check:
    model_available = check_model_availability(model_name)
    if not model_available:
        raise RuntimeError(
            f"Model '{model_name}' is not available. Please check the model name or ensure the model is loaded correctly."
        )

import concurrent.futures
import json
import threading
from concurrent.futures import ThreadPoolExecutor

generated_text = []
messages_list = messages_all

n_samples = args.n_samples
data_parallel_workers = args.data_parallel_workers
reasoning_parallel_workers = args.reasoning_parallel_workers
print(f"Number of samples to generate for each prompt: {n_samples}")

# Total operations to be performed
total_ops = len(messages_list) * n_samples
progress_bar = tqdm(total=total_ops, desc="Generating text")
# Let's use text_completion to ensure the templates are right
text_completion = True  # Set to True for text completion, False for chat completion


def apply_chat_template(messages):
    assert (
        len(messages) == 1
    ), f"Expected a single message, got {len(messages)} messages: {messages}"
    assert (
        messages[0]["role"] == "user"
    ), f"Expected the first message to be a user message, got {messages[0]['role']}"

    user_query = messages[0]["content"]

    if args.template_type == "model":
        # NOTE: This does not remove suffix <think> or <Think> at the end.
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        raise ValueError(f"Unknown template type: {args.template_type}")

    return prompt


print(f"Example chat message: {apply_chat_template(messages_list[0])}")


def generate_single_sample(prompt_token_ids, messages, stop_tokens_ids):
    if text_completion:
        # text completion
        completion = client.completions.create(
            model=model_name,
            prompt=prompt_token_ids,
            max_tokens=max_context_length - len(prompt_token_ids) - 1,
            temperature=args.temperature,
            top_p=args.top_p,
            extra_body={
                "add_special_tokens": False,
                "skip_special_tokens": False,
                "stop_tokens_ids": stop_tokens_ids,
            },
        )
        return completion.choices[0].text
    else:
        # chat completion
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_context_length - len(prompt_token_ids) - 1,
            temperature=args.temperature,
            top_p=args.top_p,
            extra_body={
                "add_special_tokens": False,
                "skip_special_tokens": False,
                "stop_tokens_ids": stop_tokens_ids,
            },
        )
        return completion.choices[0].message.content


# This is for structured generation (that parses the output).
def generate_until_any(
    model_name, tokenizer, prompt, stop, max_new_tokens, temperature, top_p
):
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    max_tokens = min(max_new_tokens, max_context_length - len(prompt_token_ids) - 1)

    if max_tokens < 0:
        raise ValueError(
            f"max_new_tokens ({max_new_tokens}) is too small for the prompt length ({len(prompt_token_ids)}) and max context length ({max_context_length})."
        )

    completion = client.completions.create(
        model=model_name,
        prompt=prompt_token_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        extra_body={"add_special_tokens": False, "skip_special_tokens": False},
    )

    gen_text = completion.choices[0].text

    finish_reason = completion.choices[0].finish_reason
    if finish_reason == "length":
        hit = None
    elif finish_reason == "stop":
        hit = completion.choices[0].matched_stop
        if hit != eos_id:
            # If it's eos token, we don't append it (and it's an id rather than a string)
            gen_text += hit  # append the stop token to the generated text
    else:
        raise ValueError(f"Unexpected finish reason: {finish_reason}")
    full_text = prompt + gen_text
    return gen_text, full_text, hit

def branching_generate(
    model_name,
    tokenizer,
    base_prompt: str,
    sampling_params: dict,
    newlines_between_path: bool = False,
    verbose: bool = False,
):
    """
    Assumes base_prompt already contains a planning block inside <Parallel>.
    Planning block is <Outlines>...</Outlines>.
    Planning entries can be <Outline>... or <Trial>... or <Subtask>...
    1) Generate up to the planning end tag (</Outlines>)
    2) Extract all plan item numbers
    3) For each, generate its <Thread>…</Thread>
    4) Merge all threads
    """

    def _generate_branch(outlines_full: str, num: str):
        branch_prompt = outlines_full + f"\n<Thread>\n{num}:"
        if verbose:
            print(colored(f"Generating branch for outline: {num}", "blue"))
            print(colored(f"Branch prompt:\n{branch_prompt}\n" + "-" * 20, "blue"))
        branch_gen, _, _ = generate_until_any(
            model_name,
            tokenizer,
            prompt=branch_prompt,
            stop=[thread_end],
            temperature=sampling_params["temperature"],
            top_p=sampling_params["top_p"],
            max_new_tokens=sampling_params["max_new_tokens"],
        )
        return branch_gen

    max_workers = max(1, reasoning_parallel_workers)
    executor = ThreadPoolExecutor(max_workers=max_workers)

    # This loop structure suggests a potential iterative process,
    # where the output of one round can be the input for the next.
    try:
        while True:
            # Step 1: generate through planning block end tag
            if verbose:
                print(colored("--- Step 1: Generating up to planning end tag ---", "blue"))
                print(f"Input prompt:\n{base_prompt}\n" + "=" * 20)

            try:
                outlines_text, outlines_full, hit = generate_until_any(
                    model_name,
                    tokenizer,
                    prompt=base_prompt,
                    stop=planning_end_tags,
                    # do_sample=sampling_params["temperature"] > 0,
                    temperature=sampling_params["temperature"],
                    top_p=sampling_params["top_p"],
                    max_new_tokens=sampling_params["max_new_tokens"],
                )
            except ValueError as e:
                # Likely during overlong generation
                print(colored(f"Error during generation: {e}", "red"))
                return base_prompt

            if verbose:
                print(
                    f"{colored('Generation result (full text):', 'green')}\n{outlines_full}\n"
                    + "=" * 20
                )

            if hit is None:
                if verbose:
                    print(
                        colored(
                            "--- No planning end tag found, returning the full output. ---",
                            "yellow",
                        )
                    )
                # no planning end tag found, just return the full output
                return outlines_full

            # Step 2: pull out outline numbers
            if verbose:
                print(colored("--- Step 2: Extracting outline numbers ---", "blue"))

            # Find the last planning block start tag and extract numbers after it
            outlines_start_index = max(outlines_text.rfind(tag) for tag in planning_start_tags)
            search_region = outlines_text if outlines_start_index == -1 else outlines_text[outlines_start_index:]
            outline_nums = outline_number_pattern.findall(search_region)

            if verbose:
                print(
                    colored(
                        f"Found outline numbers: {outline_nums}\n" + "=" * 20, "green"
                    )
                )

            if not outline_nums:
                if verbose:
                    print(
                        colored(
                            "--- No outline numbers found. Nothing to branch. Returning... ---",
                            "yellow",
                        )
                    )
                # no outlines → nothing to branch
                return outlines_full

            # Step 3: generate each <Thread> once
            if verbose:
                print(
                    colored(
                        "--- Step 3: Generating each <Thread> in parallel ---", "blue"
                    )
                )
            branches_gen = {}

            try:
                futures = {
                    executor.submit(_generate_branch, outlines_full, num): num
                    for num in outline_nums
                }
                for future in concurrent.futures.as_completed(futures):
                    num = futures[future]
                    branch_gen = future.result()
                    branches_gen[num] = branch_gen
                    if verbose:
                        print(
                            colored(
                                f"Generated branch for {num}:\n{branch_gen}\n"
                                + "-" * 20,
                                "green",
                            )
                        )
            except Exception as e:
                for future in futures:
                    future.cancel()
                # Likely during overlong generation
                print(colored(f"Error during branch generation: {e}", "red"))
                return outlines_full

            # Step 4: stitch together
            if verbose:
                print(
                    colored(
                        f"--- Step 4: Merging branches ({branches_gen.keys()}, {len(branches_gen)} in total) ---",
                        "blue",
                    )
                )
            merged = outlines_full
            end_seq = False
            for i, num in enumerate(outline_nums):
                branch_gen = branches_gen[num]
                # We extract just the generated part for the final composition
                # thread_content = branch_full.split(f"\n<Thread>\n{num}:", 1)[-1]
                thread_content = branch_gen
                if not thread_content.endswith(thread_end):
                    print(
                        f"WARNING: Thread content does not end with {thread_end}: {thread_content=}"
                    )
                    end_seq = True
                    thread_content += "</Thread>"
                # assert thread_content.endswith(thread_end), f"Thread content does not end with {thread_end}: {thread_content}"
                assert not thread_content.endswith(
                    "\n"
                ), f"Thread content should not end with a newline. {thread_content=}"
                if newlines_between_path:
                    merged += f"\n<Thread>\n{num}:{thread_content}"
                else:
                    # NOTE: there was an inconsistency in the legacy version: there should be a "\n" before the first <Thread>
                    if i == 0:
                        merged += "\n"
                    merged += f"<Thread>\n{num}:{thread_content}"

            if end_seq:
                print(
                    "WARNING: Some thread did not end properly, returning the merged text without continuing."
                )
                return merged

            merged += "\n"

            if verbose:
                print(
                    colored(
                        f"Final merged text:\n{merged}\n" + "=" * 20,
                        "green",
                    )
                )
                print(
                    colored(
                        "--- Loop will now continue with the merged text as the new base_prompt ---",
                        "blue",
                    )
                )

            # The loop continues, using the merged text as the new base_prompt.
            # To complete the generation as described in the docstring, you would
            # need a final generation step here and then a `return` or `break`.
            # As the original code stands, it re-enters the loop.
            base_prompt = merged
    finally:
        executor.shutdown(wait=True)




def generate_single_sample_branching(prompt_token_ids, base_prompt, stop_tokens_ids):
    assert (
        text_completion
    ), "Branching generation is only supported for text completion mode."
    assert (
        not stop_tokens_ids
    ), "Stop tokens are not supported for branching generation."

    # base_prompt = apply_chat_template(messages)
    gen_text = branching_generate(
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

    return gen_text


def process_sample(message_idx, sample_idx, jsonl_file_path, lock):
    """Process a single sample for a given message."""
    messages = messages_list[message_idx]
    prompt = apply_chat_template(messages)
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)

    if args.branching_generate:
        stop_tokens_ids = []  # Branching generation does not use stop tokens
    else:
        stop_tokens_ids = planning_end_ids + [thread_end_id]
        if not args.no_stop_at_eos:
            stop_tokens_ids.append(eos_id)

    if args.branching_generate:
        # Use branching generation if specified
        result = generate_single_sample_branching(
            prompt_token_ids, base_prompt=prompt, stop_tokens_ids=stop_tokens_ids
        )
    else:
        result = generate_single_sample(prompt_token_ids, messages, stop_tokens_ids)

    # Write to JSONL file with lock
    jsonl_entry = {
        "message_idx": message_idx,
        "sample_idx": sample_idx,
        "result": result,
    }

    with lock:
        with open(jsonl_file_path, "a") as f_jsonl:
            f_jsonl.write(json.dumps(jsonl_entry) + "\n")
            f_jsonl.flush()  # Ensure data is written immediately

    print(
        f"Completed and saved sample {sample_idx+1}/{n_samples} for message {msg_idx+1}/{len(messages_list)}"
    )

    progress_bar.update(1)
    return message_idx, sample_idx, result


save_base_path = model_path.split("/")[-1]
if save_base_path.startswith("global_step_"):
    save_base_path = model_path.split("/")[-2] + "_" + save_base_path
if args.suffix:
    save_base_path += f"_{args.suffix}"

# Add split suffix to filenames if using dataset splitting
split_suffix = f"_split{args.current_split}_of_{args.total_splits}" if args.total_splits > 1 else ""

if not args.debug:
    results_dir = f"{save_base_path}"
    jsonl_file = f"{results_dir}/{data_type}_{n_samples}{split_suffix}.jsonl"
    final_json_file = f"{results_dir}/{data_type}_{n_samples}{split_suffix}.json"
    print(f"Results will be saved to: {jsonl_file}")
    print(f"Final JSON will be saved to: {final_json_file}")
else:
    # Save to a temporary file for debugging
    timestamp = int(time.time())
    results_dir = f"debug_logs/{save_base_path}_{timestamp}_debug"
    jsonl_file = f"{results_dir}/{data_type}_{n_samples}{split_suffix}.jsonl"
    final_json_file = f"{results_dir}/{data_type}_{n_samples}{split_suffix}.json"
    print(
        f"Debug mode: Results will be saved to temporary files: {jsonl_file} and {final_json_file}"
    )

os.makedirs(results_dir, exist_ok=True)

# Create a lock for thread-safe JSONL writing
jsonl_lock = threading.Lock()

# Dictionary to track completed samples
# Key is (message_idx, sample_idx), value is the result
completed_samples = {}

# Load existing results from JSONL if available
if os.path.exists(jsonl_file) and not args.overwrite:
    try:
        with open(jsonl_file, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                msg_idx = data["message_idx"]
                sample_idx = data["sample_idx"]
                result = data["result"]
                completed_samples[(msg_idx, sample_idx)] = result
        print(f"Loaded {len(completed_samples)} completed samples from {jsonl_file}")
        progress_bar.update(len(completed_samples))
    except Exception as e:
        print(f"Error loading existing results: {e}")

# Create a list of tasks that still need processing
remaining_tasks = []
for msg_idx in range(len(messages_list)):
    for sample_idx in range(n_samples):
        if (msg_idx, sample_idx) not in completed_samples:
            remaining_tasks.append((msg_idx, sample_idx))

print(
    f"Remaining samples to process: {len(remaining_tasks)} out of {len(messages_list) * n_samples}"
)

# Use ThreadPoolExecutor to process remaining messages in parallel
if remaining_tasks:
    with ThreadPoolExecutor(max_workers=data_parallel_workers) as executor:
        futures = {
            executor.submit(process_sample, msg_idx, sample_idx, jsonl_file, jsonl_lock): (
                msg_idx,
                sample_idx,
            )
            for msg_idx, sample_idx in remaining_tasks
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            msg_idx, sample_idx, result = future.result()
            completed_samples[(msg_idx, sample_idx)] = result

# Verify all samples are completed
all_tasks = [
    (msg_idx, sample_idx)
    for msg_idx in range(len(messages_list))
    for sample_idx in range(n_samples)
]
missing_tasks = [task for task in all_tasks if task not in completed_samples]

if missing_tasks:
    print(
        f"Error: {len(missing_tasks)} samples are missing. Cannot create final JSON file."
    )
    for msg_idx, sample_idx in missing_tasks[:10]:  # Show first 10 missing tasks
        print(f"  Missing: message {msg_idx}, sample {sample_idx}")
    if len(missing_tasks) > 10:
        print(f"  ... and {len(missing_tasks) - 10} more.")
    sys.exit(1)

# Organize results by message_idx for the final JSON file
organized_results = [[] for _ in range(len(messages_list))]
for (msg_idx, sample_idx), result in completed_samples.items():
    organized_results[msg_idx].append((sample_idx, result))

# Sort samples within each message and extract just the results
generated_text = []
for msg_results in organized_results:
    sorted_results = [result for _, result in sorted(msg_results)]
    generated_text.append(sorted_results)

# Save the final JSON file
with open(final_json_file, "w") as f:
    json.dump(generated_text, f)
print(f"All samples completed successfully. Final results saved to {final_json_file}")

progress_bar.close()

prompts = df["prompt"]
responses = generated_text
data_sources = df["data_source"]
reward_model_data = df["reward_model"]


def rllm_reward_fn(
    data_source: str,
    llm_solution: str,
    ground_truth: Union[str, List[str]],
    extra_info={},
    **kwargs,
):
    if data_source in [
        "apps",
        "taco",
        "code_contests",
        "codeforces",
        "livecodebench",
        "kodcode",
        "leetcode",
        "primeintellect",
        "humanevalplus",
    ]:
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return False
        raise NotImplementedError(
            f"Reward function for {data_source} is not implemented yet."
        )
    else:
        return rllm_reward_fn_math(
            data_source, llm_solution, ground_truth, extra_info, **kwargs
        )

use_full_reward_fn = True
passes = 0
total = len(df)
total_scores = []
total_parallel = []
# New metric accumulators
total_acceleration_ratios = []
total_parallel_ratios = []
total_num_tokens_list = []
total_num_tokens_in_longest_thread_list = []

for i in range(total):
    response_lst = responses[i]
    data_source = data_sources[i]
    prompt = prompts[i]
    reward_data = reward_model_data[i]
    reward_fn = rllm_reward_fn
    ground_truth = reward_data["ground_truth"]
    score_lst = []
    parallel_lst = []
    # Metrics for this sample
    acceleration_ratios = []
    parallel_ratios = []
    num_tokens_list = []
    num_tokens_in_longest_thread_list = []

    for r in response_lst:
        is_parallel = "<Parallel>" in r
        parallel_lst.append(is_parallel)
        # Multiverse uses <Think> and </Think> tags, so we need to replace them with <think> and </think>
        r = r.replace("<Think>", "<think>").replace("</Think>", "</think>")
        if use_full_reward_fn:
            score = reward_fn(data_source, r, ground_truth, strip_comma_from_answer=args.strip_comma_from_answer)
        else:
            if args.strip_comma_from_answer:
                r = r.replace(",", "")
            score = grade_answer_verl(r, ground_truth)
        score_lst.append(score)

        # Compute parallel stats for each response
        response_token_ids = tokenizer.encode(r, add_special_tokens=False)
        parallel_stats = get_parallel_stats(response_token_ids, special_token_ids)
        acceleration_ratios.append(parallel_stats["acceleration_ratio"])
        parallel_ratios.append(parallel_stats["parallel_ratio"])
        num_tokens_list.append(parallel_stats["total_num_tokens"])
        num_tokens_in_longest_thread_list.append(parallel_stats["num_tokens_in_the_longest_thread"])

    max_score = np.max(score_lst)
    total_scores.append(score_lst)
    total_parallel.append(parallel_lst)
    total_acceleration_ratios.append(acceleration_ratios)
    total_parallel_ratios.append(parallel_ratios)
    total_num_tokens_list.append(num_tokens_list)
    total_num_tokens_in_longest_thread_list.append(num_tokens_in_longest_thread_list)
    if max_score == 1:
        passes += 1

pass_at_n = passes / total
pass_at_1 = np.mean(total_scores)

row_data = {
    "pass@1": pass_at_1,
    f"pass@{n_samples}": pass_at_n,
}

print(
    "With strict grading function:"
    if use_full_reward_fn
    else "With loose grading function:"
)
print(f"Pass@1: {pass_at_1} ({pass_at_1 * 100:.2f})")
print(f"Pass@{n_samples}: {pass_at_n} ({pass_at_n * 100:.2f})")

total_scores = [
    [1.0 if val else 0.0 for val in score_list] for score_list in total_scores
]

# print("Scores:", total_scores)
# True for including <Parallel> tags, False for not including
# print("Parallel responses:", total_parallel)

sampling_accs = []

for idx in range(n_samples):
    sampling_acc = np.mean([item[idx] for item in total_scores])
    sampling_accs.append(sampling_acc)

print(f"Sampling accuracies: {[f'{acc:.2f}' for acc in sampling_accs]}")

# Compute and display average metrics
print("\n" + "="*50)
print("Parallel Execution Metrics:")
print("="*50)

# Flatten all metrics
all_acceleration_ratios = [ratio for sample_ratios in total_acceleration_ratios for ratio in sample_ratios]
all_parallel_ratios = [ratio for sample_ratios in total_parallel_ratios for ratio in sample_ratios]
all_num_tokens = [tokens for sample_tokens in total_num_tokens_list for tokens in sample_tokens]
all_num_tokens_longest = [tokens for sample_tokens in total_num_tokens_in_longest_thread_list for tokens in sample_tokens]

# Compute averages
avg_acceleration_ratio = np.mean(all_acceleration_ratios) if all_acceleration_ratios else 0.0
avg_parallel_ratio = np.mean(all_parallel_ratios) if all_parallel_ratios else 0.0
avg_total_num_tokens = np.mean(all_num_tokens) if all_num_tokens else 0.0
avg_num_tokens_longest = np.mean(all_num_tokens_longest) if all_num_tokens_longest else 0.0

print(f"Average acceleration_ratio: {avg_acceleration_ratio:.4f}")
print(f"Average parallel_ratio: {avg_parallel_ratio:.4f}")
print(f"Average total_num_tokens: {avg_total_num_tokens:.2f}")
print(f"Average num_tokens_in_the_longest_thread: {avg_num_tokens_longest:.2f}")
print("="*50 + "\n")
