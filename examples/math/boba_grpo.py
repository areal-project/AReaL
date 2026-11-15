import hashlib
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.reward import get_math_verify_worker
from areal.utils.hf_utils import apply_chat_template, load_hf_tokenizer
from areal.utils.perf_tracer import record_disposition
from areal.utils.stats_logger import StatsLogger


def get_input_ids_fn(data, tokenizer, enable_thinking):
    user_token = "<｜User｜>"
    assistant_token = "<｜Assistant｜>"
    think_token = "<think>"
    has_think_token = think_token in data
    data = (
        data.replace(user_token, "")
        .replace(assistant_token, "")
        .replace(think_token, "")
    )
    input_ids = apply_chat_template(
        tokenizer,
        [{"role": "user", "content": data}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking or has_think_token,
    )
    return input_ids


def data_extract_prompt_fn(data):
    return data["prompt"]


def get_boba_math_dataset(path, tokenizer):
    dataset = load_dataset(
        path="json",
        split="train",
        data_files=path,
    )
    dataset = dataset.filter(lambda x: len(tokenizer.encode(x["prompt"])) <= 1024)
    return dataset


def boba_reward_fn(
    prompts, completions, prompt_ids, completion_ids, solutions, **kwargs
) -> float:
    try:
        worker = get_math_verify_worker()
        for sol in solutions:
            try:
                score = worker.verify(str(completions), str(sol))
                if score == 1.0:
                    return 1.0
            except Exception as exc:
                record_disposition(
                    "failure",
                    "verifier",
                    "verify_exception",
                    unit="verification",
                    exception=type(exc).__name__,
                    error=str(exc),
                    outcome="try_next_solution",
                )
        return 0.0
    except Exception as exc:
        record_disposition(
            "failure",
            "verifier",
            "verifier_exception",
            unit="verification",
            exception=type(exc).__name__,
            error=str(exc),
            outcome="zero_reward",
        )
        # Return 0 if completion parsing fails or any other error occurs
        return 0.0


def _load_inputs(args, *, record_dataset=False):
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    source = Path(config.train_dataset.path)
    record_dataset = record_dataset or (
        config.rollout.dump_to_file and source.is_file()
    )
    source_hash = None
    if record_dataset:
        with source.open("rb") as stream:
            source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    train_dataset = get_boba_math_dataset(config.train_dataset.path, tokenizer)
    if record_dataset and os.getenv("RANK", "0") == "0":
        with source.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != source_hash:
                raise ValueError("BOBA source changed while loading the dataset")
        ids = [row.get("query_id") for row in train_dataset]
        path = (
            Path(StatsLogger.get_log_path(config.stats_logger)) / "loaded-dataset.json"
        )
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "source_path": str(source.resolve()),
                    "source_sha256": source_hash,
                    "sample_count": len(train_dataset),
                    "ordered_ids": ids,
                    "filter": "len(tokenizer.encode(raw_prompt)) <= 1024",
                },
                indent=2,
            )
            + "\n"
        )
        temporary.replace(path)
    return config, tokenizer, train_dataset


def preflight(args):
    """Resolve configuration and record loaded data before creating RL workers."""
    _load_inputs(args, record_dataset=True)


def main(args):
    config, tokenizer, train_dataset = _load_inputs(args)

    workflow_kwargs = dict(
        reward_fn="examples.math.boba_grpo.boba_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=True,
        get_input_ids_fn="examples.math.boba_grpo.get_input_ids_fn",
        data_extract_prompt_fn="examples.math.boba_grpo.data_extract_prompt_fn",
    )

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=None,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow=None,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
