"""Rerun no-mem HLE for specific question IDs that got empty responses.

Usage:
    python3 scripts/rerun_nomem_empty.py \
        --config configs/rl_hle_config.nomem_gemini35flash.yaml \
        --train data/hle/hle_test.parquet \
        --empty_qids data/hle/nomem_empty_qids.json \
        --output_dir /path/to/original/local_cache \
        --judge_model gpt-4o-2024-11-20 \
        --judge_base_url https://matrixllm.alipay.com/v1/ \
        --judge_api_key <key>
"""
import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from memrl.configs.config import MempConfig
from memrl.providers.llm import OpenAILLM


def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if root.hasHandlers():
        root.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--empty_qids", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--judge_model", default="gpt-4o-2024-11-20")
    parser.add_argument("--judge_base_url", default=None)
    parser.add_argument("--judge_api_key", default=None)
    args = parser.parse_args()

    cfg = MempConfig.from_yaml(args.config)

    with open(args.empty_qids) as f:
        empty_qids = json.load(f)
    logger.info(f"Loaded {len(empty_qids)} empty question IDs to rerun")

    import pandas as pd
    df = pd.read_parquet(args.train)
    rerun_df = df[df['id'].isin(empty_qids)].reset_index(drop=True)
    logger.info(f"Matched {len(rerun_df)} questions in dataset")

    llm = OpenAILLM(
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
        model=cfg.llm.model,
        default_temperature=cfg.llm.temperature,
        default_max_tokens=cfg.llm.max_tokens,
        token_log_dir=args.output_dir,
    )

    llm_judge = OpenAILLM(
        api_key=(args.judge_api_key or cfg.llm.api_key),
        base_url=(args.judge_base_url or cfg.llm.base_url),
        model=args.judge_model,
        default_temperature=0.0,
        default_max_tokens=4096,
        token_log_dir=args.output_dir,
    )

    output_path = Path(args.output_dir) / "rerun_results.jsonl"
    max_retries = int(os.environ.get("MEMRL_RUNNER_MAX_RETRIES", "5"))

    system_prompt = (
        "Your response should be in the following format:\n"
        "Explanation: {your explanation for your final answer}\n"
        "Exact Answer: {your succinct, final answer}\n"
        "Confidence: {your confidence score between 0% and 100% for your answer}"
    )

    judge_template = (
        "You are an expert judge evaluating answer correctness.\n\n"
        "[question]: {question}\n\n"
        "[correct_answer]: {gold}\n\n"
        "[response]: {response}\n\n"
        "Extract the final answer from the response and compare it to the correct answer.\n"
        "Respond with:\n"
        "extracted_final_answer: <the answer extracted from response>\n"
        "reasoning: <your reasoning>\n"
        "correct: <yes or no>"
    )

    correct_count = 0
    total_count = 0

    for idx, row in rerun_df.iterrows():
        qid = row['id']
        question = str(row['question'])
        gold = str(row['answer'])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        kwargs = {"messages": messages, "temperature": cfg.llm.temperature}
        _re = os.environ.get("MEMRL_REASONING_EFFORT", "").strip()
        if _re and cfg.llm.model.startswith("gemini-3"):
            kwargs["reasoning_effort"] = _re
            _max_comp = int(os.environ.get("MEMRL_MAX_COMPLETION_TOKENS", "0") or "0")
            if _max_comp > 0:
                kwargs["max_completion_tokens"] = _max_comp
        elif not cfg.llm.model.startswith("gemini-3"):
            kwargs["max_tokens"] = cfg.llm.max_tokens

        output = ""
        for attempt in range(max_retries + 1):
            try:
                output = llm.generate(**kwargs)
                if output.strip():
                    break
            except Exception as e:
                logger.warning(f"[{qid}] attempt {attempt+1} failed: {e}")
            if attempt < max_retries:
                wait = 60
                logger.info(f"[{qid}] empty response, waiting {wait}s...")
                time.sleep(wait)

        if not output.strip():
            logger.error(f"[{qid}] all {max_retries+1} attempts failed, skipping")
            result = {"id": qid, "question": question, "gold": gold, "output": "", "correct": False, "error": "all_retries_failed"}
        else:
            judge_prompt = judge_template.format(question=question[:2000], gold=gold, response=output[:3000])
            judge_msgs = [{"role": "user", "content": judge_prompt}]
            try:
                judge_resp = llm_judge.generate(judge_msgs, temperature=0.0, max_tokens=4096)
                correct = "correct: yes" in judge_resp.lower() or "correct:yes" in judge_resp.lower()
            except Exception as e:
                logger.warning(f"[{qid}] judge failed: {e}")
                correct = False
                judge_resp = str(e)

            if correct:
                correct_count += 1
            total_count += 1

            result = {
                "id": qid,
                "question": question[:200],
                "gold": gold,
                "output": output[:500],
                "correct": correct,
                "judge_response": judge_resp[:300],
            }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if total_count % 10 == 0:
            logger.info(f"Progress: {total_count}/{len(rerun_df)}, correct so far: {correct_count}/{total_count}")

    logger.info(f"Done. {correct_count}/{total_count} correct ({correct_count/total_count*100:.1f}%)" if total_count else "No results")


if __name__ == "__main__":
    main()
