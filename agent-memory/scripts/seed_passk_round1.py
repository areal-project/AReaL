"""Pre-seed pass@k round 1 results from a completed no-mem experiment.

Usage:
    python3 scripts/seed_passk_round1.py \
        --nomem-ckpt /path/to/exp_hle_nomem_gemini35flash_.../local_cache \
        --passk-log-dir /path/to/exp_hle_passk9_gemini35flash_.../local_cache
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nomem-ckpt", required=True, help="Path to nomem local_cache dir")
    parser.add_argument("--passk-log-dir", required=True, help="Path to passk local_cache dir (will be created)")
    args = parser.parse_args()

    nomem_path = Path(args.nomem_ckpt) / "llm_calls.jsonl"
    passk_dir = Path(args.passk_log_dir)
    passk_dir.mkdir(parents=True, exist_ok=True)
    result_path = passk_dir / "baseline_passk_results.jsonl"

    results = []
    with open(nomem_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("type") == "judge":
                resp = d.get("response", "").lower()
                correct = (
                    "are equivalent" in resp
                    or "matches the" in resp
                    or "is equivalent" in resp
                )
                qid = d["meta"].get("question_id", "")
                results.append({
                    "round": 1,
                    "baseline": "passk",
                    "id": qid,
                    "question_id": qid,
                    "question": d["meta"].get("question", ""),
                    "gold": d["meta"].get("gold", ""),
                    "correct": correct,
                })

    with open(result_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    correct = sum(1 for r in results if r["correct"])
    print(f"Seeded {len(results)} round-1 results ({correct} correct, {correct/len(results)*100:.1f}%) -> {result_path}")


if __name__ == "__main__":
    main()
