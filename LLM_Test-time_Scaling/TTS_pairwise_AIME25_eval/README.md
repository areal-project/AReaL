# TTS Pairwise AIME25 Evaluation

Evaluates a model's ability to judge pairwise solution comparisons on AIME 2025 problems, then aggregates those judgments via a tournament to select the best solution per problem.



## Overview

Given a JSONL file of pairwise solution comparisons (two candidate solutions per math problem), this pipeline:

1. **Serves the model** via vLLM with OpenAI-compatible API
2. **Runs pairwise judgments** — the model reads each pair and decides which solution is better
3. **Grades judgments against ground truth** — extracts `\boxed{}` answers, checks correctness via `imobench.json`, and determines if the model's verdict agrees with ground truth
4. **Runs tournament aggregation** — groups comparisons by problem, tallies wins across all pairs, selects the best solution per problem, and computes pass@1
5. **Repeats** for N independent runs and averages the results

## Pipeline Stages


### Stage 0: Convert the dumped json file into jsonl file with ChatML format prompts (`convert_to_pairwise_jsonl.py`)

```
python3 convert_to_pairwise_jsonl.py \
  --input /path/to/your/dumped/json \
  --output pairwise.jsonl
```

### Stage 1: Pairwise Aggregation (`test_training_data_eval.py`)

Sends each pairwise comparison prompt to the vLLM server. For each line in the JSONL dataset:

- Extracts the ChatML prompt (problem + two solutions in `<Chunk>` blocks)
- Queries the model via `/v1/completions` (falls back to `/v1/chat/completions`)
- Parses the model's "better solution" verdict (1, 2, or 0 if unparseable) using regex patterns
- Compares model verdict against the reference verdict embedded in the dataset
- Outputs a JSONL file with per-pair results

### Stage 2: Ground-Truth Grading (`grade_against_aime25.py`)

Reads Stage 1 output and grades each verdict against AIME25 ground truth:

- Looks up the original pairwise JSONL to extract `\boxed{}` answers from both solutions
- Matches the problem to `imobench.json` ground truth (exact match, then substring fallback)
- Determines which solution is correct based on boxed answer vs ground truth
- Derives a ground-truth verdict (1 = sol1 correct, 2 = sol2 correct, 0 = tie)
- Compares model verdict against ground-truth verdict
- Reports **GT Accuracy (pass@1)**: fraction of non-tie pairs where the model picked the correct solution

### Stage 3: Tournament Aggregation (`tournament_aggregate.py`)

Groups all pairwise comparisons by problem and selects the best solution:

- Discovers unique solutions per problem from all pairs
- Tallies wins: verdict=1 gives sol1 a win, verdict=2 gives sol2 a win, verdict=0 splits 0.5/0.5
- Selects the solution with the most wins as the "best"
- Checks if the best solution's `\boxed{}` answer matches ground truth
- Reports **Tournament pass@1**: fraction of problems where the tournament-selected solution is correct

## Input Files

| File | Description |
|------|-------------|
| **Pairwise JSONL** | Each line has a `text` field containing a ChatML prompt with a problem and two solutions in `<Chunk>` blocks, plus a reference answer |
| **Ground truth JSON** (`imobench.json`) | Contains `problems` array with `problem` (text) and `ground_truth` (answer) fields |

## Quick Start

```bash
bash TTS_pairwise_AIME25_eval/run_pairwise_aime25_pipeline.sh \
  --jsonl-file pairwise_from_direct_generation_AIME25.jsonl \
  --ground-truth-file LLM_Test-time_Scaling/imobench.json \
  --model-path /path/to/model \
  --model-name qwen3-4b-newenc
```

### Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num-runs` | 8 | Number of independent runs to average |
| `--sample-size` | 0 (all) | Lines to process per run |
| `--temperature` | 0.6 | Sampling temperature |
| `--max-tokens` | 16384 | Max generation tokens |
| `--concurrency` | 1 | Concurrent requests to vLLM |
| `--tensor-parallel-size` | 4 | vLLM tensor parallelism |
| `--port` | 8000 | vLLM server port |

## Output

Results are written to `results/tts_pairwise_aime25_eval/runs/<model_name>/<timestamp>/`:

- `eval_results_runN.jsonl` — Stage 1 per-pair judgments
- `graded_results_runN.json` — Stage 2 ground-truth grading with aggregate metrics
- `tournament_results_runN.json` — Stage 3 tournament results with per-problem breakdown
- `summary_<timestamp>.csv` — averaged GT accuracy and tournament pass@1 across all runs

## Running Individual Scripts

```bash
# Stage 1 only
python3 test_training_data_eval.py \
  --dataset pairwise.jsonl \
  --api-base http://127.0.0.1:8000/v1 \
  --model-name qwen3-4b-newenc \
  --output eval_results.jsonl

# Stage 2 only
python3 grade_against_aime25.py \
  --eval-jsonl eval_results.jsonl \
  --pairwise-jsonl pairwise.jsonl \
  --ground-truth imobench.json

# Stage 3 only
python3 tournament_aggregate.py \
  --eval-jsonl eval_results.jsonl \
  --pairwise-jsonl pairwise.jsonl \
  --ground-truth imobench.json
```
