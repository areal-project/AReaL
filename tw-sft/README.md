# ThreadWeaver SFT Quickstart

## Install dependencies
```bash
pip install numpy==1.26.4 pandas==2.2.3 matplotlib==3.10.3 sympy==1.13.1 torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 torchao==0.11.0 transformers==4.51.1 datasets==3.6.0 tokenizers==0.21.1 huggingface-hub==0.31.4 safetensors==0.5.3 compressed-tensors==0.9.3 openai==1.75.0 sglang==0.4.6.post1 sgl-kernel==0.1.0 xgrammar==0.1.18 trl==0.19.0 accelerate==1.7.0 peft==0.15.2 deepspeed==0.17.0 liger_kernel==0.5.10 xformers==0.0.29.post2 wandb==0.21.0 tensorboard==2.19.0 nvidia-cuda-runtime-cu12==12.4.127 nvidia-cudnn-cu12==9.1.0.70 nvidia-nccl-cu12==2.21.5 nvidia-cublas-cu12==12.4.5.8 nvidia-cufft-cu12==11.2.1.3 nvidia-curand-cu12==10.3.5.147 nvidia-cusolver-cu12==11.6.1.9 nvidia-cusparse-cu12==12.3.1.170 nvidia-nvtx-cu12==12.4.127 nvidia-nvjitlink-cu12==12.4.127 tqdm==4.67.1 termcolor==3.1.0 packaging==25.0 typing-extensions==4.13.2 pyyaml==6.0.2 regex==2024.11.6 psutil==7.0.0 filelock==3.18.0 flask==2.3.3 fastapi==0.115.12 uvicorn==0.34.2 uvloop==0.21.0 python-multipart==0.0.20 pylatexenc==2.10 requests==2.32.3 pyzmq==26.4.0 orjson==3.10.18 partial-json-parser==0.2.1.1.post5
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## Generate data
See [data/README.md](data/README.md) to generate data for a synthetic multiplication task.

## Obtain model with longer context
```bash
huggingface-cli download Qwen/Qwen3-8B --local-dir ./Qwen/Qwen3-8B-131072
FILE="Qwen/Qwen3-8B-131072/config.json"

if ! grep -q '"max_position_embeddings": 40960,' "$FILE"; then
  echo "Error: target string not found"
else
  sed -i 's/"max_position_embeddings": 40960,/"max_position_embeddings": 131072,/' "$FILE"
fi
```

## Training
- Prepare a parquet dataset and set `TRAIN_DATA` to its path (must expose `qwen_text`).

Run:
```bash
# Train with synthetic multiplication data:
OUTPUT_DIR=ckpts/Q3-8B-131072-SFT ./train.sh

# Train with your custom training data:
# TRAIN_DATA=/path/to/train/data ./train.sh
```
Key knobs inside `train.sh`: `lr`, `epochs`, `micro_batch_size`, `gradient_accumulation_steps`, `block_size`, `attn_implementation`, `gpu_count`, and `base_model`. Outputs land in `ckpts/Q3-8B-131072-SFT-<timestamp>`.

We also offer a script that trains a sequential baseline:
```bash
OUTPUT_DIR=ckpts/Q3-8B-131072-AR-SFT ./train_ar.sh
```

Note that the sequential baseline uses the same dataset as the parallel model for fairness. Special tokens such as `<Parallel>` are treated as normal text tokens.

## Attention/Position Visualizer for Training Data
Serves tokens plus attention mask/position ids for a dataset sample using the prefix-tree collator.

Run:
```bash
python src/prefix_tree_visualizer.py \
  --dataset-path data/mult-10k-par_pq/train.parquet \
  --model-name Qwen/Qwen3-8B-131072 \
  --port 8008
```
Open `http://localhost:8008`, pick a sample index, and click tokens to inspect their attention rows. `--text-field` defaults to `qwen_text`; `--template-name` defaults to `qwen`. Uses CPU if CUDA is unavailable.***

## Quick Evaluation
This is a quick evaluation script that runs parallel generation with SGLang. We recommend using the parallel rollout implementation in veRL for evaluation (see [threadweaver_rl/README.md](threadweaver_rl/README.md)).

```bash
# Replace with your trained model path
TRAINED_MODEL="ckpts/Q3-8B-131072-SFT"

python src/simple_eval.py --data-type data/mult-10k-par_pq/train.parquet --model_name $TRAINED_MODEL --launch_server --verbose 2 --template-type model --bfloat16 --branching-generate -n 1 --max-context-length 8192
```

Reference result:
```
With strict grading function:
Pass@1: 0.9377 (93.77)
```

## Pause-Limit Sweep (4096 ~ 40960)
To evaluate one checkpoint across multiple `num_tokens_in_the_longest_thread` pause limits and generate a summary report:

```bash
./eval_pause_sweep.sh <checkpoint_path> --data-type <data_path> [-n <num>] [--bfloat16] [--verbose <level>] [extra args for simple_eval_pause.py]
```

Example:

```bash
./eval_pause_sweep.sh ckpts/Q3-8B-131072-SFT --data-type data/mult-10k-par_pq/train.parquet -n 32 --bfloat16 --verbose 2
```

Data path resolution order:
- `--data-type` / `-d` CLI argument (recommended)
- `DATA_TYPE` environment variable
- interactive terminal prompt if neither is provided

Other optional environment overrides:
- `N_SAMPLES` (default: `1`, can be overridden by `-n` / `--n-samples`)
- `MAX_CONTEXT_LENGTH` (default: `40960`)
- `TEMPLATE_TYPE` (default: `model`)
- `VERBOSE_LEVEL` (used if `--verbose` is not provided)

This runs `src/simple_eval_pause.py` at pause limits:
`4096, 8192, 16384, 24576, 32768, 40960`.

Outputs are saved under the checkpoint directory:
- `eval_pause_outputs/<run_name>/*_metrics.json` (per-run details including `num_tokens_in_longest_thread_list`)
- `eval_pause_outputs/<run_name>/*_report.md` (per-run human-readable report)
- `eval_pause_outputs/pause_sweep_report.csv`
- `eval_pause_outputs/pause_sweep_report.md`
