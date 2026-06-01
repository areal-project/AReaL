#!/usr/bin/env bash
set -euo pipefail

uid="$(date +%Y%m%d_%H%M%S)"
base_model="${BASE_MODEL:-/storage/openpis/models/Qwen__Qwen3-8B}"
lr=1e-5
epochs=8
weight_decay=1e-4
micro_batch_size=1
gradient_accumulation_steps=1
push_to_hub=false
OUTPUT_DIR=${OUTPUT_DIR:-"ckpts/Q3-8B-131072-SFT-${uid}"}
TRAIN_DATA="${TRAIN_DATA:-./data/mult-10k-par}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
RAY_WORKING_DIR="${RAY_WORKING_DIR:-$SCRIPT_DIR}"
RAY_ENTRYPOINT_NUM_GPUS="${RAY_ENTRYPOINT_NUM_GPUS:-8}"
RAY_ENTRYPOINT_NUM_CPUS="${RAY_ENTRYPOINT_NUM_CPUS:-16}"
RAY_RUNTIME_ENV_JSON="${RAY_RUNTIME_ENV_JSON:-}"
RAY_SUBMISSION_ID="${RAY_SUBMISSION_ID:-threadweaver-sft-${uid}}"
RAY_NO_WAIT=false
DRY_RUN=false
MASTER_PORT="${MASTER_PORT:-12345}"

extra_args=()

usage() {
    cat <<EOF
Usage: $0 [options] [extra_args_for_sft_threadweaver.py]

Model/data options (same as train.sh):
  --original_model_path <path>  Original/base model path
  --base_model <path>           Alias of --original_model_path
  --output_dir <path>           Output directory (default: \$OUTPUT_DIR)
  --dataset_dir <path>          Dataset path (can be a parquet file path)
  --dataset_path <path>         Alias of --dataset_dir
  --train_data <path>           Alias of --dataset_dir

Ray options:
  --ray_address <url>           Ray Jobs API address (default: \$RAY_ADDRESS)
  --ray_working_dir <path>      Working directory uploaded to Ray (default: script directory)
  --ray_num_gpus <num>          GPUs requested by Ray entrypoint (default: \$RAY_ENTRYPOINT_NUM_GPUS)
  --ray_num_cpus <num>          CPUs requested by Ray entrypoint (default: \$RAY_ENTRYPOINT_NUM_CPUS)
  --ray_runtime_env_json <json> Runtime env JSON for Ray job submit
  --ray_submission_id <id>      Ray submission id (default: auto-generated)
  --ray_no_wait                 Submit job and return immediately
  --master_port <port>          torchrun master port (default: \$MASTER_PORT)
  --dry_run                     Print the resolved Ray command and exit
  -h, --help                    Show this help message

Examples:
  bash train_ray.sh \\
    --ray_address http://127.0.0.1:8265 \\
    --ray_num_gpus 8 \\
    --dataset_dir /mnt/datasets/train.parquet \\
    --output_dir /mnt/ckpts/Q3-8B-131072-SFT-ray
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --original_model_path|--base_model|--model_name)
            [ -n "${2:-}" ] || { echo "Error: $1 requires a value."; exit 1; }
            base_model="$2"
            shift 2
            ;;
        --output_dir)
            [ -n "${2:-}" ] || { echo "Error: --output_dir requires a value."; exit 1; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dataset_dir|--dataset_path|--train_data)
            [ -n "${2:-}" ] || { echo "Error: $1 requires a value."; exit 1; }
            TRAIN_DATA="$2"
            shift 2
            ;;
        --ray_address|--ray-address)
            [ -n "${2:-}" ] || { echo "Error: $1 requires a value."; exit 1; }
            RAY_ADDRESS="$2"
            shift 2
            ;;
        --ray_working_dir|--ray-working-dir)
            [ -n "${2:-}" ] || { echo "Error: $1 requires a value."; exit 1; }
            RAY_WORKING_DIR="$2"
            shift 2
            ;;
        --ray_num_gpus|--ray-num-gpus)
            [ -n "${2:-}" ] || { echo "Error: $1 requires a value."; exit 1; }
            RAY_ENTRYPOINT_NUM_GPUS="$2"
            shift 2
            ;;
        --ray_num_cpus|--ray-num-cpus)
            [ -n "${2:-}" ] || { echo "Error: $1 requires a value."; exit 1; }
            RAY_ENTRYPOINT_NUM_CPUS="$2"
            shift 2
            ;;
        --ray_runtime_env_json|--ray-runtime-env-json)
            [ -n "${2:-}" ] || { echo "Error: $1 requires a value."; exit 1; }
            RAY_RUNTIME_ENV_JSON="$2"
            shift 2
            ;;
        --ray_submission_id|--ray-submission-id)
            [ -n "${2:-}" ] || { echo "Error: $1 requires a value."; exit 1; }
            RAY_SUBMISSION_ID="$2"
            shift 2
            ;;
        --ray_no_wait|--ray-no-wait)
            RAY_NO_WAIT=true
            shift
            ;;
        --master_port|--master-port)
            [ -n "${2:-}" ] || { echo "Error: $1 requires a value."; exit 1; }
            MASTER_PORT="$2"
            shift 2
            ;;
        --dry_run|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            extra_args+=("$1")
            shift
            ;;
    esac
done

if [ "$DRY_RUN" = false ] && ! command -v ray >/dev/null 2>&1; then
    echo "Error: ray CLI is not available. Install Ray first (e.g., pip install \"ray[default]\")."
    exit 1
fi

if [[ "$RAY_WORKING_DIR" != /* ]]; then
    RAY_WORKING_DIR="$(cd "$RAY_WORKING_DIR" && pwd)"
fi

export TRAIN_DATA

train_cmd=(
    torchrun
    --nproc-per-node
    gpu
    --master_port
    "$MASTER_PORT"
    src/sft_threadweaver.py
    --block_size=40960
    --per_device_train_batch_size="${micro_batch_size}"
    --per_device_eval_batch_size="${micro_batch_size}"
    --gradient_accumulation_steps="${gradient_accumulation_steps}"
    --num_train_epochs="${epochs}"
    --train_file_path="$TRAIN_DATA"
    --model_name="${base_model}"
    --warmup_ratio=0.05
    --deepspeed=configs/deepspeed_zero3_offload.json
    --bf16=True
    --eval_strategy=no
    --logging_steps=1
    --save_strategy=no
    --lr_scheduler_type=cosine
    --learning_rate="${lr}"
    --weight_decay="${weight_decay}"
    --adam_beta1=0.9
    --adam_beta2=0.95
    --output_dir="${OUTPUT_DIR}"
    --push_to_hub="${push_to_hub}"
    --save_only_model=True
    --gradient_checkpointing=True
    --use-liger=True
    --dataset_text_field=qwen_text
    --attn_implementation=flex_attention
    --template_name=qwen
    --report_to=wandb
)

if [ ${#extra_args[@]} -gt 0 ]; then
    train_cmd+=("${extra_args[@]}")
fi

train_cmd_escaped="$(printf '%q ' "${train_cmd[@]}")"
train_cmd_escaped="${train_cmd_escaped% }"
entrypoint_cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${train_cmd_escaped}"

ray_cmd=(
    ray
    job
    submit
    --address
    "$RAY_ADDRESS"
    --working-dir
    "$RAY_WORKING_DIR"
    --entrypoint-num-gpus
    "$RAY_ENTRYPOINT_NUM_GPUS"
)

if [ -n "$RAY_ENTRYPOINT_NUM_CPUS" ]; then
    ray_cmd+=(--entrypoint-num-cpus "$RAY_ENTRYPOINT_NUM_CPUS")
fi

if [ -n "$RAY_RUNTIME_ENV_JSON" ]; then
    ray_cmd+=(--runtime-env-json "$RAY_RUNTIME_ENV_JSON")
fi

if [ -n "$RAY_SUBMISSION_ID" ]; then
    ray_cmd+=(--submission-id "$RAY_SUBMISSION_ID")
fi

if [ "$RAY_NO_WAIT" = true ]; then
    ray_cmd+=(--no-wait)
fi

ray_cmd+=(-- bash -lc "$entrypoint_cmd")

if [ "$DRY_RUN" = true ]; then
    echo "Resolved Ray command:"
    printf '%q ' "${ray_cmd[@]}"
    echo
    exit 0
fi

"${ray_cmd[@]}"
