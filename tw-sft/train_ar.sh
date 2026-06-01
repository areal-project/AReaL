uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen3-8B-131072"
lr=1e-5
min_lr=0
epochs=1
weight_decay=1e-4
micro_batch_size=1
gradient_accumulation_steps=2
max_steps=-1
push_to_hub=false
OUTPUT_DIR=${OUTPUT_DIR:-"ckpts/Q3-8B-131072-AR-SFT-${uid}"}

export TRAIN_DATA="${TRAIN_DATA:-./data/mult-10k-par}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node gpu --master_port 12345 \
    src/sft_autoregressive.py \
    --block_size=40960 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path="$TRAIN_DATA" \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --deepspeed configs/deepspeed_zero3_offload.json \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="ckpts/Q3-8B-131072-AR-SFT-${uid}" \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --use-liger=True \
    --dataset_text_field="qwen_text" \
    --attn_implementation="flex_attention" \
    --template_name="qwen" \
    --report_to="wandb" \
    "$@"
