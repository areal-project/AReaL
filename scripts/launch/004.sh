WANDB_API_KEY=local-43c7f4f9694e7f09e8b41adfe04bf7f140a1c7a6 \
WANDB_BASE_URL=http://8.150.1.98:8080 \
python3 swe_opcd/jsonl_sft.py \
    --config swe_opcd/jsonl_sft.yaml \
    scheduler.type=ray \
    cluster.n_nodes=2 \
    experiment_name=whj-swe-opcd \
    trial_name=distill-qwen3-coder-exp5 \
    stats_logger.wandb.mode=online \
    evaluator.freq_epochs=null \
    train_dataset.path=/storage/openpsi/users/wanghuaijie.whj/datasets/swe_opcd/train400-qwen3-coder-exp5.jsonl \
    valid_dataset.path=/storage/openpsi/users/wanghuaijie.whj/datasets/swe_opcd/train400-qwen3-coder-exp5.jsonl \
    total_train_epochs=4 \
    actor.path=/storage/openpsi/models/Qwen3-Coder-30B-A3B-Instruct/ \
    actor.optimizer.lr=1e-5 \
    actor.optimizer.lr_scheduler_type=constant \
    +actor.optimizer.warmup_steps_proportion=0.1 \
