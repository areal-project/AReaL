export USE_OPTIMIZED_MODEL=0
# Some models are optimized by vllm ascend. While in some case, e.g. rlhf training,
# the optimized model may not be suitable. In this case, set this value to 0 to disable the optimized model.

python examples/multi_turn_vlm/geometry3k_grpo_mt.py \
    --config examples/multi_turn_vlm/qwen3_vl_2b_geometry3k_grpo_mt.yaml \
    scheduler.type=local \
    "$@"
