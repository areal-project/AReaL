export USE_OPTIMIZED_MODEL=0
# Some models are optimized by vllm ascend. While in some case, e.g. rlhf training,
# the optimized model may not be suitable. In this case, set this value to 0 to disable the optimized model.

python examples/vlm_npu/geometry3k_grpo.py \
    --config examples/vlm_npu/qwen3_vl_2b_geometry3k_grpo.yaml \
    scheduler.type=local
