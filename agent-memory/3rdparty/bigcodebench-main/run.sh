DATASET=bigcodebench
MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
BACKEND=vllm
NUM_GPU=2
SPLIT=complete
SUBSET=full
export E2B_API_KEY="e2b_0a231fa3b0a2b01690ab6c66a23b55c0979ce4ee"

bigcodebench.evaluate \
  --model $MODEL \
  --split $SPLIT \
  --subset $SUBSET \
  --backend $BACKEND