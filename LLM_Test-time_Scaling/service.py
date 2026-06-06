
hostname -I

export OPENAI_API_BASE=http://10.18.28.22:8000/v1
export OPENAI_API_KEY=dummy

new-encoding
export OPENAI_API_BASE=http://10.18.8.243:8000/v1
export OPENAI_API_KEY=dummy

curl http://10.18.14.145:8000/v1/models
curl http://10.18.4.140:8000/v1/models

vllm serve /storage/openpsi/models/Qwen__Qwen3-4B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --served-model-name qwen3-4b

vllm serve /storage/openpsi/models/Qwen3-30B-A3B \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 8 \
  --served-model-name qwen3-30b-a3b
  