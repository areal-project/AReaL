# MOPD example

## GSM8K MOPD (local)

This single-node example distills a Qwen3-14B teacher into a Qwen3-0.6B actor on eight
GPUs. Both checkpoints must use the same token-ID mapping.

```bash
export MOPD_STUDENT_MODEL_PATH=/path/to/Qwen3-0.6B
export MOPD_TEACHER_MODEL_PATH=/path/to/Qwen3-14B
export MOPD_GSM8K_PATH=/path/to/gsm8k
export AREAL_ADMIN_API_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"

python -m examples.mopd.gsm8k_qwen3_14b_to_0_6b \
  --config examples/mopd/gsm8k_qwen3_14b_to_0_6b_local.yaml \
  --dry-run

python -m examples.mopd.gsm8k_qwen3_14b_to_0_6b \
  --config examples/mopd/gsm8k_qwen3_14b_to_0_6b_local.yaml
```
