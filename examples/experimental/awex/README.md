# Experimental Awex GSM8K Example

This folder provides a minimal AWEX config for GSM8K GRPO in single-controller mode.

## Files

| File                              | Purpose                                        |
| --------------------------------- | ---------------------------------------------- |
| `gsm8k_grpo_awex_sample.yaml`     | Minimal config for local validation on 2 GPUs. |
| `gsm8k_grpo_awex_npu_sample.yaml` | Minimal config for local validation on 2 NPUs. |

## Install Awex

Awex may already be installed in your environment (it ships with the AReaL NPU/GPU
container images). Check first:

```bash
python -c "import awex, importlib.metadata as m; print('awex', m.version('awex'))"
```

If it is missing, install it from PyPI:

```bash
pip install awex
```

Or from source:

```bash
git clone https://github.com/inclusionAI/asystem-awex.git
cd asystem-awex
pip install -e .
```

Once installed, AReaL can use AWEX directly. The vLLM-side AWEX integration is
discovered through the installed package, so no extra `VLLM_PLUGINS` configuration is
needed in the normal AReaL workflow.

## Quickstart

Run from repo root (`AReaL/`).

```bash
python examples/math/gsm8k_rl.py \
  --config examples/experimental/awex/gsm8k_grpo_awex_sample.yaml
```

NPU sample:

```bash
python examples/math/gsm8k_rl.py \
  --config examples/experimental/awex/gsm8k_grpo_awex_npu_sample.yaml
```

## Runtime behavior

- If Awex is enabled and `awex.meta_server_addr` is empty or `auto`, `PPOTrainer` starts
  a local Awex meta server automatically before rollout initialization in
  single-controller mode.
- In SPMD mode, set `awex.meta_server_addr` explicitly instead of relying on auto-start.

## Environment variables

| Env var                       | Meaning                                                                                                 |
| ----------------------------- | ------------------------------------------------------------------------------------------------------- |
| `AREAL_AWEX_META_SERVER_ADDR` | Preferred external meta server addr (`ip:port`). If unset, the trainer follows `awex.meta_server_addr`. |
| `AWEX_META_SERVER_ADDR`       | Generic external meta server addr override.                                                             |

## Notes

- Use `actor.path`, `ref.path`, `tokenizer_path`, and `vllm.model` in the yaml to point
  to the model you want to validate.
- `gsm8k_grpo_awex_sample.yaml` targets a small local GPU validation setup.
- `gsm8k_grpo_awex_npu_sample.yaml` targets a small local NPU validation setup.

## Adapting the sample for NPU

Use `gsm8k_grpo_awex_npu_sample.yaml` directly for the single-controller NPU path. If
you need a larger NPU training config, start from `examples/math/gsm8k_grpo_npu.yaml`
and copy the AWEX settings from the NPU sample here.
