# GSM8K smoke test: 2 × 8 B300 on AIS

Run from the repository root with the AIS SDK installed (`aistudio-common==0.0.28.84`
and `aii-pypai==0.1.42.19`). Export your `AISTUDIO_USERNUMBER` and `AISTUDIO_TOKEN`. Set
`WANDB_API_KEY` at the top of your local copy of `run_ais_roce16.sh`.

```bash
export QWEN_MODEL='/storage/models/<model-directory>'
export QWEN_DATASET='/storage/<gsm8k-dataset-directory>'
export QWEN_ARENA_OUTPUT=/storage/experiments/areal

python submit.py --nodes 2 --rdma --host-network \
    --pass-env QWEN_MODEL --pass-env QWEN_DATASET --pass-env QWEN_ARENA_OUTPUT \
    --command 'bash examples/math/qwen38_27b_ray_32gpu/run_ais_roce16.sh' \
    --submit
```

Each job joins both nodes into Ray before the head runs the command. The script defaults
to 10 training steps, a checkpoint every 2 steps, and `https://slurm.alipay.com/` for
W&B. It uses the AIS job name as its trial name. Set `QWEN_ARENA_EXPERIMENT` or
`QWEN_ARENA_TRIAL` and forward it with `--pass-env` to choose a different name.

Remove `--submit` for a local preview. During training, checkpoints and logs go under
`QWEN_ARENA_OUTPUT`; launcher records are stored in `submissions/<experiment>/<trial>/`.
