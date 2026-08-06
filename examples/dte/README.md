# DTE Colocated Weight Transfer

This example runs GSM8K GRPO with Megatron training and SGLang rollout workers sharing
the same GPUs through AWEX. DTE supports full updates and incremental updates detected
either from AdamW state or from a weight snapshot.

Install the optional dependency:

```bash
python -m pip install -e ".[delta]"
```

The extra uses the public `areal-project/AReaL-DTE` repository at the revision recorded
in `uv.lock`.

Run the AdamW delta configuration:

```bash
python examples/math/gsm8k_rl.py --config examples/dte/gsm8k_dte.yaml
```

Use full transfer or snapshot-based delta detection with CLI overrides:

```bash
python examples/math/gsm8k_rl.py \
  --config examples/dte/gsm8k_dte.yaml \
  actor.dte.transfer=full

python examples/math/gsm8k_rl.py \
  --config examples/dte/gsm8k_dte.yaml \
  actor.dte.delta_method=snapshot
```

Override the model path, scheduler, and topology for the target environment. Set
`actor.dte.verify_snapshot=true` only when validating AdamW masks because it keeps an
additional CPU snapshot.
