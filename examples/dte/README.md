# DTE Weight Transfer Examples

This folder contains compact GSM8K RL configurations for DTE colocated weight updates.

## Configs

- `gsm8k_dte_full.yaml`: colocated full-weight transfer.
- `gsm8k_dte_delta_adamw.yaml`: first update is full, then DTE delta transfer with AdamW
  inversion.
- `gsm8k_dte_base.yaml`: shared two-GPU GSM8K smoke settings.

The important knobs are:

```yaml
actor:
  dte:
    enabled: true
    transfer: full        # full | delta
    delta_method: adamw   # only used when transfer=delta; snapshot | adamw
    anchor_interval: 0
    verify_snapshot: false
```

## Install

Install AReaL with the DTE dependency:

```bash
python -m pip install -e ".[delta]"
```

## Run

Run AdamW-inversion delta transfer locally:

```bash
python examples/math/gsm8k_rl.py \
  --config examples/dte/gsm8k_dte_delta_adamw.yaml
```

Run full-weight transfer instead:

```bash
python examples/math/gsm8k_rl.py \
  --config examples/dte/gsm8k_dte_full.yaml
```

Use snapshot-based delta detection without adding another config file:

```bash
python examples/math/gsm8k_rl.py \
  --config examples/dte/gsm8k_dte_delta_adamw.yaml \
  actor.dte.delta_method=snapshot
```

Override `scheduler.type`, model paths, and cluster settings for your environment. Set
`actor.dte.verify_snapshot=true` to compare AdamW-inversion masks with snapshot diffing.
