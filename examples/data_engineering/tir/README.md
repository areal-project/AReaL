# TIR data engineering

Generate an exact-answer task dataset independently of the research loop:

```bash
python3 -m areal.v2.data_engineering \
  --config examples/data_engineering/tir/data_generation.yaml \
  --output /tmp/tir-data-job \
  --export-hf /tmp/tir-training-data
```

This CPU example requires AReaL's normal Python dependencies and fresh output
directories. `data_recipe.py` defines generation, cleaning, numeric verification,
selection, and the `messages`/`answer` training view. `data_calibration.py` adapts
existing TIR evaluation reports for calibration. Custom calibration uses the Python
`Calibrator` protocol. See the
[data engineering protocol and usage](../../../docs/en/customization/data_engineering.md).

The example uses four local thread workers with 32 candidates per chunk. To continue an
interrupted job, run the same command with `--resume` and the original output directory.
If HF export already exists, omit `--export-hf`. CPU-heavy recipes can choose
`execution.backend: process`; recipe semantics stay in the example.

Generation uses 16 independent recipe streams. Scoring and eligibility/ranking run in
worker chunks; the coordinator merges family minima and recipe caps through SQLite
indexes. Set `execution.backend: process` for CPU parallelism. Complete shards, model
responses, and rank chunks are reused after failures.
