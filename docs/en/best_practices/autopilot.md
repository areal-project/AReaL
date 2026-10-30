# AReaL Autopilot (Offline Analysis)

The offline Autopilot analyzer turns existing AReaL training configuration and
PerfTracer output into a measured phase summary and a short list of next diagnostics.
It can also include compatible results from `areal.tools.benchmark_fsdp` as tuning
candidates.

The analyzer is read-only: it does not change configuration, start a training job, or
deploy resources. Phase timings show where elapsed time is observed; they do not prove a
GPU, communication, storage, or scheduler root cause.

## Collect and analyze

Enable `perf_tracer.enabled` in the training config, then run the experiment. PerfTracer
writes one `traces-r{rank}.jsonl` file per participating rank. Pass the run's YAML config
and all trace files to the analyzer:

```bash
python -m areal.tools.autopilot \
  --config examples/math/gsm8k_grpo.yaml \
  --traces /path/to/perf_tracer/traces-r*.jsonl \
  --output-dir /path/to/autopilot-report
```

The output directory must not already exist. It contains `report.json` for downstream
processing and `report.md` for review.

## Include FSDP measurements

An optional report from the FSDP train-step benchmark is considered only when the actor
uses FSDP, the allocation world size and node topology match, and the configured model
path matches the benchmark model:

```bash
python -m areal.tools.autopilot \
  --config examples/math/gsm8k_grpo.yaml \
  --traces /path/to/perf_tracer/traces-r*.jsonl \
  --benchmark-report /path/to/fsdp-benchmark/report.json \
  --output-dir /path/to/autopilot-report
```

The benchmark uses synthetic SFT tokens, not the target RL workload. Its candidates are
evidence for a follow-up RL experiment, not an automatic setting or a universal tuning
recommendation.

## How to read the report

Known `train.*` complete events are grouped into rollout, data, training, weight-update,
checkpoint, memory-transfer, evaluation, and instrumentation phases. For each operation,
the slowest rank within a role is used. Sequential operations are combined within each
role, then the slowest role is used for the phase. This avoids adding parallel rank or
role times together. Phase shares are calculated only from recognized phases present in
the input traces. The report lists known spans that were not observed overall and by
role, so missing instrumentation for a role is visible.

Each recommendation links to the observed span that triggered it and suggests the next
measurement to collect. The report lists missing instrumentation so an absent phase is
not mistaken for zero time. Use the existing profiling toolkit to capture representative
CPU/GPU activity before attributing a phase to a particular resource.

When a benchmark report is supplied, candidate rows include the measured throughput,
peak allocated memory, benchmark workload, dtype, attention implementation, optimizer,
device, and provenance. A configuration difference is reported only as a candidate to
validate when the benchmark and training configuration are compatible.
