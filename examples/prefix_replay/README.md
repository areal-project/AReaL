# Prefix replay on-policy distillation

Replay an offline trajectory up to an assistant turn, generate one fresh student action,
then score that action with MOPD teachers. The recorded target action is excluded from
the prefix. No external agent environment is started, and scalar environment rewards are
zero.

## Local recipe

`local.yaml` follows the existing eight-GPU Megatron/SGLang colocated MOPD example. It
requires a compatible AReaL runtime, local model checkpoints and a shared
student/teacher tokenizer. Set the following environment variables for your setup:

- `MOPD_STUDENT_MODEL_PATH`: student checkpoint.
- `MOPD_TEACHER_MODEL_PATH`: frozen teacher checkpoint.
- `PREFIX_REPLAY_DATA_PATH`: offline JSON/JSONL/Parquet data or dataset directory.
- `AREAL_IMAGE`: worker runtime image.
- `AREAL_ADMIN_API_KEY`: proxy administration key.
- `AREAL_FILEROOT`: optional experiment output directory.

```bash
python -m examples.prefix_replay.train_opd \
  --config examples/prefix_replay/local.yaml
```

Adjust topology and batch sizes to your hardware. This recipe has not been GPU validated
as part of the migration. The core example uses MOPD even for one teacher; legacy
`teacher.train` configurations require separate compatibility validation.

## Data and sampling

Each record may contain `messages` or SWE-style `conversations[-1].messages`:

```json
{"instance_id":"task-1","messages":[{"role":"user","content":"Solve this task."},{"role":"assistant","content":"Recorded answer."}]}
```

The loader preserves system messages by default, consecutive user messages, tools and
other record metadata. `input_mode` accepts `auto`, `trajectory` and `prefix`.
Trajectory candidates are sampled with probability `kappa**t` and a fixed seed.
Candidates end before the selected assistant action and must fit the rendered prompt
budget, with room for at least one student token. Configure template options under
`rollout.agent.chat_template_kwargs` to match generation.

`gconfig.n_samples` is handled by grouped rollout; each proxy request has `n=1`.
Length-truncated and zero-token actions fail the rollout. The recipe sets
`drop_incomplete_group: true`. Do not configure a reward-based acceptance filter for
this zero-reward task.

## Teacher routing

Use main's `train_dataset.sources[].teacher_group` and `mopd.teacher_groups`. Multiple
sources can point to different files. A teacher initialized from the student checkpoint
can use `path: ${actor.path}`; it remains a frozen MOPD teacher.

An existing mixed file can be selected by its per-record route without expanding all
prefix histories:

```yaml
train_dataset:
  sources:
    - path: ${oc.env:PREFIX_REPLAY_DATA_PATH}
      teacher_group: game
      dataset_kwargs: {route_field: task_type, route_value: game}
    - path: ${oc.env:PREFIX_REPLAY_DATA_PATH}
      teacher_group: code
      dataset_kwargs: {route_field: task_type, route_value: code}
```

Define both teacher groups in `mopd.teacher_groups`. Explicit row fields take precedence
over `prefix_replay.route_metadata_field` and `default_route`. An empty selection fails
before training. Configure every intended route explicitly: rows outside the selected
route are excluded. Main applies source-level proportional or uniform mixing, so source
ordering differs from the old single mixed-file recipe. Within each source, prefix
selection retains the seeded sampling stream.

`prepare_routed_replay.py` can attach route labels and merge input JSONL sources; run
its `--help` for source, worker and output options. Serial and parallel output are
deterministic, and conflicting labels and source/output aliases are rejected.

## Processed cache

The compact cache stores trajectories and prefix indices separately. Cache keys track
data fingerprints, tokenizer path, template, length and sampling options. Leases prevent
two trainers from using the same cache concurrently. Use a unique `trial_name` for
concurrent runs. Set `PREFIX_REPLAY_DISABLE_CACHE=1` to rebuild in memory, or
`prefix_replay.cleanup_processed_dataset=true` to remove caches after successful
training.

Experience conditioning, distributed preprocessing and Slurm-specific drivers are
separate migration PRs. Their absence in this core PR is tracked in the migration
checklist.
