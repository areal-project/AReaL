# Pixel Reasoner rollouts

Collect multi-turn image/video trajectories with a 27B model through AReaL's
`RemoteSGLangEngine`. The workflow supports image cropping, video frame selection, and
preserving reasoning in subsequent turns. It performs inference only.

## Environment

Use the GPU environment from [the quickstart](../../docs/en/tutorial/quickstart.md),
with this checkout installed or on `PYTHONPATH`.

- Use the repository's SGLang pin, **0.5.10.post1**; newer versions may break AReaL's
  server extension.
- The model's `AutoProcessor` must support `reasoning_content` and
  `preserve_thinking=True`.
- Video additionally requires **decord 0.6.0** and **FFmpeg with FFV1 support** on
  `PATH`. Install these inside the execution container. For Debian/Ubuntu:

```bash
python -m pip install decord==0.6.0
apt-get update
apt-get install -y ffmpeg
```

An existing FFmpeg binary can instead be added to `PATH`. Model/data paths must be
readable and output/cache/name-resolve directories writable inside the container. A
four-task image/video smoke passed with Python 3.12, PyTorch 2.9.1+cu129, Transformers
5.3.0, FFmpeg 7.0.2, TP=2 and concurrency 2.

## Data

Prepare a bundle containing `manifest.jsonl` and relative image/frame paths:

```json
{"source_index": 0, "qid": "example", "question": "What is shown?", "image": ["images/example.png"], "answer": ["cat"], "is_video": false}
```

For video tasks, set `is_video=true` and put the ordered frame paths in `image`. Frames
use lossless video transport at a synthetic 2 fps; odd counts repeat the final frame for
temporal patching. Original timestamps are unavailable.

## Run

The default configuration uses eight GPUs: four TP=2 replicas, 64 concurrent episodes
globally, a 256K context limit, and at most 32 tasks. Run inside an existing cluster
allocation/container. Start the model service from the repository root:

```bash
python -m areal.infra.launcher.sglang_server \
  --config examples/vlm/pixel_reasoner_config.yaml \
  tokenizer_path=/path/to/27B-model
```

In a second shell with the same environment and name-resolve directory:

```bash
python examples/vlm/pixel_reasoner.py \
  --config examples/vlm/pixel_reasoner_config.yaml \
  tokenizer_path=/path/to/27B-model data_root=/path/to/bundle \
  dump_dir=/path/to/shared/pixel-traces
```

Keep `allocation_mode` and `rollout.backend` DP/TP settings consistent. Set `limit=null`
for the full manifest; stop the server after collection. An existing compatible SGLang
service can be supplied through `AREAL_LLM_SERVER_ADDRS=host:port` (comma-separated for
multiple replicas). Otherwise the collector waits up to `rollout.setup_timeout`.

## Behavior and outputs

- `crop_image_normalized` uses normalized boxes and 1-based image indices.
  `select_frames` accepts at most eight 1-based original frame indices. After video
  tools return images, index 1 is the video and tool images start at index 2, following
  the official Pixel-Reasoner workflow at commit
  `a3a6c3a557a87d89ffaec825bd8c8d09b719fb97`.
- `gconfig.max_new_tokens` limits one response; `gconfig.max_tokens` limits prompt plus
  response; `max_output_tokens` limits cumulative episode output. `max_turns=null`
  disables the turn limit. Prompt lengths are client processor estimates.
- `rollout.max_concurrent_rollouts` is the global episode ceiling. `initial_concurrency`
  can start lower; optional `concurrency_control` enables live admission changes and
  controller heartbeats.
- Each attempt saves `episode.json` and visual assets under a UUID directory. Traces
  include messages, raw output, reasoning, token IDs, tool calls, timing and status.
  Reasoning is passed into subsequent turns; failed and truncated attempts are retained.
- The verifier normalizes and compares the boxed answer with a reference, reporting
  `matched` or `needs_review`. It is a basic verifier, not the full Pixel reward.
  Repeated runs create new attempts; resume, deduplication and dataset selection are
  external to this example.

Video currently supports the SGLang adapter only. Run a small smoke test in your actual
runtime before full collection; the existing smoke does not establish model accuracy or
portability across environments.
