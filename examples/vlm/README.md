# Pixel Reasoner GRPO

## Install

Use Python 3.12+ and the same CUDA environment on both nodes. From the repository root:

```bash
uv sync --extra cuda
source .venv/bin/activate
uv pip install decord==0.6.0
sudo apt-get update
sudo apt-get install -y ffmpeg
```

The CUDA extra installs the pinned SGLang, Megatron Core, Megatron Bridge and
flash-linear-attention packages. Video inputs also require decord and FFmpeg with FFV1
support. See [installation](../../docs/en/tutorial/installation.md) for CUDA setup.

## Run

Use Qwen3.8-27B with **8 actor GPUs (TP8)** and **8 rollout GPUs (2 replicas × TP4)** on
separate nodes. The native Slurm scheduler allocates one full node for each role.

Place the model, image/frame bundle, train/validation JSONL files and writable
experiment directory on shared storage. Each JSONL row has this format; visual paths are
relative to `data_root`. For video, set `is_video=true` and list the ordered frames in
`image`.

```json
{"question": "Which animal is shown?", "image": ["images/cat.png"], "answer": ["cat"], "is_video": false}
```

Set the shared paths and a fresh trial name in the controller environment:

```bash
export PIXEL_MODEL=/path/to/shared/Qwen3.8-27B
export PIXEL_DATA=/path/to/shared/pixel-bundle
export PIXEL_ROOT=/path/to/shared/pixel-experiment
export PIXEL_TRIAL=trial1
export NO_PROXY='*' no_proxy='*'
unset AREAL_SPMD_MODE
```

Configure the worker container image and any required Slurm reservation/node settings in
`actor.scheduling_spec`; rollout uses the same scheduling spec. Then launch:

```bash
python -m examples.vlm.pixel_reasoner \
  --config examples/vlm/pixel_reasoner_config.yaml \
  trial_name="$PIXEL_TRIAL" \
  actor.path="$PIXEL_MODEL" \
  cluster.fileroot="$PIXEL_ROOT" \
  data_root="$PIXEL_DATA" \
  train_dataset.path="$PIXEL_DATA/train.jsonl" \
  valid_dataset.path="$PIXEL_DATA/validation.jsonl"
```

AReaL submits the actor and SGLang workers through its native Slurm scheduler. The
default batch is 2 questions × 4 samples, with one actor microbatch.
