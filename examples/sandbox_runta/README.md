# Runta sandbox example

This example shows how to use `RuntaRunner` from plain Python code, such as a reward
function, evaluation script, or data-prep utility. [Runta](https://runta.com/docs/)
provides isolated cloud runtimes with Python and Docker preinstalled.

## Install

```bash
uv pip install "runta-sdk>=0.2.6,<0.3"
export RUNTA_TOKEN=your-token
```

`runta login` also works: the SDK reads the config file it writes. Set `RUNTA_ENDPOINT`
to target a non-default API endpoint.

## Run

```bash
uv run python examples/sandbox_runta/reward_example.py
```

## Notes

- `RuntaRunner` is the synchronous entry point for non-async user code;
  `AsyncRuntaRunner` offers the same operations as `astart()`, `arun()` and `aclose()`.
- Each runner instance creates one runtime and keeps it warm across multiple `run()`
  calls. Every call starts a fresh interpreter: variables do not carry over, files do.
- **The default sandbox has only the Python standard library.** Code runs on the
  runtime's own Python, where `import numpy` (or any other third-party package) fails
  with `ModuleNotFoundError: No module named 'numpy'`. `RuntaRunner` installs nothing.
  If the code you run needs packages, either pass `image=...` with a Docker image that
  already contains them, or pass `checkpoint_id=...` of a Runta checkpoint taken from a
  runtime you prepared with the packages installed.
- With `image`, the code runs inside a container started from that image. The image
  needs `python3` and `sh`; `python:3.12-slim` works but, like the default sandbox,
  ships no third-party packages.
- `network_block_all=True` closes outbound network access once the sandbox is ready.
- A program that exceeds its timeout is terminated and reported with exit code 124.
- A runtime is held on a renewable lease: Runta shuts it down `lease_seconds` (default
  ten minutes) after the last renewal, and a live runner renews about every
  `lease_seconds / 3`. A runner whose process was killed therefore loses its runtime
  within `lease_seconds` instead of leaving it running. `lease_seconds` must be longer
  than the runtime takes to prepare (the image pull).
- `max_lifetime` (default one hour) is the hard cap on a runtime's age, enforced by
  Runta: the lease is never renewed past it, and the next `run()` after it starts a
  fresh runtime.
