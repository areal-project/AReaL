# boxd sandbox example

This example shows how to use `BoxdRunner` from plain Python code, such as a reward
function, evaluation script, or data-prep utility. [boxd](https://boxd.sh?ref=AReaL)
runs each sandbox as a full Ubuntu VM.

## Install

```bash
uv sync --extra sandbox
export BOXD_API_KEY=your-key
export BOXD_REGION=fra  # optional: fra or nyc
```

Create a key with `boxd auth keys create areal` or in the boxd console. See the
[Python SDK docs](https://docs.boxd.sh/reference/python-sdk?ref=AReaL).

## Run

```bash
uv run python examples/sandbox_boxd/reward_example.py
```

## Notes

- `BoxdRunner` is the synchronous entry point for non-async user code.
- Each runner instance creates one machine and keeps it warm across multiple `run()`
  calls. Every `run()` starts a fresh Python process. `close()` deletes the machine.
- Machines are created with `isolated=True`: no private network and no credentials
  inside. The API key stays in the AReaL process.
- A run that passes its timeout is killed and returns exit code 124.
- `python_packages=["numpy", "sympy"]` installs packages into a venv in the machine when
  it starts. For many sandboxes, prepare one machine, save it as a snapshot and pass
  `snapshot="name"` instead.
- Machines carry the label `areal-sandbox=true`. `close()` deletes the machine, but a
  killed process cannot clean up. `delete_after` (seconds) has boxd delete the machine
  that long after it was created, even if it is still in use. Pick a value above your
  longest job, for example `delete_after=6 * 3600`.
- Other options: `org`, `region`, `vcpu`, `memory`, `env_vars`, `labels`, `name_prefix`.
